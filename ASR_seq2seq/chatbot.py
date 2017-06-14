from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("from_asr_train_data", "./data/train_asr.enc", "ASR training data.")
tf.app.flags.DEFINE_string("from_train_data", "./data/train.enc", "Training data.")
tf.app.flags.DEFINE_string("to_train_data", "./data/train.dec", "Training data.")
tf.app.flags.DEFINE_string("from_asr_dev_data", "./data/test_asr.enc", "ASR testing data.")
tf.app.flags.DEFINE_string("from_dev_data", "./data/test.enc", "Testing data.")
tf.app.flags.DEFINE_string("to_dev_data", "./data/test.dec", "Testing data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("attention", False,
                            "Set to True for attention decoder.")
tf.app.flags.DEFINE_boolean("pretrain", False,
                            "Set to True for pretraining.")
tf.app.flags.DEFINE_boolean("train_encoder", False,
                            "Set to True for training ASR encoder.")
tf.app.flags.DEFINE_boolean("fine_tune", False,
                            "Set to True to fine-tune ASR seq2seq.")
tf.app.flags.DEFINE_boolean("use_asr", False,
                            "Set to True to use asr_encoder_state for decoding.")
tf.app.flags.DEFINE_boolean("schedule_sampling", False,
                            "Set to True for schedule sampling.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for testing.")
tf.app.flags.DEFINE_boolean("predict", False,
                            "Set to True for predicting result of testing data")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, asr_source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  asr_data_set = [[] for _ in _buckets]

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, asr_source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  asr_data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(asr_source_path, mode="r") as asr_source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        source = source_file.readline()
        asr_source = asr_source_file.readline()
        target = target_file.readline()
        counter = 0
        while source and asr_source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          asr_source_ids = [int(x) for x in asr_source.split()]
          target_ids = [int(x) for x in target.split()]
          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            # 同時讀original以及asr data，只有當同一個pair的data被分到同一bucket時才會加進dataset裡
            # 統計的結果是training data 爲98000/110000，而testing data則是27000/30000，感覺還行
            if len(source_ids) < source_size and len(asr_source_ids) < source_size \
                                             and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              asr_data_set[bucket_id].append([asr_source_ids, target_ids])
              break
          source = source_file.readline()
          asr_source = asr_source_file.readline()
          target = target_file.readline()
  return data_set, asr_data_set

def create_model(session, forward_only, attention=False, pretrain=False,
                 train_encoder=False, fine_tune=False,
                 use_asr=False, schedule_sampling=False):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      attention=attention,
      pretrain=pretrain,
      train_encoder=train_encoder,
      fine_tune=fine_tune,
      use_asr=use_asr,
      schedule_sampling=schedule_sampling,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter("./log/",graph=session.graph)
  return model

def train():
  data = data_utils.prepare_data(
      FLAGS.data_dir,
      FLAGS.from_train_data,
      FLAGS.from_asr_train_data,
      FLAGS.to_train_data,
      FLAGS.from_dev_data,
      FLAGS.from_asr_dev_data,
      FLAGS.to_dev_data,
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size)
  from_train = data[0]
  from_asr_train = data[1]
  to_train = data[2]
  from_dev = data[3]
  from_asr_dev = data[4]
  to_dev = data[5]

  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, forward_only=False, attention=FLAGS.attention,
                         pretrain=FLAGS.pretrain, train_encoder=FLAGS.train_encoder,
                         fine_tune=FLAGS.fine_tune, use_asr=FLAGS.use_asr,
                         schedule_sampling=FLAGS.schedule_sampling)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set, asr_dev_set = read_data(from_dev, from_asr_dev, to_dev)
    train_set, asr_train_set = read_data(from_train, from_asr_train, to_train,
                                         FLAGS.max_train_data_size)

    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss, cvl = 0.0, 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      asr_encoder_inputs, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, asr_train_set, bucket_id)
      _, step_loss, step_cvl, _ = model.step(sess, asr_encoder_inputs, encoder_inputs,
                                             decoder_inputs, target_weights, bucket_id,
                                             forward_only=False, pretrain=FLAGS.pretrain,
                                             train_encoder=FLAGS.train_encoder,
                                             fine_tune=FLAGS.fine_tune)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      cvl += step_cvl / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f MSE %.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                  step_time, perplexity, cvl))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "chatbot.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss, cvl = 0.0, 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          asr_encoder_inputs, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, asr_dev_set, bucket_id)
          _, eval_loss, eval_cvl, _ = model.step(sess, asr_encoder_inputs, encoder_inputs,
                                                 decoder_inputs, target_weights, bucket_id,
                                                 forward_only=True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
          print("  eval: bucket %d perplexity %.2f MSE %.2f" % (bucket_id, eval_ppx, eval_cvl))
        sys.stdout.flush()

def test():
  data = data_utils.prepare_data(
      FLAGS.data_dir,
      FLAGS.from_train_data,
      FLAGS.from_asr_train_data,
      FLAGS.to_train_data,
      FLAGS.from_dev_data,
      FLAGS.from_asr_dev_data,
      FLAGS.to_dev_data,
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size)
  from_dev = data[3]
  from_asr_dev = data[4]
  to_dev = data[5]

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, forward_only=True, attention=FLAGS.attention,
                         pretrain=FLAGS.pretrain, use_asr=FLAGS.use_asr)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set, asr_dev_set = read_data(from_dev, from_asr_dev, to_dev)

    loss, cvl = 0.0, 0.0
    steps = 100
    for bucket_id in range(len(_buckets)):
      if len(dev_set[bucket_id]) == 0:
        print("  eval: empty bucket %d" % (bucket_id))
        continue
      bucket_loss, bucket_cvl = 0.0, 0.0
      for i in range(steps):
        asr_encoder_inputs, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            dev_set, asr_dev_set, bucket_id)
        _, step_loss, step_cvl, _ = model.step(sess, asr_encoder_inputs, encoder_inputs,
                                               decoder_inputs, target_weights, bucket_id,
                                               forward_only=True)
        bucket_loss += step_loss / steps
        bucket_cvl += step_cvl / steps
      bucket_ppx = math.exp(float(bucket_loss)) if bucket_loss < 300 else float("inf")
      print("  test: bucket %d perplexity %.2f MSE %.2f" % (bucket_id, bucket_ppx, bucket_cvl))
      loss += bucket_loss / len(_buckets)
      cvl += bucket_cvl / len(_buckets)
    ppx = math.exp(float(loss)) if loss < 300 else float("inf")
    print(" test: perplexity %.2f MSE %.2f" % (ppx, cvl))
    sys.stdout.flush()

def predict():
  data = data_utils.prepare_data(
      FLAGS.data_dir,
      FLAGS.from_train_data,
      FLAGS.from_asr_train_data,
      FLAGS.to_train_data,
      FLAGS.from_dev_data,
      FLAGS.from_asr_dev_data,
      FLAGS.to_dev_data,
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size)
  from_dev = data[3]
  from_asr_dev = data[4]
  to_dev = data[5]

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, forward_only=True, attention=FLAGS.attention,
                         pretrain=FLAGS.pretrain, use_asr=FLAGS.use_asr)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set, asr_dev_set = read_data(from_dev, from_asr_dev, to_dev)

    for bucket_id in range(len(_buckets)):
      if len(dev_set[bucket_id]) == 0:
        print("  eval: empty bucket %d" % (bucket_id))
        continue
      asr_encoder_inputs, encoder_inputs, decoder_inputs, target_weights = model.get_all(
            dev_set, asr_dev_set, bucket_id)
      model.batch_size = np.array(asr_encoder_inputs).shape[1]
      _, _, _, output_logits = model.step(sess, asr_encoder_inputs, encoder_inputs,
                                               decoder_inputs, target_weights, bucket_id,
                                               forward_only=True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [np.argmax(logit, axis=1) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      for i in range(len(outputs)):
        if data_utils.EOS_ID in outputs[i]:
          outputs[i] = outputs[i][:outputs[i].index(data_utils.EOS_ID)]
          print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))

def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, forward_only=True, attention=FLAGS.attention,
                         use_asr=FLAGS.use_asr)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      asr_encoder_inputs, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, _, output_logits = model.step(sess, asr_encoder_inputs, encoder_inputs,
                                          decoder_inputs, target_weights, bucket_id, True)
      rint(output_logits[0].shape)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in range(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.test:
    test()
  elif FLAGS.predict:
    predict()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
