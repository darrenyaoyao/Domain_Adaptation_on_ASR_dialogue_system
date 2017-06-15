from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

import data_utils
import seq2seq

class Seq2SeqModel(object):
  def __init__(self,
                source_vocab_size,
                target_vocab_size,
                buckets,
                size,
                num_layers,
                max_gradient_norm,
                batch_size,
                learning_rate,
                learning_rate_decay_factor,
                use_lstm=False,
                num_samples=512,
                forward_only=False,
                attention=False,
                pretrain=False,
                train_encoder=False,
                fine_tune=False,
                use_asr=False,
                schedule_sampling=False,
                dtype=tf.float32):
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self.global_step_incre = tf.assign_add(self.global_step, tf.constant(1))

    # If we use sampled softmax, we need an output projection
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size,],dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
                tf.nn.sampled_softmax_loss(
                    weights = local_w_t,
                    biases = local_b,
                    labels = labels,
                    inputs = local_inputs,
                    num_sampled = num_samples,
                    num_classes = self.target_vocab_size),
                    dtype)
      softmax_loss_function = sampled_loss

      # Creat the internal multi-layer cell for our RNN.
      def single_cell():
        return tf.contrib.rnn.GRUCell(size)
      if use_lstm:
        def single_cell():
          return tf.contrib.rnn.BasicLSTMCell(size)
      cell = single_cell()
      if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

      # The seq2seq function: we use embedding for the input and attention.
      # seq2seq function多了一個asr_encoder_inputs
      def seq2seq_f(asr_encoder_inputs, encoder_inputs, decoder_inputs,
                    do_decode, use_asr, schedule_sampling):
        if attention:
          print("Attention Model")
          return seq2seq.embedding_attention_seq2seq(
            asr_encoder_inputs,
            encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols=source_vocab_size,
            num_decoder_symbols=target_vocab_size,
            embedding_size=size,
            output_projection=output_projection,
            feed_previous=do_decode,
            dtype=dtype)
        else:
          print("Simple Model")
          return seq2seq.embedding_rnn_seq2seq(
            asr_encoder_inputs,
            encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols=source_vocab_size,
            num_decoder_symbols=target_vocab_size,
            embedding_size=size,
            use_asr=use_asr,
            schedule_sampling=schedule_sampling,
            global_step=self.global_step,
            output_projection=output_projection,
            feed_previous=do_decode,
            dtype=dtype)

      # Feeds for inputs.
      # 幫asr_encoder_inputs建placeholder
      self.asr_encoder_inputs = []
      self.encoder_inputs = []
      self.decoder_inputs = []
      self.target_weights = []
      with variable_scope.variable_scope("encoder"):
        for i in range(buckets[-1][0]):
          self.asr_encoder_inputs.append(tf.placeholder(tf.int32, shape=[None,],
                                                  name="asr_encoder{0}".format(i)))
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None,],
                                                  name="encoder{0}".format(i)))
      with variable_scope.variable_scope("decoder"):
        for i in range(buckets[-1][1]+1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None,],
                                                  name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None,],
                                                  name="weight{0}".format(i)))

      # Our targets are decoder inputs shifted by one.
      targets = [self.decoder_inputs[i+1]
                for i in range(len(self.decoder_inputs) - 1)]

      # Set use_asr to false when pretraining
      if pretrain:
        use_asr = False
      if train_encoder:
        use_asr = True
      if fine_tune:
        use_asr=True

      # Training outputs and losses.
      if forward_only:
        # model_with_buckets input多了self.asr_encoder_inputs，output則是self.context_vector_losses
        # seq2seq_f多了一個input
        self.outputs, self.losses, self.context_vector_losses = seq2seq.model_with_buckets(
            self.asr_encoder_inputs, self.encoder_inputs, self.decoder_inputs,
            targets, self.target_weights, buckets,
            lambda x, y, z: seq2seq_f(x, y, z, True, use_asr, schedule_sampling),
            softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
          for b in range(len(buckets)):
            self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
            ]
      else:
        self.outputs, self.losses, self.context_vector_losses = seq2seq.model_with_buckets(
            self.asr_encoder_inputs, self.encoder_inputs, self.decoder_inputs,
            targets, self.target_weights, buckets,
            lambda x, y, z: seq2seq_f(x, y, z, False, use_asr, schedule_sampling),
            softmax_loss_function=softmax_loss_function)

      # Gradients and SGD update operation for training the model.
      asr_encoder_name = 'embedding_rnn_seq2seq/asr_encoder'
      encoder_name = 'embedding_rnn_seq2seq/original_encoder'

      if pretrain:
        params = [v for v in tf.trainable_variables()]
      elif train_encoder:
        params = [v for v in tf.trainable_variables() if v.name.startswith(asr_encoder_name)]
      elif fine_tune:
        params = [v for v in tf.trainable_variables() if not v.name.startswith(encoder_name)]

      if not forward_only:
        self.gradient_norms = []
        self.gradient_norms_cvl = []
        self.updates = []
        self.updates_cvl = []
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for b in range(len(buckets)):
          # 分別對self.losses跟self.context_vector_losses作gradient
          gradients = tf.gradients(self.losses[b], params)
          gradients_cvl = tf.gradients(self.context_vector_losses[b], params)
          clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
          clipped_gradients_cvl, norm_cvl = tf.clip_by_global_norm(gradients_cvl, max_gradient_norm)
          self.gradient_norms.append(norm)
          self.gradient_norms_cvl.append(norm_cvl)
          self.updates.append(opt.apply_gradients(
              zip(clipped_gradients, params)))
          self.updates_cvl.append(opt.apply_gradients(
              zip(clipped_gradients_cvl, params)))

      self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, asr_encoder_inputs, encoder_inputs, decoder_inputs,
               target_weights, bucket_id, forward_only, pretrain=False,
               train_encoder=False, fine_tune=False):
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(asr_encoder_inputs) != encoder_size:
      raise ValueError("ASR encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(asr_encoder_inputs), encoder_size))
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.asr_encoder_inputs[l].name] = asr_encoder_inputs[l]
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      if pretrain:
        output_feed = [self.updates[bucket_id],  # Update Op for self.losses that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm for self.losses.
                       self.losses[bucket_id],  # Loss for this batch.
                       self.context_vector_losses[bucket_id]]
      elif train_encoder:
        output_feed = [self.updates_cvl[bucket_id],
                       self.gradient_norms_cvl[bucket_id],
                       self.losses[bucket_id],
                       self.context_vector_losses[bucket_id]]
      elif fine_tune:
        output_feed = [self.updates[bucket_id],
                       self.updates_cvl[bucket_id],
                       self.gradient_norms[bucket_id],
                       self.gradient_norms_cvl[bucket_id],
                       self.losses[bucket_id],
                       self.context_vector_losses[bucket_id]]
      output_feed.append(self.global_step_incre)
    else:
      output_feed = [self.losses[bucket_id],  # Loss for this batch.
                     self.context_vector_losses[bucket_id]]  # Context vector loss for this batch.
    for l in range(decoder_size):  # Output logits.
      output_feed.append(self.outputs[bucket_id][l])
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      if fine_tune:
        return outputs[2], outputs[4], outputs[5], None  # Gradient norm, loss, context vector loss, no outputs.
      else:
        return outputs[1], outputs[2], outputs[3], None  # Gradient norm, loss, context vector loss, no outputs.
    else:
      return None, outputs[0], outputs[1], outputs[2:]  # No gradient norm, loss, context vector loss, outputs.

  def get_batch(self, data, asr_data, bucket_id):
    encoder_size, decoder_size = self.buckets[bucket_id]
    asr_encoder_inputs, encoder_inputs, decoder_inputs = [], [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in range(self.batch_size):
      #這裡是先從0~len(data[bucket_id])中隨機選出一個index
      #然後根據這個index找出一筆original的encoder_input，decoder_input，以及對應的asr_encoder_input
      #後面asr_encoder_input的pad跟reindex都跟原來一樣
      idx = random.randrange(len(data[bucket_id]))
      encoder_input, decoder_input = data[bucket_id][idx]
      asr_encoder_input = asr_data[bucket_id][idx][0]
      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # ASR encoder inputs are padded and then reversed.
      asr_encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(asr_encoder_input))
      asr_encoder_inputs.append(list(reversed(asr_encoder_input + asr_encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_asr_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

    # Batch ASR encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      tmp = []
      batch_asr_encoder_inputs.append(
            np.array([asr_encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
      # We set weight to 0 if the corresponding target is a PAD symbol.
      # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_asr_encoder_inputs, batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def get_all(self, data, asr_data, bucket_id):
    encoder_size, decoder_size = self.buckets[bucket_id]
    asr_encoder_inputs, encoder_inputs, decoder_inputs = [], [], []

    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for i in range(len(data[bucket_id])):
      encoder_input, decoder_input = data[bucket_id][i]
      asr_encoder_input = asr_data[bucket_id][i][0]
      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # ASR encoder inputs are padded and then reversed.
      asr_encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(asr_encoder_input))
      asr_encoder_inputs.append(list(reversed(asr_encoder_input + asr_encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create major vectors from the data selected above.
    all_encoder_inputs, all_asr_encoder_inputs, all_decoder_inputs, all_weights = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      all_encoder_inputs.append(
            np.array([encoder_inputs[idx][length_idx]
                    for idx in range(len(data[bucket_id]))], dtype=np.int32))

    # Batch ASR encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      tmp = []
      all_asr_encoder_inputs.append(
            np.array([asr_encoder_inputs[idx][length_idx]
                    for idx in range(len(data[bucket_id]))], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
      all_decoder_inputs.append(
            np.array([decoder_inputs[idx][length_idx]
                    for idx in range(len(data[bucket_id]))], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      weight = np.ones(len(data[bucket_id]), dtype=np.float32)
      for idx in range(len(data[bucket_id])):
      # We set weight to 0 if the corresponding target is a PAD symbol.
      # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          weight[idx] = 0.0
      all_weights.append(weight)
    return all_asr_encoder_inputs, all_encoder_inputs, all_decoder_inputs, all_weights
