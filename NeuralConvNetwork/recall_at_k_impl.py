import tensorflow as tf
import numpy as np

class streaming_recall_at_k(object):
    def __init__(self, k_list):
        self.k_list = k_list
        self.num_examples = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.num_corrects = [tf.Variable(0.0, trainable=False, dtype=tf.float32)
                             for _ in range(len(k_list))]
        self.recall_inputs = tf.placeholder(tf.float32, shape=[None, 10],
                                            name="recall_inputs")
        self.recall_labels = tf.placeholder(tf.int32, shape=[None],
                                            name="recall_labels")

        self.updated_num_examples = tf.assign_add(self.num_examples,
                                        tf.to_float(tf.shape(self.recall_inputs)[0]))
        self.updated_num_corrects = [0 for _ in range((len(k_list)))]
        for i, kk in enumerate(self.k_list):
            out = tf.nn.in_top_k(self.recall_inputs, self.recall_labels, kk)
            self.updated_num_corrects[i] = tf.assign_add(self.num_corrects[i],
                                                tf.reduce_sum(tf.to_float(out)))

        self.outputs = [x / self.updated_num_examples for x in self.updated_num_corrects]

    def evaluate(self, sess, inputs, labels):
        results = sess.run([self.outputs, self.updated_num_examples],
                            feed_dict={self.recall_inputs: inputs,
                                       self.recall_labels: labels})
        return results

if __name__ == "__main__":
    np.random.seed(0)
    labels = np.zeros(64)
    with tf.Session() as sess:
        metric = streaming_recall_at_k([1, 2, 5])
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for _ in range(100):
            predictions = np.random.rand(64, 10)
            res = metric.evaluate(sess, predictions, labels)
            print(res[0], res[1])
