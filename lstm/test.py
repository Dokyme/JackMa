import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

from util import *


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(
                tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(
                tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self, ):
        # (batch*n_step, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(
            l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(
                self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(
            self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def restore():
    tf.reset_default_graph()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    # saver = tf.train.import_meta_graph('model.cpkt.meta')
    saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('./logs'))
    saver.restore(sess, "model.ckpt")
    return sess, model


def plot(y, predicts):
    y = y.reshape(-1)
    plt.plot(y, label="labeled")
    plt.plot(predicts, label="predict")
    plt.legend()
    plt.show()
    print("MSE:%f" % (metrics.mean_squared_error(y, predicts)))
    print("RMSE:%f" % np.sqrt(metrics.mean_squared_error(y, predicts)))


sess, model = restore()
x = get_time_features("test")
y = get_labeled_value("test")
predicts = []
x_length = x.shape[0]
for i in range(0, x_length-TIME_STEPS*BATCH_SIZE):
    xs = x[i:i+TIME_STEPS*BATCH_SIZE, :,
           :].reshape(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
    p = sess.run(model.pred, feed_dict={model.xs: xs})
    predicts.append(p[-1])
    print(p)
plot(y, predicts)
# f0 = pd.read_csv("data/data.csv",
#                  parse_dates=["report_date"],
#                  date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
# f0 = f0.set_index("report_date")
# time_features = pd.read_csv("data/time_features.csv",
#                             parse_dates=["11"],
#                             date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
# time_features.drop("11", axis=1, inplace=True)
# time_features.reset_index(inplace=True)


# f1 = open("data/train_norm.csv", "rb")
# seq = np.loadtxt(f1, delimiter=',', skiprows=1, usecols=(
#     1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29))
# seq = np.array(seq)
# seq = seq.reshape((260, 1, 27))
# f2 = open("data/res.csv", "rb")
# res = np.loadtxt(f2, delimiter=',', skiprows=1, usecols=(1, 2))
# res = np.array(res)

# time_features = time_features.values.reshape(
#     (int(time_features.shape[0]), 1, 12))
# f0["total_purchase_amt"] = (f0["total_purchase_amt"] - f0["total_purchase_amt"].min()) / (
#     f0["total_purchase_amt"].max() - f0["total_purchase_amt"].min())
# tres = f0["total_purchase_amt"].values
# preds = []
# for i in range(-31 - TIME_STEPS - 1, -1 - TIME_STEPS):
#     xs = time_features[i:i + TIME_STEPS, :,
#                        :].reshape((-1, TIME_STEPS, INPUT_SIZE))
#     pred = sess.run(model.pred, feed_dict={model.xs: xs})
#     preds.append(pred[-1])
#     print(pred)

# plt.plot(tres[180:], label="label")
# plt.plot(preds, label="pred")
# plt.legend()
# plt.show()
# print("MSE:%f" % (metrics.mean_squared_error(res, preds)))
# print("RMSE:%f" % np.sqrt(metrics.mean_squared_error(res, preds)))
