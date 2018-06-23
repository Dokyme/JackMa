import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import requests
import json

from util import *

def read_data():
    # 数据读取
    users = pd.read_csv("user_profile_table.csv")
    balances = pd.read_csv("user_balance_table.csv", parse_dates=[
        "report_date"], date_parser=lambda date: pd.datetime.strptime(date, "%Y%m%d"))
    shibors = pd.read_csv("mfd_bank_shibor.csv", parse_dates=[
        "mfd_date"], date_parser=lambda date: pd.datetime.strptime(date, "%Y%m%d"))
    interests = pd.read_csv("mfd_day_share_interest.csv", parse_dates=[
        "mfd_date"], date_parser=lambda date: pd.datetime.strptime(date, "%Y%m%d"))


# f0 = pd.read_csv("data/data.csv",
#                  parse_dates=["report_date"],
#                  date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
# f0 = f0.set_index("report_date")
# f1 = open("data/train_norm.csv", "rb")
# seq = np.loadtxt(f1, delimiter=',', skiprows=1, usecols=(
#     1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29))
# seq = np.array(seq)
# seq = seq.reshape((260, 1, 27))
# tseq = seq[:-31, :, :]
# f2 = open("data/res.csv", "rb")
# res = np.loadtxt(f2, delimiter=',', skiprows=1, usecols=(1, 2))
# res = np.array(res)
# res = res.reshape((260, 1, 2))
# tres = res[:-31, :, :]


def reshape_features_to_input_cell():
    return


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (52batch, 5steps)
    # xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    # seq = np.sin(xs)
    # res = np.cos(xs)
    xs = None

    sseq = tseq[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE,
                :, :].reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    rres = tres[BATCH_START:BATCH_START + TIME_STEPS * BATCH_SIZE,
                :, 1:].reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE))

    BATCH_START += TIME_STEPS
    return [sseq, rres, xs]


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


if __name__ == '__main__':
    # time_features = pd.read_csv("data/new_time_features.csv",
    #                             parse_dates=["report_date"],
    #                             date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
    # time_features.reset_index(inplace=True)
    x = get_time_features()
    x_length = x.shape[0]
    y = get_labeled_value()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    # time_features = time_features.values[:, :].reshape(
    #     (int(time_features.shape[0]), 1, 12))[
    #     :-31, :, :]
    # f0["total_purchase_amt"] = (f0["total_purchase_amt"] - f0["total_purchase_amt"].min()) / (
    #     f0["total_purchase_amt"].max() - f0["total_purchase_amt"].min())
    # tres = f0["total_purchase_amt"].values[:].reshape(
    #     (int(f0["total_purchase_amt"].shape[0]), 1, 1))[
    #     :-31, :, :]
    for j in range(ITR_STEP):
        for i in range(0, x_length-TIME_STEPS*BATCH_SIZE, TIME_STEPS):
            xs = x[i:i+TIME_STEPS*BATCH_SIZE, :,
                   :].reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
            ys = y[i:i+TIME_STEPS*BATCH_SIZE, :,
                   :].reshape((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE))
            if i == 0:
                feed_dict = {
                    model.xs: xs,
                    model.ys: ys,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: xs,
                    model.ys: ys,
                    model.cell_init_state: state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)

        if j % 20 == 0:
            print('%d\tcost:%f' % (j, round(cost, 4)))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
    # for j in range(2000):
    #     for i in range(0, 60 - TIME_STEPS * BATCH_SIZE, BATCH_SIZE):
    #         mseq = time_features[180 + i:180 + i + TIME_STEPS * BATCH_SIZE,
    #                              :, :].reshape((BATCH_SIZE, TIME_STEPS, 12))
    #         mres = tres[180 + i:180 + i + TIME_STEPS * BATCH_SIZE,
    #                     :, :].reshape((BATCH_SIZE, TIME_STEPS, 1))
    #         # mseq, mres, xs = get_batch()
    #         if i == 0:
    #             feed_dict = {
    #                 model.xs: mseq,
    #                 model.ys: mres,
    #                 # create initial state
    #             }
    #         else:
    #             feed_dict = {
    #                 model.xs: mseq,
    #                 model.ys: mres,
    #                 model.cell_init_state: state  # use last state as the initial state for this run
    #             }

    #         _, cost, state, pred = sess.run(
    #             [model.train_op, model.cost, model.cell_final_state, model.pred],
    #             feed_dict=feed_dict)
        # if j % 20 == 0:
        #     print('%d\tcost:%f' % (j, round(cost, 4)))
        #     result = sess.run(merged, feed_dict)
        #     writer.add_summary(result, i)

    # plt.plot(preds)
    # plt.show
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
