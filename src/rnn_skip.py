import math
import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
from keras.utils import to_categorical
import os
import sys
sys.path.append('../')
from collections import Counter
from tools.io import load_obj, store_obj
from tools.playlists import RecSysHelper
rec_helper = RecSysHelper('../../../workspace/data/')

class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)
    
def calc_steps_per_epoch(data, seq_len, step):
    count = 0
    for d in data:
        max_len = len(d)
        i = 0
        while i + seq_len < max_len:
            count += 1
            i += step
    return count

class BatchGenerator(object):

    def __init__(self, data, seq_length, n_batch_size, n_vocab, step=5, test=False, store_folder='step_point/'):
        """
        data: can be either training, validation or test data
        seq_length: number of words that will be fed into the time distributed input layer of the network
        skip_steps: number of words to be skipped over between training samples within each batch
        
        """
        self.data = data
        self.seq_length = seq_length
        self.n_batch_size = n_batch_size
        self.n_vocab = n_vocab
        self.store_folder = store_folder

        if not os.path.exists(self.store_folder):
            os.makedirs(self.store_folder)
        
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        if os.path.exists(os.path.join(self.store_folder, 'global_step_point.pckl')):
            self.current_idx = load_obj(os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')
        else:
            self.current_idx = 0
        
        # step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.step = step
        self.steps_per_epoch = (len(self.data)//(self.n_batch_size) - 1) // self.step # calc_steps_per_epoch(self.data, self.seq_length, self.step)

        # reload or initialize epoch and step counter
        if os.path.exists(os.path.join(self.store_folder, 'global_epoch_point.pckl')):
            self.epoch_counter = load_obj(os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')
        else:
            self.epoch_counter = 0

    def store_step_counter(self, s):
        store_obj(s, os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')

    def store_epoch_counter(self, e):
        self.epoch_counter = e
        store_obj(self.epoch_counter, os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')
        
    def generate(self):
        x = np.zeros((self.n_batch_size, self.seq_length))
        y = np.zeros((self.n_batch_size, self.seq_length))
        while True:
            for i in range(self.n_batch_size):
                if self.current_idx + self.seq_length >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                
                #print (self.data, self.current_idx, self.seq_length)
                x[i, :] = self.data[self.current_idx:self.current_idx + self.seq_length]

                y[i, :] = self.data[self.current_idx + 1:self.current_idx + self.seq_length + 1]
                self.current_idx += self.step
            yield x, y

def pad_list(l, max_seq_len=10, pad_val=0):
    l = deepcopy(l)
    while len(l) < max_seq_len:
        l.append(pad_val)
    return l

class PadOneStepGenerator(object):
    def __init__(self, data, batch_size=5, max_seq_len=10, pad_val=0):
        self.data = data
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pad_val = pad_val
        self.m_ix = 0
        self.ix = 0
        self.length = 0
        self.steps_per_epoch = self.calc_steps_per_epoch()
    
    def calc_steps_per_epoch(self):
        count = 0
        ix = 0
        m_ix = 0
        while m_ix < len(self.data):
            length = len(self.data[m_ix])
            if length <= self.max_seq_len:
                count += 1
            else:
                while ix + self.max_seq_len + 1 <= length:
                    count += 1
                    ix += 1
            m_ix += 1
            ix = 0 
        return math.ceil(count / self.batch_size)
    
    def generate(self):
        x = np.zeros((self.batch_size, self.max_seq_len))
        y = np.zeros((self.batch_size, self.max_seq_len))
        count = 0
        while True:
            while count < self.batch_size:
                if self.m_ix >= len(self.data):
                    self.m_ix = 0
                self.length = len(self.data[self.m_ix])
                if self.length < self.max_seq_len:
                    x[count, :] = pad_list(self.data[self.m_ix][:self.length-1], self.max_seq_len)
                    y[count, :] = pad_list(self.data[self.m_ix][1:self.length], self.max_seq_len)
                    count += 1
                    self.m_ix += 1
                elif self.length == self.max_seq_len:
                    self.data[self.m_ix].append(self.pad_val)
                    x[count, :] = self.data[self.m_ix][:self.length]
                    y[count, :] = self.data[self.m_ix][1:self.length+1]
                    count += 1
                    self.m_ix += 1
                else:
                    while self.ix + self.max_seq_len + 1 <= self.length:
                        if count >= self.batch_size:
                            yield x, y
                        x[count, :] = self.data[self.m_ix][self.ix:self.ix+self.max_seq_len]
                        y[count, :] = pad_list(
                            self.data[self.m_ix][self.ix+1:self.ix+self.max_seq_len+1], 
                            self.max_seq_len)
                        count += 1
                        self.ix += 1
                    self.m_ix += 1
                    self.ix = 0 
            yield x, y

def generate_sequences(data, split_char=41):
    sequences = []
    new_seq = []
    for c in data:
        if c == split_char:
            sequences.append(new_seq)
            new_seq = []
            continue
        new_seq.append(c)
    return sequences

class Seq2Track(object):

    def __init__(self, n_batch_size, seq_length, n_vocab, n_layers, latent_size=128, recommendation=False):
        
        self.n_batch_size = n_batch_size
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.latent_size = latent_size

        if recommendation:
            self.n_batch_size = 1
            self.seq_length = 1

        # create model
        # define placeholders for X and y batches
        self.X = tf.placeholder(tf.int32, [None, self.seq_length], name='X')
        self.y = tf.placeholder(tf.int32, [None, self.seq_length], name='y')

        # generate embedding matrix for data representation and initialize randomly
        self.embedding_matrix = tf.get_variable('embedding_mat', [self.n_vocab, self.latent_size], tf.float32, tf.random_normal_initializer())
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.X)

        # define an initial state for LSTM
        # since LSTM contain two states c and h we're working with the second dimension is 2
        self.initial_state = tf.placeholder(tf.float32, [self.n_layers, 2, self.n_batch_size, self.latent_size], name='initial_state')

        # states can be represented as tuples (c, h) per layer
        # to do so, we'll unstack the tensor on the layer axis
        state_list = tf.unstack(self.initial_state, axis=0)
        # and create a tuple representation for any (c, h) state representation per layer (n)
        # tuple(LSTMStateTuple(c0, h0), LSTMStateTuple(c1, h1), ..., LSTMStateTuple(cn, hn),)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_list[i][0], state_list[i][1]) for i in range(self.n_layers)]
            )

        # in case one layer is being used
        cell = tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1.0)  # different size possible?
        #devices = ['/gpu:0', '/gpu:1']
        
        def build_cells(layers, recommendation=recommendation, dropout_prob=.5):
            cells = []
            for i in range(layers):
                #cell = DeviceCellWrapper(devices[i], tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True))
                cell = tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True)
                #cell = tf.contrib.rnn.AttentionWrapper(cell, attn_length=self.seq_length, state_is_tuple=True)
                if not recommendation:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_prob)
                cells.append(cell)
            return cells

        # otherwise create multirnn cells
        if self.n_layers > 1:
            cells = build_cells(self.n_layers, recommendation, .5)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # generate state and y output per timestep
        self.output, self.state = tf.nn.dynamic_rnn(cell, self.embedding_inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # reshape so output fits into softmax function
        # [n_batch_size * seq_length, latent_size]
        self.output = tf.reshape(self.output, [-1, self.latent_size])


        # now we need to calculate the activations
        #self.W = tf.Variable(tf.random_uniform([self.latent_size, self.n_vocab], -0.05, 0.05))
        #self.b = tf.Variable(tf.random_uniform([self.n_vocab], -0.05, 0.05))
        with tf.variable_scope('lstm_vars', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W', [self.latent_size, self.n_vocab], tf.float32, tf.random_normal_initializer())
            self.b = tf.get_variable('b', [self.n_vocab], tf.float32, tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.output, self.W) + self.b
        #W_t = tf.transpose(self.W)


        # seq2seq.sequence_loss method requires [n_batch_size, seq_length, n_vocab] shaped vector
        self.logits = tf.reshape(self.logits, [self.n_batch_size, self.seq_length, self.n_vocab])

        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits,
            targets=self.y,
            weights=tf.ones([n_batch_size, seq_length], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        self.cost = tf.reduce_sum(self.loss) # / (self.n_batch_size * self.seq_length)
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 1.)

        # accuracy calculations follow
        self.softmax = tf.nn.softmax(tf.reshape(self.logits, [-1, self.n_vocab]))
        self.predict = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)
        correct_predictions = tf.equal(self.predict, tf.reshape(self.y, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        self.lr = .001
        with tf.variable_scope('lstm_vars', reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.training_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
            #self.training_op = self.optimizer.minimize(self.loss)

    
    def recommend(self, sess, start_sequence, tracks, vocab, n=100):
        def reduced_argsort(arr, size=600):
            return np.argpartition(arr, -size)[-size:]

        def subsample(preds, candidates, vocab, temp=.7):
            if temp <= 0:
                candidates.append(vocab[np.argmax(preds)])
                return
            preds = np.asarray(preds[1:]).astype('float64')
            preds = np.log(preds) / temp
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            sample = np.argmax(probas)
            candidates.append(vocab[sample])

        def artist_search(preds, candidates, vocab, seeds, c_count):
            samples = reduced_argsort(preds)
            for sample in samples:
                track = vocab[sample]
                if track in seeds:
                    continue
                if track in c_count:
                    c_count[track] += preds[sample]
                else:
                    c_count[track] = preds[sample]
                    candidates.append(track)


        state = np.zeros((self.n_layers, 2, self.n_batch_size, self.latent_size))

        candidates = []
        c_count = {}

        # iterate over seeds and generate initial state for recommendation
        for track in start_sequence:
            x = np.zeros((1, 1))
            if track not in vocab:
                continue
            x[0, 0] = vocab[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            artist_search(probabilities[0], candidates, tracks, start_sequence, c_count)
            
        track_pointer = -1
        track = start_sequence[track_pointer]
        while track not in vocab:
            track_pointer -= 1
            track = start_sequence[track_pointer]

        for n in range(n):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            artist_search(probabilities[0], candidates, tracks, start_sequence, c_count)

        return_candidates = [x[0] for x in Counter(c_count).most_common(n)]

        return [x for x in return_candidates if x not in ['<eos>', 'unknown']]



def main():

    sess = tf.Session()

    data_dir = '../../../workspace/rnn/tf'
    model_path = 'one_seq_final_lrg_5skip'
    full_model_dir = os.path.join(data_dir, model_path)

    if not os.path.exists(full_model_dir):
        print ('Created {} ...'.format(full_model_dir))
        os.makedirs(full_model_dir)

    print ('Loading data ...')
    data = load_obj('../../../workspace/final_data/id_sequence.pckl', 'pickle')
    vocab = load_obj('../../../workspace/final_data/track2id.pckl', 'pickle')
    vocab2int = vocab
    int2vocab = {v:k for k,v in vocab2int.items()}
    print (len(int2vocab))

    seq_length = 50
    n_batch_size = 18
    n_layers = 2
    epochs = 5000
    training = True
    save_steps = 5000
    latent_size = 128
    skips = 5
    
    # initialize data generator
    #data_matrix = generate_sequences(data)
    n_vocab = len(int2vocab)
    bg = BatchGenerator(
        data=data, 
        seq_length=seq_length, 
        n_batch_size=n_batch_size, 
        n_vocab=n_vocab, 
        step=skips,
        store_folder=os.path.join(full_model_dir, 'step_point'))
    current_epoch = bg.epoch_counter

    model = Seq2Track(
        n_batch_size=n_batch_size, 
        seq_length=seq_length, 
        n_vocab=n_vocab, 
        n_layers=n_layers,
        latent_size=latent_size)

    # reusing scope for recommendations
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        pred_model = Seq2Track(
            n_batch_size=n_batch_size, 
            seq_length=seq_length, 
            n_vocab=n_vocab, 
            n_layers=n_layers,
            latent_size=latent_size,
            recommendation=True)

    saver = tf.train.Saver(tf.global_variables())
    init_operation = tf.global_variables_initializer()
    sess.run(init_operation)
    if os.path.exists(os.path.join(full_model_dir, 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(full_model_dir))

    if training:
        #init_operation.run()
        for e in range(current_epoch, epochs):
            avg_epoch_cost = []
            current_state = np.zeros((n_layers, 2, n_batch_size, latent_size))
            for step in range(bg.current_idx, bg.steps_per_epoch):
                X_batch, y_batch = next(bg.generate())
                if step % 10 != 0:
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={model.X: X_batch, model.y: y_batch, model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                elif step % 1000 == 0:
                    start_time = time.time()
                    cost, _, current_state, acc = sess.run(
                        [model.cost, model.training_op, model.state, model.accuracy],
                        feed_dict={model.X: X_batch, model.y: y_batch, model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                    end_time = (time.time() - start_time)
                    print ('Epoch: {} - Step: {} / {} - Cost: {} - Accuracy: {} - Time: {}s'.format(e, step, bg.steps_per_epoch, np.mean(avg_epoch_cost), acc, end_time))
                    test_seed_sequence = ['spotify:track:14AaSKhUMiR5qbNvhjlj9L', 'spotify:track:2tznHmp70DxMyr2XhWLOW0', 'spotify:track:0uqPG793dkDDN7sCUJJIVC']
                    print ('Seeds: {} '.format([rec_helper.track_uri_to_artist_and_title(x) for x in test_seed_sequence]))
                    results = pred_model.recommend(sess, test_seed_sequence, int2vocab, vocab2int, n=500)
                    print ('Recommendations: {}'.format([rec_helper.track_uri_to_artist_and_title(x) for x in results]))
                else:
                    start_time = time.time()
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={model.X: X_batch, model.y: y_batch, model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                    end_time = (time.time() - start_time)
                    print ('Epoch: {} - Step: {} / {} - Cost: {} - Time: {}s'.format(e, step, bg.steps_per_epoch, np.mean(avg_epoch_cost), end_time))
                
                # Save the model and the vocab
                if step != 0 and step % save_steps == 0:
                    # Save model
                    bg.store_step_counter(step)
                    bg.store_epoch_counter(e)

                    model_file_name = os.path.join(full_model_dir, 'model')
                    saver.save(sess, model_file_name, global_step = step)
                    print('Model Saved To: {}'.format(model_file_name))
        # if epoch is over
        bg.store_epoch_counter(e)
        bg.current_idx = 0
        bg.store_step_counter(0)
    
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        if os.path.exists(os.path.join(full_model_dir, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(full_model_dir))
        test_seed_sequence = ['spotify:track:7DH9OOPwnURxwqBKiFN3Pk', 'spotify:track:6WK9XJMNifCNiVtE5YjX1I']
        print ('Seeds: {} '.format([rec_helper.track_uri_to_artist_and_title(x) for x in test_seed_sequence]))

        results = pred_model.recommend(sess, test_seed_sequence, int2vocab, vocab2int, n=500)

        print ('Recommendations: {}'.format([rec_helper.track_uri_to_artist_and_title(x) for x in results]))


if __name__ == "__main__":
    main()
