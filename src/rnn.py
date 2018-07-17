import math
import numpy as np
import tensorflow as tf
import time
import os
import sys
sys.path.append('../')

from collections import Counter
from copy import deepcopy
from keras.utils import to_categorical
from tools.io import extract_pids, load_obj, store_obj, write_recommendations_to_file

print ('#' * 80)
print ('Track2Seq Model')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

t2s_config = load_obj('../config.json', 'json')
input_folder = t2s_config['RESULTS_FOLDER']  # data of pre-processing steps
model_folder = t2s_config['T2S_MODEL_FOLDER']  # where model checkpoints are stored
model_name = t2s_config['T2S_MODEL_NAME']  # name of model
full_model_path = os.path.join(model_folder, model_name)

# generate folder
if not os.path.exists(full_model_path):
    print ('Created {} ...'.format(full_model_path))
    os.makedirs(full_model_path)

print ('Loading data ...')
data = load_obj(os.path.join(input_folder, 'id_sequence.pckl'), 'pickle')
vocab = load_obj(os.path.join(input_folder, 'track2id.pckl'), 'pickle')
track2int = vocab
int2track = {v:k for k,v in track2int.items()}
print ('There are {} tokens in the vocabulary'.format(len(int2track)))

##################################################################
######################### HYPER PARAMETERS #######################
##################################################################

seq_length = 50  # how long are training sequences
n_batch_size = 18  # how many sequences per batch
n_layers = 2  # amount of lstm layers
epochs = 1000  # epochs to train on
training = False  # is training active - if not, recommendation process starts / continues
save_steps = 5000  # after how many steps should the progress be saved
latent_size = 128  # latent size of LSTM and embedding layer
skips = 5  # how many skips in between sequences 

##################################################################
########################## TRAINING SETUP ########################
##################################################################

evaluation_set_fname = os.path.join(input_folder,'filled_dev_playlists_dict.pckl')
results_folder = 'recommendations/'
result_fname = os.path.join(results_folder, 'seq2track_recommendations.csv')

if not os.path.exists(results_folder):
    print('Creating results folder: {}'.format(results_folder))
    os.makedirs(results_folder)


##################################################################
####################### RECOMMENDATION SETUP #####################
##################################################################

challenge_track = t2s_config['TEAM_TRACK']
team_name = t2s_config['TEAM_NAME']
contact_info = t2s_config['TEAM_CONTACT']

##################################################################
############################# METHODS ############################
##################################################################


class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    """
    Helper class if LSTM layers should be divided
    on multiple GPUs.
    """
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
    

class BatchGenerator(object):

    def __init__(self, data, seq_length, n_batch_size, n_vocab, step=5, test=False, store_folder='step_point/'):
        """
        data: can be either training, validation or test data
        seq_length: number of tracks that will be fed into the network
        step: number of words to be skipped over between training samples within each batch
        
        """
        self.data = data
        self.seq_length = seq_length
        self.n_batch_size = n_batch_size
        self.n_vocab = n_vocab
        self.store_folder = store_folder

        if not os.path.exists(self.store_folder):
            os.makedirs(self.store_folder)
        
        # current_idx will save progress and serve as pointer
        # will reset to 0 once end is reached
        if os.path.exists(os.path.join(self.store_folder, 'global_step_point.pckl')):
            self.current_idx = load_obj(os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')
        else:
            self.current_idx = 0
        
        self.step = step
        # calculate steps per epoch
        self.steps_per_epoch = (len(self.data)//(self.n_batch_size) - 1) // self.step

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
                
                x[i, :] = self.data[self.current_idx:self.current_idx + self.seq_length]
                y[i, :] = self.data[self.current_idx + 1:self.current_idx + self.seq_length + 1]
                self.current_idx += self.step
            yield x, y


##################################################################
############################## MODEL #############################
##################################################################

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
        
        #devices = ['/gpu:0', '/gpu:1']  # multi gpu layout - amount of devices ==  amount of layers
        def build_cells(layers, recommendation=recommendation, dropout_prob=.5):
            cells = []
            for i in range(layers):
                cell = tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True)
                #cell = DeviceCellWrapper(devices[i], tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True))
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
        with tf.variable_scope('lstm_vars', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W', [self.latent_size, self.n_vocab], tf.float32, tf.random_normal_initializer())
            self.b = tf.get_variable('b', [self.n_vocab], tf.float32, tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.output, self.W) + self.b

        # seq2seq.sequence_loss method requires [n_batch_size, seq_length, n_vocab] shaped vector
        self.logits = tf.reshape(self.logits, [self.n_batch_size, self.seq_length, self.n_vocab])

        # targets are expected to be of shape [seq_len, 1] where the second dimension represents the class as int
        # we can introduce weights regarding the tracks, this might be interesting for
        # an emulated attention mechanism or if we use artist / genre level recommendations
        # could also be used to weigh the first tracks or last tracks of a sequence 
        # with more importance
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits,
            targets=self.y,
            weights=tf.ones([n_batch_size, seq_length], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        self.cost = tf.reduce_sum(self.loss)
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

    
    def recommend(self, sess, start_sequence, int2track, track2int, n=100):
        def reduced_argsort(arr, size=n+100):
            return np.argpartition(arr, -size)[-size:]

        def subsample(preds, candidates, int2track, temp=.7):
            if temp <= 0:
                candidates.append(int2track[np.argmax(preds)])
                return
            preds = np.asarray(preds[1:]).astype('float64')
            preds = np.log(preds) / temp
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            sample = np.argmax(probas)
            candidates.append(int2track[sample])

        def artist_search(preds, candidates, int2track, seeds, c_count):
            samples = reduced_argsort(preds)
            for sample in samples:
                track = int2track[sample]
                if track in seeds:
                    continue
                if track in c_count:
                    c_count[track] += preds[sample]
                else:
                    c_count[track] = preds[sample]
                    candidates.append(track)

            # return index of highest probability
            pointer = -1

            # filter out eos and unknown token for stream of conciousness
            while int2track[samples[pointer]] in ['<eos>', 'unknown']:
                pointer -= 1
            return samples[pointer]

        state = np.zeros((self.n_layers, 2, self.n_batch_size, self.latent_size))

        candidates = []
        c_count = {}

        # iterate over seeds and generate initial state for recommendation
        for track in start_sequence:
            x = np.zeros((1, 1))
            if track not in track2int:
                continue
            x[0, 0] = track2int[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            _ = artist_search(probabilities[0], candidates, int2track, start_sequence, c_count)
        

        track_pointer = -1
        track = start_sequence[track_pointer]
        while track not in track2int:
            track_pointer -= 1
            try:
                track = start_sequence[track_pointer]
            except:
                return []

        truth_flag = False
        truth_pointer = 0
        valid_sequence = [x for x in start_sequence if x in track2int]

        for n in range(n):
            track = np.random.choice([x for x in start_sequence if x in track2int], 1)[0]
            x = np.zeros((1, 1))
            x[0, 0] = track2int[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            track_int = artist_search(probabilities[0], candidates, int2track, start_sequence, c_count)
            
            # Semi-guided prediction
            if truth_flag:
                truth_flag = False
                if truth_pointer == len(valid_sequence):
                    truth_pointer = 0
                track = start_sequence[truth_pointer]
            else:
                truth_flag = True
                track = int2track[track_int]

        # return most probable candidates
        return_candidates = [x[0] for x in Counter(c_count).most_common(n)]

        return [x for x in return_candidates if x not in ['<eos>', 'unknown']]



##################################################################
############################## MAIN ##############################
##################################################################


def main():
    # in case a specific GPU should be used
    #gpu_options = tf.GPUOptions(visible_device_list='0')
    #config = tf.ConfigProto(gpu_options=gpu_options)
    #sess = tf.Session(config=config)

    sess = tf.Session()
    
    # initialize data generator
    n_vocab = len(int2track)
    bg = BatchGenerator(
        data=data, 
        seq_length=seq_length, 
        n_batch_size=n_batch_size, 
        n_vocab=n_vocab, 
        step=skips,
        store_folder=os.path.join(full_model_path, 'step_point'))
    
    current_epoch = bg.epoch_counter

    # intialize model for training
    model = Seq2Track(
        n_batch_size=n_batch_size, 
        seq_length=seq_length, 
        n_vocab=n_vocab, 
        n_layers=n_layers,
        latent_size=latent_size)

    # initialize model for prediction
    # reusing scope for recommendations
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        pred_model = Seq2Track(
            n_batch_size=n_batch_size, 
            seq_length=seq_length, 
            n_vocab=n_vocab, 
            n_layers=n_layers,
            latent_size=latent_size,
            recommendation=True)

    # pick up the process where we left off - if possible
    saver = tf.train.Saver(tf.global_variables())
    init_operation = tf.global_variables_initializer()
    sess.run(init_operation)

    # check if a model exists, if so - load it
    if os.path.exists(os.path.join(full_model_path, 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(full_model_path))

    # training routine
    if training:
        # run epochs
        for e in range(current_epoch, epochs):
            avg_epoch_cost = []  # store average cost per epoch

            # for any epoch initialize state as zeros
            current_state = np.zeros((n_layers, 2, n_batch_size, latent_size))
            for step in range(bg.current_idx, bg.steps_per_epoch):
                X_batch, y_batch = next(bg.generate())  # generate fresh training batch
                
                if step % 10 == 0:  # show progress every 10 steps
                    start_time = time.time()
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={model.X: X_batch, model.y: y_batch, model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                    end_time = (time.time() - start_time)
                    print ('Epoch: {} - Step: {} / {} - Cost: {} - Time: {}s'.format(
                        e, step, bg.steps_per_epoch, np.mean(avg_epoch_cost), end_time))

                elif step % 1000 == 0:  # show recommendation examples every 1000 steps
                    start_time = time.time()
                    
                    cost, _, current_state, acc = sess.run(
                        [model.cost, model.training_op, model.state, model.accuracy],
                        feed_dict={
                            model.X: X_batch, 
                            model.y: y_batch, 
                            model.initial_state: current_state})

                    # Compute cost and accuracy
                    avg_epoch_cost.append(cost)
                    end_time = (time.time() - start_time)
                    print ('Epoch: {} - Step: {} / {} - Cost: {} - Accuracy: {} - Time: {}s'.format(
                        e, step, bg.steps_per_epoch, np.mean(avg_epoch_cost), acc, end_time))

                    # Show recommendations
                    # can be changed to incorporate any track that's in int2track
                    sample_seed_sequence = [
                        'spotify:track:14AaSKhUMiR5qbNvhjlj9L', 
                        'spotify:track:2tznHmp70DxMyr2XhWLOW0', 
                        'spotify:track:0uqPG793dkDDN7sCUJJIVC']
                    
                    print ('Seeds: {} '.format(x for x in sample_seed_sequence))
                    results = pred_model.recommend(sess, sample_seed_sequence, int2track, track2int, n=500)
                    print ('Recommendations: {}'.format([x for x in results]))

                else:
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={
                            model.X: X_batch, 
                            model.y: y_batch, 
                            model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                
                # Save the model and the vocab
                if step != 0 and step % save_steps == 0:
                    # Save model
                    bg.store_step_counter(step)
                    bg.store_epoch_counter(e)

                    model_file_name = os.path.join(full_model_path, 'model')
                    saver.save(sess, model_file_name, global_step = step)
                    print('Model Saved To: {}'.format(model_file_name))
        # if epoch is over
        bg.store_epoch_counter(e)
        bg.current_idx = 0
        bg.store_step_counter(0)
        model_file_name = os.path.join(full_model_path, 'model')
        saver.save(sess, model_file_name, global_step = step)
        print('Model Saved To: {}'.format(model_file_name))
    
    else:
        pid_collection = extract_pids(result_fname)
        all_challenge_playlists = load_obj(challenge_set_fname, 'pickle')

        init = tf.global_variables_initializer()
        sess.run(init)
        if os.path.exists(os.path.join(full_model_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(full_model_path))

        num_playlists = len(all_challenge_playlists)

        print('Recommending tracks for {:,} playlists...'.format(num_playlists))

        avg_time = []
        for k in all_challenge_playlists:
            for ix, playlist in enumerate(all_challenge_playlists[k]):
                start_wall_time = time.time()

                if playlist['pid'] in pid_collection:
                    continue
                reco_per_playlist = []
                reco_store = []

                try:
                    reco_per_playlist = pred_model.recommend(sess, playlist['tracks'], int2track, track2int, n=600)
                    if not reco_per_playlist:
                        print('Something went wrong with playlist {}'.format(playlist['pid']))
                        continue
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as err:
                    print('Something went wrong with playlist {} (Error: {})'.format(playlist['pid'], err))
                    continue

                # store recommendations
                reco_per_playlist = reco_per_playlist[:500]
                pid_collection.append(playlist['pid'])
                time_elapsed = time.time() - start_wall_time
                avg_time.append(time_elapsed)

                print(
                    'Recommended {} songs ({} / {}). Avg time per playlist: {:.2f} seconds.'.format(
                        len(reco_per_playlist),
                        ix,
                        num_playlists,
                        np.mean(avg_time)))

                write_recommendations_to_file(challenge_track, team_name, contact_info, pid, recos, fname)
                
                with open(result_fname, 'a') as f:
                    f.write(str(playlist['pid']) + ', ')
                    f.write(', '.join([x for x in reco_per_playlist]))
                    f.write('\n\n')


if __name__ == "__main__":
    main()
