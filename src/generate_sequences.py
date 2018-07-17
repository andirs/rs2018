import collections
import gensim
import json
import math
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import string
import sys
import time

from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tools.io import load_obj, store_obj

print ('#' * 80)
print ('Track2Seq Preprocessing')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

PLAYLIST_FOLDER = '../../workspace/data'  # set folder of playlist information
RESULTS_FOLDER = '../../workspace/final_submission/results/'  # all information will be stored here
recompute = True
random_state = 2018
np.random.seed(random_state)

# download `GoogleNews-vectors-negative300.bin.gz` from 
# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
w2v_fname = '../../workspace/gw2v/GoogleNews-vectors-negative300.bin.gz'


##################################################################
############################# METHODS ############################
##################################################################


def load_inclusion_tracks(dev_playlist_dict, test_playlist_dict):
    print ('... Loading dev and test set tracks for inclusion ...')
    inclusion_tracks = set()

    # load dev set
    for k in dev_playlist_dict:
        for playlist in dev_playlist_dict[k]:
            for track in playlist['tracks']:
                inclusion_tracks.add(track)

    # load test set
    for k in test_playlist_dict:
        for playlist in test_playlist_dict[k]:
            for track in playlist['tracks']:
                inclusion_tracks.add(track)

    return inclusion_tracks


def _build_vocabulary(track_sequence):

    print ('Creating dictionaries ...')
    counter = collections.Counter(track_sequence)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    track2id = dict(zip(words, range(len(words))))

    return track2id, track_sequence

def _filter_sequence(sequence, track2id, min_val, challenge_tracks):
    
    # count all tracks
    counter = {}
    for t in sequence:
        if t in counter:
            counter[t] += 1
        else:
            counter[t] = 1
    print ('Finished counting ...')
    
    # create filter dict
    new_track2id = {}
    n = len(counter)
    for ix, track in enumerate(counter):
        if ix % 1000 == 0:
            print ('{} / {}'.format(ix, n), end='\r')
        if counter[track] > min_val or track in challenge_tracks:
            new_track2id[track] = counter[track]
    
    #new_track2id = {k:v for k,v in counter.items() if (v > min_val) or (k in challenge_tracks)}
    unk_val = len(new_track2id)
    new_track2id['<unk>'] = unk_val
    counter = collections.Counter(new_track2id)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    new_track2id = dict(zip(words, range(len(words))))
    print (len(new_track2id))


    return new_track2id, sequence


def sequences_to_ids(sequence, track2id):
    return_sequence = [track2id[x] for x in sequence if x in track2id]

    return return_sequence


class Statistician(object):
    def __init__(self, playlist_folder, results_folder):
        self.playlist_folder = playlist_folder
        self.results_folder = results_folder
        self.all_playlist_filenames = [
            os.path.join(self.playlist_folder, x) for x in os.listdir(self.playlist_folder) if 'mpd' in x]
        self.n_playlists = len(self.all_playlist_filenames)
        self.track_popularity_dict = None
        self.playlist_df = None
        self.all_playlists_dict = {}

    def create_track_popularity_dict(self, recompute=False):
        """
        Iteration method leveraging count_artists_and_tracks method 
        to aggregate information out of all playlist collections.
        
        Parameters:
        --------------
        recompute:    bool flag determining whether precomputed results should be used or not
        
        Returns:
        --------------
        track_popularity_dict:     dict mapping track uris to their popularity count in all playlists
        """
        track_popularity_dict_fname = os.path.join(self.results_folder, 'track_popularity_dict.pckl')
        all_playlists_dict_fname = os.path.join(self.results_folder, 'all_playlists_dict.pckl')
        
        if not os.path.exists(track_popularity_dict_fname) or recompute:
            track_popularity_dict = {}
            total_files = len(self.all_playlist_filenames)
            counter = 0
            for playlist_file in self.all_playlist_filenames:
                counter += 1
                print ("Working on slice {} ({:.2f} %) (File Name:  {} || Total Slices: {})".format(
                    counter, (counter / total_files) * 100, playlist_file, total_files), end='\r')
                playlist_collection = load_obj(playlist_file, 'json')
                for playlist in playlist_collection['playlists']:

                    self.all_playlists_dict[playlist['pid']] = {
                    'pid': playlist['pid'],
                    'name': playlist['name'], 
                    'tracks': []}
                    
                    for t in playlist['tracks']:
                        track_uri = t['track_uri']
                        # create popularity dict
                        if track_uri in track_popularity_dict:
                            track_popularity_dict[track_uri] += 1
                        else:
                            track_popularity_dict[track_uri] = 1

                        # create all playlist dict
                        self.all_playlists_dict[playlist['pid']]['tracks'].append(track_uri)

            # store dict
            store_obj(track_popularity_dict, track_popularity_dict_fname, 'pickle')
            store_obj(self.all_playlists_dict, all_playlists_dict_fname, 'pickle')
            self.track_popularity_dict = track_popularity_dict
        else:
            self.track_popularity_dict = load_obj(track_popularity_dict_fname, 'pickle')
            self.all_playlists_dict = load_obj(all_playlists_dict_fname, 'pickle')
        
        return self.track_popularity_dict


    def get_playlist_df(self, recompute):
        """
        Method that iterates over a playlist collection and retrieves all potential information 
        to store in one list of lists. This list can be used to create a well-formed pandas
        DataFrame.
        
        Parameters:
        ---------------
        columns: list storing all available and additional features for playlists
        artist_popularity_dict: lookup dict for artist popularity metrics
        artist_popularity_dict: lookup dict for track popularity metrics
        playlist_collection: retrieved playlist json
        
        Returns:
        ---------------
        tmp_playlist_list: list of lists every list containing features of a playlist
        columns: list of column names
        """
        playlist_df_fname = os.path.join(self.results_folder, 'playlist_df.csv')

        if not os.path.exists(playlist_df_fname) or recompute:

            # check if popularity dict has been created and loaded
            if not self.track_popularity_dict:
                _ = self.create_track_popularity_dict(recompute)

            playlist_popularity = []
            for playlist_coll_fname in self.all_playlist_filenames:
                tmp_playlist_list = []
                playlist_coll = load_obj(playlist_coll_fname, 'json')
                for playlist in playlist_coll['playlists']:
                    tmp_track_pop = []
                    track_count = 0
                    columns = [x for x in playlist.keys() if 'tracks' not in str(x) and 'description' not in str(x)]
                    columns.extend(['track_popularity_median', 'description', 'num_tracks'])
                    tmp_playlist_features = [
                        playlist[x] for x in playlist.keys() if 'tracks' not in str(x) and 'description' not in str(x)]
                    for track in playlist['tracks']:
                        track_count += 1
                        artist = track['artist_uri']
                        track = track['track_uri']
                        tmp_track_pop.append(self.track_popularity_dict[track])
                    tmp_playlist_features.extend(
                        [np.median(tmp_track_pop)])
                    tmp_playlist_features.append(playlist['description'] if 'description' in playlist.keys() else None)
                    tmp_playlist_features.append(track_count)
                    tmp_playlist_list.append(tmp_playlist_features)
                playlist_popularity.extend(tmp_playlist_list)
            
            self.playlist_df = pd.DataFrame(playlist_popularity, columns=columns)
            # store DataFrame to HDD
            self.playlist_df.to_csv(playlist_df_fname)
        else:
            self.playlist_df = pd.read_csv(playlist_df_fname, index_col=0)

        return self.playlist_df


def qs(q):
    """
    Helper method for quantile list. 
    Calculates quantile steps for q number of quantiles.
    
    Parameters:
    --------------
    q:    int, number of quantiles
    
    Returns:
    --------------
    quantile_list:    list of quantile steps
    """
    step = 100 / q
    quantile_list = []
    
    for i in range(1,q+1):
        quantile_list.append(i * step / 100)
    return quantile_list


def get_quantile_list(df, feature, bins=10):
    """
    Returns q quantile boundaries for a feature in a dataset. 
    
    Parameters:
    --------------
    df:      pandas.DataFrame
    feature: str, column name of feature
    bins   : amount of bins
    
    Returns:
    --------------
    quantile_list: list of boundaries to get quantile distribution
    """
    error_msg = None
    # create quantile pointers
    quantile_steps = qs(bins)
    quantile_list = []
    for step in quantile_steps:
        quant = df[feature].quantile(step)
        if quant not in quantile_list:
            quantile_list.append(quant)
        else:
            error_msg = 'Warning: Reduced bin size to'
    if error_msg:
        error_msg += ' {} bins.'.format(len(quantile_list))
        print (error_msg)
    return quantile_list


def get_feature_class(row, quantile_list):
    """
    Helper method for stratification. 
    Returns class label based on quantile boundaries.
    
    Parameters:
    --------------
    row:            int, data point or series entry
    quantile_list:  list with quantile measures
    
    Returns:
    --------------
    class:          int, range(0,len(quantile_list)) - determining class
    """
    if row <= quantile_list[0]:
        return 0
    for q in range(len(quantile_list) - 1):
        if row > quantile_list[q] and row <= quantile_list[q+1]:
            return q+1


def adjust_bins(df, feature, bins, sets=3):
    histogram_tuple = np.histogram(playlist_df[feature], bins)
    return_boundaries = histogram_tuple[1]
    del_count = 0
    del_list = []
    for idx, c in enumerate(histogram_tuple[0]):
        if c < sets:
            del_list.append(idx)
    return_boundaries = np.delete(return_boundaries, del_list)
    if del_list:
        print ('Reduced bin size to {}.'.format(bins - len(del_list)))
    return return_boundaries


def create_stratification_classes(df):
    nt_quantile_list = get_quantile_list(df, 'num_tracks', bins=10)
    df['num_tracks_class_quantile'] = df['num_tracks'].apply(get_feature_class, args=(nt_quantile_list, ))

    ma_quantile_list = get_quantile_list(df, 'modified_at', bins=10)
    df['modified_at_class_quantile'] = df['modified_at'].apply(get_feature_class, args=(ma_quantile_list, ))

    pop_quantile_list = get_quantile_list(df, 'track_popularity_median', bins=10)
    df['track_popularity_median_class_quantile'] = df['track_popularity_median'].apply(
        get_feature_class, args=(pop_quantile_list, ))
    return df


def split_playlist_df(df, random_state, all_playlists_dict, results_folder, recompute=False):
        x_train_pids_fname = os.path.join(results_folder, 'x_train_pids.pckl')
        x_dev_pids_fname = os.path.join(results_folder, 'x_dev_pids.pckl')
        x_test_pids_fname = os.path.join(results_folder, 'x_test_pids.pckl')

        if recompute:
            # To meet the second criteria for all tracks in the dev 
            # and test sets to be in the training set 
            # a bigger split is being produced. 

            X_train_full, X_test = train_test_split(
                df, 
                test_size=.1, 
                random_state=random_state, 
                stratify=df[[
                    'track_popularity_median_class_quantile', 
                    'num_tracks_class_quantile', 
                    'modified_at_class_quantile']])

            # filter playlist for rare tracks that occur only in one set but not in the other
            x_train_pids = X_train_full.pid.values
            x_test_pids = X_test.pid.values

            all_tracks = set()
            test_playlists = {}

            for p in all_playlists_dict:
                if p in x_train_pids:
                    for track in all_playlists_dict[p]['tracks']:
                        all_tracks.add(track)
                elif p in x_test_pids:
                    test_playlists[p] = all_playlists_dict[p]
            
            missing_pid = {}
            candidates = []
            for p in test_playlists:
                is_candidate = True
                for track in test_playlists[p]['tracks']:
                    if track not in all_tracks:
                        is_candidate = False
                        if p not in missing_pid:
                            missing_pid[p] = 1
                        else:
                            missing_pid[p] += 1
                if is_candidate:
                    candidates.append(p)

            # do final dev / test split
            dev_test = np.random.choice(candidates, 20000, replace=False)
            dev_test = shuffle(dev_test, random_state=random_state)
            x_dev_pids, x_test_pids = dev_test[:10000], dev_test[10000:]
            print ('Storing train, dev and test playlist ids ...')
            store_obj(x_train_pids, x_train_pids_fname, 'pickle')
            store_obj(x_dev_pids, x_dev_pids_fname, 'pickle')
            store_obj(x_test_pids, x_test_pids_fname, 'pickle')
        else:
            x_train_pids = load_obj(x_train_pids_fname, 'pickle')
            x_dev_pids = load_obj(x_dev_pids_fname, 'pickle')
            x_test_pids = load_obj(x_test_pids_fname, 'pickle')

        return x_train_pids, x_dev_pids, x_test_pids

def generate_all_train_playlist_set(x_train_pids, statistician, results_folder, recompute):
    all_train_playlist_set_fname = os.path.join(results_folder, 'all_train_playlist_set.pckl')
    if recompute:
        all_train_playlist_set = {}
        for pid in x_train_pids:
            all_train_playlist_set[pid] = statistician.all_playlists_dict[pid]
        store_obj(all_train_playlist_set, all_train_playlist_set_fname, 'pickle')
    else:
        all_train_playlist_set = load_obj(all_train_playlist_set_fname, 'pickle')

    return all_train_playlist_set


# Dev & Test Set Bucketing Method

def get_complete_testing_sets(playlists, test_indices_dict):
    """
    Generates dictionary with test buckets according to provided indices. 
    Adds additional seed and groundtruth lists to playlists.
    
    Parameters:
    --------------
    playlists:         list, original playlists included in test set
    test_indices_dict: dict, dictionary including the indices for every split
    
    Returns:
    --------------
    return_dict:       dict, {bucket_no: [playlist1, playlist2, ..., playlistn], ...}
    """
    # prepare return_dict
    return_dict = {}
    for bucket in test_indices_dict.keys():
        return_dict[bucket] = [y for x, y in enumerate(playlists) if x in test_indices_dict[bucket]]
    
    # add seed tracks and ground_truth to playlists
    for key in return_dict.keys():
        for playlist in return_dict[key]:
            playlist['seed'] = [x for x in playlist['tracks'][:key]]
            playlist['groundtruth'] = [x for x in playlist['tracks'][key:]]
    
    return return_dict


def get_testing_indices(lengths, buckets=[0, 1, 5, 10, 25, 100]):
    """
    Compute random indices for all k-seed options of challenge. 
    Sorts lenghts and divides 
    Splits depending on 50th percentile and preselects higher values for
    upper boundaries. Afterwards playlists are being sorted to fit the highest possible bucket.
    
    Parameters:
    --------------
    lengths:      list, length values (int) in order of indices
    random_seed:  int, determines shuffle seed for numpy.shuffle
    
    Returns:
    --------------
    indices_dict: dict, {bucket_no: [idx1, idx2, ..., idxn], ...}
    """
    sorted_lengths = sorted(enumerate(lengths), key=lambda x: x[1], reverse=False)
    bucket_size = math.floor(len(lengths) / len(buckets))
    ret_dict = {}
    final_offset = 0  # to add uneven counts to last bucket
    for idx, bucket in enumerate(buckets):
        if idx == len(buckets)-1:
            final_offset = len(lengths) % len(buckets)
        ret_dict[bucket] = [x[0] for x in sorted_lengths][idx*bucket_size:(idx+1)*bucket_size+final_offset]
    return ret_dict  

def bucketing_eval_playlists(x_dev_pids, x_test_pids, all_playlists_dict, RESULTS_FOLDER, recompute):
    test_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'test_playlist_dict.pckl')
    dev_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'dev_playlist_dict.pckl')

    if recompute:
        dev_playlists = []
        test_playlists = []
        dev_pid_order = []
        test_pid_order = []

        for pid in x_dev_pids:
            dev_playlists.append(all_playlists_dict[pid])
        
        for pid in x_test_pids:
            test_playlists.append(all_playlists_dict[pid]) 

        

        # gather lengths to generate buckets
        dev_lengths = [len(x['tracks']) for x in dev_playlists]
        test_lengths = [len(x['tracks']) for x in test_playlists]

        dev_indices = get_testing_indices(dev_lengths)
        test_indices = get_testing_indices(test_lengths)
        
        dev_playlist_dict = get_complete_testing_sets(dev_playlists, dev_indices)
        test_playlist_dict = get_complete_testing_sets(test_playlists, test_indices)
        
        store_obj(dev_playlist_dict, dev_playlist_dict_fname, 'pickle')
        store_obj(test_playlist_dict, test_playlist_dict_fname, 'pickle')
    else:
        dev_playlist_dict = load_obj(dev_playlist_dict_fname, 'pickle')
        test_playlist_dict = load_obj(test_playlist_dict_fname, 'pickle')

    return dev_playlist_dict, test_playlist_dict

# LEVENSHTEIN METHODS
class Levenshtein(object):
    def __init__(self):
        version = '0.1'

    @staticmethod
    def pre_process(playlist_name):
        """
        Preprocess a playlist name through tokenization, transforming it
        to lowercase and filtering out music playlist related stop-words.
        """
        try:
            word = [x for x in playlist_name.lower().split() if x not in stop_words]
            if len(word) > 1:
                word = '_'.join([x for x in word])
            if isinstance(word, list):
                return word[0]
            return word
        except Exception as e:
            return playlist_name


    @staticmethod
    def get_closest(df_row, playlist_name, return_dict, comp_memory):
        lowest = return_dict['lowest']
        targets = return_dict['targets']
        try:
            if df_row not in comp_memory:
                distance = nltk.edit_distance(df_row, playlist_name)
                comp_memory[df_row] = distance
            else:
                distance = comp_memory[df_row]
        except:
            return_dict['counter'] += 1
            return None
        if not lowest or distance <= lowest[0]:
            lowest.insert(0, distance)
            targets.insert(0, return_dict['counter'])
        return_dict['counter'] += 1


    @staticmethod
    def get_seed_tracks(playlist_df, return_dict, all_playlists_dict):
        sim_count = len([x for x in return_dict['lowest'] if x == return_dict['lowest'][0]])
        tmp_pids = return_dict['targets']

        if sim_count > 100:
            tmp_pids = tmp_pids[:sim_count]
            np.random.shuffle(tmp_pids)
            candidate_list = {}
            candidate_counts = 0
            for i in range(100):
                tmp_pid = playlist_df[playlist_df.index == tmp_pids[i]]['pid'].values[0]
                tmp_tracks = all_playlists_dict[tmp_pid]['tracks']
                if track_uri == '<eos>':
                    continue
                for track_uri in tmp_tracks:
                    if track_uri not in candidate_list:
                        candidate_list[track_uri] = 0
                    else:
                        candidate_list[track_uri] += 1
        else:
            candidate_list = {}
            candidate_counts = 0
            i = 0
            while candidate_counts < 100 and i < len(tmp_pids):

                tmp_pid = playlist_df[playlist_df.index == tmp_pids[i]]['pid'].values[0]
                tmp_tracks = all_playlists_dict[tmp_pid]['tracks']
                for track_uri in tmp_tracks:
                    if track_uri == '<eos>':
                        continue
                    if track_uri not in candidate_list:
                        candidate_list[track_uri] = 0
                        candidate_counts += 1
                    else:
                        candidate_list[track_uri] += 1
                i += 1

        return Counter(candidate_list).most_common(100)


    @staticmethod
    def generate_levenshtein_seed_dict(
        zero_seed_playlists, 
        all_playlist_names, 
        all_playlists_dict, 
        playlist_df, 
        RESULTS_FOLDER, 
        filename, 
        recompute):
        fname = os.path.join(RESULTS_FOLDER, filename)
        if recompute:
            comp_memory = {}
            seed_set = {}
            for idx, playl in enumerate(zero_seed_playlists):
                playlist_name = Levenshtein.pre_process(playl['name'])
                print ('Retrieving levenshtein similarities for \'{}\' ({:.2f} %)'.format(
                    playlist_name, ((idx + 1) / len(zero_seed_playlists)) * 100), end='\r')
                return_dict = {}
                return_dict['counter'] = 0
                return_dict['lowest'] = []
                return_dict['targets'] = []
                _ = all_playlist_names.apply(Levenshtein.get_closest, args=(playlist_name, return_dict, comp_memory))
                seeds = Levenshtein.get_seed_tracks(playlist_df, return_dict, all_playlists_dict)
                seed_set[playl['pid']] = [x[0] for x in seeds]

            store_obj(seed_set, fname, 'pickle')
        else:
            seed_set = load_obj(fname, 'pickle')

        return seed_set


# W2V Methoden
# load decoding dicts
emoji_dict = load_obj('dicts/emoji_dict.pckl', 'pickle')
urban_dict = load_obj('dicts/urban_dict.pckl', 'pickle')

def remove_punct_replace_emoji_with_meaning(token):
        new_string = ''
        for c in token:
            if c in set(string.punctuation).difference(set('_')):
                continue
            elif c in ['_']:
                new_string += ' '
            else:
                new_string += c
        return new_string.strip()

def check_whitespace_word(input_string):
    ws_counter = {'space': 0,
                  'other': []}
    for c in input_string:
        if c == ' ':
            ws_counter['space'] += 1
        else:
            ws_counter['other'].append(c)
    if not len(ws_counter['other']) - 1 > ws_counter['space']:
        return ''.join(ws_counter['other'])
    return input_string

def pre_process_title(title):
    t = str(title).lower().strip()
    for word in urban_dict:
        if word in t:
            t = t.replace(word, ' '.join(urban_dict[word]))
    for emoji in emoji_dict:
        if emoji in t:
            t = t.replace(emoji, ' ' + ' '.join(emoji_dict[emoji]))
    t = check_whitespace_word(t)
    t = t.split(' ')
    t = [x.strip() for x in t]
    stop_words = ['playlist', 'music']
    t = [remove_punct_replace_emoji_with_meaning(x) for x in t]
    return_tokens = []
    for ti in t:
        if ti in urban_dict:
            return_tokens.extend(urban_dict[ti])
        if ti not in stop_words:
            return_tokens.append(ti)
    return return_tokens

def get_vecs(row, return_vecs):
    playlist_title = row['name']
    playlist_tokens = pre_process_title(playlist_title)
    return_vec = []
    for token in playlist_tokens:
        try:
            return_vec.append(np.array(model.wv.word_vec(token)))
        except KeyError as e:
            continue
    return_vecs[row['pid']] = return_vec

def mean_and_unify(vector_dict):
    count = 0
    new_vector_dict = {}
    for i in vector_dict:
        length = len(np.array(vector_dict[i]))
        if length > 1 and length < 300:
            new_vector_dict[i] = np.mean(vector_dict[i], axis=0)
        else:
            new_vector_dict[i] = np.array(vector_dict[i]).flatten()
    return new_vector_dict

def avg_vector_to_matrix(avg_vectors):
    translation_dict = {}
    matrix = []
    idx = 0
    for item in avg_vectors:
        if len(avg_vectors[item]) > 0:
            translation_dict[idx] = item
            matrix.append(avg_vectors[item])
            idx += 1
    return np.array(matrix), translation_dict

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    return dot_product / (np.linalg.norm(a) * np.linalg.norm(b))

def get_vec_by_tokens(tokens):
    return_vec = []
    for token in tokens:
        try:
            return_vec.append(np.array(model.wv.word_vec(token)))
        except KeyError as e:
            print ('{} not in dataset'.format(token))
            continue
    if len(return_vec) > 1 and len(return_vec) < 300:
        return np.mean(return_vec, axis=0)
    else:
        return np.array(return_vec).flatten()

def get_similar_playlists(title, k=10):
    tokens = pre_process_title(title)
    vec = get_vec_by_tokens(tokens)
    
    dists = []
    for i in range(len(playlist_title_2_vec)):
        dists.append(cos_sim(vec, playlist_title_2_vec[i]))
    arg_sort_dist = np.argsort(dists)[-k:]
    dsts = np.sort(dists)[-k:]
    return [translation_dict[x] for x in arg_sort_dist][::-1], dsts[::-1]

def get_correspondant_list(pid_to_name, seed_k, results_folder, recompute):
    list_fname = os.path.join(results_folder, 'w2v_dev_correspondant_list.pckl')
    probs_fname = os.path.join(results_folder, 'w2v_dev_correspondant_list_probas.pckl')

    if recompute:
        correspondant_list = {}
        correspondant_list_probs = {}
        for ix, pid in enumerate(pid_to_name):
            
            print ('Retrieving CWVA for \'{}\' ({:.2f} %)'.format(
                pid_to_name[pid], ((ix+1) / len(pid_to_name)) * 100 ), end='\r')
            try:
                playlists, probabilities = get_similar_playlists(pid_to_name[pid], seed_k)
                correspondant_list[pid] = playlists
                correspondant_list_probs[pid] = probabilities
            except KeyboardInterrupt:
                break
            except:
                print ('Something went wrong with playlist: \'{}\' (pid: {})'.format(pid_to_name[pid], pid))
        store_obj(correspondant_list, list_fname, 'pickle')
        store_obj(correspondant_list_probs, probs_fname, 'pickle')
    else:
        correspondant_list = load_obj(list_fname, 'pickle')
        correspondant_list_probs = load_obj(probs_fname, 'pickle')
    
    return correspondant_list, correspondant_list_probs


def get_seed_tracks_probs(old_pid, seed_pid_list, seed_pid_probs, all_playlists_dict, k=100, include_probs=False):
    candidate_list = {}
    candidate_counts = 0
    i = 0
    for pid, prob in zip(seed_pid_list, seed_pid_probs):
        try:
            for track_uri in all_playlists_dict[pid]['tracks']:
                if track_uri == '<eos>':
                    continue
                if track_uri not in candidate_list:
                    candidate_list[track_uri] = prob
                else:
                    candidate_list[track_uri] += prob
        except:
            continue
    if include_probs:
        return [x for x in Counter(candidate_list).most_common(k)]
    return [x[0] for x in Counter(candidate_list).most_common(k)]


##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":

    W2V_FOLDER = os.path.join(RESULTS_FOLDER, 'w2v/')
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print ('Results folder created ...')

    if not os.path.exists(W2V_FOLDER):
        os.makedirs(W2V_FOLDER)

    # PRE PROCESSING
    statistician = Statistician(PLAYLIST_FOLDER, RESULTS_FOLDER)

    print ('Generating popularity dict ...')
    track_popularity_dict = statistician.create_track_popularity_dict(recompute=recompute)
    # generate sampling information
    # stratification on median track popularity, number of tracks and modified at
    print ('Generating playlist DataFrame inclusive aggregate features ...', end='\r\n')
    playlist_df = statistician.get_playlist_df(recompute=recompute)

    # binning for stratification process
    print ('Generating stratification classes ...')
    playlist_df = create_stratification_classes(playlist_df)

    print ('Splitting data into train, test and dev sets ...')
    x_train_pids, x_dev_pids, x_test_pids = split_playlist_df(
        playlist_df, random_state, statistician.all_playlists_dict, RESULTS_FOLDER, recompute=recompute)

    print ('Bucketing dev & test playlists ...')
    dev_playlist_dict, test_playlist_dict = bucketing_eval_playlists(
        x_dev_pids, x_test_pids, statistician.all_playlists_dict, RESULTS_FOLDER, recompute=recompute)

    print ('Loading training set ...')
    all_train_playlists = generate_all_train_playlist_set(
        x_train_pids, statistician, RESULTS_FOLDER, recompute=recompute)
    
    c_set_tracks = load_inclusion_tracks(dev_playlist_dict, test_playlist_dict)

    id_sequence_fname = os.path.join(RESULTS_FOLDER, 'id_sequence.pckl')
    track2id_fname = os.path.join(RESULTS_FOLDER, 'track2id.pckl')

    if recompute:
        # load playlists
        x_train = []
        print ('Working on training set ...')
        for p in all_train_playlists:
            tmp_playlist = all_train_playlists[p]['tracks']
            tmp_playlist.append('<eos>')
            x_train.extend(tmp_playlist)

        print ('Extracting sequences and building vocabulary ...')
        track2id, track_sequence = _build_vocabulary(x_train)

        print ('Filtering sequences ...')
        track2id, track_sequence = _filter_sequence(track_sequence, track2id, 5, c_set_tracks)

        print ('Transforming track-uri sequences in int sequences ...')
        track_sequence = sequences_to_ids(track_sequence, track2id)

        print ('Storing id_sequence file ...')
        store_obj(track_sequence, id_sequence_fname, 'pickle')
        print ('Storing vocabulary file ...')
        store_obj(track2id, track2id_fname, 'pickle')
    else:
        track_sequence = load_obj(id_sequence_fname, 'pickle')
        track2id = load_obj(track2id_fname, 'pickle')

    # LEVENSHTEIN
    print ('Computing Levenshtein distance ...')
    train_playlist_df = playlist_df[playlist_df.pid.isin(x_train_pids)].copy()
    train_playlist_df.reset_index(inplace=True)

    all_playlist_names = train_playlist_df['name'].apply(Levenshtein.pre_process)
    
    zero_dev = dev_playlist_dict[0]
    #zero_test = test_playlist_dict[0]
    
    # iterate over first 0-seed playlists
    dev_leve_seed_dict = Levenshtein.generate_levenshtein_seed_dict(
        zero_dev, all_playlist_names, statistician.all_playlists_dict, 
        train_playlist_df, RESULTS_FOLDER, 'dev_leve_seed_dict.pckl', recompute)
    #test_leve_seed_dict = Levenshtein.generate_levenshtein_seed_dict(
    #    zero_test, all_playlist_names, statistician.all_playlists_dict, 
    #    train_playlist_df, RESULTS_FOLDER, 'test_leve_seed_dict.pckl', recompute)

    # WORD2VEC - CWVA
    print ('Loading word2vec embeddings ...')
    playlist_title_2_vec_fname = os.path.join(W2V_FOLDER, 'playlist_title_2_vec.pkl')
    translation_dict_fname = os.path.join(W2V_FOLDER, 'translation_dict.pkl')
    
    if recompute:
        if not os.path.exists(w2v_fname):
            raise ValueError(
                'Download pre-computed word embeddings from https://github.com/mmihaltz/word2vec-GoogleNews-vectors')
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_fname, binary=True)

        print ('Calculating average tokens for playlist titles ...')
        df = playlist_df[playlist_df['pid'].isin(x_train_pids)]
    
        return_vecs = {}
        _ = df.apply(get_vecs, axis=1, args=(return_vecs, ))
        return_vecs_norm = mean_and_unify(return_vecs)

        playlist_title_2_vec, translation_dict = avg_vector_to_matrix(return_vecs_norm)

        store_obj(playlist_title_2_vec, playlist_title_2_vec_fname, 'pickle')
        store_obj(translation_dict, translation_dict_fname, 'pickle')
    else:
        playlist_title_2_vec = load_obj(playlist_title_2_vec_fname, 'pickle')
        translation_dict = load_obj(translation_dict_fname, 'pickle')

    complete_dev_seed_list_fname = os.path.join(W2V_FOLDER, 'complete_dev_seed_list.pckl')
    #complete_test_seed_list_fname = os.path.join(W2V_FOLDER, 'complete_test_seed_list.pckl')

    if recompute:
        dev_pid_to_name = {}
        for dplaylist in zero_dev:
            dev_pid_to_name[dplaylist['pid']] = dplaylist['name']

        #test_pid_to_name = {}
        #for dplaylist in zero_test:
        #    test_pid_to_name[dplaylist['pid']] = dplaylist['name']

        dev_correspondant_list, dev_correspondant_list_probs = get_correspondant_list(
            dev_pid_to_name, seed_k=100, results_folder=RESULTS_FOLDER, recompute=recompute)

        # turn playlists into tracks
        print ('Completing dev and test proxis ...')
        complete_dev_seed_list = {}
        for p in dev_correspondant_list:
            complete_dev_seed_list[p] = get_seed_tracks_probs(
                p, dev_correspondant_list[p], dev_correspondant_list_probs[p], statistician.all_playlists_dict)

        # if vectors are missing, fill up with levenshtein proxis
        for pid in [x['pid'] for x in zero_dev]:
            if pid not in complete_dev_seed_list:
                complete_dev_seed_list[pid] = dev_leve_seed_dict[pid]
        
        store_obj(complete_dev_seed_list, complete_dev_seed_list_fname, 'pickle')


        #test_correspondant_list, test_correspondant_list_probs = get_correspondant_list(
        #    test_pid_to_name, seed_k=100, results_folder=RESULTS_FOLDER, recompute=recompute)


        #complete_test_seed_list = {}
        #for p in test_correspondant_list:
        #    complete_test_seed_list[p] = get_seed_tracks_probs(
        #        p, test_correspondant_list[p], test_correspondant_list_probs[p], statistician.all_playlists_dict)

        #for pid in [x['pid'] for x in zero_test]:
        #    if pid not in complete_test_seed_list:
        #        complete_test_seed_list[pid] = test_leve_seed_dict[pid]

        store_obj(complete_dev_seed_list, complete_dev_seed_list_fname, 'pickle')
        #store_obj(complete_test_seed_list, complete_test_seed_list_fname, 'pickle')
    else:
        complete_dev_seed_list = load_obj(complete_dev_seed_list_fname, 'pickle')
        #complete_test_seed_list = load_obj(complete_test_seed_list_fname, 'pickle')
    if recompute:
        # add seeds to dev playlists
        for p in dev_playlist_dict[0]:
            if p['pid'] in complete_dev_seed_list:
                p['seed'] = complete_dev_seed_list[p['pid']]

        print ('Stored final dev set in results folder ...')
        store_obj(dev_playlist_dict, os.path.join(RESULTS_FOLDER, 'filled_dev_playlists_dict.pckl'), 'pickle')
    else:
        dev_playlist_dict = load_obj(os.path.join(RESULTS_FOLDER, 'filled_dev_playlists_dict.pckl'), 'pickle')