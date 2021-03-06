import collections
import gensim
import json
import math
import numpy as np
import os
import pandas as pd

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

t2s_config = load_obj('config.json', 'json')  # all configuration files can be set manually as well
PLAYLIST_FOLDER = t2s_config['PLAYLIST_FOLDER']  # set folder of playlist information
RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']  # all information will be stored here
RANDOM_STATE = t2s_config['RANDOM_STATE']
recompute = True  

np.random.seed(RANDOM_STATE)

##################################################################
############################# METHODS ############################
##################################################################

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
        track_uri_to_track_artist_string_fname = os.path.join(self.results_folder, 'track_uri_to_track_artist_string.pckl')
        
        if not os.path.exists(track_popularity_dict_fname) or recompute:
            track_uri_to_track_artist_string = {}  # TODO: fill with goods
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
            print ('\nStoring all_playlist and popularity dicts ...')
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


def load_inclusion_tracks(dev_playlist_dict, test_playlist_dict):
    print ('Loading dev and test set tracks for inclusion ...')
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


def build_vocabulary(track_sequence):

    print ('Creating dictionaries ...')
    counter = collections.Counter(track_sequence)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    track2id = dict(zip(words, range(len(words))))

    return track2id, track_sequence


def filter_sequence(sequence, track2id, min_val, challenge_tracks):
    
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


##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print ('Results folder created ...')

    # PRE PROCESSING
    statistician = Statistician(PLAYLIST_FOLDER, RESULTS_FOLDER)

    print ('Generating popularity dict ...')
    track_popularity_dict = statistician.create_track_popularity_dict(recompute=recompute)
    # generate sampling information
    # stratification on median track popularity, number of tracks and modified at
    print ('Generating playlist DataFrame including feature aggregations ...')
    playlist_df = statistician.get_playlist_df(recompute=recompute)

    # binning for stratification process
    print ('Generating stratification classes ...')
    playlist_df = create_stratification_classes(playlist_df)

    print ('Splitting data into train, test and dev sets ...')
    x_train_pids, x_dev_pids, x_test_pids = split_playlist_df(
        playlist_df, RANDOM_STATE, statistician.all_playlists_dict, RESULTS_FOLDER, recompute=recompute)

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
        track2id, track_sequence = build_vocabulary(x_train)

        print ('Filtering sequences ...')
        track2id, track_sequence = filter_sequence(track_sequence, track2id, 5, c_set_tracks)

        print ('Transforming track-uri sequences in int sequences ...')
        track_sequence = sequences_to_ids(track_sequence, track2id)

        print ('Storing id_sequence file ...')
        store_obj(track_sequence, id_sequence_fname, 'pickle')
        print ('Storing vocabulary file ...')
        store_obj(track2id, track2id_fname, 'pickle')
    else:
        track_sequence = load_obj(id_sequence_fname, 'pickle')
        track2id = load_obj(track2id_fname, 'pickle')

    ## END OF PRE-PROCESSING
    print ('Generated all files for next steps ...')
