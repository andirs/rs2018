import collections
import json
import os
import pandas as pd
import pickle
import sys
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_obj(fname, dtype='json'):
    """
    Object loader function to simplify code.
    
    Parameters:
    --------------
    fname: str, file name of stored object
    dtype: str, 'json', 'pickle' or 'pandas'
    
    Returns:
    --------------
    return_obj: depending on dtype returns json object, pickle representation of dict/list or pd.DataFrame
    """
    if not os.path.exists(fname):
        raise IOError('{} does not exist and needs to be recomputed. Set recompute flag to \'True\''.format(fname))
    else:
        if dtype == 'json':
            with open(fname, 'r') as f:
                return_obj = json.load(f)
        elif dtype == 'pickle':
            with open(fname, 'rb') as f:
                return_obj = pickle.load(f)
        elif dtype == 'pandas':
            return_obj = pd.read_csv(fname, sep='\t', header=None)
        else:
            raise ValueError('Data type {} does not exist. Use json, pickle or pandas'.format(dtype))
        return return_obj

def store_obj(obj, fname, dtype='pickle'):
    """
    Object storing function to simplify code.
    
    Parameters:
    --------------
    fname: str, file name of stored object
    dtype: str, 'json' or 'pickle'
    
    Returns:
    --------------
    None
    """
    folder_fname = os.path.dirname(fname)
    if not os.path.exists(folder_fname):
        print ('{} does not exist and has been created'.format(folder_fname))
        os.makedirs(folder_fname)
    else:
        if dtype == 'pickle':
            with open(fname, 'wb') as f:
                pickle.dump(obj, f)
        elif dtype == 'json':
            with open(fname, 'w') as f:
                json.dump(obj, f)

store_folder = '../../workspace/dev_data/'
data_fname = '../../workspace/data/'
print ('Loading training data ...')
c_set_fname = '../../workspace/challenge_data/challenge_set.json'
d_set_fname = '../../recsys2018/src/title2vec/result/dev_set.json'
t_set_fname = '../../recsys2018/src/title2vec/result/test_set.json'


if not os.path.exists(store_folder):
    print ('Creating {} ...'.format(store_folder))
    os.makedirs(store_folder)

def load_inclusion_tracks(c_set_fname, d_set_fname, t_set_fname):
    print ('... Loading challenge, dev and test set tracks ...')
    # load challenge set
    #c_set = load_obj(c_set_fname, 'json')
    c_set_tracks = set()

    #for p in c_set['playlists']:
    #    for t in p['tracks']:
    #        c_set_tracks.add(t['track_uri'])

    # load dev set
    d_set = load_obj(d_set_fname, 'json')
    for d in d_set:
        for t in d['tracks']:
            c_set_tracks.add(t)

    # load test set
    t_set = load_obj(t_set_fname, 'json')
    for d in t_set:
        for t in d['tracks']:
            c_set_tracks.add(t)

    return c_set_tracks


#sys.exit()

#print ('Storing challenge tracks ...')
#store_obj(
#    c_set_tracks, 
#    os.path.join(store_folder, 'challenge_set_tracks.pckl'),
#    'pickle')

def _build_vocabulary(track_sequence):

    print ('... Creating dictionaries ...')
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
    print ('... Finished counting ...')
    
    # create filter dict
    new_track2id = {}
    n = len(counter)
    for ix, track in enumerate(counter):
        if ix % 1000 == 0:
            print ('{} / {}'.format(ix, n))
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
        self.all_playlist_filenames = [os.path.join(self.playlist_folder, x) for x in os.listdir(self.playlist_folder) if 'mpd' in x]
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
            total_files = len(os.listdir(self.playlist_folder))
            counter = 0
            for playlist_file in self.all_playlist_filenames:
                counter += 1
                print ("Working on slice {} ({:.2f} %) (File Name:  {} || Total Slices: {})".format(
                    counter, (counter / total_files) * 100, playlist_file, total_files), end='                   \r')
                playlist_collection = load_obj(playlist_file, 'json')
                for playlist in playlist_collection['playlists']:
                    self.all_playlists_dict[playlist['pid']] = {'pid': playlist['pid'], 'tracks': []}
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
                    tmp_playlist_features = [playlist[x] for x in playlist.keys() if 'tracks' not in str(x) and 'description' not in str(x)]
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

import time

if __name__ == "__main__":
    #s = s
    #print ('Sleeping for {} seconds'.format(s))
    #time.sleep(s)

    PLAYLIST_FOLDER = '../../workspace/data'
    RESULTS_FOLDER = '../../workspace/final_submission/results'
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print ('Results folder created ...')

    # dicts
    # map track or artist uri with their string
    ## Compute Popularity of Songs

    recompute = True
    random_state = 2018
    np.random.seed(random_state)
    # first sample
    statistician = Statistician(PLAYLIST_FOLDER, RESULTS_FOLDER)

    print ('Generating popularity dict ...')
    track_popularity_dict = statistician.create_track_popularity_dict(recompute=False)
    # generate sampling information
    # stratification on median track popularity, number of tracks and modified at
    print ('Generating playlist DataFrame inclusive aggregate features ...')
    playlist_df = statistician.get_playlist_df(recompute=False)

    # binning for stratification process
    print ('Generating stratification classes ...')
    playlist_df = create_stratification_classes(playlist_df)

    print ('Splitting data into train, test and dev sets ...')
    x_train_pids, x_dev_pids, x_test_pids = split_playlist_df(playlist_df, random_state, statistician.all_playlists_dict, RESULTS_FOLDER, recompute)
    
    print ('Loading training set ...')
    all_train_playlists = generate_all_train_playlist_set(x_train_pids, statistician, RESULTS_FOLDER, recompute)
    
    c_set_tracks = load_inclusion_tracks(c_set_fname, d_set_fname, t_set_fname)

    # load playlists
    x_train = []
    print ('Working on training set ...')
    for p in all_train_playlists:
        tmp_playlist = all_train_playlists[p]['tracks']
        tmp_playlist.append('<eos>')
        x_train.extend(tmp_playlist)

    print ('Extracting sequences and building vocabulary ...')
    track2id, track_sequence = _build_vocabulary(x_train)
    
    print ('Load inclusion tracks from dev and test sets ...')
    c_set_tracks = load_inclusion_tracks(c_set_fname, d_set_fname, t_set_fname)
    print ('Filtering sequences ...')
    track2id, track_sequence = _filter_sequence(track_sequence, track2id, 5, c_set_tracks)

    print ('Transforming track-uri sequences in int sequences ...')
    track_sequence = sequences_to_ids(track_sequence, track2id)

    print ('Storing id_sequence file ...')
    store_obj(track_sequence, os.path.join(RESULTS_FOLDER, 'id_sequence.pckl'), 'pickle')
    print ('Storing vocabulary file ...')
    store_obj(track2id, os.path.join(RESULTS_FOLDER   , 'track2id.pckl'), 'pickle')
