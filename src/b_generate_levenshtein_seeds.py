import nltk
import numpy as np
import os
import pandas as pd

from collections import Counter
from tools.io import load_obj, store_obj

print ('#' * 80)
print ('Track2Seq Levenshtein Seeds')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

t2s_config = load_obj('config.json', 'json')  # all configuration files can be set manually as well
RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']  # all information will be stored here
RANDOM_STATE = t2s_config['RANDOM_STATE']
recompute = True  

np.random.seed(RANDOM_STATE)

##################################################################
############################# METHODS ############################
##################################################################

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
    def get_seed_tracks(playlist_df, return_dict, all_playlists_dict, seed_k=100):
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
            while candidate_counts < seed_k and i < len(tmp_pids):

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

        return Counter(candidate_list).most_common(seed_k)


    @staticmethod
    def generate_levenshtein_seed_dict(
        zero_seed_playlists, 
        all_playlist_names, 
        all_playlists_dict, 
        playlist_df, 
        RESULTS_FOLDER, 
        filename, 
        recompute,
        seed_k=100):
        fname = os.path.join(RESULTS_FOLDER, filename)
        if recompute:
            comp_memory = {}
            seed_set = {}
            for idx, playl in enumerate(zero_seed_playlists):
                playlist_name = Levenshtein.pre_process(playl['name'])
                print ('\r{:.2f} % :: Retrieving levenshtein similarities for \'{}\''.format(
                    ((idx + 1) / len(zero_seed_playlists)) * 100, playlist_name), end='')
                return_dict = {}
                return_dict['counter'] = 0
                return_dict['lowest'] = []
                return_dict['targets'] = []
                _ = all_playlist_names.apply(Levenshtein.get_closest, args=(playlist_name, return_dict, comp_memory))
                seeds = Levenshtein.get_seed_tracks(playlist_df, return_dict, all_playlists_dict, seed_k=seed_k)
                seed_set[playl['pid']] = [x[0] for x in seeds]

            store_obj(seed_set, fname, 'pickle')
        else:
            seed_set = load_obj(fname, 'pickle')

        return seed_set


##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":
    playlist_df_fname = os.path.join(RESULTS_FOLDER, 'playlist_df.csv')
    x_train_pids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl')
    dev_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'dev_playlist_dict.pckl')
    all_playlists_dict_fname = os.path.join(RESULTS_FOLDER, 'all_playlists_dict.pckl')

    playlist_df = pd.read_csv(playlist_df_fname, index_col=0)
    x_train_pids = load_obj(x_train_pids_fname, 'pickle')
    
    # LEVENSHTEIN
    print ('Computing Levenshtein distance ...')
    train_playlist_df = playlist_df[playlist_df.pid.isin(x_train_pids)].copy()
    train_playlist_df.reset_index(inplace=True)
    del(playlist_df)

    print ('Preprocessing all playlist names in training set ...')
    all_playlist_names = train_playlist_df['name'].apply(Levenshtein.pre_process)

    print ('Loading evaluation data set ...')
    dev_playlist_dict = load_obj(dev_playlist_dict_fname, 'pickle')
    zero_dev = dev_playlist_dict[0]
    
    print ('Starting with Levenshtein seeds ...')
    print ('Loading all playlists ...')
    all_playlists_dict = load_obj(all_playlists_dict_fname, 'pickle')
    # iterate over first 0-seed playlists
    dev_leve_seed_dict = Levenshtein.generate_levenshtein_seed_dict(
        zero_dev, all_playlist_names, all_playlists_dict, 
        train_playlist_df, RESULTS_FOLDER, 'dev_leve_seed_dict.pckl', recompute, seed_k=100)  # change to 500 to get recommendation format