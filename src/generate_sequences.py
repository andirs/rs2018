import collections
import json
import os
import pandas as pd
import pickle
import sys

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

store_folder = '../../workspace/final_data/'
data_fname = '../../workspace/data/'
print ('Loading training data ...')
c_set_fname = '../../workspace/challenge_data/challenge_set.json'

if not os.path.exists(store_folder):
    print ('Creating {} ...'.format(store_folder))
    os.makedirs(store_folder)

print ('Loading challenge set tracks ...')
c_set = load_obj(c_set_fname, 'json')
c_set_tracks = set()

for p in c_set['playlists']:
    for t in p['tracks']:
        c_set_tracks.add(t['track_uri'])

#c_set_tracks = list(c_set_tracks)
print ('Storing challenge tracks ...')
store_obj(
    c_set_tracks, 
    os.path.join(store_folder, 'challenge_set_tracks.pckl'),
    'pickle')

def _build_vocab(track_sequence):

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


if __name__ == "__main__":
    all_train_playlists_fname = '../../workspace/artist2vec/all_train_playlists.pkl'

    
    # load playlists
    x_train = []
    print ('Loading training set ...')
    all_train_playlists = load_obj(all_train_playlists_fname, 'pickle')

    print ('Working on training set ...')
    for p in all_train_playlists:
        tmp_playlist = all_train_playlists[p]
        tmp_playlist.append('<eos>')
        x_train.extend(tmp_playlist)

    print ('Extracting sequences and building vocabulary ...')
    track2id, track_sequence = _build_vocab(x_train)
    
    print ('Filtering sequences ...')
    track2id, track_sequence = _filter_sequence(track_sequence, track2id, 5, c_set_tracks)

    print ('Transforming track-uri sequences in int sequences ...')
    track_sequence = sequences_to_ids(track_sequence, track2id)

    print ('Storing id_sequence file ...')
    store_obj(track_sequence, os.path.join(store_folder, 'id_sequence.pckl'), 'pickle')
    print ('Storing vocabulary file ...')
    store_obj(track2id, os.path.join(store_folder, 'track2id.pckl'), 'pickle')
