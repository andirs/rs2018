import os
import pickle
import json
import pandas as pd

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
            with open(fname, 'rb') as f:
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
            with open(fname, 'rb') as f:
                json.dump(obj, f)

def load_challenge_set(fname='../../../workspace/challenge_data/challenge_set.json'):
    return load_obj(fname, 'json')