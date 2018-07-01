"""
Metrics module for RecSys 2018 challenge. Implements r-precision,
normalized discounted cumulative gain, recommend songs count and recall.
"""

import numpy as np

def r_precision(ground_truth, prediction):
    """
    R-precision is the number of retrieved relevant tracks divided by the number of known relevant tracks.

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list of predictions a given algorithm returns.
    
    Returns:
    ------------
    relevant_tracks: list of all relevant tracks in order of appearance in prediction set
    r-precision metric: float measure of r-precision
    """
    relevant_tracks = []
    for idx, track in enumerate(prediction):
        if track in ground_truth and idx < len(ground_truth):
            relevant_tracks.append(track)
    return relevant_tracks, (len(relevant_tracks) / float(len(ground_truth)))


def get_relevance(ground_truth, item):
    """
    Returns relevance measure for playlist predictions.

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations.
    item: recommendation that needs to be checked

    Returns:
    ------------
    relevance: 1 if track is in ground_truth, 0 otherwise
    """
    if item in ground_truth:
        return 1
    return 0

def dcg(ground_truth, prediction):
    """
    Discounted cumulative gain (DCG) measures the ranking quality of the recommended tracks.
    DCG increases when relevant tracks are placed higher in the list. 

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list of predictions a given algorithm returns.
    
    Returns:
    ------------
    relevance: float representing the relevance metric for a given playlist prediction
    """
    relevance = None
    for idx, track in enumerate(prediction):
        if not relevance:
            relevance = get_relevance(ground_truth, track)
        else:
            relevance += get_relevance(ground_truth, track) / float(
                np.log2(idx + 1))
    return relevance


def idcg(ground_truth, prediction):
    """
    Maximum (ideal) discounted cumulative gain measure 
    for a prediction set. The ideal DCG simulates a situation
    in which the recommended tracks are perfectly ranked.

    Parameters:
    ------------
    ground_truth: list, elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list, predictions a given algorithm returns.

    Returns:
    ------------
    IDCG: float, ideal discounted cumulative gain
    """
    relevance = None
    idx = 0
    for track in set(ground_truth).intersection(set(prediction)):
        idx += 1
        if not relevance:
            relevance = 1
        else:
            relevance += 1 / float(np.log2(idx))
    return relevance


def ndcg(ground_truth, prediction):
    """
    Normalized discounted cumulative gain (NDCG). 

    > NDCG = DCG / IDCG

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list of predictions a given algorithm returns.
    
    Returns:
    ------------
    NDCG: float - discounted cumulative gain given the ideal discounted cumulative gain
    
    """
    idcg_val = idcg(ground_truth, prediction)
    if idcg_val and idcg_val > 0.0:
        return dcg(ground_truth, prediction) / float(
            idcg_val)
    else:
        return 0.0


def rsc(ground_truth, prediction):
    """
    Recommended Songs is a Spotify feature that, given a set of 
    tracks in a playlist, recommends 10 tracks to add to the playlist. 
    The list can be refreshed to produce 10 more tracks. 
    Recommended Songs clicks is the number of refreshes 
    needed before a relevant track is encountered
    
    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list of predictions a given algorithm returns.
    
    Returns:
    ------------
    counter: amount of clicks needed for the first relevant song to appear
    """
    counter = 0
    for idx, track in enumerate(prediction):
        if idx % 10 == 0:
            counter += 1
        if track in ground_truth:
            return counter
    return counter + 1


def recall(ground_truth, prediction):
    """
    Returns recall for a given retrieval task. 
    Recall can be defined as the number of relevant predictions given
    all relevant documents. 

    Parameters:
    --------------
    ground_truth: list, elements representing all known relevant items
    prediction:   list, predictions
    """
    return len(set(prediction).intersection(set(ground_truth))) / len(ground_truth)


def evaluate(pred_set, test_set, exclude_cold=False):
    """
    RecSys specific evaluation method. Returns a dictionary
    with a summary of all metric calculations.

    Parameters:
    --------------
    pred_set:     dict, {'k': []} k = seed bucket, maps to list of lists with 500 recommendations each
    test_set:     dict, {'k': []}
    exclude_cold: bool, flag if set True, 0 seed is being excluded

    Returns:
    --------------
    result_dict:  dict, {'metric_name': float}
    """
    result_dict = {}
    for key in test_set.keys():
        if exclude_cold and key == '0':
            continue
        result_dict[key] = {}
        all_r_precs = []
        all_ndcg = []
        all_rsc = []
        all_recall = []
        preds = pred_set[key]
        gt = [x['groundtruth'] for x in test_set[key]]
        for x, y in zip(gt, preds):
            all_r_precs.append(r_precision(x, y)[1])
            all_ndcg.append(ndcg(x, y))
            all_rsc.append(rsc(x, y))
            all_recall.append(recall(x, y))
        result_dict[key]['r_precision'] = np.mean(all_r_precs)
        result_dict[key]['ndcg'] = np.mean(all_ndcg)
        result_dict[key]['rsc'] = np.mean(all_rsc)
        result_dict[key]['recall'] = np.mean(all_recall)
    return result_dict


def print_results(result_dict):
    """
    Prints recommendation result statement.

    Parameters:
    --------------
    result_dict: dict, output from evaluate method {'metric': float}

    Returns:
    --------------
    None
    """
    print ('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('k', 'r_precision', 'ndcg', 'rsc', 'recall'))
    print ('='*100)
    sorted_keys = sorted([int(x) for x in result_dict.keys()])
    for k in sorted_keys:
        print ('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            k, result_dict[str(k)]['r_precision'], 
            result_dict[str(k)]['ndcg'], 
            result_dict[str(k)]['rsc'],
            result_dict[str(k)]['recall']))


def main():
    # calc r-precision
    ground_truth = ['1', '2', '3', '5', '8', '99']
    prediction = ['5', '8', '13', '3']
    print (r_precision(ground_truth, prediction))

    # calc normalized discounted cumulative gain
    print(ndcg(ground_truth, prediction))

    # calculate recommended songs count
    ground_truth_rsc_one = [1]
    ground_truth_rsc_two = [499]
    ground_truth_rsc_three = [500]
    prediction_rsc_one = range(500)
    
    print (rsc(ground_truth_rsc_one, prediction_rsc_one))

if __name__ == "__main__":
    main()
