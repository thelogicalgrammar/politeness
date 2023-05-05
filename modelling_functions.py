import numpy as np
import seaborn as sns
import pandas as pd

import pymc as pm
try:
    import aesara
    import aesara.tensor as at
except ModuleNotFoundError:
    # If pymc v5 is installed
    import pytensor as aesara
    import pytensor.tensor as at
import arviz as az


def normalize(arr, axis):
    return arr / np.sum(arr, axis, keepdims=True)


def masked_mean(arr, mask):
    """
    Calculate the (row)mean of only those values
    in arr where mask is true
    """
    
    arr, mask = np.broadcast_arrays(
        arr, mask
    )
    
    return np.ma.masked_where(
        np.logical_not(mask), 
        arr
    ).mean(1).data.reshape(-1,1)


def get_data(path="../yoon_data/speaker_production.csv",
             path_meaning='../yoon_data/literal_semantics.csv'):

    dt = pd.read_csv(path)

    us = [
        'terrible',
        'bad',
        'good',
        'amazing',
        'not terrible',
        'not bad',
        'not good',
        'not amazing'
    ]

    ids, _ = dt.subid.factorize()
    dt.loc[:,'id'] = ids
    dt = dt.drop(columns='subid')
    dt.loc[:,'true_state'] = dt['true_state'].str.lstrip('heart').astype(int)

    dt.loc[:,'utterance_full'] = np.where(
        dt['positivity']=='no_neg',
        dt.utterance,
        'not ' + dt.utterance
    )
    utt_i = {u:i for i,u in enumerate(us)}
    dt.loc[:,'utterance_index'] = dt.utterance_full.replace(utt_i)

    goal_id, goals = dt.goal.factorize()
    dt.loc[:,'goal_id'] = goal_id

    dt_meaning = pd.read_csv(path_meaning)

    dt_meaning.loc[:,'neg'] = np.where(
        (dt_meaning.positivity == 'it was ___'),
        '',
        'not '  
    )
    
    dt_meaning.loc[:,'utterance_index'] = (
        dt_meaning['neg'] 
        + dt_meaning['utterance']
    ).replace(utt_i)
    
    return dt, utt_i, goal_id, goals, dt_meaning
