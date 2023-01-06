import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import pymc as pm
import aesara
import aesara.tensor as at
import arviz as az

from modelling_functions import (
    normalize, 
    masked_mean,
    get_data
)


def calculate_polite_pymc(s1_rationality,
                          verosimilitude, 
                          pretend_temp,
                          S1_weights,
                          values):
    """
    Parameters
    ----------
    s1_rationality: array
        Shape: (id, goal)
        Rationality parameter of the pragmatic speaker
    verosimilitude: array
        Shape: (id, goal)
        Between 0 and 1
        When 1, participnat only cares about being truthful,
        When 0 participant only cares about being kind.
        Encodes the importance of sticking close to the true value.
        Can be thought of as the inverse of the social punishment 
        for straying from the truth.
    pretend_temp: array
        Shape: (id, goal)
        >=0
        The temperature parameter of the pretend distribution.
        At 0, participant is equally likely to pretend to be in all states.
        As ->\infty, tends to pick only most useful state
    S1_weights: array
        Shape: (id, goal, 3)
        The importance of each of the speaker's utility components:
        0: informativeness, 1: social value, 2: cost
    Returns
    -------
    """
    
    def lognormalize(x, axis):
        return x - pm.logsumexp(x, axis=axis, keepdims=True)
    
    def normalize(x, axis):
        return x / x.sum(axis=axis, keepdims=True)
    
    ###### Stuff in normal numpy
    
    # Smooth out slightly the truth values
    # So that gradient is defined
    likestate_utterance_compatibility = np.where(
        np.array([
            # terrible
            [1, 0, 0, 0, 0],
            # bad
            [1, 1, 0, 0, 0],
            # good
            [0, 0, 0, 1, 1],
            # excellent
            [0, 0, 0, 0, 1],
            # not terrible
            [0, 1, 1, 1, 1],
            # not bad
            [0, 0, 1, 1, 1],
            # not good
            [1, 1, 1, 0, 0],
            # not excellent
            [1, 1, 1, 1, 0]
        ]),
        1,
        1e-10
    )
    
    costs = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    L0 = normalize(likestate_utterance_compatibility,1)
    
    Xs, Zs = np.indices((5,5))    
    
    ##### Stuff with pymc
    
    # Dims: (utterance)
    # NOTE: This only works if likestate_utterance_compatibility
    # is smoothed. Otherwise you have to use a masked mean.
    # Expected social value of each utterance
    expected_social_value = at.mean(
        values * L0, 
        1,
        keepdims=True
    )
    
    # unnormalized logp_present in log space
    # shape (id, goal, true state, pretend state)
    logp_pretend_unnorm = pretend_temp[:,:,None,None] * (
        (1-verosimilitude[:,:,None,None]) * values
        # add a state dimension
        - verosimilitude[:,:,None,None] * (np.abs(Xs - Zs)**2)
    )
    logp_pretend = lognormalize(logp_pretend_unnorm, 3)
    
    # shape (id, goal, utterance, state)
    log_S1_unnorm = s1_rationality[:,:,None,None]*(
          S1_weights[:,:,0,None,None]*np.log(L0)
        # add 
        + S1_weights[:,:,1,None,None]*expected_social_value
        # None adds state dimension
        - S1_weights[:,:,2,None,None]*costs[:,None]
    )
    log_S1 = lognormalize(log_S1_unnorm,2)

    # shape: (id, goal, true state, pretend state, utterance)
    log_y = (
        # prob given each true state of pretending 
        # to be in each of the other states 
        # (for each partic in each condition)
        logp_pretend[:,:,:,:,None] 
        # add a dimension for the true state
        + log_S1.dimshuffle(0,1,'x',3,2)
    )
    
    # shape: (id, goal, utterance, observed state)
    log_S1_polite = pm.logsumexp(log_y,3).dimshuffle(0,1,3,2)
    
    return log_S1_polite


def my_model_factory(df):
    
    with pm.Model() as model:
        
        ##### DATA
        
        goal = pm.MutableData(
            'goal_id', 
            df.goal_id
        )
        true_state = pm.MutableData(
            'true_state', 
            df.true_state
        )
        part_id = pm.MutableData(
            'part_id', 
            df.id
        )
        utterance_index = pm.MutableData(
            'utterance_index', 
            df.utterance_index
        )
        
        ##### PRIORS
        
        # Rationality is specified for each participant
        s1_rationality = pm.Exponential(
            "s1_rationality",
            lam=0.9,
            shape=203
        )
        s1_rationality = s1_rationality[:,None]
        
        # verosimilitude is specified for each condition as a whole
        # plus a hierarchical participant offset
        
        # condition baselines
        verosim_condition = pm.Normal(
            "verosim_condition",
            shape=3
        )
        # condition baseline stds
        verosim_sigma = pm.HalfNormal(
            'verosim_sigma',
            shape=3
        )
        # participants' normalized offsets
        verosim_zs = pm.Normal(
            "verosim_zs",
            shape=203
        )
        # verosimilitude param 
        # Shape (participant, condition)
        verosimilitude = pm.Deterministic(
            "verosimilitude",
            at.expit(
                # overall condition means
                verosim_condition + 
                # participant-wise offsets
                (verosim_zs[:,None]*verosim_sigma)
            )
        )
        
        # temperature of pretend, 
        # one for each participant
        pretend_temp = pm.Exponential(
            "pretend_temp",
            lam=0.9,
            shape=203
        )
        pretend_temp = pretend_temp[:,None]
        
        ### TODO: Find better parameterization of this prior
        ### Currently it's just a sum of all positive values!
        ### E.g. use a normal for S1_weight_zs instead of half-normal
        
        # overall S1_weight of each component
        # for each condition 
        S1_weight_condition = pm.HalfNormal(
            "S1_weight_condition",
            shape=(1,3,3)
        )
        # overall baseline std of each component
        # for each condition
        S1_weight_sigma = pm.HalfNormal(
            "S1_weight_sigma",
            shape=(1,3,3)
        )
        # offsets of each participant
        S1_weight_zs = pm.HalfNormal(
            "S1_weight_zs",
            shape=(203,1,1)
        )
            
        # for each participant, 
        # the weight of the three components
        # in each condition
        S1_weights = pm.Dirichlet(
            "S1_weights",
            a=S1_weight_condition+(S1_weight_sigma*S1_weight_zs),
            shape=(203,3,3)
        )
        
        # values = np.array([-8., -2., 0., 1., 1.5])
        values = pm.Normal(
            "values",
            mu=0,
            sigma=5,
            shape=5,
            initval=[-8, -2, 0, 1, 1.2],
            transform=pm.distributions.transforms.ordered
        )
        
        #### LIKELIHOOD

        # shape (id, goal, utterance, state)
        log_S1 = calculate_polite_pymc(
            s1_rationality,
            verosimilitude, 
            pretend_temp,
            S1_weights,
            values
        )
        
        # remove the middle state from S1
        log_S1 = log_S1[:,:,:,[0,1,3,4]]

        # shape (observation, possible utterance)
        logp_production = log_S1[part_id,goal,:,true_state]

        chosen_utterances = pm.Categorical(
            'chosen_utterances',
            p=pm.math.exp(logp_production),
            observed=utterance_index
        )
    return model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    
    model = my_model_factory(dt)
    
    with model:
        trace = pm.sample(
            target_accept=0.99
        )
    
    trace.to_netcdf("traces/trace_mymodel.cdf")
