import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import pymc as pm
try:
    import aesara
    import aesara.tensor as at
except ModuleNotFoundError:
    import pytensor as aesara
    import pytensor.tensor as at
import arviz as az

from modelling_functions import (
    normalize, 
    masked_mean,
    get_data
)


def lognormalize(x, axis):
    return x - pm.logsumexp(x, axis=axis, keepdims=True)


def normalize(x, axis):
    return x / x.sum(axis=axis, keepdims=True)
    

def yoon_S1(values, alpha, costs, L, phi_grid_n=40, lib='np'):

    if lib == 'at':
        lib = at
    else:
        lib == np
    
    # dimensions (utterance, state)
    L0 = normalize(L,1)

    # expected value given each utterance
    # Shape (utterance)
    # NOTE: This only works if L is not
    # exactly 0 anywhere!
    exp_values = lib.mean(
        values*L0,
        axis=1
    )
        
    # using a grid approximation for phi
    phis = np.linspace(0,1,phi_grid_n)

    # p(u | s, phi)
    # where phi is essentially 
    # a politeness weight
    # Dimensions (phi, utterance, state)
    # NOTE: This is the same for all goal conditions
    S1 = normalize(
        lib.exp(alpha*(
            # informational component
            # (value of phi, utterance, state)
              phis    [:,None,None]*lib.log(L0)
            # social component
            # (value of phi, utterance, 1)
            + (1-phis)[:,None,None]*exp_values[:,None]
            # (utterance, 1)
            - costs[:,None]
        )),
        # normalize by utterance
        # for each discretized value of phi
        axis=1
    )
    
    return S1


def yoon_utilities(S1, phi, values):
        
    # Prob of state and phi given utterance
    # Dimensions (phi, utterance, state)
    L1_s_phi_given_w_grid = normalize(
        S1,
        (0,2)
    )
    
    # grid-marginalize over phi
    L1_s_given_w = L1_s_phi_given_w_grid.sum(0)
    
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (utterance, state)
    u_inf = pm.math.log(L1_s_given_w)

    # expected (value of state)
    # for each utterance as produced by L1
    # NOTE: Same for all goal conditions
    u_soc = at.mean(
        values*L1_s_given_w,
        1
    )

    # u_pres is the only component where phi is not marginalized 
    # and therefore I need to use the inferred phi (argument)
    # rather than grid approximation
    
    # Dimensions: (goal condition, utterance, state)
    # Get the probabilities with speaker's actual phi
    # pm.Deterministic('phi_val', aesara.printing.Print()(phi))
    L1_s_phi_given_w = L1_s_phi_given_w_grid[phi]
    
    # print("L1_s_phi_given_w: ", L1_s_phi_given_w.eval())
    
    # equation 3 in 2020 paper
    # (Note that the log doesn't appear in supplementary material!)
    # Shape (goal condition, utterance)
    u_pres = pm.math.log(
        L1_s_phi_given_w
        # marginalize across state
        # to get prob of each utterance given phi
        .sum(2)
    )
          
    # print("U soc: ", u_soc.eval())
    # print("U inf: ", u_inf.eval())
    # print("U pres: ", u_pres.eval())

    # shape (utility component, goal condition, utterance, state)
    utilities = at.stack(
        at.broadcast_arrays(
            # (utterance, state)
            u_inf,
            # (utterance,1)
            u_soc[:,None],
            # (goal condition, utterance, state)
            u_pres[:,:,None]
        ),
        axis=0
    )
    
    return utilities


def yoon_likelihood(alpha, values, omega, costs, phi, L, phi_grid_n=100, lib='np'):

    S1 = yoon_S1(values, alpha, costs, L, phi_grid_n, lib)

    utilities = yoon_utilities(S1, phi, values)
    
    # print("omega: ", (
    #     omega
    # ).eval())

    # shape (goal condition, utterance, state)
    util_total = (
        # shape (goal condition, utterance, state)
        (
            # dims: (utility component, goal condition, 1, 1)
            omega.T[:,:,None,None] 
            # dims: (utility component, goal condition, utterance, state)
            * utilities
        # sum weighted utility components together
        ).sum(0) 
        # (utterance, 1)
        - costs[:,None]
    )

    # print("util_total: ", util_total.eval())
    
    # for each goal condition,
    # prob of utterance given state
    # Shape: (goal_condition, utterance, state)
    S2 = normalize(
        pm.math.exp(alpha*util_total), 
        1
    )
    
    return S2


def factory_yoon_model(dt, dt_meaning):
    
    dt_meaning_pymc = (
        dt_meaning
        .groupby(['state', 'utterance_index'])
        ['judgment']
        .agg([np.sum, 'count'])
        .reset_index()
    )

    with pm.Model() as yoon_model:

        # literal semantic compatibility
        # shape: (utterances, states)
        # L = np.array([
        #     [0.961, 0.627, 0.039, 0.039],
        #     [0.980, 0.882, 0.039, 0.020],
        #     [0.001, 0.020, 0.941, 0.999],
        #     [0.001, 0.001, 0.216, 0.980],
        #     [0.001, 0.353, 0.980, 0.863],
        #     [0.020, 0.157, 0.999, 0.902],
        #     [0.961, 0.863, 0.039, 0.020],
        #     [0.961, 0.980, 0.627, 0.039]
        # ])

        L = pm.Uniform(
            'L',
            lower=0,
            upper=1,
            shape=(8,4)
        )

        pm.Binomial(
            'L_observed',
            n=dt_meaning_pymc['count'],
            p=L[
                dt_meaning_pymc['utterance_index'],
                dt_meaning_pymc['state']
            ],
            observed=dt_meaning_pymc['sum']
        )

        negative_cost = pm.Uniform(
            'c',
            lower=1,
            upper=10
        )

        costs = at.concatenate((
            at.ones(4),
            at.repeat(negative_cost,4)
        ))

        # >= 0 
        alpha = pm.Uniform(
            'alpha',
            lower=0,
            upper=20
        ) 

        values = np.array([0, 1, 2, 3])

        # goal weights
        # [informational, prosocial, presentational]
        # A triplet of goal weights for each goal condition!
        # Shape: (goal condition, utility component)
        omega = pm.Dirichlet(
            'omega',
            [1,1,1],
            shape=(3,3)
        )

        phi_grid_n = 100

        # politeness weight
        # One for each goal condition!
        # Shape: (goal condition)
        phi = pm.Categorical(
            'phi',
            p=np.ones(phi_grid_n)/phi_grid_n,
            shape=(3)
        )

        # Shape (condition, utterance, state)
        S2 = yoon_likelihood(
            alpha, 
            values, 
            omega, 
            costs, 
            phi,
            L,
            phi_grid_n=phi_grid_n,
            lib='at'
        )

        # each combination of goal and state
        # should give a prob vector over utterances
        print(S2.eval().sum((1)))

        p_production = S2[
            dt.goal_id,
            :,
            dt.true_state
        ]

        pm.Categorical(
            "chosen",
            p_production,
            observed=dt.utterance_index.values,
            shape=len(dt)
        )
        
    return yoon_model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    
    yoon_model = yoon_model_functions.factory_yoon_model(
        dt,
        dt_meaning
    )
    
    with yoon_model:
        yoon_trace = pm.sample(
            draws=10000
        )
    
    az.to_netcdf(
        yoon_trace, 
        'traces/yoon_trace_nonhierarchical.cdf'
    )
    