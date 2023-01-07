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
    

def yoon_S1_hierarchical(values, alpha, costs, L, phi_grid_n=40, lib='np'):

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
    # Dimensions (participant, phi value, utterance, state)
    # NOTE: This is the same for all goal conditions
    S1 = normalize(
        lib.exp(alpha[:,None,None,None]*(
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


def yoon_utilities_hierarchical(S1, phi, values):
    """
    Parameters
    ----------
    S1: float tensor
        dims: (participant, phi value, utterance, state)
    phi: int tensor
        dims: (participant, goal condition)
    values: float tensor
        dims: (utterance)
    """
        
    # Prob of state and phi given utterance and participant
    # Dimensions (participant, phi, utterance, state)
    L1_s_phi_given_w_grid = normalize(
        S1,
        (1,3)
    )
    
    # grid-marginalize over phi
    # dims: (participant, utterance, state)
    L1_s_given_w = L1_s_phi_given_w_grid.sum(1)
    
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (participant, utterance, state)
    u_inf = pm.math.log(L1_s_given_w)

    # expected *value of state*
    # for each utterance as produced by L1
    # NOTE: Same for all goal conditions
    # dims: (participant, utterance)
    u_soc = at.mean(
        values*L1_s_given_w,
        2
    )

    # u_pres is the only component where phi is not marginalized 
    # and therefore I need to use the inferred phi (argument)
    # rather than grid approximation
    
    # dims: (participant, goal condition, utterance, state)
    # Get the probabilities with speaker's actual phi
    # pm.Deterministic('phi_val', aesara.printing.Print()(phi))
    L1_s_phi_given_w = L1_s_phi_given_w_grid[
        # for each participant
        at.arange(L1_s_phi_given_w_grid.shape[0])[:,None],
        # get one production array per condition
        phi
    ]
    
    # print("L1_s_phi_given_w: ", L1_s_phi_given_w.eval())
    
    # equation 3 in 2020 paper
    # Shape (participant, goal condition, utterance)
    u_pres = pm.math.log(
        L1_s_phi_given_w
        # marginalize across state
        # to get prob of each utterance given phi
        .sum(3)
    )
          
    print("U soc: ", u_soc.eval().shape)
    print("U inf: ", u_inf.eval().shape)
    print("U pres: ", u_pres.eval().shape)
 
    # U soc:  (203,8) (participant, utterance)
    # U inf:  (203,8,4) (participant, utterance, state)
    # U pres: (203,3,8) (participant, goal condition, utterance)
    
    # shape (participant, goal condition, utterance, state)
    utilities = at.stack(
        at.broadcast_arrays(
            # (participant, 1, utterance, state)
            u_inf[:,None,:,:],
            # (participant, 1, utterance,1)
            u_soc[:,None,:,None],
            # (participant, goal condition, utterance, 1)
            u_pres[:,:,:,None]
        ),
        axis=0
    )
    
    return utilities


def yoon_likelihood_hierarchical(alpha, values, omega, costs, phi, L, phi_grid_n=100, lib='np'):
    """
    Parameters
    ----------
    alpha: tensor
        dims: participant
    values: tensor
        dims: state
    omega: tensor
        dims: (participant, goal condition, utility component)
    costs: tensor
        dims: (utterance)
    phi: tensor
        dims: (participant, goal condition)
    L: tensor
        dims: (utterance, state)
    Returns
    -------
    tensor
        The utterance probabilities
        Dims: (participant, goal condition, utterance, state)
    """
    
    # dims: (participant, phi value, utterance, state)
    S1 = yoon_S1_hierarchical(values, alpha, costs, L, phi_grid_n, lib)

    # dims: (participant, utility component, goal condition, utterance, state)
    utilities = yoon_utilities_hierarchical(S1, phi, values)
    
    # print("omega: ", (
    #     omega
    # ).eval())

    # shape (participant, goal condition, utterance, state)
    util_total = (
        # shape (participant, goal condition, utterance, state)
        (
            # dims: (utility component, participant, goal condition, 1, 1)
            omega.dimshuffle(2,0,1)[:,:,:,None,None] 
            # dims: (utility component, participant, goal condition, utterance, state)
            * utilities
        # sum weighted utility components together
        ).sum(0) 
        # (utterance, 1)
        - costs[:,None]
    )

    # print("util_total: ", util_total.eval())
    
    # for each goal condition,
    # prob of utterance given state
    # Shape: (participant, goal_condition, utterance, state)
    S2 = normalize(
        pm.math.exp(alpha[:,None,None,None]*util_total), 
        2
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

        ##### DATA

        goal = pm.MutableData(
            'goal_id', 
            dt.goal_id
        )
        true_state = pm.MutableData(
            'true_state', 
            dt.true_state
        )
        part_id = pm.MutableData(
            'part_id', 
            dt.id
        )
        utterance_index = pm.MutableData(
            'utterance_index', 
            dt.utterance_index
        )

        ##### PRIORS

        # Shape (participant, utterance, state)
        # Keep this non-hierarchical!
        # Since it (argubly) encodes a public fact about language
        # (and anyway it'd be impossible to fit)
        L = pm.Uniform(
            'L',
            lower=0,
            upper=1,
            shape=(8,4)
        )

        ### Costs prior
        # I assume that it's the same for all participants
        # So not hierarchical

        negative_cost = pm.Exponential(
            'c',
            lam=1
        )

        # dims: (utterance)
        costs = at.concatenate((
            at.ones(4),
            at.repeat(negative_cost+1,4)
        ))

        alpha_mu = pm.Normal(
            'alpha_mu'
        )

        alpha_sigma = pm.HalfNormal(
            'alpha_sigma'
        )

        # >= 0 
        # Dims: (participant)
        alpha = pm.LogNormal(
            'alpha',
            mu=alpha_mu,
            sigma=alpha_sigma,
            shape=203
        ) 

        values = np.array([0, 1, 2, 3])

        ##### Omega prior

        # overall omega of each component
        # for each condition 
        omega_condition = pm.HalfNormal(
            "omega_condition",
            shape=(1,3,3)
        )
        # overall baseline std of each component
        # for each condition
        omega_sigma = pm.HalfNormal(
            "omega_sigma",
            shape=(1,3,3)
        )
        # offsets of each participant
        # for each component
        omega_zs = pm.HalfNormal(
            "omega_zs",
            shape=(203,1,3)
        )

        # goal weights
        # [informational, prosocial, presentational]
        # A triplet of goal weights for each goal condition!
        # Dims: (participant, goal condition, utility component)
        omega = pm.Dirichlet(
            "omega",
            a=omega_condition+(omega_sigma*omega_zs),
            shape=(203,3,3)
        )

        ##### phi prior

        phi_grid_n = 100

        # mean of population-level distribution 
        # over phi parameter
        phi_goal_means = pm.Normal(
            'phi_goal_means',
            shape=3
        )

        phi_goal_stds = pm.HalfNormal(
            'phi_goal_stds',
            shape=3
        )

        phi_participant_zs = pm.Normal(
            'phi_participants_zs',
            shape=203
        )

        # Binomial parameter
        # for each participant for each goal
        # Dims: (participant, goal condition)
        phi_ps = pm.Deterministic(
            'phi_ps',
            at.expit(
                phi_goal_means
                + (phi_participant_zs[:,None] * phi_goal_stds)
            )
        )

        # politeness weight of each agent 
        # in each goal condition!
        # Shape: (participant, goal condition)
        phi = pm.Binomial(
            'phi',
            n=phi_grid_n-1,
            p=phi_ps,
            shape=(203, 3)
        )


        ##### LIKELIHOOD

        # Dims: (goal condition, participant, utterance, state)
        S2 = yoon_likelihood_hierarchical(
            # dims: participant
            alpha, 
            # dims: state
            values, 
            # dims: (participant, goal condition, utility component)
            omega, 
            # dims: (utterance)
            costs, 
            # dims: participant, goal condition
            phi,
            # dims: (utterance, state)
            L,
            phi_grid_n=phi_grid_n,
            lib='at'
        )

        p_production = S2[
            part_id,
            goal,
            :,
            true_state
        ]

        ##### OBSERVED

        pm.Categorical(
            "chosen",
            p_production,
            observed=utterance_index,
            shape=len(dt)
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
    
    return yoon_model


if __name__=='__main__':
    
    dt, utt_i, goal_id, goals, dt_meaning = get_data()
    
    yoon_model = factory_yoon_model(
        dt,
        dt_meaning
    )
    
    with yoon_model:
        yoon_trace = pm.sample(
            draws=5000,
            target_accept=0.95
        )
    
    az.to_netcdf(
        yoon_trace, 
        'traces/yoon_trace_hierarchical.cdf'
    )
