import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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
    get_data,
    Integrate
)


def lognormalize(x, axis):
    return x - pm.logsumexp(x, axis=axis, keepdims=True)


def normalize(x, axis):
    return x / x.sum(axis=axis, keepdims=True)
    

def yoon_S1(phi, alpha, costs, L0, exp_values):
    """
    Return the production probabilities of S1
    with the given values of alpha, phi, costs, and L
    
    If phi is a scalar, returns shape (utterance, state)
    if phi has shape (goal condition), return (goal condition, utterance, state)
    """
    
    if L0.ndim == 1:
        # if L0 is passed as a vector, 
        # reshape before computations
        L0 = L0.reshape((8,4))
    
    if phi.ndim == 0:
        phi_ = phi
    elif phi.ndim == 1:
        # (goal_condition, utterance, state)
        phi_ = phi[:,None,None]
    else:
        raise ValueError("Phi has a strange shape")

    # p(u | s, phi)
    # where phi is essentially a politeness weight
    # Dimensions (goal_condition, utterance, state)
    # or (utterance, state)
    # NOTE: This is the same for all goal conditions
    S1 = normalize(
        at.exp(alpha*(
            # informational component
              phi_*at.log(L0)
            # social component
            + (1-phi_)*exp_values[:,None]
            # (utterance, 1)
            - costs[:,None]
        )),
        # normalize each state
        axis=-2
    )
    
    return S1


def yoon_utilities(S1, phi, values, S1_integrated_phi):
    
    # sum over states to get  
    # the normalization factor for L1
    # Dims: (utterance,1)
    S1_norm = S1_integrated_phi.sum(-1, keepdims=True)
    
    # Prob of state and phi given utterance
    # Get the probabilities with speaker's actual phi
    # dims: (goal_condition, utterance, state)
    L1_s_phi_given_w = S1 / S1_norm
        
    # marginalize over phi
    # By normalizing S1_integrated_phi
    # across states (each utterance is a prob vector)
    # See politeness.ipynb for explanation
    # Dims: (utterance, state)
    L1_s_given_w = normalize(
        S1_integrated_phi,
        -1
    )
    
    # informativity of utterances given state
    # with utterances produced by L1
    # Shape (utterance, state)
    u_inf = pm.math.log(L1_s_given_w)

    # expected (value of state)
    # for each utterance as produced by L1
    # NOTE: Should be the same for all goal conditions
    # Dims: (utterance)
    u_soc = at.mean(
        values*L1_s_given_w,
        -1
    )
    
    # print("L1_s_phi_given_w: ", L1_s_phi_given_w.eval())
    
    # equation 3 in 2020 paper
    # (Note that the log doesn't appear in supplementary material!)
    # Shape (goal condition, utterance)
    u_pres = pm.math.log(
        L1_s_phi_given_w
        # marginalize across state
        # to get prob of each utterance given phi
        .sum(-1)
    )
          
    print("U soc: ", u_soc.eval())
    print("U soc shape: ", u_soc.eval().shape)
    print("U inf: ", u_inf.eval())
    print("U inf shape: ", u_inf.eval().shape)
    print("U pres: ", u_pres.eval())
    print("U pres shape: ", u_pres.eval().shape)

    # shape (utility component, goal condition, utterance, state)
    utilities = at.stack(
        at.broadcast_arrays(
            # (utterance, state)
            u_inf,
            # (goal condition, utterance,1)
            u_soc[:,None],
            # (goal condition, utterance, 1)
            u_pres[:,:,None]
        ),
        axis=0
    )
    
    return utilities


def yoon_likelihood(alpha, values, omega, costs, phi, L):

    # dimensions (utterance, state)
    L0 = normalize(L,1)

    # expected value given each utterance
    # Shape (utterance)
    # NOTE: This only works if L is not
    # exactly 0 anywhere!
    exp_values = at.mean(
        values*L0,
        axis=1
    )
    
    # Get the speaker with the actual value of phi
    # dims (goal condition, utterance, state)
    S1 = yoon_S1(
        phi, 
        alpha, 
        costs, 
        L0, 
        exp_values
    )
    
    ###### Define integration Op
    phi_ = at.dscalar('phi')
    phi_.tag.test_value = np.array(0.2)
    alpha_ = at.dscalar('alpha')
    alpha_.tag.test_value = np.array(2.)
    costs_ = at.dvector('costs')
    costs_.tag.test_value = np.array(
        [1,1,1,1,2,2,2,2]
    ).astype(float)
    L0_testvalue = np.array([
        [0.961, 0.627, 0.039, 0.039],
        [0.980, 0.882, 0.039, 0.020],
        [0.001, 0.020, 0.941, 0.999],
        [0.001, 0.001, 0.216, 0.980],
        [0.001, 0.353, 0.980, 0.863],
        [0.020, 0.157, 0.999, 0.902],
        [0.961, 0.863, 0.039, 0.020],
        [0.961, 0.980, 0.627, 0.039]
    ])
    L0_ = at.dvector('L0')
    L0_.tag.test_value = L0_testvalue.flatten()
    exp_values_ = at.dvector('exp_values')
    values_testvalue = np.array(
        [0,1,2,3]
    ).astype(float)
    exp_values_.tag.test_value = np.mean(
        values_testvalue*L0_testvalue,
        axis=1
    )

    # dims (utterance, state)
    # define the expressions to integrate
    S1_component = yoon_S1(
        phi_, 
        alpha_, 
        costs_, 
        L0_, 
        exp_values_
    ).flatten()

    integrate = Integrate(
        # "cost": expression we're integrating
        S1_component, 
        # wrt: variable wrt which we're integrating
        # NOTE: phi is a scalar here!!
        phi_, 
        # other variables
        alpha_, 
        costs_, 
        L0_, 
        exp_values_
    )
    
    ###### Use integration Op

    start = aesara.shared(0.)
    stop = aesara.shared(1.)
    # Now we plug in the values from the model.
    # The result is S1 with phi integrated out.
    # Dims: (utterance, state)
    S1_integrated_phi = integrate(
        start, 
        stop, 
        alpha, 
        costs, 
        L0.flatten(), 
        exp_values
    ).reshape(L0.shape)
    
    print("S1_integrated_phi: ", S1_integrated_phi)
    print("S1_integrated_phi ndim: ", S1_integrated_phi.ndim)

    utilities = yoon_utilities(
        S1, 
        phi, 
        values,
        S1_integrated_phi
    )
    
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

        values = np.array([0., 1., 2., 3.])

        # goal weights
        # [informational, prosocial, presentational]
        # A triplet of goal weights for each goal condition!
        # Shape: (goal condition, utility component)
        omega = pm.Dirichlet(
            'omega',
            [1,1,1],
            shape=(3,3)
        )

        # politeness weight
        # One for each goal condition!
        # Shape: (goal condition)
        phi = pm.Beta(
            'phi',
            alpha=1,
            beta=1,
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
        )

        # each combination of goal and state
        # should give a prob vector over utterances
        # print(S2.eval().sum((1)))

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
    
#     dt, utt_i, goal_id, goals, dt_meaning = get_data()
    
#     yoon_model = factory_yoon_model(
#         dt,
#         dt_meaning
#     )
    
#     with yoon_model:
#         yoon_trace = pm.sample(
#             draws=1000
#         )
    
#     az.to_netcdf(
#         yoon_trace, 
#         'traces/yoon_trace_nonhierarchical_integrate.cdf'
#     )
    
    
    ### Example of usage

    y_obs = np.array([8.3, 8.0, 7.8])
    start = aesara.shared(1.)
    stop = aesara.shared(2.)

    with pm.Model() as basic_model:

        a = pm.Uniform(
            'a', 
            1.5, 
            3.5
        )
        b = pm.Uniform(
            'b', 
            4., 
            6., 
            shape=(3)
        )

        # Define the function to integrate in plain pytensor
        t = at.dscalar('t')
        t.tag.test_value = np.array(0.)

        a_ = at.dscalar('a_')
        a_.tag.test_value = np.array(2.)

        b_ = at.dvector('b_')
        b_.tag.test_value = np.ones((3))*5.
        # b_ = at.dscalar('b_')
        # b_.tag.test_value = np.array(5.0)

        func = t**a_ + b_
        integrate = Integrate(
            # Function we're integrating
            func, 
            # variable we're integrating
            t, 
            # other variables
            a_, b_,
        )

        # Now we plug in the values from the model.
        # The `a_` and `b_` from above corresponds 
        # to the `a` and `b` here.
        mu = integrate(
            start, 
            stop, 
            a, 
            b
        )
        
        y = pm.Normal(
            'y', 
            mu=mu, 
            sigma=0.4, 
            observed=y_obs
        )
        
    with basic_model:
        trace = pm.sample(
            1500, 
            tune=500, 
            cores=2, 
            chains=2,
            return_inferencedata=True
        )