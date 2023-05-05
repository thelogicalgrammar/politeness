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
    # Needed for integration
    from scipy.integrate import quad, quad_vec
    from pytensor.graph.op import Op
    from pytensor.graph.basic import Apply
    from pytensor import clone_replace

import arviz as az

from modelling_functions import (
    normalize, 
    masked_mean,
    get_data,
)


class Integrate(Op):
    
    # Class to perform integration of one variable
    # on a bounded interval
    
    # Adapted from:
    # https://discourse.pymc.io/t/
    # custom-theano-op-to-do-numerical-integration/734/12
    # With some modifications!
    
    def __init__(self, expr, var, *extra_vars):
        """
        Parameters
        ----------
        expr: Aesara Variable
            The expression encoding the output
        var: Aesara Variable
            The input variable
        """
        super().__init__()
        
        # function we're integrating
        self._expr = expr
        
        # input var we're integrating over
        self._var = var
        
        # other variables
        self._extra_vars = extra_vars
        
        print("Input types: ", [i.type for i in [var] + list(extra_vars)])
        print("Output type: ", self._expr.type)
        print("Output dtype: ", self._expr.dtype)
        
        # transform expression into callable function
        self._func = aesara.function(
            # a list with all the inputs
            [var] + list(extra_vars),
            # output
            self._expr,
            on_unused_input='ignore'
        )
        
        self._func.trust_input=True
    
    def make_node(self, start, stop, *extra_vars):
        """
        This is called by Op.__call__
        
        creates an Apply node representing the application 
        of the Op on the inputs provided.
        
        - First:
            It first checks that the input Variables types are compatible 
            with the current Op. 
            If the Op cannot be applied on the provided input types, 
            it must raises an exception (such as TypeError).
        - Second:
            it operates on the Variables found in *inputs in Aesara’s 
            symbolic language to infer the type 
            of the symbolic output Variables. It creates output 
            Variables of a suitable symbolic Type to serve as 
            the outputs of this Op’s application.
            
        Create a node to be included in the expression graph. 
        This runs when we apply our Op (integrate) to the Variable inputs. 
        When an Op has multiple inputs, their order in the inputs argument to
        Apply is important: 
        Aesara will call make_node(*inputs) to copy the graph, so it is important 
        not to change the semantics of the expression 
        by changing the argument order.
        """
        
        self._extra_vars_node = [
            at.as_tensor_variable(ex)
            for ex in extra_vars
        ]

        # make sure that the same number of extra variables
        # are passed here as were specified when defining the Op
        assert len(self._extra_vars) == len(extra_vars)
        
        # Define the bounds of integration
        self._start = at.as_tensor_variable(start)
        self._stop = at.as_tensor_variable(stop)
                
        print("Make node input types: ", [
            (i.type, i.dtype)
            for i in
            [self._start, self._stop] + list(self._extra_vars_node)
        ])
        print(
            "Make node output type: ", 
            self._expr.type()
        )
        # return an Apply instance with the input and output Variable
        return Apply(
            # op: The operation that produces `outputs` given `inputs`.
            op=self, 
            # inputs: The arguments of the expression 
            # modeled by the `Apply` node.
            inputs=[self._start, self._stop] + list(self._extra_vars_node), 
            # outputs: The outputs of the expression modeled by the `Apply` node.
            # NOTE: This is a scalar if self._expr is a scalar,
            # and a vector if self._expr is a vector. Etc.
            outputs=[self._expr.type()]
        )
    
    def perform(self, node, inputs, out):
        """
        Out is the output storage.
        Inputs are passed by value.
        A single output is returned indirectly 
        as the first element of single-element lists (out)
        
        NOTE: There's a restriction, namely the variable to integrate
        has to be a scalar, even though the other variables can be any shape.
        
        Parameters
        ----------
        node: Apply node
            The output of make_node
        inputs: List of data
            The data can be operated on with numpy
        out: List
            output_storage is a list of storage cells where the output 
            is to be stored. There is one storage cell for each output of the Op. 
            The data put in output_storage must match the type 
            of the symbolic output. 
            It is forbidden to change the length of the list(s) 
            contained in output_storage. 
            A function Mode may allow output_storage elements 
            to persist between evaluations, 
            or it may reset output_storage cells to hold a value of None. 
            It can also pre-allocate some memory for the Op to use. 
            This feature can allow perform to reuse memory between calls, 
            for example. 
            If there is something preallocated in the output_storage, 
            it will be of the good dtype, but can have the wrong shape and 
            have any stride pattern.
        """
        # Runs the computation in python
        start, stop, *args = inputs
        
        # print(inputs)
        # print([type(a) for a in inputs])
        # print([a.dtype for a in inputs])
                
        if self._expr.ndim == 0:
            val = quad(
                self._func, 
                start, 
                stop, 
                args=tuple(args),
                # epsabs=1e-5
            )[0]
        elif self._expr.ndim == 1:
            # if the function is vector-valued
            # (e.g., the gradient of a vector),
            # use quad_vec
            val = quad_vec(
                self._func,
                start,
                stop,
                args=tuple(args),
                # epsabs=1e-5
            )[0]
        else:
            # first reshape into an array
            
            shape = self._func(
                start,
                *args
            ).shape
            
            def helpfunc(*args):
                return self._func(*args).flatten()
            
            # put back into original shape
            val = quad_vec(
                helpfunc,
                start,
                stop,
                args=tuple(args),
                # epsabs=1e-5
            )[0].reshape(shape)
        
        # in-place modification of "out".
        # out is a single-element list
        out[0][0] = np.array(val)
        
    def grad(self, inputs, grads):
        """
        NOTE: This function does not calculate the gradient
        but rather implements part of the chain rule,
        i.e. multiplies the grads by the gradient wrt to the cost
        See https://aesara.readthedocs.io/en/latest/extending/op.html
        for an explanation
        
        Inputs in this case contains: 
        [lower integration bound, upper integration bound, ...
        other variables of function]
        
        It takes two arguments inputs and output_gradients, 
        which are both lists of Variables, and those must be operated on 
        using Aesara’s symbolic language. 
        
        The Op.grad() method must return a list containing one 
        Variable for each input. 
        
        Each returned Variable represents the gradient with respect to that input 
        computed based on the symbolic gradients with respect to each output. 
        
        If the output is not differentiable with respect to an input 
        then this method should be defined to return a variable of type NullType 
        for that input. 
        Likewise, if you have not implemented the gradient computation 
        for some input, 
        you may return a variable of type NullType for that input. 
        
        Please refer to Op.grad() for a more detailed view.
        """
        
        # unpack the input
        start, stop, *args = inputs
        out, = grads
        
        # dictionary with the extra variables as keys
        # and the extra variables in "inputs" as values
        replace = dict(zip(
            self._extra_vars, 
            args
        ))
        
        # The derivative of integral wrt to the upper limit of integration
        # is just the value of the function at that limit
        # (for lower limit, it's minus the function)
        # See e.g.,
        # https://math.stackexchange.com/questions/984111/
        # differentiating-with-respect-to-the-limit-of-integration
        replace_ = replace.copy()
        replace_[self._var] = start
        dstart = out * clone_replace(
            # Clone a graph and replace subgraphs within it. 
            # It returns a copy of the initial subgraph with the corresponding
            # substitutions.
            -self._expr, 
            # Dictionary describing which subgraphs should be replaced by what.
            replace=replace_
        )
        
        replace_ = replace.copy()
        replace_[self._var] = stop
        dstop = out * clone_replace(
            self._expr, 
            replace=replace_
        )

        # calculate the symbolic gradient of self._expr
        # wrt each extra variable.
        # This can be done because they're symbolic aesara variables!
        # This corresponds to the gradient of the expression
        # *inside* the integral (the inner part of Leibniz'
        # integral rule)
        grads = at.jacobian(
            # cost
            self._expr, 
            # wrt
            self._extra_vars
        ) 
        
        # print("Inputs: ", inputs)
        # print("Ndim Inputs: ", [i.ndim for i in inputs])
        # print("Out: ", out)
        # print("Ndim Out: ", out.ndim)
        # print("Grads: ", grads)
        # print('Grads ndim: ', [g.ndim for g in grads])
        
        dargs = []
        # loop over the gradients of the extra vars
        for grad in grads:
            
            # define an Apply node
            # for that gradient
            integrate = Integrate(
                # integrate gradient
                grad, 
                # wrt to same variable as 
                # main expression
                self._var, 
                *self._extra_vars
            )
                        
            # Apply Leibniz' integral rule:
            # call integrate, which evaluates
            # the integral of the gradient.
            # And then multiply with previous gradient
            # that was passed in the input.
            # NOTE: This is not actually doing the operation,
            # but rather calling make_node, which *creates the node*
            # that does the operation
            jacobian = integrate(
                start, stop, 
                *args
            )
            
            darg = at.dot(
                jacobian.T, 
                out
            )
            
            dargs.append(darg)
            
            # print('evaluate darg: ', darg.ndim)
        
#         print("dargs with jac: ", dargs)
#         print("dargs ndims: ", [d.ndim for d in dargs])
        
        # return a list with one Variable for each input in inputs
        return [dstart, dstop] + dargs


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