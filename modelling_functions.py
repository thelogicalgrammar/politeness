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

# Needed for integration
from scipy.integrate import quad, quad_vec
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from pytensor import clone_replace


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


class Integrate(Op):
    
    # Class to perform integration of one variable
    # on a bounded interval
    
    # Adapted from:
    # https://discourse.pymc.io/t/custom-theano-op-to-do-numerical-integration/734/12
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
        print("Output type: ", self._expr.type, self._expr.dtype)
        
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
        not to change the semantics of the expression by changing the argument order.
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
            # inputs: The arguments of the expression modeled by the `Apply` node.
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
            The data put in output_storage must match the type of the symbolic output. 
            It is forbidden to change the length of the list(s) contained in output_storage. 
            A function Mode may allow output_storage elements to persist between evaluations, 
            or it may reset output_storage cells to hold a value of None. 
            It can also pre-allocate some memory for the Op to use. 
            This feature can allow perform to reuse memory between calls, for example. 
            If there is something preallocated in the output_storage, 
            it will be of the good dtype, but can have the wrong shape and 
            have any stride pattern.
        """
        # Runs the computation in python
        start, stop, *args = inputs
        
        print(inputs)
        print([type(a) for a in inputs])
        print([a.dtype for a in inputs])
                
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
        [lower integration bound, upper integration bound, ...other variables of function]
        
        It takes two arguments inputs and output_gradients, 
        which are both lists of Variables, and those must be operated on 
        using Aesara’s symbolic language. 
        
        The Op.grad() method must return a list containing one Variable for each input. 
        
        Each returned Variable represents the gradient with respect to that input 
        computed based on the symbolic gradients with respect to each output. 
        
        If the output is not differentiable with respect to an input 
        then this method should be defined to return a variable of type NullType 
        for that input. 
        Likewise, if you have not implemented the gradient computation for some input, 
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

