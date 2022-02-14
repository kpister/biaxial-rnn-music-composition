from collections import OrderedDict
import aesara
import aesara.tensor as at
from aesara.tensor.random.utils import RandomStream
import numpy as np

srng = RandomStream(1234)
np_rng = np.random.RandomState(1234)


def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, "initial_hidden_state")


def apply_dropout(x, mask):
    if mask is not None:
        return mask * x
    else:
        return x


def Dropout(shape, prob):
    """
    Return a dropout mask on x.
    The probability of a value in x going to zero is prob.
    Inputs
    ------
    x    aesara variable : the variable to add noise to
    prob float, variable : probability of dropping an element.
    size tuple(int, int) : size of the dropout mask.
    Outputs
    -------
    y    aesara variable : x with the noise multiplied.
    """

    mask = srng.binomial(1, 1 - prob, size=shape)
    return at.cast(mask, aesara.config.floatX)


def MultiDropout(shapes, dropout=0.0):
    """
    Return all the masks needed for dropout outside of a scan loop.
    """
    return [Dropout(shape, dropout) for shape in shapes]


def random_initialization(size):
    return (np_rng.standard_normal(size) * 1.0 / size[0]).astype(aesara.config.floatX)


def create_shared(out_size, in_size=None, name=None):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.
    Inputs
    ------
    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.
    Outputs
    -------
    aesara shared : the shared matrix, with random numbers in it
    """

    if in_size is None:
        return aesara.shared(random_initialization((out_size,)), name=name)
    else:
        return aesara.shared(random_initialization((out_size, in_size)), name=name)


class Layer:
    """
    Base object for neural network layers.
    A layer has an input set of neurons, and
    a hidden activation. The activation, f, is a
    function applied to the affine transformation
    of x by the connection matrix W, and the bias
    vector b.
    > y = f ( W * x + b )
    """

    def __init__(self, input_size, hidden_size, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.is_recursive = False
        self.create_variables()

    def create_variables(self):
        """
        Create the connection matrix and the bias vector
        """
        self.linear_matrix = create_shared(
            self.hidden_size, self.input_size, name="Layer.linear_matrix"
        )
        self.bias_matrix = create_shared(self.hidden_size, name="Layer.bias_matrix")

    def activate(self, x):
        """
        The hidden activation of the network
        """
        if x.ndim > 1:
            return self.activation(
                at.dot(self.linear_matrix, x.T) + self.bias_matrix[:, None]
            ).T
        else:
            return self.activation(at.dot(self.linear_matrix, x) + self.bias_matrix)

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())


class RNN(Layer):
    """
    Special recurrent layer than takes as input
    a hidden activation, h, from the past and
    an observation x.
    > y = f ( W * [x, h] + b )
    Note: x and h are concatenated in the activation.
    """

    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.is_recursive = True

    def create_variables(self):
        """
        Create the connection matrix and the bias vector,
        and the base hidden activation.
        """
        self.linear_matrix = create_shared(
            self.hidden_size,
            self.input_size + self.hidden_size,
            name="RNN.linear_matrix",
        )
        self.bias_matrix = create_shared(self.hidden_size, name="RNN.bias_matrix")
        self.initial_hidden_state = create_shared(
            self.hidden_size, name="RNN.initial_hidden_state"
        )

    def activate(self, x, h):
        """
        The hidden activation of the network
        """
        if x.ndim > 1:
            return self.activation(
                at.dot(self.linear_matrix, at.concatenate([x, h], axis=1).T)
                + self.bias_matrix[:, None]
            ).T
        else:
            return self.activation(
                at.dot(self.linear_matrix, at.concatenate([x, h])) + self.bias_matrix
            )

    @property
    def params(self):
        return [self.linear_matrix, self.bias_matrix]

    @params.setter
    def params(self, param_list):
        self.linear_matrix.set_value(param_list[0].get_value())
        self.bias_matrix.set_value(param_list[1].get_value())


class LSTM(RNN):
    """
    The structure of the LSTM allows it to learn on problems with
    long term dependencies relatively easily. The "long term"
    memory is stored in a vector of memory cells c.
    Although many LSTM architectures differ in their connectivity
    structure and activation functions, all LSTM architectures have
    memory cells that are suitable for storing information for long
    periods of time. Here we implement the LSTM from Graves et al.
    (2013).
    """

    def create_variables(self):
        """
        Create the different LSTM gates and
        their variables, along with the initial
        hidden state for the memory cells and
        the initial hidden activation.
        """
        # input gate for cells
        self.in_gate = Layer(
            self.input_size + self.hidden_size,
            self.hidden_size,
            at.sigmoid,
        )
        # forget gate for cells
        self.forget_gate = Layer(
            self.input_size + self.hidden_size,
            self.hidden_size,
            at.sigmoid,
        )
        # input modulation for cells
        self.in_gate2 = Layer(
            self.input_size + self.hidden_size,
            self.hidden_size,
            self.activation,
        )
        # output modulation
        self.out_gate = Layer(
            self.input_size + self.hidden_size,
            self.hidden_size,
            at.sigmoid,
        )

        # keep these layers organized
        self.internal_layers = [
            self.in_gate,
            self.forget_gate,
            self.in_gate2,
            self.out_gate,
        ]

        # store the memory cells in first n spots, and store the current
        # output in the next n spots:
        self.initial_hidden_state = create_shared(
            self.hidden_size * 2, name="LSTM.initial_hidden_state"
        )

    @property
    def params(self):
        """
        Parameters given by the 4 gates and the
        initial hidden activation of this LSTM cell
        layer.
        """
        return [param for layer in self.internal_layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    def postprocess_activation(self, x, *args):
        if x.ndim > 1:
            return x[:, self.hidden_size :]
        else:
            return x[self.hidden_size :]

    def activate(self, x, h):
        """
        The hidden activation, h, of the network, along
        with the new values for the memory cells, c,
        Both are concatenated as follows:
        >      y = f( x, past )
        Or more visibly, with past = [prev_c, prev_h]
        > [c, h] = f( x, [prev_c, prev_h] )
        """

        if h.ndim > 1:
            # previous memory cell values
            prev_c = h[:, : self.hidden_size]

            # previous activations of the hidden layer
            prev_h = h[:, self.hidden_size :]
        else:

            # previous memory cell values
            prev_c = h[: self.hidden_size]

            # previous activations of the hidden layer
            prev_h = h[self.hidden_size :]

        # input and previous hidden constitute the actual
        # input to the LSTM:
        if h.ndim > 1:
            obs = at.concatenate([x, prev_h], axis=1)
        else:
            obs = at.concatenate([x, prev_h])
        # TODO could we combine these 4 linear transformations for efficiency? (e.g., http://arxiv.org/pdf/1410.4615.pdf, page 5)
        # how much to add to the memory cells
        in_gate = self.in_gate.activate(obs)

        # how much to forget the current contents of the memory
        forget_gate = self.forget_gate.activate(obs)

        # modulate the input for the memory cells
        in_gate2 = self.in_gate2.activate(obs)

        # new memory cells
        next_c = forget_gate * prev_c + in_gate2 * in_gate

        # modulate the memory cells to create the new output
        out_gate = self.out_gate.activate(obs)

        # new hidden output
        next_h = out_gate * at.tanh(next_c)

        if h.ndim > 1:
            return at.concatenate([next_c, next_h], axis=1)
        else:
            return at.concatenate([next_c, next_h])


class StackedCells:
    """
    Sequentially connect several recurrent layers.
    celltypes can be RNN or LSTM.
    """

    def __init__(
        self,
        input_size,
        celltype=RNN,
        layers=None,
        activation=lambda x: x,
    ):
        if layers is None:
            layers = []
        self.input_size = input_size
        self.create_layers(layers, activation, celltype)

    def create_layers(self, layer_sizes, activation_type, celltype):
        self.layers = []
        prev_size = self.input_size
        for _, layer_size in enumerate(layer_sizes):
            layer = celltype(
                prev_size,
                layer_size,
                activation_type,
            )
            self.layers.append(layer)
            prev_size = layer_size

    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    def forward(self, x, prev_hiddens=None, dropout=None):
        """
        Return new hidden activations for all stacked RNNs
        """
        if dropout is None:
            dropout = []
        if prev_hiddens is None:
            prev_hiddens = [
                (
                    at.repeat(
                        at.shape_padleft(layer.initial_hidden_state), x.shape[0], axis=0
                    )
                    if x.ndim > 1
                    else layer.initial_hidden_state
                )
                if has_hidden(layer)
                else None
                for layer in self.layers
            ]

        out = []
        layer_input = x
        for k, layer in enumerate(self.layers):
            level_out = layer_input
            if len(dropout) > 0:
                level_out = apply_dropout(layer_input, dropout[k])
            if layer.is_recursive:
                level_out = layer.activate(level_out, prev_hiddens[k])
            else:
                level_out = layer.activate(level_out)
            out.append(level_out)
            # deliberate choice to change the upward structure here
            # in an RNN, there is only one kind of hidden values
            if hasattr(layer, "postprocess_activation"):
                # in this case the hidden activation has memory cells
                # that are not shared upwards
                # along with hidden activations that can be sent
                # updwards
                if layer.is_recursive:
                    level_out = layer.postprocess_activation(
                        level_out, layer_input, prev_hiddens[k]
                    )
                else:
                    level_out = layer.postprocess_activation(level_out, layer_input)

            layer_input = level_out

        return out


# TODO
def create_optimization_updates(
    cost,
    params,
    updates=None,
    max_norm=5.0,
    lr=0.01,
    eps=1e-6,
    rho=0.95,
    method="adadelta",
    gradients=None,
):
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.
    Returns the shared variables for the gradient caches,
    and the updates dictionary for compilation by a
    aesara function.
    Inputs
    ------
    cost     aesara variable : what to minimize
    params   list            : list of theano variables
                               with respect to which
                               the gradient is taken.
    max_norm float           : cap on excess gradients
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.
    Outputs:
    --------
    updates  OrderedDict   : the updates to pass to a
                             aesara function
    gsums    list          : gradient caches for Adagrad
                             and Adadelta
    xsums    list          : gradient caches for AdaDelta only
    lr       aesara shared : learning rate
    max_norm aesara shared : normalizing clipping value for
                             excessive gradients (exploding).
    """
    lr = aesara.shared(np.float64(lr).astype(aesara.config.floatX))
    eps = np.float64(eps).astype(aesara.config.floatX)
    rho = aesara.shared(np.float64(rho).astype(aesara.config.floatX))
    if max_norm is not None and max_norm is not False:
        max_norm = aesara.shared(np.float64(max_norm).astype(aesara.config.floatX))

    gsums = [
        aesara.shared(np.zeros_like(param.get_value(borrow=True)))
        if (method == "adadelta" or method == "adagrad")
        else None
        for param in params
    ]
    xsums = [
        aesara.shared(np.zeros_like(param.get_value(borrow=True)))
        if method == "adadelta"
        else None
        for param in params
    ]

    gparams = at.grad(cost, params) if gradients is None else gradients

    if updates is None:
        updates = OrderedDict()

    for gparam, param, gsum, xsum in zip(gparams, params, gsums, xsums):
        # clip gradients if they get too big
        if max_norm is not None and max_norm is not False:
            grad_norm = gparam.norm(L=2)
            gparam = (at.minimum(max_norm, grad_norm) / (grad_norm + eps)) * gparam

        if method == "adadelta":
            updates[gsum] = at.cast(
                rho * gsum + (1.0 - rho) * (gparam**2), aesara.config.floatX
            )
            dparam = -at.sqrt((xsum + eps) / (updates[gsum] + eps)) * gparam
            updates[xsum] = at.cast(
                rho * xsum + (1.0 - rho) * (dparam**2), aesara.config.floatX
            )
            updates[param] = at.cast(param + dparam, aesara.config.floatX)
        elif method == "adagrad":
            updates[gsum] = at.cast(gsum + (gparam**2), aesara.config.floatX)
            updates[param] = at.cast(
                param - lr * (gparam / (at.sqrt(updates[gsum] + eps))),
                aesara.config.floatX,
            )
        else:
            updates[param] = param - gparam * lr

    if method == "adadelta":
        lr = rho

    return updates, gsums, xsums, lr, max_norm
