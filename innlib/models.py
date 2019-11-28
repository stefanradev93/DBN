import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class Permutation(tf.keras.Model):
    """Implements a permutation layer to permute the input dimensions of the cINN block."""

    def __init__(self, input_dim):
        """
        Creates a permutation layer for a conditional invertible block.
        ----------

        Arguments:
        input_dim  : int -- the dimensionality of the input to the c inv block.
        """

        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, x, inverse=False):
        """Permutes the bach of an input."""

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(x), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(x), self.inv_permutation))


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).
        ----------

        Arguments:
        meta  : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer.
        n_out : int  -- number of outputs of the coupling net
        """

        super(CouplingNet, self).__init__()

        self.dense = tf.keras.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']] +
            # Output layer
            [tf.keras.layers.Dense(n_out,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, theta, x):
        """
        Concatenates x and y and performs a forward pass through the coupling net.
        Arguments:
        theta : tf.Tensor of shape (batch_size, inp_dim)     -- the parameters x ~ p(x|y) of interest
        x     : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = sum(y)
        """

        inp = tf.concat((theta, x), axis=-1)
        out = self.dense(inp)
        return out


class ConditionalInvertibleBlock(tf.keras.Model):
    """Implements a conditional version of the INN block."""

    def __init__(self, meta, theta_dim, alpha=1.9, permute=False):
        """
        Creates a conditional invertible block.
        ----------

        Arguments:
        meta      : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer. All coupling nets are assumed to be equal.
        theta_dim : int  -- the number of outputs of the invertible block (eq. the dimensionality of the latent space)
        """

        super(ConditionalInvertibleBlock, self).__init__()
        self.alpha = alpha
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
        if permute:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None
        self.s1 = CouplingNet(meta, self.n_out1)
        self.t1 = CouplingNet(meta, self.n_out1)
        self.s2 = CouplingNet(meta, self.n_out2)
        self.t2 = CouplingNet(meta, self.n_out2)

    def call(self, theta, x, inverse=False, log_det_J=True):
        """
        Implements both directions of a conditional invertible block.
        ----------

        Arguments:
        theta     : tf.Tensor of shape (batch_size, theta_dim) -- the parameters theta ~ p(theta|y) of interest
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest x = sum(x)
        inverse   : bool -- flag indicating whether to tun the block forward or backwards
        log_det_J : bool -- flag indicating whether to return the log determinant of the Jacobian matrix
        ----------

        Output:
        (v, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        u               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        # --- Forward pass --- #
        if not inverse:

            if self.permutation is not None:
                theta = self.permutation(theta)

            u1, u2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

            # Pre-compute network outputs for v1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            t1 = self.t1(u2, x)
            v1 = u1 * tf.exp(s1) + t1

            # Pre-compute network outputs for v2
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            t2 = self.t2(v1, x)
            v2 = u2 * tf.exp(s2) + t2
            v = tf.concat((v1, v2), axis=-1)

            if log_det_J:
                # log|J| = log(prod(diag(J))) -> according to inv architecture
                return v, tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
            return v

        # --- Inverse pass --- #
        else:

            v1, v2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

            # Pre-Compute s2
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            u2 = (v2 - self.t2(v1, x)) * tf.exp(-s2)

            # Pre-Compute s1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            u1 = (v1 - self.t1(u2, x)) * tf.exp(-s1)
            u = tf.concat((u1, u2), axis=-1)

            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
            return u


class DeepConditionalModel(tf.keras.Model):
    """Implements a chain of conditional invertible blocks."""

    def __init__(self, meta, n_blocks, theta_dim, alpha=1.9, summary_net=None, permute=False):
        """
        Creates a chain of cINN blocks and chains operations.
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        n_blocks    : int  -- the number of invertible blocks
        theta_dim   : int  -- the dimensionality of the parameter space to be learned
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of x
        permute     : bool -- whether to permute the inputs to the cINN
        """

        super(DeepConditionalModel, self).__init__()

        self.cINNs = [ConditionalInvertibleBlock(meta, theta_dim, alpha=alpha, permute=permute) for _ in range(n_blocks)]
        self.summary_net = summary_net
        self.theta_dim = theta_dim

    def call(self, theta, x, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        ----------

        Arguments:
        theta     : tf.Tensor of shape (batch_size, inp_dim) -- the parameters theta ~ p(theta|x) of interest
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data x
        inverse   : bool -- flag indicating whether to tun the chain forward or backwards
        ----------

        Output:
        (z, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        x               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        if self.summary_net is not None:
            x = self.summary_net(x)
        if inverse:
            return self.inverse(theta, x)
        else:
            return self.forward(theta, x)

    def forward(self, theta, x):
        """Performs a forward pass though the chain."""

        z = theta
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, x)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, z, x):
        """Performs a reverse pass through the chain."""

        theta = z
        for cINN in reversed(self.cINNs):
            theta = cINN(theta, x, inverse=True)
        return theta

    def sample(self, x, n_samples, to_numpy=False, training=False):
        """
        Samples from the inverse model given a single instance y or a batch of instances.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, summary_dim) -- the conditioning data of interest
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        training  : bool -- flag used to indicate that samples are drawn are training time (BatchNorm)
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """

        # Summarize obs data if summary net available
        if self.summary_net is not None:
            x = self.summary_net(x, training=training)

        # In case x is a single instance
        if int(x.shape[0]) == 1:
            z_normal_samples = tf.random_normal(shape=(n_samples, self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.tile(x, [n_samples, 1]))
        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random_normal(shape=(n_samples, int(x.shape[0]), self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.stack([x] * n_samples))

        if to_numpy:
            return theta_samples.numpy()
        return theta_samples


class InvariantModule(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, h_dim, n_dense=3):
        """
        Creates an invariant function with mean pooling.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        """

        super(InvariantModule, self).__init__()

        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform')
            for _ in range(n_dense)
        ])

        self.post_pooling_dense = tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform')

    def call(self, x):
        """
        Transofrms the input into an invariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x = self.module(x)
        x = tf.reduce_mean(x, axis=1)
        out = self.post_pooling_dense(x)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, h_dim, n_dense=3):
        """
        Creates an equivariant neural network consisting of a FC network with
        equal number of hidden units in each layer and an invariant module
        with the same FC structure.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        """

        super(EquivariantModule, self).__init__()

        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu')
            for _ in range(n_dense)
        ])

        self.invariant_module = InvariantModule(h_dim, n_dense)

    def call(self, x):
        """
        Transofrms the input into an equivariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x_inv = self.invariant_module(x)
        x_inv = tf.stack([x_inv] * int(x.shape[1]), axis=1) # Repeat x_inv n times
        x = tf.concat((x_inv, x), axis=-1)
        out = self.module(x)
        return out


class InvariantNetwork(tf.keras.Model):
    """
    Implements a network which parameterizes a
    permutationally invariant function according to Bloem-Reddy and Teh (2019).
    """

    def __init__(self, h_dim, n_dense=3, n_equiv=2):
        """
        Creates a permutationally invariant network
        consisting of two equivariant modules and one invariant module.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        n_equiv : int -- the number of equivariant modules
        """

        super(InvariantNetwork, self).__init__()

        self.equiv = tf.keras.Sequential([
            EquivariantModule(h_dim, n_dense)
            for _ in range(n_equiv)
        ])
        self.inv = InvariantModule(h_dim, n_dense)

    def call(self, x, **kwargs):
        """
        Transofrms the input into a permutationally invariant
        representation by first passing it through multiple equivariant
        modules in order to increase representational power.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or
        'samples' dimensions over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x = self.equiv(x)
        out = self.inv(x)
        return out






class DeepEvidentialModel(tf.keras.Model):

    def __init__(self):
        pass
