from ...operations.operation_base import Operation
from ...tensor_base import Tensor
from numbers import Integral
import numpy as np

from numba import njit


@njit
def _dot_tanh_accum(x, W):
    for n in range(len(x) - 1):
        x[n + 1] += np.dot(x[n], W)
        x[n + 1] = np.tanh(x[n + 1])


class RecurrentUnit(Operation):
    """ Defines a basic recurrent unit for a RNN.

        This unit operates on a sequence of data {X_j | (0 <= j <= T - 1)}, producing
        a sequence of "hidden descriptors": {S_i | (0 <= i <= T}}, via the trainable parameters
        U and W

                                S_{t} = tanh(U X_{t-1} + W S_{t-1})

        For a language model, S_{t} is traditionally mapped to a prediction of X_t via: softmax(V S_t),
        where V is yet another trainable parameter (not built into the recurrent unit)."""

    def __call__(self, X, U, W, s0=None, bp_lim=None):
        """ Performs a forward pass of sequential data through a simple RNN layer, returning
            the 'hidden-descriptors' arrived at by utilizing the trainable parameters U and V:

                                S_{t} = tanh(U X_{t-1} + W S_{t-1})

            Parameters
            ----------
            X : mygrad.Tensor, shape=(T, N, C)
               The sequential data to be passed forward.

            U : mygrad.Tensor, shape=(D, C)
               The weights used to map sequential data to its hidden-descriptor representation

            W : mygrad.Tensor, shape=(D, D)
                The weights used to map a hidden-descriptor to a hidden-descriptor.

            s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
                The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
                of zeros of shape (N, D) is created.

            bp_lim : Optional[int]
                The (non-zero) limit of the number of back propagations through time are
                performed


            Returns
            -------
            mygrad.Tensor
                The sequence of 'hidden-descriptors' produced by the forward pass of the RNN."""
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 < bp_lim <= len(X)
        self.bp_lim = bp_lim if bp_lim is not None else len(X)

        self.X = X
        self.U = U
        self.W = W
        self._hidden_seq = []

        seq = self.X.data
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.U.shape[-1]))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.U.data, out=out[1:])
        _dot_tanh_accum(out, self.W.data)

        self._hidden_seq = Tensor(out, _creator=self)

        return self._hidden_seq.data


    def backward(self, grad):
        """ Performs back propagation through time (with optional truncation), using the
            following notation:
                s_t = tanh(f_t)
                f_t = U x_{t-1} + W s_{t-1}
        """
        if self.U.constant and self.W.constant and self.X.constant:
            return None

        s = self._hidden_seq

        dst_dft = (1 - s.data ** 2)  # ds_{t} / d_f{t}
        dLt_dst = np.copy(grad)  # dL_{t} / ds_{t}
        old_dst = np.zeros_like(grad)

        for i in range(self.bp_lim):
            # dL_{n} / ds_{t+1} -> dL_{n} / df_{t+1}  | ( n > t )
            index = slice(2, len(grad) - i)
            dLn_ft1 = dst_dft[index] * (dLt_dst[index] - old_dst[index])
            old_dst = np.copy(dLt_dst)
            dLt_dst[1:len(grad) - (i + 1)] += np.dot(dLn_ft1, self.W.data.T)  # dL_{t} / ds_{t} + ... + dL_{n} / ds_{t}

        self._hidden_seq.grad = dLt_dst  # element t: dL_{t} / ds_{t} + ... + dL_{T_lim} / ds_{t}

        dLt_dft = dLt_dst[1:] * dst_dft[1:]  # dL_{t} / df_{t} + ... + dL_{T_lim} / df_{t}

        if not self.U.constant:
            self.U.backward(np.einsum("ijk, ijl -> kl", self.X.data, dLt_dft))  # dL_{1} / dU + ... + dL_{T} / dU
        if not self.W.constant:
            self.W.backward(np.einsum("ijk, ijl -> kl", s.data[:-1], dLt_dft))  # dL_{1} / dW + ... + dL_{T} / dW
        if not self.X.constant:
            self.X.backward(np.dot(dLt_dft, self.U.data.T))  # dL_{1} / dX + ... + dL_{T} / dX

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for x in [self.X, self.U, self.W]:
            x.null_gradients()


def simple_RNN(X, U, W, s0=None, bp_lim=None):
    """ Performs a forward pass of sequential data through a simple RNN layer, returning
        the 'hidden-descriptors' arrived at by utilizing the trainable parameters U and V:

                            S_{t} = tanh(U X_{t-1} + W S_{t-1})

        Parameters
        ----------
        X : mygrad.Tensor, shape=(T, N, C)
           The sequential data to be passed forward.

        U : mygrad.Tensor, shape=(D, C)
           The weights used to map sequential data to its hidden-descriptor representation

        W : mygrad.Tensor, shape=(D, D)
            The weights used to map a hidden-descriptor to a hidden-descriptor.

        s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
            The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
            of zeros of shape (N, D) is created.

        bp_lim : Optional[int]
            The (non-zero) limit of the depth of back propagation through time to be
            performed. If `None` back propagation is passed back through the entire sequence.

            E.g. `bp_lim=3` will propagate gradients only up to 3 steps backward through the
            recursive sequence.

        Returns
        -------
        mygrad.Tensor
            The sequence of 'hidden-descriptors' produced by the forward pass of the RNN.

        Notes
        -----
        T : Sequence length
        N : Batch size
        C : Length of single datum
        D : Length of 'hidden' descriptor"""
    s = Tensor._op(RecurrentUnit, X, U, W, op_kwargs=dict(s0=s0, bp_lim=bp_lim))
    s.creator._hidden_seq = s
    return s





@njit
def _gru_layer(s, z, r, h, Wz, Wr, Wh, bz, br, bh):
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz) + bz
        z[n] = 1 / (1 + np.exp(-z[n]))

        r[n] += np.dot(s[n], Wr) + br
        r[n] = 1 / (1 + np.exp(-r[n]))

        h[n] += np.dot(r[n] * s[n], Wh) + bh
        h[n] = np.tanh(h[n])

        s[n + 1] = (1 - z[n]) * h[n] + z[n] * s[n]


def _gru_dLds(s, z, r, h, dLds, Wz, Wr, Wh):
    """
    Returns
    --------
        partial dL / ds(t+1) * ds(t+1) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dz(t) * dz(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / dr(t) * dr(t) / ds(t)
    """
    dz = 1 - z # note: not actually ds(t+1) / dz(t)
    dh = 1 - h ** 2
    dLdh = np.dot(dLds * dh * dz, Wh.T)

    return z * dLds + \
           np.dot((dLds * (s - h)) * z * dz, Wz.T) + \
           dLdh * r + \
           np.dot(dLdh * s * r * (1 - r), Wr.T)

def _gru_dLdx(s, z, r, h, dLds, Uz, Ur, Uh):
    """
    Returns
    --------
        partial dL / ds(t+1) * ds(t+1) / dz(t) * dz(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / dr(t) * dr(t) / ds(t)
    """
    dz = 1 - z # note: not actually derivative of sigmoid
    dh = 1 - h ** 2
    dLdh = np.dot(dLds * dh * dz, Uh.T)

    return np.dot((dLds * (s - h)) * z * dz, Uz.T) + \
           dLdh * r + \
           np.dot(dLdh * s * r * (1 - r), Ur.T)


class GRUnit(Operation):
    def __call__(self, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None):
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 < bp_lim <= len(X)
        self.bp_lim = bp_lim if bp_lim is not None else len(X)

        self.X = X

        self.Uz = Uz
        self.Wz = Wz
        self.bz = bz

        self.Ur = Ur
        self.Wr = Wr
        self.br = br

        self.Uh = Uh
        self.Wh = Wh
        self.bh = bh

        self._hidden_seq = []

        self._z = []
        self._r = []
        self._h = []

        seq = self.X.data

        z = np.zeros((seq.shape[0], seq.shape[1], self.Uz.shape[-1]))
        r = np.zeros((seq.shape[0], seq.shape[1], self.Ur.shape[-1]))
        h = np.zeros((seq.shape[0], seq.shape[1], self.Uh.shape[-1]))
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.Uz.shape[-1]))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.Uz.data, out=z)
        np.dot(seq, self.Ur.data, out=r)
        np.dot(seq, self.Uh.data, out=h)

        _gru_layer(out, z, r, h,
                   self.Wz.data, self.Wr.data, self.Wh.data,
                   self.bz.data, self.br.data, self.bh.data)


        self._hidden_seq = Tensor(out, _creator=self)
        self._z = Tensor(z, _creator=self)
        self._r = Tensor(r, _creator=self)
        self._h = Tensor(h, _creator=self)

        return self._hidden_seq


    def backward(self, grad):
        if self.X.constant and self.Uz.constant and self.Wz.constant  and self.bz.constant \
           and self.Ur.constant and self.Wr.constant and self.br.constant \
           and self.Uh.constant and self.Wh.constant and self.bh.constant:
            return None

        s = self._hidden_seq.data[:-1]
        z = self._z.data
        r = self._r.data
        h = self._h.data

        dLds = grad[1:]
        old_dLds = np.zeros_like(dLds)

        """
        for i in range(self.bp_lim):
            # dL_{n} / ds_{t+1} -> dL_{n} / df_{t+1}  | ( n > t )
            index = slice(2, len(grad) - i)
            dLn_ft1 = dst_dft[index] * (dLt_dst[index] - old_dst[index])
            old_dst = np.copy(dLt_dst)
            dLt_dst[1:len(grad) - (i + 1)] += np.dot(dLn_ft1, self.W.data.T)  # dL_{t} / ds_{t} + ... + dL_{n} / ds_{t}
        """

        for i in range(self.bp_lim):
            #  dL(t) / ds(t) + dL(t+1) / ds(t)
            dt = dLds[1:len(dLds) - i] - old_dLds[1:len(dLds) - i]
            old_dLds = np.copy(dLds)
            dLds[:len(dLds) - (i + 1)] += _gru_dLds(s[1:len(dLds) - i],
                                                   z[1:len(dLds) - i],
                                                   r[1:len(dLds) - i],
                                                   h[1:len(dLds) - i],
                                                   dt,
                                                   self.Wz.data,
                                                   self.Wr.data,
                                                   self.Wh.data)

        zgrad = dLds * (s - h) # dL / dz
        hgrad = dLds * (1 - z) # dL / dh
        rgrad = np.dot((1 - h ** 2) * hgrad, self.Wh.data.T) * s # dL / dr

        self._hidden_seq.grad = dLds
        self._z.grad = zgrad
        self._r.grad = rgrad
        self._h.grad = hgrad

        dz = zgrad * z * (1 - z)
        dr = rgrad * r * (1 - r)
        dh = hgrad * (1 - h ** 2)

        self.Uz.backward(np.einsum("ijk, ijl -> kl", self.X.data, dz))
        self.Wz.backward(np.einsum("ijk, ijl -> kl", s, dz))
        self.bz.backward(np.einsum("ijk -> k", dz))

        self.Ur.backward(np.einsum("ijk, ijl -> kl", self.X.data, dr))
        self.Wr.backward(np.einsum("ijk, ijl -> kl", s, dr))
        self.br.backward(np.einsum("ijk -> k", dr))

        self.Uh.backward(np.einsum("ijk, ijl -> kl", self.X.data, dh))
        self.Wh.backward(np.einsum("ijk, ijl -> kl", (s * r), dh))
        self.bh.backward(np.einsum("ijk -> k", dh))

        if not self.X.constant:
            dz = 1 - z # note: not actually derivative of sigmoid
            dh = 1 - h ** 2

            self.X.backward(np.dot((dLds * (s - h)) * z * dz, self.Uz.data.T) + \
                   np.dot(dLds * dh * dz, self.Uh.data.T) + \
                   np.dot(np.dot(dLds * dh * dz, self.Wh.data.T) * s * r * (1 - r), self.Ur.data.T))


        def null_gradients(self):
            """ Back-propagates `None` to the gradients of the operation's input Tensors."""
            for x in [self.X, self.Uz, self.Wz, self.Ur, self.Wr, self.Uh, self.Wh]:
                x.null_gradients()


def GRU(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None):
    """ Performs a forward pass of sequential data through a Gated Recurrent Unit layer, returning
        the 'hidden-descriptors' arrived at by utilizing the trainable parameters as follows:

                            Z_{t} = sigmoid(Uz X_{t} + Wz S_{t-1} + bz)
                            R_{t} = sigmoid(Ur X_{t} + Wr S_{t-1} + br)
                            H_{t} = tanh(Uh X_{t} + Wh (R{t} * S_{t-1}) + bh)
                            S_{t} = (1 - Z{t}) * H{t} + Z{t} * S_{t-1}

        Parameters
        ----------
        X : mygrad.Tensor, shape=(T, N, C)
           The sequential data to be passed forward.

        Uz/Ur/Uh : mygrad.Tensor, shape=(D, C)
           The weights used to map sequential data to its hidden-descriptor representation

        Wz/Wr/Wh : mygrad.Tensor, shape=(D, D)
            The weights used to map a hidden-descriptor to a hidden-descriptor.

        bz/br/bh : mygrad.Tensor, shape=(D,)
           The biases used to scale a hidden-descriptor.

        s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
            The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
            of zeros of shape (N, D) is created.

        bp_lim : Optional[int]
            The (non-zero) limit of the depth of back propagation through time to be
            performed. If `None` back propagation is passed back through the entire sequence.

            E.g. `bp_lim=3` will propagate gradients only up to 3 steps backward through the
            recursive sequence.

        Returns
        -------
        mygrad.Tensor
            The sequence of 'hidden-descriptors' produced by the forward pass of the RNN.

        Notes
        -----
        T : Sequence length
        N : Batch size
        C : Length of single datum
        D : Length of 'hidden' descriptor"""
    s = Tensor._op(GRUnit, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, op_kwargs=dict(s0=s0, bp_lim=bp_lim))
    s.creator._hidden_seq = s
    return s
