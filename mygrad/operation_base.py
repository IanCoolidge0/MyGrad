""" Defines the base class for mathematical operations capable of back-propagating
    gradients to their input tensors."""

__all__ = ["Operation",
           "BroadcastableOp"]


class Operation:
    """ Base class for all tensor operations that support back-propagation
        of gradients.

        Consider the Operation-instance ``f``. A forward-pass through ``f`` is defined
        via ``f.__call__``. Thus, given tensors ``a`` and ``b``, a computational
        graph is defined ``f.__call__(a, b) -> c``, where the "creator" of tensor ``c``
        is recorded as ``f``.

        (node: a) --+
                     -> [operation: f(a, b)] --> (node: c)
        (node: b) --+

        Thus back-propagating through ``c`` will instruct ``f`` to back-propagate
        the gradient to its inputs, which are recorded as ``a`` and ``b``. Each
        node then back-propagates to any Operation-instance that is recorded
        as its creator, and so on.

        If an operation class has `scalar_only=True`, then the terminal node of a
        computational graph involving that operation can only trigger back-propagation
        from a 0-dimensional tensor (i.e. a scalar). This is `False` for operations that
        manifest as trivial element-wise operations over tensors. In such cases, the
        gradient of the operation can also be treated element-wise, and thus be computed
        unambiguously.

        The matrix-multiplication operation, for example, is a scalar-only operation because
        computing the derivative of A_{ij} B_{jk} (summed over j)) with respect to A_{pq}
        produces a 4-tensor. This is the case unless the terminal node of this graph is a
        scalar, in which  case the elements of the 2-tensor d(scalar)/ d(A_{pq}) has a trivial
        correspondence to the elements of A_{pq}.
        """
    scalar_only = False

    def __call__(self, *input_vars):
        """ Performs a forward pass, f, of this Operation:
                  f(x1, ...., xn) -> out

            Parameters
            ----------
            *input_vars : mygrad.Tensor
                The input-arguments of f. The tuple (x1, ...., xn)
                should be bound to the instance-attribute `self.variables`

            Returns
            -------
            numpy.ndarray
                The output of the forward pass function."""

        self.variables = input_vars
        raise NotImplementedError

    def backward_var(self, grad, index, **kwargs):
        """ Given ``grad = d(out)/d(f)``, computes ``d(out)/d(var)``, and passes this result
            to ``var.backward()``, where var is the tensor-argument at position ``index``.

            Parameters
            ----------
            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present
                operation (`f`): d(out)/df

            index : int
                The index-location of ``var`` in ``self.variables``

            Other Parameters
            ----------------
            _broadcastable : bool, optional (default:False)
                Devs-only: Indicates whether or not the up-stream operation
                can utilize broadcasting.

            Raises
            ------
            NotImplemented Error"""
        raise NotImplementedError

    def backward(self, grad, **kwargs):
        """ Back-propagates the gradient through all of the operation's inputs.
            Constant tensors do not propagate a gradient.

            grad : numpy.ndarray
                The back-propagated total derivative with respect to the present
                operation (`f`): d(out)/df

            Other Parameters
            ----------------
            _broadcastable : bool, optional (default:False)
                Devs-only: Indicates whether or not the up-stream operation
                can utilize broadcasting."""
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index, _force=False, **kwargs)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input tensors,
            and to all preceding tensors in the computational graph."""
        for var in self.variables:
            var.null_gradients()


class BroadcastableOp(Operation):
    """ Signals that an Operation's forward pass can broadcast its tensor
        arguments."""
    pass
