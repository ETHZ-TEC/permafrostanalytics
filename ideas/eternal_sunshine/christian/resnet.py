#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file has been copied and modified from another project of mine in
order to be used for the purpose of this hackathon.

This module implements the class of Resnet networks described in section 4.2 of
the following paper:

    "Deep Residual Learning for Image Recognition", He et al., 2015
    https://arxiv.org/abs/1512.03385
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from batchnorm_layer import BatchNormLayer


class ResNet(nn.Module):
    """A resnet with :math:`6n+2` layers with :math:`3n` residual blocks,
    consisting of two layers each.

    Args:
        in_shape: The shape of an input sample.

            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_outs: The number of output neurons.
        verbose: Allow printing of general information about the generated
            network (such as number of weights).
        n: The network will consist of :math:`6n+2` layers. In the
            paper :math:`n` has been chosen to be 3, 5, 7, 9 or 18.
        no_weights: If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.

            Note, this also affects the affine parameters of the
            batchnorm layer. I.e., if set to ``True``, then the argument
            ``affine`` of :class:`utils.batchnorm_layer.BatchNormLayer`
            will be set to ``False`` and we expect the batchnorm parameters
            to be passed to the :meth:`forward`.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.
        use_batch_norm: Whether batch normalization should used. It will be
            applied after all convolutional layers (before the activation).
        bn_track_stats: If batch normalization is used, then this option
            determines whether running statistics are tracked in these
            layers or not (see argument ``track_running_stats`` of class
            :class:`utils.batchnorm_layer.BatchNormLayer`).

            If ``False``, then batch statistics are utilized even during
            evaluation. If ``True``, then running stats are tracked. When
            using this network in a continual learning scenario with
            different tasks then the running statistics are expected to be
            maintained externally. The argument ``stats_id`` of the method
            :meth:`utils.batchnorm_layer.BatchNormLayer.forward` can be
            provided using the argument ``condition`` of method
            :meth:`forward`.

            Example:
                To maintain the running stats, one can simply iterate over
                all batch norm layers and checkpoint the current running
                stats (e.g., after learning a task when applying a Continual
                Learning scenario).

                .. code:: python

                    for bn_layer in net.batchnorm_layers:
                        bn_layer.checkpoint_stats()
        distill_bn_stats: If ``True``, then the shapes of the batchnorm
            statistics will be added to the attribute
            :attr:`mnets.mnet_interface.MainNetInterface.\
hyper_shapes_distilled` and the current statistics will be returned by the
            method :meth:`distillation_targets`.

            Note, this attribute may only be ``True`` if ``bn_track_stats``
            is ``True``.
    """

    def __init__(
        self,
        in_shape=[536, 356, 3],
        num_outs=1,
        verbose=True,
        n=5,
        no_weights=False,
        init_weights=None,
        use_batch_norm=True,
        bn_track_stats=True,
        distill_bn_stats=False,
    ):
        nn.Module.__init__(self)

        self._num_outs = num_outs
        self._in_shape = in_shape
        self._n = n

        self._use_context_mod = False

        assert init_weights is None or not no_weights
        self._no_weights = no_weights

        assert not use_batch_norm or (not distill_bn_stats or bn_track_stats)

        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._distill_bn_stats = distill_bn_stats and use_batch_norm

        self._kernel_size = [3, 3]
        self._filter_sizes = [16, 16, 32, 64]

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        # We don't use any output non-linearity.
        self._has_linear_out = True

        self._param_shapes = []
        self._weights = None if no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None if not no_weights else []

        ###########################
        ### Print infos to user ###
        ###########################
        # Compute the total number of weights in this network and display
        # them to the user.
        # Note, this complicated calculation is not necessary as we can simply
        # count the number of weights afterwards. But it's an additional sanity
        # check for us.
        fs = self._filter_sizes
        num_weights = (
            np.prod(self._kernel_size)
            * (
                in_shape[2] * fs[0]
                + np.sum(
                    [fs[i] * fs[i + 1] + (2 * n - 1) * fs[i + 1] ** 2 for i in range(3)]
                )
            )
            + (fs[0] + 2 * n * np.sum([fs[i] for i in range(1, 4)]))
            + (fs[-1] * num_outs + num_outs)
        )

        if use_batch_norm:
            # The gamma and beta parameters of a batch norm layer are
            # learned as well.
            num_weights += 2 * (fs[0] + 2 * n * np.sum([fs[i] for i in range(1, 4)]))

        if verbose:
            print(
                "A ResNet with %d layers and %d weights is created."
                % (6 * n + 2, num_weights)
            )

        ################################################
        ### Define and initialize batch norm weights ###
        ################################################
        self._batchnorm_layers = nn.ModuleList() if use_batch_norm else None

        if use_batch_norm:
            if distill_bn_stats:
                self._hyper_shapes_distilled = []

            bn_ind = 0
            for i, s in enumerate(self._filter_sizes):
                if i == 0:
                    num = 1
                else:
                    num = 2 * n

                for j in range(num):
                    bn_layer = BatchNormLayer(
                        s, affine=not no_weights, track_running_stats=bn_track_stats
                    )
                    self._batchnorm_layers.append(bn_layer)

                    if distill_bn_stats:
                        self._hyper_shapes_distilled.extend(
                            [list(p.shape) for p in bn_layer.get_stats(0)]
                        )

                    if not no_weights and init_weights is not None:
                        assert len(bn_layer.weights) == 2
                        for ii in range(2):
                            assert np.all(
                                np.equal(
                                    list(init_weights[bn_ind].shape),
                                    list(bn_layer.weights[ii].shape),
                                )
                            )
                            bn_layer.weights[ii].data = init_weights[bn_ind]
                            bn_ind += 1

            if init_weights is not None:
                init_weights = init_weights[bn_ind:]

        # Note, method `_compute_hyper_shapes` doesn't take context-mod into
        # consideration.
        self._param_shapes.extend(self._compute_hyper_shapes(no_weights=True))
        assert num_weights == np.sum([np.prod(l) for l in self._param_shapes])

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        if no_weights:
            if self._hyper_shapes_learned is None:
                self._hyper_shapes_learned = self._compute_hyper_shapes()
            else:
                # Context-mod weights are already included.
                self._hyper_shapes_learned.extend(self._compute_hyper_shapes())

            self._is_properly_setup()
            return

        if use_batch_norm:
            for bn_layer in self._batchnorm_layers:
                self._weights.extend(bn_layer.weights)

        ############################################
        ### Define and initialize layer weights ###
        ###########################################
        ### Does not include context-mod or batchnorm weights.
        # First layer.
        self._layer_weight_tensors.append(
            nn.Parameter(
                torch.Tensor(
                    self._filter_sizes[0], self._in_shape[2], *self._kernel_size
                ),
                requires_grad=True,
            )
        )
        self._layer_bias_vectors.append(
            nn.Parameter(torch.Tensor(self._filter_sizes[0]), requires_grad=True)
        )

        # Each block consists of 2n layers.
        for i in range(1, len(self._filter_sizes)):
            in_filters = self._filter_sizes[i - 1]
            out_filters = self._filter_sizes[i]

            for _ in range(2 * n):
                self._layer_weight_tensors.append(
                    nn.Parameter(
                        torch.Tensor(out_filters, in_filters, *self._kernel_size),
                        requires_grad=True,
                    )
                )
                self._layer_bias_vectors.append(
                    nn.Parameter(torch.Tensor(out_filters), requires_grad=True)
                )
                # Note, that the first layer in this block has potentially a
                # different number of input filters.
                in_filters = out_filters

        # After the average pooling, there is one more dense layer.
        self._layer_weight_tensors.append(
            nn.Parameter(
                torch.Tensor(num_outs, self._filter_sizes[-1]), requires_grad=True
            )
        )
        self._layer_bias_vectors.append(
            nn.Parameter(torch.Tensor(num_outs), requires_grad=True)
        )

        # We add the weights interleaved, such that there are always consecutive
        # weight tensor and bias vector per layer. This fulfils the requirements
        # of attribute `mask_fc_out`.
        for i in range(len(self._layer_weight_tensors)):
            self._weights.append(self._layer_weight_tensors[i])
            self._weights.append(self._layer_bias_vectors[i])

        ### Initialize weights.
        if init_weights is not None:
            num_layers = 6 * n + 2
            assert len(init_weights) == 2 * num_layers
            offset = 0
            if use_batch_norm:
                offset = 2 * (6 * n + 1)
            assert len(self._weights) == offset + 2 * num_layers
            for i in range(len(init_weights)):
                j = offset + i
                assert np.all(
                    np.equal(list(init_weights[i].shape), list(self._weights[j].shape))
                )
                self._weights[j].data = init_weights[i]
        else:
            for i in range(len(self._layer_weight_tensors)):
                init_params(self._layer_weight_tensors[i], self._layer_bias_vectors[i])

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            x: Input image.

                .. note::
                    We assume the Tensorflow format, where the last entry
                    denotes the number of channels.
            weights (list or dict): If a list of parameter tensors is given and
                context modulation is used (see argument ``use_context_mod`` in
                constructor), then these parameters are interpreted as context-
                modulation parameters if the length of ``weights`` equals
                :code:`2*len(net.context_mod_layers)`. Otherwise, the length is
                expected to be equal to the length of the attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                Alternatively, a dictionary can be passed with the possible
                keywords ``internal_weights`` and ``mod_weights``. Each keyword
                is expected to map onto a list of tensors.
                The keyword ``internal_weights`` refers to all weights of this
                network except for the weights of the context-modulation layers.
                The keyword ``mod_weights``, on the other hand, refers
                specifically to the weights of the context-modulation layers.
                It is not necessary to specify both keywords.
            distilled_params: Will be passed as ``running_mean`` and
                ``running_var`` arguments of method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.
            condition (optional, int or dict): If ``int`` is provided, then this
                argument will be passed as argument ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.

                If a ``dict`` is provided instead, the following keywords are
                allowed:

                    - ``bn_stats_id``: Will be handled as ``stats_id`` of the
                      batchnorm layers as described above.
                    - ``cmod_ckpt_id``: Will be passed as argument ``ckpt_id``
                      to the method
                      :meth:`utils.context_mod_layer.ContextModLayer.forward`.

        Returns:
            y: The output of the network.
        """
        if self._no_weights and weights is None:
            raise Exception(
                "Network was generated without weights. "
                + 'Hence, "weights" option may not be None.'
            )

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        # FIXME code mostly copied from MLP forward method.
        n_cm = 0

        if weights is None:
            weights = self._weights

            int_weights = weights
        else:
            int_weights = None

            if isinstance(weights, dict):
                assert "internal_weights" in weights.keys()
                if "internal_weights" in weights.keys():
                    int_weights = weights["internal_weights"]
            else:
                assert len(weights) == len(self.param_shapes)
                int_weights = weights

            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        ########################
        ### Parse condition ###
        #######################

        bn_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                assert "bn_stats_id" in condition.keys()
                if "bn_stats_id" in condition.keys():
                    bn_cond = condition["bn_stats_id"]
            else:
                bn_cond = condition

        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            # There are 6*n+1 layers that use batch normalization.
            lbw = 2 * (6 * self._n + 1)

            bn_weights = int_weights[:lbw]
            layer_weights = int_weights[lbw:]

            nn = len(self._batchnorm_layers)
            running_means = [None] * nn
            running_vars = [None] * nn
        else:
            layer_weights = int_weights

        if distilled_params is not None:
            if not self._distill_bn_stats:
                raise ValueError(
                    'Argument "distilled_params" can only be '
                    + "provided if the return value of "
                    + 'method "distillation_targets()" is not None.'
                )
            shapes = self.hyper_shapes_distilled
            assert len(distilled_params) == len(shapes)
            for i, s in enumerate(shapes):
                assert np.all(np.equal(s, list(distilled_params[i].shape)))

            # Extract batchnorm stats from distilled_params
            for i in range(0, len(distilled_params), 2):
                running_means[i // 2] = distilled_params[i]
                running_vars[i // 2] = distilled_params[i + 1]

        elif self._use_batch_norm and self._bn_track_stats and bn_cond is None:
            for i, bn_layer in enumerate(self._batchnorm_layers):
                running_means[i], running_vars[i] = bn_layer.get_stats()

        ###############################################
        ### Extract weight tensors and bias vectors ###
        ###############################################
        assert self._has_bias
        w_weights = []
        b_weights = []
        for i, p in enumerate(layer_weights):
            if i % 2 == 1:
                b_weights.append(p)
            else:
                w_weights.append(p)

        ###########################
        ### Forward Computation ###
        ###########################
        bn_ind = 0
        layer_ind = 0

        ### Helper function to process convolutional layers.
        def conv_layer(h, stride, shortcut=None):
            """Compute the output of a resnet conv layer including batchnorm,
            context-mod, non-linearity and shortcut.

            The order if the following:

            conv-layer -> context-mod (if pre-activation) -> batch-norm ->
            shortcut -> non-linearity -> context-mod (if post-activation)

            This method increments the indices ``layer_ind``, ``cm_ind`` and
            ``bn_ind``.

            Args:
                h: Input activity.
                stride: Stride of conv. layer (padding is set to 1).
                shortcut: Is set, this tensor will be added to the activation
                    before the non-linearity is applied.

            Returns:
                Output of layer.
            """
            nonlocal layer_ind, bn_ind

            h = F.conv2d(
                h,
                w_weights[layer_ind],
                bias=b_weights[layer_ind],
                stride=stride,
                padding=1,
            )
            layer_ind += 1

            # Batch-norm
            if self._use_batch_norm:
                h = self._batchnorm_layers[bn_ind].forward(
                    h,
                    running_mean=running_means[bn_ind],
                    running_var=running_vars[bn_ind],
                    weight=bn_weights[2 * bn_ind],
                    bias=bn_weights[2 * bn_ind + 1],
                    stats_id=bn_cond,
                )
                bn_ind += 1

            # Note, as can be seen in figure 5 of the original paper, the
            # shortcut is performed before the ReLU is applied.
            if shortcut is not None:
                h += shortcut

            # Non-linearity
            h = F.relu(h)

            return h

        # x = x.view(*([-1] + self._in_shape))
        # x = x.permute(0, 3, 1, 2)
        h = x

        ### Initial convolutional layer.
        h = conv_layer(h, 1, shortcut=None)

        ### Three blocks, each containing n resnet units.
        for i in range(3):
            # Only the first layer in a block may be a strided convolution.
            if i == 0:
                stride = 1
            else:
                stride = 2

            # For each resnet unit. A resnet unit consists of 2 convolutional
            # layers.
            for j in range(self._n):
                if stride == 1:
                    shortcut_h = h
                else:
                    # The original paper uses zero padding for added output
                    # feature dimensions. Since we apply a strided conv, we
                    # additionally have to subsample the input.
                    # This implementation is motivated by
                    #    https://git.io/fhcfk
                    fs = self._filter_sizes[i + 1]
                    shortcut_h = F.pad(
                        h[:, :, ::2, ::2], (0, 0, 0, 0, fs // 4, fs // 4), "constant", 0
                    )

                h = conv_layer(h, stride, shortcut=None)

                stride = 1

                h = conv_layer(h, stride, shortcut=shortcut_h)

        ### Average pool all activities within a feature map.
        h = F.avg_pool2d(h, [h.size()[2], h.size()[3]])
        h = h.view(h.size(0), -1)

        ### Apply final fully-connected layer and compute outputs.
        h = F.linear(h, w_weights[layer_ind], bias=b_weights[layer_ind])

        return h

    def _compute_hyper_shapes(self, no_weights=None):
        r"""Helper function to compute weight shapes of this network for
        externally maintained weights.

        Returns a list of lists of integers denoting the shape of every
        weight tensor that is not a trainable parameter of this network (i.e.,
        those weight tensors whose shapes are specified in
        :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_distilled`).

        If batchnorm layers are used, then the first :math:`2 * (6n+1)` lists
        will denote the shapes of the batchnorm weights
        :math:`[\gamma_1, \beta_1, \gamma_2, ..., \beta_{6n+1}]`.

        The remaining :math:`2 * (6n+2)` entries are weight tensors and bias
        vectors of each convolutional or fully-connected (last two entries)
        layer in this network.

        Args:
            no_weights (optional): If specified, it will overwrite the private
                member :code:`self._no_weights`.

                If set to ``True``, then all weight shapes of the network
                are computed independent of whether they are maintained
                internally or externally.

        Returns:
            A list of lists of integers.
        """
        if no_weights is None:
            no_weights = self._no_weights

        ret = []
        if no_weights is False:
            return ret

        fs = self._filter_sizes
        ks = self._kernel_size
        n = self._n

        if self._use_batch_norm:
            for i, s in enumerate(fs):
                if i == 0:
                    num = 1
                else:
                    num = 2 * n

                for _ in range(2 * num):
                    ret.append([s])

        f_in = self._in_shape[-1]
        for i, s in enumerate(fs):
            f_out = s
            if i == 0:
                num = 1
            else:
                num = 2 * n

            for _ in range(num):
                ret.append([f_out, f_in, *ks])
                ret.append([f_out])
                f_in = f_out
        ret.append([self._num_outs, fs[-1]])
        ret.append([self._num_outs])

        return ret

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the current batch statistics of all batch
        normalization layers if ``distill_bn_stats`` and ``use_batch_norm``
        was set to ``True`` in the constructor.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        if self.hyper_shapes_distilled is None:
            return None

        ret = []
        for bn_layer in self._batchnorm_layers:
            ret.extend(bn_layer.get_stats())

        return ret

    def _compute_layer_out_sizes(self):
        """Compute the output shapes of all layers in this network.

        This method will compute the output shape of each layer in this network,
        including the output layer, which just corresponds to the number of
        classes.

        Returns:
            (list): A list of shapes (lists of integers). The first entry will
            correspond to the shape of the output of the first convolutional
            layer. The last entry will correspond to the output shape.

            .. note:
                Output shapes of convolutional layers will adhere PyTorch
                convention, i.e., ``[C, H, W]``, where ``C`` denotes the channel
                dimension.
        """
        in_shape = self._in_shape
        fs = self._filter_sizes
        ks = self._kernel_size
        pd = 1  # all paddings are 1.
        assert len(ks) == 2
        assert len(fs) == 4
        n = self._n

        # Note, `in_shape` is in Tensorflow layout.
        assert len(in_shape) == 3
        in_shape = [in_shape[2], *in_shape[:2]]

        ret = []

        C, H, W = in_shape

        # Recall the formular for convolutional layers:
        # W_new = (W - K + 2P) // S + 1

        # First conv layer (stride 1).
        C = fs[0]
        H = (H - ks[0] + 2 * pd) // 1 + 1
        W = (W - ks[1] + 2 * pd) // 1 + 1
        ret.append([C, H, W])

        # First block (no strides).
        C = fs[1]
        H = (H - ks[0] + 2 * pd) // 1 + 1
        W = (W - ks[1] + 2 * pd) // 1 + 1
        ret.extend([[C, H, W]] * (2 * n))

        # Second block (first layer has stride 2).
        C = fs[2]
        H = (H - ks[0] + 2 * pd) // 2 + 1
        W = (W - ks[1] + 2 * pd) // 2 + 1
        ret.extend([[C, H, W]] * (2 * n))

        # Third block (first layer has stride 2).
        C = fs[3]
        H = (H - ks[0] + 2 * pd) // 2 + 1
        W = (W - ks[1] + 2 * pd) // 2 + 1
        ret.extend([[C, H, W]] * (2 * n))

        # Final fully-connected layer (after avg pooling), i.e., output size.
        ret.append([self._num_outs])

        assert len(ret) == 6 * n + 2

        return ret


def init_params(weights, bias=None):
    """Initialize the weights and biases of a linear or (transpose) conv layer.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    Args:
        weights: The weight tensor to be initialized.
        bias (optional): The bias tensor to be initialized.
    """
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


if __name__ == "__main__":
    pass
