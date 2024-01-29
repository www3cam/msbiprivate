#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 08:24:38 2021

@author: cameron

https://notes.quantecon.org/submission/5f44808ad24fdb001162a53b
"""

#from sbi.inference import infer
import torch
import numpy as np
from numpy import genfromtxt
from torch.utils.tensorboard import SummaryWriter
from sbi.inference.base import infer
import sbi
from torch.distributions.distribution import Distribution
import time
# from SW import solve_and_sim_SW
from transformer_module import TransformerModel, SummaryNetLSTM, SummaryNetFeedForward

from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNRE_B
from sbi.utils.get_nn_models import posterior_nn
#from sbi import analysis

from typing import Callable
import pickle as pk

from torch import nn

from sbi.neural_nets.classifier import (
    build_linear_classifier,
    build_mlp_classifier,
    build_resnet_classifier,
)
from warnings import warn

from nflows.transforms.linear import Linear

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tanh, tensor, uint8

from nflows.transforms.base import Transform

from sbi.utils.sbiutils import standardizing_net, standardizing_transform
from sbi.utils.torchutils import create_alternating_binary_mask

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from nflows.utils import torchutils

import matlab.engine

import torch
from torch.distributions.distribution import Distribution

import torch
import torch.distributions as dist

from torch.distributions import constraints
from torch.distributions.transforms import PowerTransform

from pyro.distributions.torch import Gamma, TransformedDistribution



class InvSigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def inverse(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet

class NaiveLinearRange(Linear):
    """A general linear transform that uses an unconstrained weight matrix.
    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.
    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, lb, ub, using_cache=False):
        """Constructor.
        Args:
            features: int, number of input features.
            orthogonal_initialization: bool, if True initialize weights to be a random
                orthogonal matrix.
        Raises:
            TypeError: if `features` is not a positive integer.
        """
        features = len(lb)
        super().__init__(features, using_cache)

        self.bias = torch.nn.Parameter(torch.tensor(lb, dtype=torch.float32), requires_grad=False)
        self._weight = torch.diag(torch.tensor(ub, dtype=torch.float32, requires_grad=False) - torch.tensor(lb, dtype=torch.float32, requires_grad=False))


    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet = torchutils.logabsdet(self._weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        outputs, lu = torch.linalg.solve(self._weight, outputs.t()), torch.lu(outputs.t())[0] # Linear-system solver.
        outputs = outputs.t()
        # The linear-system solver returns the LU decomposition of the weights, which we
        # can use to obtain the log absolute determinant directly.
        logabsdet = -torch.sum(torch.log(torch.abs(torch.diag(lu))))
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(1)
        """
        return self._weight

    def weight_inverse(self):
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        return torch.inverse(self._weight)

    def weight_inverse_and_logabsdet(self):
        """
        Cost:
            inverse = O(D^3)
            logabsdet = O(D)
        where:
            D = num of features
        """
        # If both weight inverse and logabsdet are needed, it's cheaper to compute both together.
        identity = torch.eye(self.features, self.features)
        weight_inv, lu = torch.solve(identity, self._weight)  # Linear-system solver.
        logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
        return weight_inv, logabsdet

    def logabsdet(self):
        """Cost:
            logabsdet = O(D^3)
        where:
            D = num of features
        """
        return torchutils.logabsdet(self._weight)

def build_maf_range(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    lowbound = [0.000001,0.000001,0.000001,0.000001],
    upbound = [.999999,.999999,.999999,.999999],
    **kwargs,
) -> nn.Module:
    
    
    """Builds MAF p(x|y).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    if x_numel == 1:
        warn(f"In one-dimensional output space, this flow is limited to Gaussians")

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=x_numel,
                        hidden_features=hidden_features,
                        context_features=y_numel,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=x_numel),
                ]
            )
            for _ in range(num_transforms)
            ]
    )
    transform = transforms.CompositeTransform([NaiveLinearRange(lb = lowbound, ub = upbound), InvSigmoid(), transform])
    
    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


def range_bound_maf_nn(
    z_score_theta: bool = False,
    z_score_x: bool = False,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    lowerbound = [0.000001,0.000001,0.000001,0.000001],
    upperbound = [.999999,.999999,.999999,.999999]
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the posterior.
    This function will usually be used for SNPE. The returned function is to be passed
    to the inference class when using the flexible interface.
    Args:
        model: only works on maf curretly
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
                "lowbound",
                "upbound"
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                embedding_net,
                num_components,
                lowerbound,
                upperbound
            ),
        )
    )
    def build_fn(batch_theta, batch_x):
        return build_maf_range(batch_x=batch_theta, batch_y=batch_x, **kwargs)
    return build_fn


'''
nnet = range_bound_maf_nn()

datax = genfromtxt('k.csv',delimiter=',')

theta = torch.tensor([.7, .15, .2, .7], dtype = torch.float32).reshape(1,-1)

tdata = torch.tensor(datax.ravel().reshape(1,-1), dtype = torch.float32)
tdata2 = tdata.tile((2,1))
theta2 = theta.tile((2,1))

neuralnet1 = nnet(batch_theta = theta2, batch_x=tdata2)

print(neuralnet1.sample(1, tdata2))


'''

eng = matlab.engine.start_matlab()

def dynareSW(vecin):
    try:
        vec1 = vecin.reshape(-1).numpy().tolist()
        # vec = [0 for i in range(36)]
        #get rid of this for SW priors
        # vec[35] = vec1[0]
        # vec[17] = vec1[1]
        # vec[25] = vec1[2]
        # vec[34] = vec1[3]
        # vec[16] = vec1[4]
        # vec[18] = vec1[5]
        # vec[19] = vec1[6]
        # vec[20] = vec1[7]
        # vec[21] = vec1[8]
        # vec[22] = vec1[9]
        # vec[23] = vec1[10]
        # vec[24] = vec1[11]
        # vec[26] = vec1[12]
        # vec[27] = vec1[13]
        # vec[28] = vec1[14]
        # vec[29] = vec1[15]
        # vec[7] = vec1[16]
        # vec[8] = vec1[17]
        # vec[9] = vec1[18]
        # vec[10] = vec1[19]
        # vec[11] = vec1[20]
        # vec[12] = vec1[21]
        # vec[13] = vec1[22]
        # vec[14] = vec1[23]
        # vec[15] = vec1[24]
        # vec[30] = vec1[25]
        # vec[31] = vec1[26]
        # vec[32] = vec1[27]
        # vec[33] = vec1[28]
        # vec[0] = vec1[29]
        # vec[1] = vec1[30]
        # vec[2] = vec1[31]
        # vec[3] = vec1[32]
        # vec[4] = vec1[33]
        # vec[5] = vec1[34]
        # vec[6] = vec1[35]
        
        vec1 = matlab.double(vec1, size = (36,1))
         
        x = np.asarray(eng.simDSGEdata(vec1)).reshape((1,7*200))
    except:
        x = np.empty((1,7*200))
        x.fill(np.nan)
    
    return torch.tensor(x, dtype = torch.float32)


start = time.time()

num_rounds = 50

writer = SummaryWriter()


#vecin = [0.491692949132718, 0.348294030602838, 0.665231992527837,0.557840210250528,0.219038092751674,0.208093187137879,0.256317924519313,0.972635066864697,0.129121393775878,0.958628355930555,0.602064421092848,0.189932908919849,0.975604730882909,0.971780669995855,0.823984792605355,0.923696712042029,6.30438354438921,1.25790675965434,0.795613779353936,0.756803508578363,2.51006941593423,0.520392268102388,0.524504314169190,0.167907316224608,0.349694333086695,1.65696356884638,1.85846020646764,0.863865858276508,0.110347234197639,0.118157950835201,0.626466700805271,0.102616274945699,1.31631181800318,0.501327099746232,0.578101790099014,0.192430595782335]

datax = np.loadtxt('realdataSW.csv', delimiter = ',') #genfromtxt('krbc.csv',delimiter=',')

# s = prior2.sample()

# z = prior2.log_prob(s)
tdata = torch.tensor(datax.ravel().reshape(1,-1), dtype = torch.float32)
    
    # calfa = .01, csigma = .25, cfc = 1.0, cgy = .01, \
    # csadjcost = 2., chabb = .001, cprobw = .5, csigl = .25, \
    # cprobp = .5, cindw = .01, cindp = .01, czcap = .01, \
    # crpi = 1.0, crr = .5, cry = .001, crdy = .001, crhoa = .01, \
    # crhob = .01, crhog = .01, crhoqs = .01, \
    # crhoms = 0.01, crhopinf = 0.01, crhow = 0.001, cmap = 0.01, cmaw = 0.01, constepinf = .1, \
    # constebeta = .01, constelab = -10., ctrend = .1, ea = 0.01,\
    # eb = 0.025, eg = .01, eqs = .01, em = .01, epinf = .01, ew = .01
    
    # calfa = 1.0, csigma = 1.5, cfc = 3., cgy = 2., \
    # csadjcost = 15, chabb = .99, cprobw = .95, csigl = 10., \
    # cprobp = .95, cindw = .99, cindp = .99, czcap = 1., \
    # crpi = 3., crr = .975, cry = .5, crdy = .5, crhoa = .9999, \
    # crhob = .9999, crhog = .9999, crhoqs = .9999, \
    # crhoms = 0.9999, crhopinf = 0.9999, crhow = 0.9999, cmap = 0.9999, cmaw = 0.9999, constepinf = 2., \
    # constebeta = 2.0, constelab = 10., ctrend = .8, ea = 3.,\
    # eb = 5., eg = 3., eqs = 3., em = 3., epinf = 3., ew = 3.
   


# Data from the table
table = [
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.025, 5, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.001, 0.9999, "BETA_PDF", 0.5, 0.20],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.2],
    [0.01, 0.9999, "BETA_PDF", 0.5, 0.2],
    [2, 15, "NORMAL_PDF", 4, 1.5],
    [0.25, 3, "NORMAL_PDF", 1.5, 0.375],
    [0.001, 0.99, "BETA_PDF", 0.7, 0.1],
    [0.3, 0.95, "BETA_PDF", 0.5, 0.1],
    [0.25, 10, "NORMAL_PDF", 2, 0.75],
    [0.5, 0.95, "BETA_PDF", 0.5, 0.10],
    [0.01, 0.99, "BETA_PDF", 0.5, 0.15],
    [0.01, 0.99, "BETA_PDF", 0.5, 0.15],
    [0.01, 1, "BETA_PDF", 0.5, 0.15],
    [1, 3, "NORMAL_PDF", 1.25, 0.125],
    [1, 3, "NORMAL_PDF", 1.5, 0.25],
    [0.5, 0.975, "BETA_PDF", 0.75, 0.10],
    [0.001, 0.5, "NORMAL_PDF", 0.125, 0.05],
    [0.001, 0.5, "NORMAL_PDF", 0.125, 0.05],
    [0.1, 2, "GAMMA_PDF", 0.625, 0.1],
    [0.01, 2, "GAMMA_PDF", 0.25, 0.1],
    [-10, 10, "NORMAL_PDF", 0, 2.0],
    [0.1, 0.8, "NORMAL_PDF", 0.4, 0.10],
    [0.01, 2, "NORMAL_PDF", 0.5, 0.25],
    [0.01, 1, "NORMAL_PDF", 0.3, 0.05]
]

# table = [
#     [1, 3, "NORMAL_PDF", 1.25, 0.125],
#     [1, 3, "NORMAL_PDF", 1.5, 0.25],
#     [0.5, 0.975, "BETA_PDF", 0.75, 0.10],
#     [0.001, 0.5, "NORMAL_PDF", 0.125, 0.05],
#     [0.001, 0.5, "NORMAL_PDF", 0.125, 0.05],
#     [0.1, 2, "GAMMA_PDF", 0.625, 0.1],
#     [0.01, 2, "GAMMA_PDF", 0.25, 0.1],
#     [-10, 10, "NORMAL_PDF", 0, 2.0],
#     [0.1, 0.8, "NORMAL_PDF", 0.4, 0.10],
#     [0.01, 2, "NORMAL_PDF", 0.5, 0.25],
#     [0.01, 1, "NORMAL_PDF", 0.3, 0.05],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.025, 5, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 3, "INV_GAMMA_PDF", 0.1, 2],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.001, 0.9999, "BETA_PDF", 0.5, 0.20],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.2],
#     [0.01, 0.9999, "BETA_PDF", 0.5, 0.2],
#     [2, 15, "NORMAL_PDF", 4, 1.5],
#     [0.25, 3, "NORMAL_PDF", 1.5, 0.375],
#     [0.001, 0.99, "BETA_PDF", 0.7, 0.1],
#     [0.3, 0.95, "BETA_PDF", 0.5, 0.1],
#     [0.25, 10, "NORMAL_PDF", 2, 0.75],
#     [0.5, 0.95, "BETA_PDF", 0.5, 0.10],
#     [0.01, 0.99, "BETA_PDF", 0.5, 0.15],
#     [0.01, 0.99, "BETA_PDF", 0.5, 0.15],
#     [0.01, 1, "BETA_PDF", 0.5, 0.15],

# ]

# vec1 = [[] for i in range(36)]

# vec[35] = vec1[0]
# vec[17] = vec1[1]
# vec[25] = vec1[2]
# vec[34] = vec1[3]
# vec[16] = vec1[4]
# vec[18] = vec1[5]
# vec[19] = vec1[6]
# vec[20] = vec1[7]
# vec[21] = vec1[8]
# vec[22] = vec1[9]
# vec[23] = vec1[10]
# vec[24] = vec1[11]
# vec[26] = vec1[12]
# vec[27] = vec1[13]
# vec[28] = vec1[14]
# vec[29] = vec1[15]
# vec[7] = vec1[16]
# vec[8] = vec1[17]
# vec[9] = vec1[18]
# vec[10] = vec1[19]
# vec[11] = vec1[20]
# vec[12] = vec1[21]
# vec[13] = vec1[22]
# vec[14] = vec1[23]
# vec[15] = vec1[24]
# vec[30] = vec1[25]
# vec[31] = vec1[26]
# vec[32] = vec1[27]
# vec[33] = vec1[28]
# vec[0] = vec1[29]
# vec[1] = vec1[30]
# vec[2] = vec1[31]
# vec[3] = vec1[32]
# vec[4] = vec1[33]
# vec[5] = vec1[34]
# vec[6] = vec1[35]

# print(vec1)




class constrained_dist(dist.Distribution):
    
    def __init__(self, base_dist, lb, ub):
        super(constrained_dist, self).__init__()
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        self.lb = lb
        self.ub = ub
        try:
            self.factor = 1/(base_dist.cdf(ub) - base_dist.cdf(lb))
        except:
            sam1 = base_dist.sample((1000000,))
            self.factor = 1000000/torch.sum(torch.gt(sam1,self.lb)*torch.lt(sam1,self.ub))
        self.base = base_dist

    
    def sample(self, sample_shape=torch.Size()):
        #tested
        samps = self.base.sample(sample_shape)
        any1 = True
        while any1:
            outbounds = torch.gt(samps,self.lb)*torch.lt(samps,self.ub)
            any1 = torch.any(~outbounds)
            outbounds = outbounds.float()
            samps2 = self.base.sample(sample_shape)
            samps = outbounds*samps + (1-outbounds)*samps2
        return samps
    
    def log_prob(self, samps):
        size = samps.size()
        eles = torch.numel(samps)
        outbounds = torch.gt(samps,self.lb)*torch.lt(samps,self.ub)
        any1 = torch.any(~outbounds)
        assert not any1
        lp = self.base.log_prob(samps) + torch.log(self.factor)
        return lp

# bdsit = dist.Normal(0,1)

# dist = constrained_dist(bdsit, -.5, .5)

# sa = dist.sample((100,))

# lp1 = dist.log_prob(sa)


# print(lp1)
# print(dist.factor)

# print(1/(bdsit.cdf(torch.tensor(1)) - bdsit.cdf(torch.tensor(-1))))



class InverseGamma(TransformedDistribution):
    r"""
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.

        X ~ Gamma(concentration, rate)
        Y = 1/X ~ InverseGamma(concentration, rate)

    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        base_dist = Gamma(concentration, rate)
        super().__init__(
            base_dist,
            PowerTransform(-base_dist.rate.new_ones(())),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate

# Create a list to hold the distributions
distributions = []

for row in table:
    lower_bound, upper_bound, distribution_type, mean, std_dev = row
    if distribution_type == "INV_GAMMA_PDF":
        alpha = mean**2 / std_dev**2 + 2
        beta = mean*(alpha-1)
        distribution = InverseGamma(alpha,beta)
    elif distribution_type == "BETA_PDF":
        # For Beta distribution, parameters alpha and beta can be derived from mean and variance
        # Variance = mean*(1-mean) / (alpha+beta+1)
        # We can rewrite this equation to express alpha in terms of beta: alpha = beta*mean/(1-mean)
        # And then substitute this into the variance equation to solve for beta.
        variance = std_dev**2
        beta = (mean*(1-mean)/variance - 1)*(1-mean)
        alpha = beta * mean / (1-mean)
        
        # Adjust parameters to be within the defined bounds
        concentration0 = alpha
        concentration1 = beta
        distribution = dist.Beta(concentration0, concentration1)
    elif distribution_type == "NORMAL_PDF":
        distribution = dist.Normal(mean, std_dev)
    elif distribution_type == "GAMMA_PDF":
        # For Gamma distribution, parameters alpha (shape) and beta (rate) can be derived from mean and variance
        # mean = alpha/beta and variance = alpha/beta^2
        alpha = mean**2 / std_dev**2
        beta = mean / std_dev**2
        distribution = dist.Gamma(alpha, beta)
    # ... similarly for other distribution types
    constrained_distribution = constrained_dist(distribution, lower_bound, upper_bound)


    distributions.append(constrained_distribution)


class thirtysixprior(dist.Distribution):
    def __init__(self, distarray):
        validate_args=False
        super(thirtysixprior, self).__init__()
        self.distarr = distarray

    def sample(self, sample_shape=torch.Size()):
        #tested
        samps = []
        for distrib in self.distarr:
            samps.append(distrib.sample(sample_shape))
        s = torch.stack(samps, axis = -1)
        return s
    
    def log_prob(self, samps):
        shape1 = samps.shape
        assert shape1[-1] == 36
        # samps1 = torch.reshape(samps, (-1,36))
        # shape2 = samps1.shape
        # lprobv = torch.zeros(shape1[:-1])
        lprob = torch.zeros(shape1[:-1])
        for i in range(36):
            add = self.distarr[i].log_prob(samps[...,i])
            lprob = lprob + add
        return lprob



prior_min =  [vec[0] for vec in table]
prior_max = [vec[1] for vec in table]

 
# prior_min = [.01, .25, 1.0, .01, 2., .001, .5, .25, .5, .01, .01, .01, 1.0, .5, .001, 
#              .001, .01, .01, .01, .01, 0.01, 0.01, 0.001, 0.01, 0.01, .1, .01, -10., 
#              .1, 0.01, 0.025, .01, .01, .01, .01, .01]
    
# prior_max = [1.0, 1.5, 3., 2., 15, .99, .95, 10., .95, .99, .99, 1., 3., .975,
#              .5, .5, .9999, .9999, .9999, .9999, 0.9999, 0.9999, 0.9999, 0.9999, 
#              0.9999, 2., 2.0, 10., .8, 3., 5., 3., 3., 3., 3., 3.]


# class constrained_dist(Distribution):
    
#     def __init__(self, base_dist, lb, ub):
#         super(constrained_dist, self).__init__()
#         self.factor = 1/(base_dist.cdf(ub) - base_dist.cdf(lb))
#         self.base = base_dist
#         self.lb = lb
#         self.ub = ub
    
#     def sample(self, sample_shape=torch.Size()):
#         lbmat = self.lb.expand(sample_shape)
#         ubmat = self.ub.expand(sample_shape)
#         samps = self.base.sample(sample_shape)
#         any1 = True
#         while any1:
#             outbounds = torch.lt(samps,lbmat)*torch.gt(samps,ubmat)
#             any1 = torch.any(outbounds)
#             samps2 = self.base.sample(sample_shape)
#             samps = (1-outbounds)*samps + outbounds*samps2
#         return samps
    
#     def log_prob(self, samps):
#         size = samps.size()
#         eles = torch.numel(samps)
#         lbmat = self.lb.expand(size)
#         ubmat = self.ub.expand(size)
#         outbounds = torch.lt(samps,lbmat)*torch.gt(samps,ubmat)
#         any1 = torch.any(outbounds)
#         assert not any1
#         logp = torch.log(self.factor)*(eles)+base.dist.log_prob(samps)
#         return logp

#can build a constrianted distribution like this:
# import torch
# import torch.distributions as dist

# # Define a normal distribution with mean and standard deviation
# mu = torch.tensor(0.0)
# sigma = torch.tensor(1.0)
# normal_distribution = dist.Normal(mu, sigma)

# # Define the lower and upper truncation bounds
# lower_bound = torch.tensor(-1.0)
# upper_bound = torch.tensor(1.0)

# # Apply constraints to truncate the distribution
# constrained_distribution = dist.transform_to(dist.constraints.interval(lower_bound, upper_bound))(normal_distribution)

# # Sample from the constrained distribution
# sample = constrained_distribution.sample()
# print("Sample from the constrained distribution:", sample)

# # Calculate the log probability of a sample
# log_prob = constrained_distribution.log_prob(sample)
# print("Log probability of the sample:", log_prob)
  
#add independent
# import torch
# import torch.distributions as dist

# # Set the means and standard deviations for the normal distributions
# means = torch.tensor([0.0, 1.0])
# std_devs = torch.tensor([1.0, 2.0])

# # Create a list of univariate normal distributions
# normal_distributions = [dist.Normal(loc=mu, scale=sigma) for mu, sigma in zip(means, std_devs)]

# # Set the shape parameter for the gamma distribution
# shape_param = torch.tensor([2.0, 3.0])

# # Create a list of univariate gamma distributions
# gamma_distributions = [dist.Gamma(concentration=alpha, rate=beta) for alpha, beta in zip(shape_param, shape_param)]

# # Combine the normal and gamma distributions into a multivariate distribution
# # Using Independent to create a multivariate distribution with independent components
# multivariate_dist = dist.Independent(dist.Categorical(torch.tensor([0.5, 0.5])), reinterpreted_batch_ndims=1)

# # Sample from the multivariate distribution
# sample = multivariate_dist.sample()
# print("Sample from the multivariate distribution:", sample)

# # Calculate the log probability of a sample
# log_prob = multivariate_dist.log_prob(sample)
# print("Log probability of the sample:", log_prob)


        
        
#prior_min = [0.,0.,0.,0.]
#prior_max = [1.,1.,1.,1.]
# prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
#                                     high=torch.as_tensor(prior_max))

prior = thirtysixprior(distributions)
    
#SNPE  = sbi.inference.snpe.snpe_c.SNPE_C(summary_writer=writer)

#device = torch.device("cuda")

model = SummaryNetFeedForward(timesteps = 200, covariates = 7)#SummaryNetLSTM TransformerModel(10, 10, 100, 5, 32, 4, 0.0)

simulator, prior = prepare_for_sbi(dynareSW, prior)


lobound1 = (np.array(prior_min) + .000001).tolist()

upbound1 = (np.array(prior_max) - .000001).tolist()

nnet = range_bound_maf_nn(embedding_net=model, lowerbound=lobound1, upperbound = upbound1)

inference = SNPE(prior=prior, summary_writer=writer, device='cpu', density_estimator=nnet)

posteriors = []#pk.load(open(r"RBC_posteriors_realdata_NF1.pickle", "rb"))

#inference = SNRE_B(prior=prior,  classifier='mlp', summary_writer=writer, num_workers=8)

proposal = prior# posteriors[-1]


sampsvec = []
sampsstd = []
xsamp = []
thetasamp = []

prpoposal = prior

folder = 'data/dynareSW_realprior'
name = 'dense1_NF2'

#post11 = pk.load(open(folder + "SW_posteriors_"+name+".pickle", "rb"))

    
#proposal = post11[-1].set_default_x(tdata)
    
theta1, x1 = simulate_for_sbi(dynareSW, proposal, num_simulations=10)
print(x1.shape)
print(theta1.shape)
xsamp.append(x1)
thetasamp.append(theta1)

print(x1)
print(theta1)

 # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`

# condition = theta1 > 0.000001
# row_cond1 = condition.all(1)
# condition = theta1 < .999999
# row_cond2 = condition.all(1)
# rcond = torch.logical_and(row_cond1, row_cond2)
# x = x1[rcond, :]
# theta = theta1[rcond, :]

density_estimator = inference.append_simulations(theta1, x1, proposal=proposal).train(training_batch_size=1000, discard_prior_samples = True)
#density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
posteriors.append(posterior)
proposal = posterior.set_default_x(tdata)

for r in range(num_rounds):
    sims = 10
    theta1, x1 = simulate_for_sbi(dynareSW, proposal, num_simulations=sims)
    print(x1.shape)
    print(theta1.shape)
    xsamp.append(x1)
    thetasamp.append(theta1)
    
    # if r > 0:
    meanvec = posteriors[-1].sample((1000,), x = tdata).mean(0)
    stdvec = posteriors[-1].sample((1000,), x = tdata).std(0)
    print(meanvec)
    print(stdvec)
    print(np.array([.24, 1.5, 1.5, .51, 6.0144, .6361, .8087, 1.9423, 
                    .6, .3243, .47, .2696, 1.488, .8762, .0593, .2347, .9977, 
                    .5799, .9957, .7165, 0.015, 0.015, 0.015, 0.015, 0.015, .7, 
                    .7420, 0., .3982, 0.4618, 1.8513, .6090, .6017, .2397, .1455, .2089]))
    sampsvec.append(meanvec)
    sampsstd.append(stdvec)
     # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
    
    # condition = theta1 > 0.0
    # row_cond1 = condition.all(1)
    # condition = theta1 < 1.0
    # row_cond2 = condition.all(1)
    # rcond = torch.logical_and(row_cond1, row_cond2)
    # x = x1[rcond, :]
    # theta = theta1[rcond, :]
    
    density_estimator = inference.append_simulations(theta1, x1, proposal=proposal).train(training_batch_size=1000)
    #density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(tdata)
    # print(posteriors[-1].log_prob(torch.tensor([[.7, .7,.7,.7],[.7, .15,.2,.7],[.1, .15,.2,.1]]),x=tdata))

    #posterior = infer(run_and_simulate, prior, method='SNPE', num_simulations=1000)

    folder = 'data/dynareSW1234567/'
    name = 'x'
    samples = posterior.sample((100000,), x=tdata)
    
    print(time.time()-start)
    
    sampnum = samples.numpy()
    np.savetxt(folder + 'SW_thetas'+name+str(r)+'.csv', sampnum, delimiter = ',')
    pk.dump(sampsvec, open(folder + "SW_imeans_"+name+".pickle", "wb"))
    pk.dump(sampsstd, open(folder + "SW_istd_"+name+".pickle", "wb"))
    pk.dump(posteriors, open(folder + "SW_posteriors_"+name+".pickle", "wb"))
    pk.dump(xsamp, open(folder + "SW_x_"+name+".pickle", "wb"))
    pk.dump(thetasamp, open(folder + "SW_thetas_value_"+name+".pickle", "wb"))
    # fig, ax = analysis.pairplot(sampnum, 
    #                          points=np.array([.24, 1.5, 1.5, .51, 6.0144, .6361, .8087, 1.9423, 
    #                                           .6, .3243, .47, .2696, 1.488, .8762, .0593, .2347, .9977, 
    #                                           .5799, .9957, .7165, 0.015, 0.015, 0.015, 0.015, 0.015, .7, 
    #                                           .7420, 0., .3982, 0.4618, 1.8513, .6090, .6017, .2397, .1455, .2089]),
    #                          labels=['calfa', 'csigma', 'cfc', 'cgy', 'csadjcost', 'chabb', 'crobw', 
    #                                  'csigl', 'cprobp', 'cindw', 'cindp', 'czcap', 'crpi', 'crr', 'cry',
    #                                  'crdy', 'crhoa', 'crhob', 'crhog', 'crhoqs', 'crhoms', 'crhopinf',
    #                                  'crhow', 'cmap', 'cmaw', 'constepinf', 'constebeta', 'constelab',
    #                                  'ctrend', 'ea', 'eb', 'eg', 'eqs', 'em', 'epinf', 'ew'],
    #                          limits=[[0.01, 1.0],[0.25, 1.5], [1.0, 3.0], [0.01, 2.0], [2.0, 15.0],
    #                                  [0.001, 0.99], [0.5, 0.95], [0.25, 10.0], [0.5, 0.95], [0.01, 0.99],
    #                                  [0.01, 0.99], [0.01, 1.0], [1.0, 3.0], [0.5, 0.975], [0.001, 0.5],
    #                                  [0.001, 0.5], [0.01, 0.9999], [0.01, 0.9999], [0.01, 0.9999], 
    #                                  [0.01, 0.9999], [0.01, 0.9999], [0.01, 0.9999], [0.001, 0.9999],
    #                                  [0.01, 0.9999], [0.01, 0.9999], [0.1, 2.0], [0.01, 2.0], [-10.0, 10.0],
    #                                  [0.1, 0.8], [0.01, 3.0], [0.025, 5.0], [0.01, 3.0], [0.01, 3.0],
    #                                  [0.01, 3.0], [0.01, 3.0], [0.01, 3.0]],
    #                          ticks=[[0.01, 1.0],[0.25, 1.5], [1.0, 3.0], [0.01, 2.0], [2.0, 15.0],
    #                                  [0.001, 0.99], [0.5, 0.95], [0.25, 10.0], [0.5, 0.95], [0.01, 0.99],
    #                                  [0.01, 0.99], [0.01, 1.0], [1.0, 3.0], [0.5, 0.975], [0.001, 0.5],
    #                                  [0.001, 0.5], [0.01, 0.9999], [0.01, 0.9999], [0.01, 0.9999], 
    #                                  [0.01, 0.9999], [0.01, 0.9999], [0.01, 0.9999], [0.001, 0.9999],
    #                                  [0.01, 0.9999], [0.01, 0.9999], [0.1, 2.0], [0.01, 2.0], [-10.0, 10.0],
    #                                  [0.1, 0.8], [0.01, 3.0], [0.025, 5.0], [0.01, 3.0], [0.01, 3.0],
    #                                  [0.01, 3.0], [0.01, 3.0], [0.01, 3.0]],
    #                          points_colors='r',
    #                          points_offdiag={'markersize': 6},
    #                          figsize=[15., 15.])
# method_fun: Callable = getattr(sbi.inference, 'SNPE')
# inference = method_fun(prior)
