#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 08:24:38 2021

@author: cameron
"""

import torch
import numpy as np
from numpy import genfromtxt
from torch.utils.tensorboard import SummaryWriter
from sbi.inference.base import infer
import sbi
from torch.distributions.distribution import Distribution
import time
from HANK_Reiter_2_like_mp3 import solve_and_sim_reiter
from transformer_module import TransformerModel, SummaryNetLSTM, SummaryNetFeedForward

from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNRE_B
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.inference import ratio_estimator_based_potential, VIPosterior


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
        outputs, lu = torch.solve(outputs.t(), self._weight)  # Linear-system solver.
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
start = time.time()

num_rounds = 50

writer = SummaryWriter()


#"true" data from file
datax = genfromtxt('reiter_RBC_po2.csv',delimiter=',')

# s = prior2.sample()

# z = prior2.log_prob(s)
tdata = torch.tensor(datax.ravel().reshape(1,-1), dtype = torch.float32)


prior_min = [0.,0.,0.0,0.0,1.02,0.0,1.0,0.01,0.01,0.4]
prior_max =  [1.,1.,3.,.3,1.4,1.,2.,.2,.3,1.]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))
    
#create embedding network
# model = SummaryNetFeedForward(timesteps = 200, covariates = 2)#SummaryNetLSTM TransformerModel(10, 10, 100, 5, 32, 4, 0.0)

simulator, prior = prepare_for_sbi(solve_and_sim_reiter, prior)


#create normalizing flow
#nnet = range_bound_maf_nn(embedding_net=model, lowerbound=[0.,0.,0.0,0.0,1.02,0.0,1.0,0.01,0.01,0.4], upperbound = [1.,1.,3.,.3,1.4,1.,2.,.2,.3,1.])
inference = SNRE_B(prior=prior, summary_writer=writer, device='cpu', classifier = 'mlp')

posteriors = []#pk.load(open(r"RBC_posteriors_realdata_NF1.pickle", "rb"))

proposal = prior# posteriors[-1]


sampsvec = []
sampsstd = []
xsamp = []
thetasamp = []

folder = 'HANK/'#'data/Winbery_dynare/'
name = 'real_distributed_dense_NF3'

theta, x = simulate_for_sbi(solve_and_sim_reiter, proposal, num_simulations=10, num_workers=1)
print(x.shape)
print(theta.shape)
#xsamp.append(x)
#thetasamp.append(theta)

# meanvec = posteriors[-1].sample((1000,), x = tdata).mean(0)
# stdvec = posteriors[-1].sample((1000,), x = tdata).std(0)
# print(meanvec)
# print(stdvec)
# sampsvec.append(meanvec)
# sampsstd.append(stdvec)
 # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`

# condition = theta1 > 0.0
# row_cond1 = condition.all(1)
# condition = theta1 < 1.0
# row_cond2 = condition.all(1)
# rcond = torch.logical_and(row_cond1, row_cond2)
# x = x1[rcond, :]
# theta = theta1[rcond, :]

density_estimator = inference.append_simulations(theta, x).train(training_batch_size=200)
#density_estimator = inference.append_simulations(theta, x).train()
# posterior = inference.build_posterior(density_estimator, sample_with='vi')
# posteriors.append(posterior)
# proposal = posterior.set_default_x(tdata)
potential_fn, parameter_transform = ratio_estimator_based_potential(
    ratio_estimator = density_estimator, prior = prior, x_o = tdata)
posterior = VIPosterior(potential_fn, prior, theta_transform=parameter_transform)
posterior.set_default_x(tdata) 
posterior.vi_method = 'rKL'
posterior.train() 
posteriors.append(posterior)

samples = posterior.sample((10000,))

time_cur = time.time()-start

sampnum = samples.numpy()
pk.dump(time_cur, open(folder + "HANK_reiter_time_"+name+".pickle", "wb"))
np.savetxt(folder + 'HANK_reiter_thetas'+name+'.csv', sampnum, delimiter = ',')
pk.dump(sampsvec, open(folder + "HANK_reiter_imeans_"+name+".pickle", "wb"))
pk.dump(sampsstd, open(folder + "HANK_reiter_istd_"+name+".pickle", "wb"))
# pk.dump(posteriors, open(folder + "HANK_reiter_posteriors_"+name+".pickle", "wb"))
#pk.dump(xsamp, open(folder + "HANK_reiter_x_"+name+".pickle", "wb"))
#pk.dump(thetasamp, open(folder + "HANK_reiter_thetas_value_"+name+".pickle", "wb"))


for r in range(num_rounds):
    sims = 10
    #simulate data
    theta, x = simulate_for_sbi(solve_and_sim_reiter, proposal, num_simulations=sims, num_workers=1)
    print(x.shape)
    print(theta.shape)
    
    print(r)
    #estimate the normalizing flow on simulated data
    # inference = SNPE(prior=prior, summary_writer=writer, device='cpu', density_estimator=nnet)
    density_estimator = inference.append_simulations(theta, x).train(training_batch_size=2000)
    potential_fn, parameter_transform = ratio_estimator_based_potential(
        ratio_estimator = density_estimator, prior = prior, x_o = tdata)
    posterior = VIPosterior(potential_fn, prior, theta_transform=parameter_transform)
    posterior.set_default_x(tdata) 
    posterior.vi_method = 'rKL'
    posterior.train() 
    posteriors.append(posterior)

    # posterior = inference.build_posterior(density_estimator, sample_with='vi')
    # posteriors.append(posterior)
    # proposal = posterior.set_default_x(tdata)
    # potential_fn, parameter_transform = likelihood_estimator_based_potential(density_estimator, prior, tdata)
    # posterior = VIPosterior(potential_fn, proposal=prior, theta_transform=parameter_transform)
    # posteriors.append(posterior)
    # #fix new proposal distrubtion as current posterior conditioned on real data
    # proposal = posterior.set_default_x(tdata)
    
    meanvec = posteriors[-1].sample((1000,)).mean(0)
    stdvec = posteriors[-1].sample((1000,)).std(0)
    print(meanvec)
    print(stdvec)
    print([.95,.80,.8,.1,1.2, .75,1.5,.06,.15,.6])
    sampsvec.append(meanvec)
    sampsstd.append(stdvec)

    samples = posterior.sample((10000,))
    
    time_cur = time.time()-start
    #save data to disk
    sampnum = samples.numpy()
    pk.dump(time_cur, open(folder + "HANK_reiter_time_"+name+".pickle", "wb"))
    np.savetxt(folder + 'HANK_reiter_thetas'+name+'.csv', sampnum, delimiter = ',')
    pk.dump(sampsvec, open(folder + "HANK_reiter_imeans_"+name+".pickle", "wb"))
    pk.dump(sampsstd, open(folder + "HANK_reiter_istd_"+name+".pickle", "wb"))
    # pk.dump(posteriors, open(folder + "HANK_reiter_posteriors_"+name+".pickle", "wb"))
    #pk.dump(xsamp, open(folder + "HANK_reiter_x_"+name+".pickle", "wb"))
    #pk.dump(thetasamp, open(folder + "HANK_reiter_thetas_value_"+name+".pickle", "wb"))
    # print(posteriors[-1].log_prob(torch.tensor([[.7, .7,.7,.7],[.7, .15,.2,.7],[.1, .15,.2,.1]]),x=tdata))

print(time.time()-start)

