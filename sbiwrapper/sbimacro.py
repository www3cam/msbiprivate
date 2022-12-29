# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:10:20 2022

@author: camer
"""

import torch
import numpy as np
from numpy import genfromtxt
from torch.utils.tensorboard import SummaryWriter
from sbi.inference.base import infer
import sbi
from torch.distributions.distribution import Distribution
import time
from transformer_module import SummaryNetFeedForward

from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNRE_B
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.inference import ratio_estimator_based_potential, VIPosterior

import multiprocessing


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

def sbi_macro(X, boundmin, boundmax, prior, generator, netparams = None, num_rounds = 10, 
              method = 'SNPE', folder = None, init_simulations = None, round_simulations = 10000,
              numworkers = 1, batch_size = 1000):
    """
    Takes a structural model and returns the posterior parameters that fit the
    data using simulation based inference

    Parameters
    ----------
    X : numpy array
        the input data to condition posterior on,
        should be correct shape even though the data will be flattened in 
        simulation. Will be flattened to have shape (1,-1)
    boundmin : list
        lower bound of parameters (if using a uniform prior, for both min and max sometimes it helps to have a bound
                                   epsilon above and below the uniform prior up and lower bound)
    boundmax : list
        upper bound of parameters (if boundmin and boundmax are none, uses a mixture density network)
    prior: torch distributions
        the prior of the parameters
    generator : python callable:
        should take in a torch tensor of parameters and output a torch tensor 
        of simulated data (flattened with shape (1,-1))
    netparams : a tuple of 2 values         
         the tuple contains: (hiddendimesnions, output/embedding dimension)
         if None, infers timestesp and covariate shape from X and uses default 
         other values
    num_rounds : int
        number of rounds to do sbi for
    method : string one of SNPE or SNVI
        Tells the model to use either SNPE or SNVI
    folder : path (string)
        folder to save the data if None doesn't save the data
    init_simulations : int or None:
        Number of simulations for the first round.  If None same number as 
        round_simulations
    round_simulations : int
        number of simulations after round 1.  
    numworkers : int or None
        number of cpus to use. If set to none uses all available cpus
    batch_size : int
        The batch size to train the density estimator
    Returns
    -------
    sampnum : numpy array
        A numpy array of samples from the posterior
    posteriors: list of posterior objects
        A list of sbi posteriors for each round of inference.  The object of interest
        is the last posterior ie posteriors[-1]

    """
    r,c = X.shape
    Xtorch = torch.tensor(X.ravel().reshape(1,-1), dtype = torch.float32)

    writer = SummaryWriter()
    if init_simulations == None:
        init_simulations = round_simulations
            
    if numworkers == None:
        numworkers = multiprocessing.cpu_count()
        
    if method == 'SNPE':    
        if netparams == None:
            net = SummaryNetFeedForward(r, c)
        else:
            net = SummaryNetFeedForward(r, c, netparams[0], netparams[1])
    
    simulator, prior = prepare_for_sbi(generator, prior)
        
    if method == 'SNVI':#TODO doesn't have upper bound or lower bound
        inference = SNRE_B(prior=prior, summary_writer=writer, device='cpu', classifier='mlp')
        posteriors = []
        
        proposal = prior
        
        print("Round: 1")
        
        theta, x = simulate_for_sbi(generator, proposal, num_simulations=init_simulations,
                                    num_workers=numworkers)
        
        density_estimator = inference.append_simulations(theta, x).train(training_batch_size=batch_size)
        potential_fn, parameter_transform = ratio_estimator_based_potential(
            ratio_estimator = density_estimator, prior = prior, x_o = Xtorch)
        posterior = VIPosterior(potential_fn, prior, theta_transform=parameter_transform)
        posterior.set_default_x(Xtorch)
        posterior.vi_method = 'rKL'
        posterior.train()
        
        posteriors.append(posterior)
        
        
        
        if folder != None:
            samples = posterior.sample((10000,), x=Xtorch)            
            sampnum = samples.numpy()
            np.savetxt(folder + '/thetas'+method+'.csv', sampnum, delimiter = ',')
                        
        for r in range(num_rounds - 1):
            #simulate data
            print("Round: " + str(r+2))
            theta, x = simulate_for_sbi(generator, proposal, num_simulations=round_simulations, num_workers=numworkers)


            density_estimator = inference.append_simulations(theta, x).train(training_batch_size=batch_size)
            potential_fn, parameter_transform = ratio_estimator_based_potential(
                ratio_estimator = density_estimator, prior = prior, x_o = Xtorch)
            posterior = VIPosterior(potential_fn, prior, theta_transform=parameter_transform)
            posterior.set_default_x(Xtorch)
            posterior.vi_method = 'rKL'
            posterior.train()
            posteriors.append(posterior)
            
            meanvec = posteriors[-1].sample((1000,), x = Xtorch).mean(0)
            stdvec = posteriors[-1].sample((1000,), x = Xtorch).std(0)
            print("Parameter Means:")
            print(meanvec)
            print("Parameter Standard Deviations:")
            print(stdvec)
            
            #save data to disk
            if folder != None:
                samples = posterior.sample((10000,), x=Xtorch)
                sampnum = samples.numpy()
                np.savetxt(folder + '/thetas'+method+'.csv', sampnum, delimiter = ',')
        return sampnum, posteriors
        
    if method == 'SNPE':
        if boundmax != None:
            nnet = range_bound_maf_nn(embedding_net=net, lowerbound=boundmin,
                                      upperbound=boundmax)
            inference = SNPE(prior=prior, summary_writer=writer, device='cpu', density_estimator=nnet)
        else:
            inference = SNPE(prior=prior, summary_writer=writer, device='cpu', density_estimator='mdn')
        
        posteriors = []
        proposal = prior
        
        for r in range(num_rounds):
            print("Round: " + str(r+1))
            
            if r == 0:
                sims = init_simulations
            else:
                sims = round_simulations
            theta1, x1 = simulate_for_sbi(generator, proposal, num_simulations=sims, num_workers=numworkers)
            
            density_estimator = inference.append_simulations(theta1, x1, proposal=proposal).train(training_batch_size=batch_size)

            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            
            proposal = posterior.set_default_x(Xtorch)
            
            meanvec = posteriors[-1].sample((1000,), x = Xtorch).mean(0)
            stdvec = posteriors[-1].sample((1000,), x = Xtorch).std(0)
            print("Parameter Means:")
            print(meanvec)
            print("Parameter Standard Deviations:")
            print(stdvec)
            
            if folder != None:
                samples = posterior.sample((10000,), x=Xtorch)
                sampnum = samples.numpy()
                np.savetxt(folder + '/thetas'+method+'.csv', sampnum, delimiter = ',')


        return sampnum, posteriors

    
    
    
