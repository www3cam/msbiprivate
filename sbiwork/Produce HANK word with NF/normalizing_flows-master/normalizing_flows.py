#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from generate_points import create_uniform_points, create_points, visualize_data
from time import time

tfd = tfp.distributions
tfb = tfp.bijectors


tf.random.set_seed(0)
np.random.seed(seed=0)

settings = {
    'batch_size': 2000,
    'method': 'MAF',
    'num_bijectors': 12,
    'learning_rate': 3e-4,#2.5e-7,
    'train_iters': 4e5, #4e5,
    'visualize_data': False,
}


class Flow(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Flow, self).__init__(**kwargs)
        flow = None

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    @tf.function
    def train_step(self, X, optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class MAF(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        self.bijector_fns = []

        bijectors = []
        for i in range(settings['num_bijectors']):
            self.bijector_fns.append(tfb.masked_autoregressive_default_template(hidden_layers=[1024, 1024, 1024]))
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=self.bijector_fns[-1]
                )
            )

            # if i%2 == 0:
            #     bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag = [.4,.75]),
            bijector=bijector)
        
class MAF3(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        self.bijector_fns = []

        bijectors = []
        for i in range(settings['num_bijectors']):
            self.bijector_fns.append(tfb.masked_autoregressive_default_template(hidden_layers=[1024, 1024]))
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=self.bijector_fns[-1]
                )
            )

            # if i%2 == 0:
            #     bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=[2, 0, 1]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0, 0.0], scale_diag = [.25,.5, 1.0]),
            bijector=bijector)


class RealNVP(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(RealNVP, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        self.bijector_fns = []
        self.bijector_fn = tfp.bijectors.real_nvp_default_template(hidden_layers=[512, 512])

        bijectors = []
        for i in range(settings['num_bijectors']):
            # Note: Must store the bijectors separately, otherwise only a single set of tf variables is created for all layers
            self.bijector_fns.append(tfp.bijectors.real_nvp_default_template(hidden_layers=[512, 512]))
            bijectors.append(
                tfb.RealNVP(num_masked=self.num_masked,
                            shift_and_log_scale_fn=self.bijector_fns[-1])
            )

            if i % 3 == 0:
                bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        # bijector = tfb.Chain(bijectors[:-1])

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag = [.25, .5]),
            bijector=bijector)


def plot_layers(dist, final=False):
    """
    Generate samples from the base distribution and visualize the motion of the points after each 
    layer transformation
    """
    x = dist.distribution.sample(8000)
    samples = [x]
    names = [dist.distribution.name]
    for bijector in reversed(dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x[:,:2])
        names.append(bijector.name)

    results = samples

    X0 = results[0].numpy()

    rows = 4
    cols = int(len(results) / rows) + (len(results) % rows > 0)

    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    i = 0
    # for i in range(len(results)):
    for r in range(rows):
        for c in range(cols):
            if i >= len(results):
                break
            X1 = results[i].numpy()
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[r, c].scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
            arr[r, c].set_xlim([-5, 5])
            arr[r, c].set_ylim([-5, 5])
            arr[r, c].set_title(names[i])
            arr[r, c].axis('equal')
            i += 1
    plt.show()

    if not final:
        return

    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    plt.scatter(X1[idx, 0], X1[idx, 1], s=5, color='black')
    plt.axis('equal')
    plt.show()


def train(model, optimizer, print_period=100):#ds here
    """
    Train `model` on dataset `ds` using optimizer `optimizer`, 
      prining the current loss every `print_period` iterations
    """
    start = time()
    bloss = 2.5
    count = 0

    # for i in range(int(2e5 + 1)):
    for i in range(int(settings['train_iters'] + 1)):
        ds, _= create_dataset(settings['batch_size'])
        itr = ds.__iter__()
        X = next(itr)
        loss = model.train_step(X, optimizer).numpy()
        count = count + 1
        # plot_layers(model.flow, final=True)
        if loss < bloss:
            model.save_weights('./checkpoints/my_checkpoint2')
            bloss = loss
            count = 0
        if count == 5000 and optimizer.learning_rate > 4e-7:
            optimizer.learning_rate = optimizer.learning_rate/10
            count = 0
#        if loss < 1.5:
#            plot_layers(model.flow, final=True)
             #optimizer.learning_rate = 1e-10
        if i % print_period == 0:
            print(optimizer.learning_rate)
            print("{} loss: {}, {}s".format(i, loss, time() - start))
            # 
            if np.isnan(loss):
                break
        if i % 1000 == 0:
            plot_layers(model.flow, final=True)
    return loss


def print_settings():
    """
    display the settings used when creating the model
    """
    print("Using settings:")
    for k in settings.keys():
        print('{}: {}'.format(k, settings[k]))


def build_model(model):
    """
    Run a pass of the model to initialize the tensorflow network
    """
    x = model.flow.distribution.sample(8000)
    for bijector in reversed(model.flow.bijector.bijectors):
        x = bijector.forward(x)


def create_dataset(samps):
    # pts = create_uniform_points(1000)
    # pts = create_points('two_moons.png', 10000)
    pts = create_points('HANK2.png', samps)#'HANK2.png'

    if settings['visualize_data']:
        visualize_data(pts)

    ds = tf.data.Dataset.from_tensor_slices(pts)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.prefetch(3 * settings['batch_size'])
    ds = ds.batch(settings['batch_size'])

    return ds, pts

#def create_ds(samps):
#    # pts = create_uniform_points(1000)
#    # pts = create_points('two_moons.png', 10000)
#
#    ds = tf.data.Dataset.from_tensor_slices(pts)
#    ds = ds.repeat()
#    ds = ds.shuffle(buffer_size=10000)
#    ds = ds.prefetch(3 * settings['batch_size'])
#    ds = ds.batch(settings['batch_size'])
#
#    return ds


def train_and_run_model(display=True):
    print_settings()

    ds, pts = create_dataset(10000)
    
    if settings['method'] == 'MAF':
        model = MAF(output_dim=2, num_masked=1)
    elif settings['method'] == 'NVP':
        model = RealNVP(output_dim=2, num_masked=1)

    model(pts)
    build_model(model)
    if display:
        model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learning_rate'])
    loss = train(model, optimizer)


    if display:
        XF = model.flow.sample(2000)
        plot_layers(model.flow, final=True)

    return loss


def run_statistics_trial():
    """
    Runs 10 trials and reports the number of times training fails
    """
    final_loss = []
    for i in range(10):
        print()
        final_loss.append(train_model(display=False))
        print("Final loss for trial {} is {}".format(i, final_loss[-1]))

    print("Training failed {} of the time".format(np.sum(np.isnan(final_loss)) * 1.0 / len(final_loss)))


if __name__ == "__main__":
    train_and_run_model()
    # run_statistics_trial()
