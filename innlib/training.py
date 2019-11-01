"""
This module implements the basic functions for training deep learning models
for parameter estimation and model selection.
"""


__version__ = '0.1'
__author__ = 'Stefan Radev'

from collections.abc import Iterable

import tensorflow as tf
import numpy as np

from .utils import clip_gradients, apply_gradients


def train_online(model, optimizer, data_gen, loss_fun, iterations, batch_size, p_bar=None,
                 clip_value=None, clip_method='global_norm', global_step=None, n_smooth=100):
    """
    Performs a number of training iterations... #TODO 
    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_gen        : callable -- a function providing batches of data
    loss_fun        : callable -- a function computing the loss given model outputs
    iterations      : int -- the number of training loops to perform
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dict for storing losses
    losses = {
        'loss': [],
        'regularization': []
    }

    # Run training loop
    for it in range(1, iterations+1):

        with tf.GradientTape() as tape:
            # Generate inputs for the network
            inputs = data_gen(batch_size)
            # Forward pass (check if inputs is a tuple or a single value)
            if isinstance(inputs, Iterable):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            # Loss computation and backward pass
            if isinstance(outputs, Iterable):
                loss = loss_fun(*outputs)
            else:
                loss = loss_fun(outputs)
            # Compute loss + regularization, if any
            w_decay = tf.add_n(model.losses)
            total_loss = loss + w_decay

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['regularization'].append(w_decay.numpy())
        losses['loss'].append(loss.numpy())
        running_loss = loss.numpy() if it < n_smooth else np.mean(losses['loss'][-n_smooth:])

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Iteration: {0}, Loss: {1:.3f},Running Loss: {2:.3f},Regularization: {3:.3f}"
            .format(it, loss.numpy(), running_loss, w_decay.numpy()))
            p_bar.update(1)

    return losses


def train_offline(model, optimizer, dataset, loss_fun, batch_size, p_bar=None, clip_value=None, 
                  clip_method='global_norm', global_step=None):
    """
    Loops throuhg a dataset  #TODO 
    ----------

    Arguments:
    model           : tf.keras.Model -- a neural network model implementing a __call__() method
    optimizer       : tf.train.Optimizer -- the optimizer used for backprop
    data_generator  : callable -- a function providing batches of data
    loss_fun        : callable -- a function computing the loss given model outputs
    batch_size      : int -- the batch_size used for training
    ----------

    Keyword Arguments:
    p_bar           : ProgressBar or None -- an instance for tracking the training progress
    clip_value      : float       -- the value used for clipping the gradients
    clip_method     : str         -- the method used for clipping (default 'global_norm')
    global_step     : tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay  
    ----------

    Returns:
    losses : a dictionary with regularization and loss evaluations at each training iteration
    """
    
    # Prepare a dictionary to track losses
    losses = {
        'loss': [],
        'regularizarion': []
    }

    # Loop through dataset
    for bi, inputs in enumerate(dataset):

        with tf.GradientTape() as tape:
            # Forward pass (check if inputs is a tuple or a single value)
            if isinstance(inputs, Iterable):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)
            # Loss computation and backward pass
            if isinstance(outputs, Iterable):
                loss = loss_fun(*outputs)
            else:
                loss = loss_fun(outputs)
            # Compute loss + regularization, if any
            w_decay = tf.add_n(model.losses)
            total_loss = loss + w_decay

        # One step backprop
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if clip_value is not None:
            gradients = clip_gradients(gradients, clip_value, clip_method)
        apply_gradients(optimizer, gradients, model.trainable_variables, global_step)  

        # Store losses
        losses['regularization'].append(w_decay.numpy())
        losses['loss'].append(loss.numpy())

        # Update progress bar
        if p_bar is not None:
            p_bar.set_postfix_str("Batch: {0},Loss: {1:.3f},Regularization: {2:.3f}"
            .format(bi, loss.numpy(), w_decay.numpy()))
            p_bar.update(1)

    return losses