# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

    
def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    #int(np.log2(G.output_shapes[-1]))
    total_latent_size = G.input_shapes[0][1:][0]
    c_size = 1 + 10 + 1
    random_latent_size = total_latent_size - c_size
    c_3_ind = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
#     c_3 = tf.one_hot(c_3_ind, 2)
    c_4_ind = tf.random_uniform([minibatch_size], 0, 10, dtype = tf.int32)
    c_4 = tf.one_hot(c_4_ind, 10)
    c_5_ind = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
#     c_5 = tf.one_hot(c_5_ind, 2)
    
    test = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
    c_3 = tf.reshape(c_3_ind, [minibatch_size, 1])
#     c_4 = c_4_ind
    c_5 = tf.reshape(c_5_ind, [minibatch_size, 1])
    
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out, qf3, qf4, qf5, lod_in = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    
    loss3 = tf.losses.mean_squared_error(c_3, qf3)
    #loss3 = tf.Print(loss3, [loss3], message="loss3")
    loss4 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_4, logits=qf4)
    #loss4 = tf.Print(loss4, [tf.reduce_mean(loss4)], message="loss4")
    loss5 = tf.losses.mean_squared_error(c_5, qf5)
    #loss5 = tf.Print(loss5, [loss5], message="loss5")
    #loss = tf.Print(loss, [tf.reduce_mean(loss)], message="loss")
    
    
    #print(loss.shape, loss3.shape, loss4.shape, loss5.shape, type(loss), type(loss3))
    loss = loss + tf.clip_by_value((2 - lod_in), 0.0, 1.0)*(2*loss3 + 0.2*loss4 + 2*loss5)
    #loss = loss + loss3 + loss4 + loss5
    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
    loss = tfutil.autosummary('Loss/GInfoLoss3', loss3)
    loss = tfutil.autosummary('Loss/GInfoLoss4', loss4)
    loss = tfutil.autosummary('Loss/GInfoLoss5', loss5)    
    loss = tfutil.autosummary('Loss/GFinalLoss', loss)
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.
    
    total_latent_size = G.input_shapes[0][1:][0]
    c_size = 1 + 10 + 1
    random_latent_size = total_latent_size - c_size
    c_3_ind = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
#     c_3 = tf.one_hot(c_3_ind, 2)
    c_4_ind = tf.random_uniform([minibatch_size], 0, 10, dtype = tf.int32)
    c_4 = tf.one_hot(c_4_ind, 10)
    c_5_ind = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
#     c_5 = tf.one_hot(c_5_ind, 2)
    
    test = tf.random_uniform([minibatch_size], 0, 1, dtype = tf.float32)
    c_3 = tf.reshape(c_3_ind, [minibatch_size, 1])
#     c_4 = c_4_ind
    c_5 = tf.reshape(c_5_ind, [minibatch_size, 1])
    
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    
    #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out, qr3, qr4, qr5, lod_in = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out, qf3, qf4, qf5, lod_in = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out, q3, q4, q5, lod_in = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    loss = tfutil.autosummary('Loss/DFinalLoss', loss)    
    return loss

#----------------------------------------------------------------------------
