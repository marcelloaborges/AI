import os, gc, threading, logging, time
import tensorflow as tf
import cv2 as cv
import numpy as np
from functools import partial
import json
from random import randint, random
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal

# ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

#
from sess import TfSess
from manager import tf_global_initializer, tf_load, tf_save

# ann
from ann import conv2d, fully, flatten, quantiles_layer, fraction_proposal_layer,\
                calc_fraction_loss, calculate_huber_loss, soft_update, dropout, dconv2d,\
                nfully, se

#
from rl_train_memory import ReplayMemory

import gc

def stack_images( features ):
    
    grid = int( math.sqrt( features.shape.as_list()[-1] ) )
    index = 0
    cl = []
    for l in range( grid ):
        cc = []
        for c in range( grid ):                    
            cc.append( features[:,:,:,index] )
            index += 1
        cl.append( tf.concat( cc, axis = 1 ) )
    im = tf.concat( cl, axis = 2 )

    return im[:,:,:,tf.newaxis]

def _reparameterize(mu, logvar, SAMPLES, random=True):
        
    samples_z = []
    std = 0.5 * tf.exp( logvar )
    for _ in range(SAMPLES):
        
        if random:
            eps = tf.compat.v1.random_normal( shape = tf.shape( std ), mean = 0, stddev = 1, dtype = tf.float32 )
            z = mu + ( eps * std )
        else:
            z = mu + std

        samples_z.append( z )  
    return samples_z

def base_encoder(c, is_training, summary=False):

    with tf.compat.v1.variable_scope( 'encoder', reuse = tf.compat.v1.AUTO_REUSE ):

        c1 = conv2d( c, 5, 2, 32, "c1", is_training = is_training )
        # c1 = se( c1, 'se1', ratio = 2 )
        c1 = dropout( c1, 0.25, 'dp1' )

        c2 = conv2d( c1, 5, 2, 64, "c2", is_training = is_training )
        # c2 = se( c2, 'se2', ratio = 4 )
        c2 = dropout( c2, 0.25, 'dp2' )

        c3 = conv2d( c2, 3, 2, 64, "c3", is_training = is_training )
        # c3 = se( c3, 'se3', ratio = 4 )
        c3 = dropout( c3, 0.15, 'dp3' )

        c4 = conv2d( c3, 3, 2, 128, "c4", is_training = is_training )
        # c4 = se( c4, 'se4', ratio = 4 )
        c4 = dropout( c4, 0.1, 'dp4' )

        c5 = conv2d( c4, 3, 2, 128, "c5", is_training = is_training )
        # c5 = se( c5, 'se5', ratio = 4 )
        c5 = dropout( c5, 0.1, 'dp5' )

        features = flatten( c5 )
    
    if summary:

        tf.compat.v1.summary.image( family = 'encoder', name = 'c1', tensor = stack_images( c1 ), max_outputs = 1 )
        tf.compat.v1.summary.image( family = 'encoder', name = 'c2', tensor = stack_images( c2 ), max_outputs = 1 )
        tf.compat.v1.summary.image( family = 'encoder', name = 'c3', tensor = stack_images( c3 ), max_outputs = 1 )
        tf.compat.v1.summary.image( family = 'encoder', name = 'c4', tensor = stack_images( c4 ), max_outputs = 1 )
        tf.compat.v1.summary.image( family = 'encoder', name = 'c5', tensor = stack_images( c4 ), max_outputs = 1 )

        tf.compat.v1.summary.histogram( name = 'c1', values = c1, family = 'encoder' )
        tf.compat.v1.summary.histogram( name = 'c2', values = c2, family = 'encoder' )
        tf.compat.v1.summary.histogram( name = 'c3', values = c3, family = 'encoder' )
        tf.compat.v1.summary.histogram( name = 'c4', values = c4, family = 'encoder' )
        tf.compat.v1.summary.histogram( name = 'c5', values = c5, family = 'encoder' )
            
    return features, tf.shape(c5)

def base_lattent_space(features, is_training, summary=False):

    with tf.compat.v1.variable_scope( 'distribution', reuse = tf.compat.v1.AUTO_REUSE ):
        
        logvar = fully( features, features.shape[-1], "logvar_lin", is_training = is_training )
        mu = fully( features, features.shape[-1], "mu_lin", is_training = is_training )
    
    if summary:

        tf.compat.v1.summary.histogram( name = 'mu', values = mu, family = 'encoder' )
        tf.compat.v1.summary.histogram( name = 'logvar', values = logvar, family = 'encoder' )

    return mu, logvar

def base_decoder(z, name, output, is_training):

    with tf.compat.v1.variable_scope( 'decoder', reuse = tf.compat.v1.AUTO_REUSE ):
        
        d1 = dconv2d( z, 3, 2, 64, "c6", is_training = is_training )
        d2 = dconv2d( d1, 3, 2, 32, "c5", is_training = is_training )
        d3 = dconv2d( d2, 3, 2, 32, "c4", is_training = is_training )
        d4 = dconv2d( d3, 3, 2, 32, "c3", is_training = is_training )
        d5 = dconv2d( d4, 5, 2, 32, "c2", is_training = is_training )

    with tf.compat.v1.variable_scope( name, reuse = tf.compat.v1.AUTO_REUSE ):
        
        co = conv2d( d5, 5, 1, 16, "co1", is_training = is_training )
        co = conv2d( co, 3, 1, 8, "co2", is_training = is_training )
        #co = conv2d( co, 2, 1, 8, "co3", is_training = is_training )
        #co = conv2d( co, 1, 1, 4, "co4", is_training = is_training )
        outp = conv2d( co, 1, 1, output, "out", is_training = is_training, relu = False )

    return outp

def encode_image(x, loss=False, summary=False):

    features, shape = base_encoder( x, True, summary = True )
    mu, logvar = base_lattent_space( features, True, True )

    r = _reparameterize( mu, logvar, 1, True )

    dec = base_decoder( tf.reshape( r[0], shape ), 'dec', 3, True )

    if summary:
        tf.compat.v1.summary.image( family = 'decoder', name = 'out', tensor = dec, max_outputs = 1 )

    if not loss: return mu

    _, var = tf.nn.moments( x, axes = [ 0,1,2,3 ] )
    rec_loss = tf.reduce_mean( ( dec - x ) ** 2  ) / var # Normalized MSE

    lat_loss = -0.5 * tf.reduce_mean( 1.0 + logvar - tf.pow( mu, 2 ) - tf.exp( logvar ), axis = 1 )
    lat_loss /= tf.cast( logvar.shape[-1], tf.float32 )
    lat_loss = tf.reduce_mean( lat_loss )

    tf.compat.v1.summary.scalar( 'rec_loss', rec_loss, family = 'base_loss' )
    tf.compat.v1.summary.scalar( 'lat_loss', lat_loss, family = 'base_loss' )
    
    total_loss = rec_loss + lat_loss

    return mu, total_loss

def gather( values, indexs ):

    one_hot = tf.one_hot( tf.tile( indexs[:,tf.newaxis], [1, values.shape[1]] ), values.shape[-1], dtype = tf.float32 )
    val = tf.reduce_sum( one_hot * values, axis = -1 )
    return val

def train_model(features,  a, r, s_, d, params):

    actions, rewards, next_states, dones = ( a, r, s_, d )

    embedding, vloss = encode_image( features, True, True )
    next_state_embedding_loc = encode_image( next_states )

    taus, taus_, entropy = fraction_proposal_layer( tf.stop_gradient( embedding ), fully, 'fpn', params['N'] )

    # Get expected Q values from local model
    F_Z_expected = quantiles_layer( 'local_quantiles', taus_, embedding, params['ac'], fully, N = params['N'] )
    Q_expected = gather( F_Z_expected, actions )
    
    # calc fractional loss 
    F_Z_tau = quantiles_layer( 'local_quantiles', taus[:, 1:-1], tf.stop_gradient( embedding ), params['ac'], fully, N = params['N'] )
    FZ_tau = tf.stop_gradient( gather( F_Z_tau, actions ) )
    
    with tf.compat.v1.variable_scope( 'fractional_loss' ):
        
        frac_loss = calc_fraction_loss( tf.stop_gradient( Q_expected ), FZ_tau, taus )
        entropy_loss = params['entropy_coeff'] * tf.reduce_mean( entropy ) 
        frac_loss += entropy_loss

    n_taus, n_taus_, _ = fraction_proposal_layer( next_state_embedding_loc, fully, 'fpn', params['N'] )
    F_Z_next = quantiles_layer( 'local_quantiles', n_taus_, next_state_embedding_loc, params['ac'], fully, N = params['N'] )
    
    Q_targets_next = tf.reduce_sum( ( ( n_taus[:, 1:, tf.newaxis] - n_taus[:, :-1, tf.newaxis] ) ) * F_Z_next, axis = 1 )
    action_indx = tf.argmax( Q_targets_next, axis = 1 )

    F_Z_next = quantiles_layer( 'target_quantiles', taus_, next_state_embedding_loc, params['ac'], fully, N = params['N'] )
    Q_targets_next = gather( F_Z_next, action_indx )
    Q_targets = tf.cast( rewards[:,tf.newaxis], tf.float32 ) + ( params['GAMMA'] ** params['n_step'] * Q_targets_next * tf.cast( 1 - dones[:,tf.newaxis], tf.float32 ) )
        
    # Quantile Huber loss
    td_error = Q_targets - Q_expected
    huber_l = calculate_huber_loss( td_error, 1.0 )
    quantil_l = abs( taus_ - tf.cast( tf.stop_gradient( td_error ) < 0, tf.float32 ) ) * huber_l / 1.0

    loss = tf.reduce_mean( tf.reduce_sum( quantil_l, axis = 1 ) )
    # loss = tf.reduce_mean( td_error )

    frac_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'fpn' in vr.name ]
    enc_vision_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'encoder' in vr.name ]
    dec_vision_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'decoder' in vr.name ]
    dst_vision_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'distribution' in vr.name ]
    quantile_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'local_quantiles' in vr.name ]
    quantile_vars_t = [ vr for vr in tf.compat.v1.trainable_variables() if 'target_quantiles' in vr.name ]
    global_step = tf.compat.v1.get_variable( 'global_step', [], initializer = tf.constant_initializer(0), trainable = False )

    optimizer_v = tf.compat.v1.train.AdamOptimizer( learning_rate = 2e-4 )
    optimizer_v = tf.train.experimental.enable_mixed_precision_graph_rewrite( optimizer_v, loss_scale='dynamic' )
    grads_and_vars_v = optimizer_v.compute_gradients( vloss, enc_vision_vars + dst_vision_vars + dec_vision_vars )
    train_op_v = optimizer_v.apply_gradients( grads_and_vars_v )

    optimizer_f = tf.compat.v1.train.RMSPropOptimizer( learning_rate = 2e-3 * 0.000001 )
    optimizer_f = tf.train.experimental.enable_mixed_precision_graph_rewrite( optimizer_f, loss_scale='dynamic' )
    grads_and_vars_f = optimizer_f.compute_gradients( frac_loss, frac_vars )
    train_op_f = optimizer_f.apply_gradients( grads_and_vars_f )

    optimizer_p = tf.compat.v1.train.RMSPropOptimizer( learning_rate = 2e-3 )
    optimizer_p = tf.train.experimental.enable_mixed_precision_graph_rewrite( optimizer_p, loss_scale='dynamic' )
    grads_and_vars_p = optimizer_p.compute_gradients( loss, enc_vision_vars + [ vr for vr in dst_vision_vars if 'mu' in vr.name ] + quantile_vars )
    grads_and_vars_p_ = []
    for vl in grads_and_vars_p:
        if 'encode' in vl[1].name or 'decode' in vl[1].name:
            grads_and_vars_p_.append( ( vl[0] * 1e-15, vl[1] ) )
        else:
            grads_and_vars_p_.append( ( vl[0], vl[1] ) )
    train_op_p = optimizer_p.apply_gradients( grads_and_vars_p_ )

    soft = soft_update( quantile_vars, quantile_vars_t, params['TAU'] )
    
    update_global_step = tf.compat.v1.assign( global_step, global_step + 1, name = 'update_global_step' )
    
    train_op = tf.group( [ train_op_v, train_op_p, train_op_f, soft, update_global_step ] )

    tf.compat.v1.summary.scalar( 'reward', tf.reduce_mean( rewards ), family = 'rd' )
    tf.compat.v1.summary.scalar( 'reward_p', tf.reduce_mean( tf.reduce_sum( Q_expected, axis = -1 ) ), family = 'rd' )
    tf.compat.v1.summary.scalar( 'reward_t', tf.reduce_mean( tf.reduce_sum( Q_targets, axis = -1 ) ), family = 'rd' )

    tf.compat.v1.summary.scalar( 's_reward', tf.reduce_sum( rewards ), family = 'rd' )
    tf.compat.v1.summary.scalar( 's_reward_p', tf.reduce_sum( tf.reduce_sum( Q_expected, axis = -1 ) ), family = 'rd' )
    tf.compat.v1.summary.scalar( 's_reward_t', tf.reduce_sum( tf.reduce_sum( Q_targets, axis = -1 ) ), family = 'rd' )

    tf.compat.v1.summary.scalar( 'f_loss', frac_loss, family = 'base_loss' )
    tf.compat.v1.summary.scalar( 'qloss', loss, family = 'base_loss' )

    return train_op, global_step

def eval_model(features,  params):
    
    embedding = encode_image( features )

    taus, taus_, _ =  fraction_proposal_layer( embedding, fully, 'fpn', params['N'] )
    F_Z = quantiles_layer( 'local_quantiles', taus_, embedding, params['ac'], fully, N = params['N'] )
    action_values = tf.reduce_sum( ( taus[:, 1:][...,tf.newaxis] - taus[:, :-1][...,tf.newaxis]) * F_Z, axis = 1 )            
    pred = tf.cast( tf.argmax( action_values, axis = 1 )[0], tf.int32 )

    action_r = tf.range( params['ac'] )
    p = tf.random.uniform( [params['ac']] )
    samples = tf.multinomial( tf.math.log( [ p ] ), 1) # note log-prob
    pred_r = action_r[ tf.cast( samples[0][0], tf.int32 ) ]

    explore = tf.random.uniform( [1] ) > params['eps']
    return tf.where( explore, [pred], [pred_r] )[-1], ( ( taus[:, 1:][...,tf.newaxis] - taus[:, :-1][...,tf.newaxis]) * F_Z )[0]

def render(env, q, fig, ax):

    axs.clear()
    for ix in range( q.shape[-1] ):
        axs.bar( np.arange(len(COMPLEX_MOVEMENT)), q[:,ix] )

    fig.canvas.draw_idle()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv.resize( cv.cvtColor(img,cv.COLOR_RGB2BGR), (500,500) )
    env = cv.resize( env, (500,500) )
    img = np.hstack( ( env[:,:,::-1], img ) )
   
    # display image with opencv or any operation you like
    cv.imshow("game",img)
    cv.waitKey(1)

# input
size = ( 128, 128 )

# train
bs = 128
buffer_memory = 10000
t_steps = 1000
n_episodes = 100000

# vars
env = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0'), COMPLEX_MOVEMENT )
rl_mem = ReplayMemory( buffer_memory, bs )

# params
with open('test3.json', "r") as f:
    params = json.load(f)
    params['ac'] = env.action_space.n
    params['BATCH_SIZE'] = bs

# session
config = tf.ConfigProto( log_device_placement = False ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = TfSess( "mario_simple_vae", gpu = True, config = config, folder='logs/' )

## model
s = tf.compat.v1.placeholder( tf.float32, [ None ] + list(size)[::-1] + [ 3 ], name = "state"  )
s_ = tf.compat.v1.placeholder( tf.float32, [ None ] + list(size)[::-1] + [ 3 ], name = "state_" )
a = tf.compat.v1.placeholder( tf.int32, [None,], name = "a" )
r = tf.compat.v1.placeholder( tf.float32, [None,], name = "r" )
d = tf.compat.v1.placeholder( tf.int32, [None,], name = "d" )

with tf.compat.v1.variable_scope( 'mario_simple_vae', reuse = tf.AUTO_REUSE ):
    model_predict = eval_model( s, params )
    model_train, g_step = train_model( s, a, r, s_, d, params )

tf_global_initializer( sess )

sess.merge_summary()

variables = tf.trainable_variables()
# tf_load( 'saved/simple/', variables, 'mario', sess, True )

sess.tensorboard_graph()

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

t_step = 0
max_stoped = 150
dists = list(np.arange(max_stoped))
state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 127.0
for episode in range(n_episodes):

    total_reward = 0
    max_x = 0
    for t in range(t_steps):

        action, q = sess( model_predict, { s: [ state ] } )
        ns, reward, done, info = env.step( action )
        next_state = cv.resize( ns, ( size[0], size[1] ) ) / 127.0
        total_reward += reward

        render( ns, q, fig, axs )

        if max_x < info['x_pos']:
            max_x = info['x_pos']
        else:
            if max_x - info['x_pos'] > 50:
                reward -= 5
                done = True
        
        if np.std( dists ) <= 3:
            reward -= 15
            dists = list(np.arange(max_stoped))
            done = True
            vls = np.repeat( 5, max_stoped )
            vls = signal.lfilter([1], [1, -0.5], vls, axis=0) * -1
            for il, vl in zip( range(max_stoped), range( len( rl_mem.memory ) - max_stoped, len( rl_mem.memory ) -1 )):
                rl_mem.memory[vl][2] += vls[il]

        rl_mem.add( state, action, reward, next_state, done )
        
        dists.append( info['x_pos'] )
        dists.pop(0)
                                    
        if done:
            state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 127.0
            print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (total_reward, info['x_pos']))            
            if 'distance' in info: print('Mario Distance Covered:', info['distance'])
            # if len(rl_mem) > params['BATCH_SIZE']:
            #     for x in range( 10 ):
            #         experiences = rl_mem.sample()
            #         if not params['per']:
            #             step = int( sess( g_step ) )
            #             sess( model_train, { s: experiences[0], a: experiences[1], r: experiences[2], s_: experiences[3], d: experiences[4] }, True, step )
            #         else:
            #             # loss, entropy = self.learn_per(experiences)
            #             pass
            #     print( step )
            gc.collect()
            break

        state = next_state
    tf_save( 'saved/simple/', variables, 'mario', sess, True )

env.close()