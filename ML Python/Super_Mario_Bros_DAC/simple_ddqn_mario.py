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

# ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

#
from sess import TfSess
from manager import tf_global_initializer, tf_load, tf_save

# ann
from ann import conv2d, fully, flatten, quantiles_layer, fraction_proposal_layer, calc_fraction_loss, calculate_huber_loss, soft_update

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

def encode_image(x):

    with tf.compat.v1.variable_scope( 'encoder', reuse = tf.compat.v1.AUTO_REUSE ):

        x1 = tf.nn.relu( conv2d( x, 5, 2, 32, 'c1' ) )
        x2 = tf.nn.relu( conv2d( x1, 3, 2, 32, 'c2' ) )
        x3 = tf.nn.relu( conv2d( x2, 3, 2, 32, 'c3' ) )
        x4 = tf.nn.relu( conv2d( x3, 2, 2, 32, 'c4' ) )

    tf.compat.v1.summary.image( family = 'encoder', name = 'c1', tensor = stack_images( x1 ) )
    tf.compat.v1.summary.image( family = 'encoder', name = 'c2', tensor = stack_images( x2 ) )
    tf.compat.v1.summary.image( family = 'encoder', name = 'c3', tensor = stack_images( x3 ) )
    tf.compat.v1.summary.image( family = 'encoder', name = 'c4', tensor = stack_images( x4 ) )
    
    tf.compat.v1.summary.histogram( name = 'c1', values = x1, family = 'encoder' )
    tf.compat.v1.summary.histogram( name = 'c2', values = x2, family = 'encoder' )
    tf.compat.v1.summary.histogram( name = 'c3', values = x3, family = 'encoder' )
    tf.compat.v1.summary.histogram( name = 'c4', values = x4, family = 'encoder' )

    return x4

def gather( values, indexs ):

    one_hot = tf.one_hot( tf.tile( indexs[:,tf.newaxis], [1, values.shape[1]] ), values.shape[-1], dtype = tf.float32 )
    val = tf.reduce_sum( one_hot * values, axis = -1 )
    return val

def train_model(features,  a, r, s_, d, params):

    actions, rewards, next_states, dones = ( a, r, s_, d )

    embedding = flatten( encode_image( features ) )

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

    if not params['munchausen']:

        # embeding
        next_state_embedding_loc = flatten( encode_image( next_states ) )

        n_taus, n_taus_, _ = fraction_proposal_layer( next_state_embedding_loc, fully, 'fpn', params['N'] )
        F_Z_next = quantiles_layer( 'local_quantiles', n_taus_, next_state_embedding_loc, params['ac'], fully, N = params['N'] )
        
        Q_targets_next = tf.reduce_sum( ( ( n_taus[:, 1:, tf.newaxis] - n_taus[:, :-1, tf.newaxis] ) ) * F_Z_next, axis = 1 )
        action_indx = tf.argmax( Q_targets_next, axis = 1 )

        F_Z_next = quantiles_layer( 'target_quantiles', taus_, next_state_embedding_loc, params['ac'], fully, N = params['N'] )
        Q_targets_next = gather( F_Z_next, action_indx )
        Q_targets = tf.cast( rewards[:,tf.newaxis], tf.float32 ) + ( params['GAMMA'] ** params['n_step'] * Q_targets_next * tf.cast( 1 - dones[:,tf.newaxis], tf.float32 ) )
        
    else:
        pass

    # Quantile Huber loss
    td_error = Q_targets - Q_expected    
    huber_l = calculate_huber_loss( td_error, 1.0 )
    quantil_l = abs( taus_ - tf.cast( tf.stop_gradient( td_error ) < 0, tf.float32 ) ) * huber_l / 1.0

    loss = tf.reduce_mean( tf.reduce_sum( quantil_l, axis = 1 ) )
    # loss = tf.reduce_mean( td_error )

    frac_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'fpn' in vr.name ]
    vision_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'encoder' in vr.name ]
    quantile_vars = [ vr for vr in tf.compat.v1.trainable_variables() if 'local_quantiles' in vr.name ]
    quantile_vars_t = [ vr for vr in tf.compat.v1.trainable_variables() if 'target_quantiles' in vr.name ]
    global_step = tf.compat.v1.get_variable( 'global_step', [], initializer = tf.constant_initializer(0), trainable = False )

    optimizer_f = tf.compat.v1.train.RMSPropOptimizer( learning_rate = params['lr'] * 0.000001 )
    optimizer_f = tf.train.experimental.enable_mixed_precision_graph_rewrite( optimizer_f, loss_scale='dynamic' )
    grads_and_vars_f = optimizer_f.compute_gradients( frac_loss, frac_vars )
    train_op_f = optimizer_f.apply_gradients( grads_and_vars_f )

    optimizer_p = tf.compat.v1.train.RMSPropOptimizer( learning_rate = params['lr'] )
    optimizer_p = tf.train.experimental.enable_mixed_precision_graph_rewrite( optimizer_p, loss_scale='dynamic' )
    grads_and_vars_p = optimizer_p.compute_gradients( loss, vision_vars + quantile_vars )
    train_op_p = optimizer_p.apply_gradients( grads_and_vars_p )

    soft = soft_update( quantile_vars, quantile_vars_t, params['TAU'] )
    
    update_global_step = tf.compat.v1.assign( global_step, global_step + 1, name = 'update_global_step' )
    
    train_op = tf.group( [ train_op_p, train_op_f, soft, update_global_step ] )

    tf.compat.v1.summary.scalar( 'reward', tf.reduce_mean( r ), family = 'base_loss' )
    tf.compat.v1.summary.scalar( 'f_loss', frac_loss, family = 'base_loss' )
    tf.compat.v1.summary.scalar( 'qloss', loss, family = 'base_loss' )

    return train_op, global_step

def eval_model(features,  params):

    global_step = tf.compat.v1.get_variable( 'global_step', [], initializer = tf.constant_initializer(0), trainable = False )
    
    embedding = flatten( encode_image( features ) )
    taus, taus_, _ =  fraction_proposal_layer( embedding, fully, 'fpn', params['N'] )
    F_Z = quantiles_layer( 'local_quantiles', taus_, embedding, params['ac'], fully, N = params['N'] )
    action_values = tf.reduce_sum( ( taus[:, 1:][...,tf.newaxis] - taus[:, :-1][...,tf.newaxis]) * F_Z, axis = 1 )            
    pred = tf.cast( tf.argmax( action_values, axis = 1 )[0], tf.int32 )

    action_r = tf.range( params['ac'] )
    p = tf.random.uniform( [params['ac']] )
    samples = tf.multinomial( tf.math.log( [ p ] ), 1) # note log-prob
    pred_r = action_r[ tf.cast( samples[0][0], tf.int32 ) ]

    explore = tf.random.uniform( [1] ) > ( params['eps'] * ( 1 - ( global_step / 10000000 ) ) )
    return tf.where( explore, [pred], [pred_r] )[-1], action_values

def render(env, q, fig, ax):

    axs.clear()
    axs.bar(np.arange(len(COMPLEX_MOVEMENT)), q[0,:], color='red')

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
bs = 512
buffer_memory = 50000
t_steps = 1000
n_episodes = 100000

# vars
env = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0'), COMPLEX_MOVEMENT )
rl_mem = ReplayMemory( buffer_memory, bs )

# params
with open('test1.json', "r") as f:
    params = json.load(f)
    params['ac'] = env.action_space.n
    params['BATCH_SIZE'] = bs

# session
config = tf.ConfigProto( log_device_placement = False ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = TfSess( "mario_simple", gpu = True, config = config, folder='logs/' )

## model
s = tf.compat.v1.placeholder( tf.float32, [ None ] + list(size)[::-1] + [ 3 ], name = "state"  )
s_ = tf.compat.v1.placeholder( tf.float32, [ None ] + list(size)[::-1] + [ 3 ], name = "state_" )
a = tf.compat.v1.placeholder( tf.int32, [None,], name = "a" )
r = tf.compat.v1.placeholder( tf.float32, [None,], name = "r" )
d = tf.compat.v1.placeholder( tf.int32, [None,], name = "d" )

with tf.compat.v1.variable_scope( 'simple_mario', reuse = tf.AUTO_REUSE ):
    model_predict = eval_model( s, params )
    model_train, g_step = train_model( s, a, r, s_, d, params )

tf_global_initializer( sess )

sess.merge_summary()

variables = tf.trainable_variables()
tf_load( 'saved/simple/', variables, 'mario', sess, True )

sess.tensorboard_graph()

fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

t_step = 0
dists = list(np.arange(200))
state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 127.0
for episode in range(n_episodes):

    total_reward = 0
    for t in range(t_steps):

        action, q = sess( model_predict, { s: [ state ] } )
        ns, reward, done, info = env.step( action )
        next_state = cv.resize( ns, ( size[0], size[1] ) ) / 127.0
        total_reward += reward

        # render( ns, q, fig, axs )
        
        if np.std( dists ) <= 1:
            reward = - 5
            dists = list(np.arange(200))
            done = True

        rl_mem.add( state, action, reward, next_state, done )

        dists.append( info['x_pos'] )
        dists.pop(0)
                                    
        if done:
            state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 127.0
            print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (total_reward, info['x_pos']))            
            if 'distance' in info: print('Mario Distance Covered:', info['distance'])
            if len(rl_mem) > params['BATCH_SIZE']:
                for x in range( 10 ):
                    experiences = rl_mem.sample_inverse_dist()
                    if not params['per']:
                        step = int( sess( g_step ) )
                        sess( model_train, { s: experiences[0], a: experiences[1], r: experiences[2], s_: experiences[3], d: experiences[4] }, True, step )
                    else:
                        # loss, entropy = self.learn_per(experiences)
                        pass
                print( step )
            gc.collect()
            break

        # if t % params['UPDATE_EVERY'] == 0:
        #     if len(rl_mem) > params['BATCH_SIZE']:
        #         for x in range( 10 ):
        #             experiences = rl_mem.sample_inverse_dist()
        #             if not params['per']:
        #                 step = int( sess( g_step ) )
        #                 sess( model_train, { s: experiences[0], a: experiences[1], r: experiences[2], s_: experiences[3], d: experiences[4] }, True, step )
        #             else:
        #                 # loss, entropy = self.learn_per(experiences)
        #                 pass
        #         print( step )
        
        state = next_state
    tf_save( 'saved/simple/', variables, 'mario', sess, True )

env.close()