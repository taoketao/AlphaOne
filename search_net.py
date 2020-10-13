'''
search_net.py:
    The root 'script' that runs the machine learning training regimen.
    Training begins at line <<  for e in range(episodes):  >>
      at which the untrained network begins iterations of forward-props
      generating actions, reward responses from the environment, and
      backpropagation passes.

[[ please ignore the commented-out code! If I/you were to revisit this,
    housekeeping and deep cleaning would be in order.  ]] 

'''



# IPython log file


import gym
import keras
import baselines
import tensorflow as tf

from baselines import deepq
from keras.models import Model
import baselines.common.tf_util as U
from huber_loss import huber
#from state_saver import WithSnapshots as ws
from copy import deepcopy as dc
#from gym.envs.board_game.go import GoState
from custom_go import CustomGoState
from pachi_py import WHITE, BLACK
from random import random
import time
import numpy as np
import pickle

#env = gym.make('Go9x9-v0')
from gym.envs.registration import register
game_name = 'CustumGo9x9-v0'; # 'Go9x9-v0'
register(
        id=game_name,
        #entry_point='gym.envs.board_game:GoEnv',
        entry_point='custom_go:CustomGoEnv',
        kwargs={
            'player_color': 'black',
            'opponent': 'pachi:uct:_2400',
            'observation_type': 'image3c',
            'illegal_move_mode': 'lose',
            'board_size': 9,
        },
        nondeterministic=True,
    )
env = gym.make(game_name)
env.seed(1) # from gym/gym/envs/__init__.py: allows for reproducibility
env.reset()
env.render()

#from layers_tied.layers_tied import Convolution2D_tied as conv2d_tied
#model = keras.Sequential()
#
#with tf.
#input_var
#
#def pick_action(orig_state, curr_state, state_stack, threshold, depth_left):
#    pass
#


# IPython log file

from keras.layers import Conv2D, Flatten, Dense, Reshape, Input

cons_state_var = Input( [9,9,3] )
orig_state_var = Input( [9,9,3] )

original_state = Flatten()(Conv2D(64, (7,7))(orig_state_var))
considered_state = Flatten()(Conv2D(64, (7,7))(cons_state_var))
merge_layer = keras.layers.concatenate([original_state, considered_state])
representation_layer = keras.layers.Dense(64)(merge_layer)
action_outputs = keras.layers.Dense(83)(representation_layer)

confidence = keras.layers.Dense(1)(representation_layer)

_model = Model(inputs=[orig_state_var, cons_state_var], outputs=[action_outputs, confidence])

SENTINEL='SENTINEL - reset environment'

gamma = 0.95
episodes=500
max_actions=100
max_search_iterations=9
max_search_depth=3
max_buf_len = 1000
update_concentration = 16
lapl = laplace_like_smoothing_factor = 1.

def to_valid_moves(action_pref, state_obs, moves_log, decl):
    M = np.multiply(action_pref[0,:81], np.reshape(state_obs,(81,)))
    for i in moves_log: M[i]=np.min(M)
    for seq in decl:
        pass
    return M

def get_obs_GoS(_state):
    l = _state.board.get_stones(0) 
    a = np.zeros((9,9))
    for x,y in l: a[x,y]=1
    return a
def get_obs_GoE(_env): return get_obs_GoS(_env.state)

def get_inp_state(go_state):
    x = np.expand_dims(np.transpose(go_state.board.encode(), (2,1,0)), 0)
    return x
def copy_go_state(st): return CustomGoState(st.board.clone(), st.color)
class Exps(object):
    def __init__(self): 
        self.experiences = []

    def add(self, orig_state, action, rew, new_state, done):
        self.experiences.append( [orig_state, action, rew, new_state, done, lapl] )
        while len(self.experiences)>max_buf_len:
            self.experiences.pop( np.argmax([e[-1] for e in self.experiences])[0] )

    def get_any(self, n=1): 
        return np.random.sample(self.experiences,n)[0]
    def get_samp(self):
        ps = np.array([x[-1]**-2 for x in self.experiences])
        self.experiences[ np.argmax(ps)[0] ][-1] += 1.
        return np.random.choice(self.experiences, p=ps/sum(ps))[0]

Experiences = Exps()

for e in range(episodes):
#    env.reset()
    print('e',e)
    if e<300 or random.random()<0.5: 
        #orig_state = np.reshape(env.reset(), (1,9,9,3))
        orig_state = np.expand_dims(np.transpose(env.reset(), (2,1,0)), 0)
#        orig_GoState = env.state.clone()
#        orig_GoState = GoState(env.state.board.clone(), env.state.color)
        orig_GoState = copy_go_state(env.state)
    else: 
        orig_GoState = Experiences.get_samp()
    cur_GoState = copy_go_state(orig_GoState)
    #scratch_env = dc(env)

    for game_turn in range(max_actions):
        print('>'*80+'\n> gt',game_turn,'.  True moves:',env.moves_log)
        # On this turn, pick an action. Control flow: using the scratch env set
        # originally to be a copy of the true state, search up or down trees
        current_depth = 0
        running_champion_action = -1
        #state_stack = []; 
        declined_actions = set() # reset per exploration

        for d in range(max_search_iterations):
            time.sleep(0.4)
#                print(orig_state.shape, cur_state.shape)
#                if len(orig_state.shape)<4:orig_state=np.expand_dims(orig_state,0)
#                if len(cur_state.shape)<4:cur_state=np.expand_dims(cur_state,0)
            cur_state = get_inp_state(cur_GoState)
            q_actions, confidence = _model.predict([orig_state, cur_state])
            confidence=confidence[0][0]
            accept = True if confidence > 0 else False
            if current_depth==0: accept = True
            if current_depth==max_search_depth: accept=False

            print('>>> itr',d,'depth',current_depth,'conf',confidence)

            if not accept:
                if not current_depth==max_search_depth:
#                    #plyr_actions = [env.moves[i] for i in range(len(env.moves)) if i%2==0]
                    declined_actions.add( env.moves[-1] ) # add move that led to this situation
                current_depth -= 1
                print('Considering stack:',env.revert(game_turn),\
                        '; current best option:',running_champion_action)
#                cur_GoState = env.state
                #env.state=None
#                print('\n\nvvvvvvvvvvvvvvv')
#                env.render()
                #env.state = cur_GoState = pickle.loads(state_stack.pop(-1) )
#                env.render()
#                print('^^^^^^^^^^6\n\n')
            elif accept:
#                env.render()
                valid_moves = to_valid_moves(q_actions, get_obs_GoS(cur_GoState), \
                                env.moves_log, declined_options)
                tmp_cur = copy_go_state(env.state)
                actn = np.argmax(valid_moves)
                print('\t\tattempting',actn)
                env.state = cur_GoState # just in case...
                obs, rew, done, st_dict = env.step(actn)
                print(st_dict['state'])

#                while str(obs) in declined_options:  # str: hack
#                    env.state = tmp_cur
#                    print('>>>>  action',actn,'ruled out')
#                    valid_moves[actn]=np.min(valid_moves)
#                    actn = np.argmax(valid_moves)
#                    obs, rew, done, st = env.step(actn)
#                print(">>>>> Trying action ",actn, 'with stack:', env.moves_log)
#                if str(obs) in declined_options:  # str: hack
#                    #env.state=None
#                    #env.state = cur_GoState = pickle.loads(state_stack.pop(-1) )
#                    env.revert(game_turn)
#                    cur_GoState = env.state
#                    current_depth -= 1
#                    if random.random()<0.5:
#                        env.state = cur_GoState = state_stack.pop(-1) 
#                        current_depth -= 1
#                else:
#                    if current_depth==0: 
#                        running_champion_action = actn
#                    current_depth += 1
##                    state_stack.append( cur_GoState )
##                    env.state = cur_GoState # This line is test crucial
#                    #state_stack.append( pickle.dumps(env.state) )
#                    cur_state = obs
#                    cur_GoState = copy_go_state(env.state)
#        
        print('Acknowledged moves:', env.moves_log)
        env.state = orig_GoState
        env.revert(game_turn, step='true')
#        print("Placing tile at ",env.state.board.coord_to_ij(running_champion_action))
#        obs_new_state, rew, done, _ = env.step(running_champion_action)
        print("Placing tile at ",env.state.board.coord_to_ij(env.moves_log[-1]))
        env.moves_log.pop(-1) # easier than circumventing env.step function...
        obs_new_state, rew, done, _ = env.step(env.moves_log[-1])
        env.render()

        Experiences.add(orig_state, running_champion_action, rew, obs_new_state, done)
        orig_GoState = copy_go_state(env.state)
        cur_GoState = copy_go_state(orig_GoState)

    if e%1==0: 
        for upd in range(update_concentration):
            state, action, reward, next_st, done, count = Experiences.get_samp()
            target_a=reward
            if not done:
                if len(next_state.shape)<4:next_state=np.expand_dims(next_state,0)
                pred = _model.predict([next_state, next_state])
                target_a = reward + gamma*np.argmax( to_valid_moves(pred[0], \
                            next_state, env.moves_log, declined_options) )
                target_c = reward + gamma*pred[1]
                
            if len(state.shape)<4:state=np.expand_dims(state,0)
            target_f = to_valid_moves(_model.predict([state, state]), env.moves_log, declined_options)
            target_f[action] = target
            _model.fit(state, target_f, epochs=1, verbose=0)
        


# Now that the model's built, plug this into openai gym's baselines' deepq system.
#tf.reset_default_graph()
#inp_orig = tf.placeholder(tf.float32, [None, 9, 9, 3], 'inp_orig')
#inp_cons = tf.placeholder(tf.float32, [None, 9, 9, 3], 'inp_cons')
#
#sess=tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
##optm = [tf.train.RMSPropOptimizer(learning_rate=1e-3),  'rmsprop']
##
#def model(m, num_actions, scope): 
#    return _model.compile(optimizer=optm[1], loss=huber_loss)
#
#act, train, update_target, debug = deepq.build_train(\
#        make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name), 
#        q_func=model,
#        num_actions=81, 
#        optimizer=optm[0], 
#        reuse=tf.AUTO_REUSE)


