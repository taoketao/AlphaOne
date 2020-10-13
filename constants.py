'''constants for this extended program module.'''
import numpy as np

''' programming facilities/globals: '''
BLK = 111
WHT = 222
x=_x = 0
y=_y = 1
_who = _team = 0
_when = _turn = 1
N,S,E,W = 45,46,47,48
NSEW = [N,S,E,W]


''' development constants: '''
harness_mode = 'random' # vs. interactive, auto


''' experiment constants: '''
board_size = board_shape = (3,3)
human_mode = True
viz_delay = 2 # [seconds] if human-mode, outside test harness



''' optimization constants & hyperparameters: '''
gamma = 0.95
episodes=5000
max_actions=100
max_search_iterations=9
max_search_depth=3
max_buf_len = 1000
update_concentration = 16
merge_adversaries_interval = 200

burn_in = 250

network_params = {  \
        'board_shape': board_shape, \
        'inp_lyr_size' : 64, \
        'rep_lyr_size': 64, \
        'out_lyr_size' : int(np.prod(board_shape)+2), \
        'default_confidence': 0.00001, \
        'version': 'TRAINABLE' \
        }

