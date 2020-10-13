'''
Morgan, 12.9.17 

OpenAI interfacer. CustomGoEnv is the callable
class that initializes the game-playing paradigm in which
an agent can step through the enviroment.

The purpose was to build a custom enviroment as per the specifications
of gym. 

Left off: Adapting this environment to handle reversion to previous states.
'''



from gym import error
try:
    import pachi_py
except ImportError as e:
    # The dependency group [pachi] should match the name is setup.py.
    raise error.DependencyNotInstalled('{}. (HINT: you may need to install the Go dependencies via "pip install gym[pachi]".)'.format(e))

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six


# The coordinate representation of Pachi (and pachi_py) is defined on a board
# with extra rows and columns on the margin of the board, so positions on the board
# are not numbers in [0, board_size**2) as one would expect. For this Go env, we instead
# use an action representation that does fall in this more natural range.

def _pass_action(board_size):
    return board_size**2

def _resign_action(board_size):
    return board_size**2 + 1

def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
    i, j = board.coord_to_ij(c)
    return i*board.size + j

def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == _pass_action(board.size): return pachi_py.PASS_COORD
    if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)

def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s.encode()))

class CustomGoState(object):
    '''
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in [pachi_py.BLACK, pachi_py.WHITE], 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            1) a new CustomGoState with the new board and the player switched
            2) a move log item of {action taken, by who}
        '''
#        print(action, self.color, _action_to_coord(self.board, action))
        coord=_action_to_coord(self.board, action)
#            self.moves_log.append({'action':action, 'board':self.state.board,
#                'color':self.state.color})
        return CustomGoState(self.board.play(coord, self.color),\
                        pachi_py.stone_other(self.color)), coord
#                    {'action_to_coord':coord, 'color':self.color}

    def __repr__(self):
        return 'To play: {}\n{}'.format(six.u(pachi_py.color_to_str(self.color)), self.board.__repr__().decode())


### Adversary policies ###
def make_random_policy(np_random):
    def random_policy(curr_state, prev_state, prev_action):
        b = curr_state.board
        legal_coords = b.get_legal_coords(curr_state.color)
        return _coord_to_action(b, np_random.choice(legal_coords))
    return random_policy

def make_pachi_policy(board, engine_type='uct', threads=1, pachi_timestr=''):
    engine = pachi_py.PyPachiEngine(board, engine_type, six.b('threads=%d' % threads))

    def pachi_policy(curr_state, prev_state, prev_action):
        print('         pachi checkpoint 1662136')
        if prev_state is not None:
            #assert engine.curr_board == prev_state.board, 'Engine internal board 
            #is inconsistent with provided board. The Pachi engine must be called
            #consistently as the game progresses.'
            prev_coord = _action_to_coord(prev_state.board, prev_action)
            engine.notify(prev_coord, prev_state.color)
            engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = engine.genmove(curr_state.color, pachi_timestr)
        out_action = _coord_to_action(curr_state.board, out_coord)
        engine.curr_board.play_inplace(out_coord, curr_state.color)
        return out_action

    return pachi_policy


def _play(black_policy_fn, white_policy_fn, board_size=19):
    '''
    Samples a trajectory for two player policies.
    Args:
        black_policy_fn, white_policy_fn: functions that maps a 
        CustomGoState to a move coord (int)
    '''
    moves = []

    prev_state, prev_action = None, None
    curr_state = CustomGoState(pachi_py.CreateBoard(board_size), BLACK)

    while not curr_state.board.is_terminal:
        a = (black_policy_fn if curr_state.color == BLACK else \
                    white_policy_fn)(curr_state, prev_state, prev_action)
        next_state, ignored__log_move = curr_state.act(a)
        moves.append((curr_state, a, next_state))

        prev_state, prev_action = curr_state, a
        curr_state = next_state

    return moves

class CustomGoEnv(gym.Env):
    '''
    Go environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        self._seed()

        colormap = {
            'black': pachi_py.BLACK,
            'white': pachi_py.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent_policy = None
        self.opponent = opponent

        assert observation_type in ['image3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'image3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        shape = pachi_py.CreateBoard(self.board_size).encode().shape
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        # One action for each board position, pass, and resign
        self.action_space = spaces.Discrete(self.board_size**2 + 2)

        # Filled in by _reset()
        self.state = None
        #self.prev_state_stack = []
        self.moves_log = []
        self.done = True

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        pachi_py.pachi_srand(seed2)
        return [seed1, seed2]

    def _reset(self):
        # Don't touch moves_log!
        self.state = CustomGoState(pachi_py.CreateBoard(self.board_size), pachi_py.BLACK)

        # (re-initialize) the opponent
        # necessary because a pachi engine is attached to a game via internal data in a board
        # so with a fresh game, we need a fresh engine
        self._reset_opponent(self.state.board)

        # Let the opponent play if it's not the agent's turn
        opponent_resigned = False
        if self.state.color != self.player_color:
            self.state, opponent_resigned = self._exec_opponent_play(self.state, None, None)

        # We should be back to the agent color
        assert self.state.color == self.player_color

        self.done = self.state.board.is_terminal or opponent_resigned
        return self.state.board.encode()

    def _close(self):
        self.opponent_policy = None
        self.state = None
        self.moves_log = []
        #self.prev_state_stack = []

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile

    ''' STEP ---------------------------------------------------------- '''

    def _step(self, action):
        assert self.state.color == self.player_color

        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}

        # If resigned, then we're done
        if action == _resign_action(self.board_size):
            self.done = True
            return self.state.board.encode(), -1., True, {'state': self.state}

        # Play
        #self.prev_state_stack.append( self.state )
        prev_state = self.state
        try:
            self.state, logged_move = self.state.act(action)
            self.moves_log.append(logged_move)
        except pachi_py.IllegalMove:
            if self.illegal_move_mode == 'raise':
                six.reraise(*sys.exc_info())
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state.board.encode(), -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(\
                                    self.illegal_move_mode))


        # Opponent play
        if not self.state.board.is_terminal:
#            self.state, opponent_resigned = self._exec_opponent_play(self.state, \
#                            self.prev_state_stack[-1], action)
            self.state, opponent_resigned = self._exec_opponent_play(self.state, \
                            prev_state, action)
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color

            # If the opponent resigns, then the agent wins
            if opponent_resigned:
                self.done = True
                return self.state.board.encode(), 1., True, {'state': self.state}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.board.is_terminal:
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal
        self.done = True
        white_wins = self.state.board.official_score > 0
        black_wins = self.state.board.official_score < 0
        player_wins = (white_wins and self.player_color == pachi_py.WHITE) or \
                (black_wins and self.player_color == pachi_py.BLACK)
        reward = 1. if player_wins else -1. if (white_wins or black_wins) else 0.
        return self.state.board.encode(), reward, True, {'state': self.state}

    ''' REVERT ---------------------------------------------------------- '''

    def revert(self, game_turn, step='imagined', which_reversion_type='penultimate'):
        ''' Revert: reset the environment to new game, then retrace all the 
            actions taken by both players as recorded. Note, since the 
            opponent's actions are nondeterministic, they must be stored not
            reattempted. '''
        assert which_reversion_type=='penultimate' # for now
        self._reset()
        if step=='imagined' and len(self.moves_log)>1 and len(self.moves_log)>game_turn*2:
            self.moves_log.pop(-1)
            self.moves_log.pop(-1)
        elif step=='true':
            self.moves_log = self.moves_log[:2*(1+game_turn)] # ?
        for move in self.moves_log:
            self.state, _ = self.state.act(move)
        return self.moves_log[:]

    def _exec_opponent_play(self, curr_state, prev_state, prev_action):
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        opponent_resigned = opponent_action == _resign_action(self.board_size)
        new_state, logged_move = curr_state.act(opponent_action)
        self.moves_log.append(logged_move)
        return new_state, opponent_resigned

#    @property
#    def _state(self):
#        return self.state

    def _reset_opponent(self, board):
        if self.opponent == 'random':
            self.opponent_policy = make_random_policy(self.np_random)
        elif self.opponent == 'pachi:uct:_2400':
            self.opponent_policy = make_pachi_policy(board=board, engine_type=six.b('uct'), pachi_timestr=six.b('_2400')) # TODO: strength as argument
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

