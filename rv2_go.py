'''
In this file we are building a new quick-n-dirty go game.

update: Upon discovering the limitations of Heroku, and not finding
an alternative readily, and thinking that Go would be simple to implement,
I attempted to implement the game of Go in this file.
Turns out that, despite the apparent simplicity of the game, it is a bit
more involved than seemed worthwhile.


'''
print('>>\t>> ',__name__)
import numpy as np
import random, time
import constants as C

class Move(object):
    def __init__(m, team, loc=None, nth=-1, board_size=None, loc2=None):
        if not (loc2==None)==(nth==-1): raise Exception("Ambiguous")
        m._team = team
        try:
            assert(len(loc)==2)
            m._loc=loc
        except:
            try: 
                int(loc)
                if loc2:
                    m._loc=(loc,loc2)
                elif board_size:
                    m._loc=(loc//board_size[0],loc%board_size[0])
            except: raise Exception(loc, board_size)
        m._nth=nth

    def set_n(m,n): 
        if m._nth==-1:  ret=1
        else:           ret=-1
        m._nth = n
        return ret

    def loc(m):     return m._loc # tuple
    def loc_id(m):  return m.lx()*board_size[0]+m.ly() # x,y-> unitary number
    def lx(m):      return m._loc[C.x]
    def ly(m):      return m._loc[C.y]
    def team(m):    return m._team
    def n(m):       return m._nth
    def displ(m,ret=False):   
        s='('+str(m._loc[C.x])+','+str(m._loc[C.y])+')/'+\
                {C.WHT:'w',C.BLK:'b'}[m._team]
        if m._nth>0: s += '/'+str(m._nth)
        if ret: return s
        else: print(s)



class Board(object):
    def __init__(brd, shape, moves=None):
        assert(type(shape)==tuple and len(shape)==2 and shape[0]==shape[1])
        brd.shape = shape
        brd.wipe_board()
        if moves:
            brd.instantiate_board(moves)

    '''
    brd.board data structure is the gameboard's size, with pair (which team,
        nth overall game turn). 
    brd._record appends (x,y,team,nth_turn).
    '''

    def wipe_board(brd):
        brd._board = np.zeros((brd.shape[C.x], brd.shape[C.y], 2))
        brd._record = []
        brd._n = 0

    def is_occ(brd, move):
        return brd._board[move.lx(), move.ly(), C._team]>0
    def is_suicide(brd, move):
        return False; # Stub for now -- returns false negatives.

#    def view(brd): return Board(brd.shape, brd.view_record())
    def get_team(brd,i,j=None): 
        if j==None: return brd._board[i[C.x],i[C.y],0]
        else: return brd._board[i,j,0]
    def view_n(brd): return brd._n
    def view_record(brd): return brd._record.copy()
    def view_moves(brd): return brd.view_record()
    def view_move(brd): return brd.view_record()[-1]
    def view_last_move_loc(brd):  return brd.view_record()[0:2] # [...)
    def view_last_move_team(brd): return brd.view_record()[2]
    def view_board(brd):  return brd._board.copy()
    def view_padded_board(brd): 
        return np.pad(brd.view_board(), 1,'constant', constant_values = \
                -1)[:,:,1:-2]
    '''
    attempts to make a move on this board.
    If the return value is >0, the board was changed; if <0, board is same.
    returns: 
        2 if success and captures were made, 
        1 if success,
       -1 if the team played the last move also,
       -2 if spot taken,
       -3 if attempted suicide is stopped,
    '''
    def player_moves(brd, move, display=False):
        if display:
            print('Player move:',move.displ(True), end=';  ')
            print('priors:', ';  '.join([r.displ(True) for r in brd._record]))
        assert(move.team() in (C.WHT, C.BLK) \
               and len(move.loc())==2 \
               and move.lx()<=brd.shape[C.x] and move.ly()<brd.shape[C.y])
        if len(brd._record)>0 and move.team() == brd._record[-1].team(): 
            return -1
        if brd.is_occ(move): return -2
        if brd.is_suicide(move): return -3
        brd._make_move(move)
        any_captures = brd.check_captures(move)
        if display: brd.display_board()
        if any_captures: return 2
        else: return 1

    ''' _make_move [internal]: do checks beforehand! Does not do captures. '''
    def _make_move(brd, move):
        brd._board[move.loc()] = [move.team(), brd._n]
        move.set_n(brd._n)
        brd._record.append( move )
        brd._n += 1

    ''' view_board: make a copy of the board, and return it '''
    def get_board_copy(brd):
        new_b = Board(brd.shape)
        for rec_move in brd._record:
            assert(new_b.player_moves(rec_move) > 0 )
        return new_b

    def virtual_move(brd, move, display=False):
        return brd.get_board_copy().player_moves(move, display)

    def instantiate_board(brd, moves):
        if not brd._n==0: raise Exception('wipe me first')
        brd.update_board(moves)
    def restore(brd, targ_b): 
        return brd.instantiate_board(targ_b.view_record())

    def update_board(brd, moves):
        for m in moves:
            try:    assert(brd.player_moves(m)>0)
            except: raise Exception(m)
    def validate_moves(brd):
        return B._board[np.where(B._board>0)]

    def display_board(brd, mode='default'):
        print('')
        for i in range(brd.shape[0]):
            for j in range(brd.shape[1]):
                if brd._board[i,j,0]==0: print('- ', end='')
                elif brd._board[i,j,0]==C.WHT: print('x ', end='')
                elif brd._board[i,j,0]==C.BLK: print('o ', end='')
                else: print('? ', end='')
            print('')
        print('')


    ''' check_captures, eliminate pieces, and score are the algorithmic
        core of this class.
        
        check_capture naive method: add stone; scan stones and search
        along them & tag with colors to identify groups, for each color;
        again search for liberties in each group by pinging blocked-by-
        -other/free/recurse-on-friend.

        0: unchecked
        1: simple 'visited' check
        '''
    def check_captures(brd, move):
        # 1. identify groups
        groups = []
        new_group_map = np.zeros(brd.shape)
        itr=0

        checked = np.zeros(brd.shape, dtype=bool)
        for i in range(brd.shape[C.x]):
            for j in range(brd.shape[C.y]):
                if checked[i,j]: continue
                new_group = np.zeros(brd.shape)
                brd.find_group(i,j,new_group)
                checked += new_group.astype(bool)
                itr += 1
                new_group_map += new_group*itr
                groups.append(zip(*np.where(new_group_map)))

#        for group in groups:
#            team = brd._board[group[0][C.x],group[0][C.y], 0]

        # 2. find liberties
        any_captures=False
        for g in groups:
            if not brd.find_liberties(g):
                if brd.get_team(g[0])==move.team():
                    # case: suicide
                    if not any_captures: # dual annihilation??
                        return False     # No, it is a valid move
                if brd.get_team(g[0])=={C.WHT:C.BLK,C.BLK:C.WHT}[move.team()]:
                    any_captures=True
                    for i,j in g:
                        brd._board[i,j,0] = 0
                        brd._board[i,j,1] = brd._n
        return any_captures

    ''' returns: True if this group has liberties, False if this group is
        captured / suicide '''
    def find_liberties(brd, g):
        #g_team = brd.get_team(g[0])
        #liberty_found = False
        checked = np.zeros(brd.shape, dtype=bool)
        #otherteam = {C.WHT:C.BLK, C.BLK:C.WHT}[team]
        for i,j in g:
            for d in C.NSEW:
                q_x,q_y={C.N:(i,j+1),C.S:(i,j-1),C.E:(i+1,j),C.W:(i-1,j)}[d]
                if checked[q_x,q_y]: continue
                if q_x<0 or q_x>=brd.shape[C.x] or  \
                        q_y<0 or q_y>=brd.shape[C.y]: continue
                checked[q_x,q_y] = True

                q_team = brd.get_team(q_x,q_y)
                if q_team==0: return True
        return False


    def find_group(brd,i,j,checked,from_dir=None):
        for nxt in C.NSEW: # tbh NSEW usage is arbitrary to symmetry
            adjval = { C.N:(i,j+1), C.S:(i,j-1), C.E:(i+1,j), C.W:(i-1,j)}[nxt]
            if any(adjval[z]<0 or adjval[z]>=brd.shape[z] for z in [C.x,C.y]):
                continue
            same_clr = all(brd._board[(i,j)] == brd._board[adjval])
            if not same_clr: continue
            if bool(checked[adjval])==False and same_clr:
                checked[adjval]=int(True)
                brd.find_group(adjval[C.x],adjval[C.y],checked,\
                            {C.N:C.S,C.S:C.N,C.E:C.W,C.W:C.E}[nxt])

''' Build the network '''
import tensorflow as tf


''' a non-batchified custom network ready for reinforcement learning. '''
class Net(object):
    def __init__(network, **params):
        assert('board_shape' in params.keys())
        network.version = params.get('version', 'RANDOM') # vs. 'TRAINABLE',...
        network.sess = None
        network.batch_on = params.get('batch_on', False)
        network.default_confidence = params.get('default_confidence', 0.0)
        network.net_params = params.copy()
                                   
        network.build_net(**params)
        
    def build_net(network, **params):
        input_shape = [2+i for i in params.get('board_shape')]+[1]
        batchsize = params.get('batch_size', 1)
        if not batchsize==1:
            assert(network.batch_on) 
        input_shape = [batchsize]+input_shape

        ''' inputs '''
        network.input_var = \
                tf.placeholder( \
                    dtype = tf.float32,  \
                    shape = input_shape)

        network.team_var = \
                tf.placeholder( \
                    dtype = tf.float32,  \
                    shape = [batchsize, 1])

        ''' intermediate layers '''
        network.conv_layer =  \
                tf.layers.conv2d( \
                    inputs = network.input_var,  \
                    filters = params.get('inp_lyr_size'),  \
                    kernel_size = params.get('kernel_size', (5,5)),  \
                    activation = tf.nn.relu)

        network.merge_layer =  tf.concat(  [  tf.reshape(  \
                        network.conv_layer, shape=(1, \
                        np.prod(network.conv_layer.get_shape().as_list()))),\
                        network.team_var ], axis=1 )

        network.reprs_layer = tf.layers.dense( \
                    inputs = network.merge_layer, \
                    units = params.get('rep_lyr_size'),  \
                    activation = tf.nn.relu) 

        ''' intermediate outputs '''
        network.action_outputs =  \
                tf.layers.dense( \
                    inputs = network.reprs_layer,  \
                    units = params.get('out_lyr_size'),  \
                    activation = tf.nn.softmax) 

        network.confidence =  \
                tf.layers.dense( \
                    inputs = network.reprs_layer,  \
                    units = 1,  \
                    activation = None)

    def set_sess(network, s):
        if not network.sess==None:
            print("Warning: sess is already initialized, overwriting")
        network.sess=s

    def create_sess(network):
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        network.set_sess(s)
        return s

    def _format_team_var_input(network, team):
        try: 
            int(team); return np.reshape((team,), (1,1))
        except: pass
        try: 
            float(team); return np.reshape((team,), (1,1))
        except: pass



    def get_predictions(network, state=None, which_team_turn=None):
        assert(which_team_turn in [C.WHT,C.BLK] or all(t in [C.WHT,C.BLK] for t \
                in which_team_turn))
        Qs = None
        if network.version=='RANDOM':
            return [network._random_action_Qs(), network.default_confidence]
        else:
            if type(state)==Board: _state = state.view_board()
            if type(state)==np.ndarray: _state = state

            if network.sess==None: 
                print('network sess unset. creating new one as [net].sess')
                networ.create_sess
            if _state.ndim==3: _state = np.expand_dims(_state, 0)
            assert( _state.ndim==4)
            
            team_var_input = network._format_team_var_input(which_team_turn)

            Q_sa = network.sess.run( \
                    fetches = [network.action_outputs, network.confidence], \
                    feed_dict = { network.input_var: _state, \
                                  network.team_var: team_var_input } )
            return Q_sa
            #return { True:Q_sa, False: Q_sa[0] }[network.batch_on]

    def _random_action_Qs(network):
        sz = network.net_params['out_lyr_size']
        return np.ones(shape=(sz,)) * (sz**-1)
        


net = Net(**C.network_params)
net.create_sess()

''' container for game states '''
class Episode(object):
    def __init__(ep, board, replay_or_update ):
        ep.num_accesses = 1

class Buffer(object):
    def __init__(bf, max_num_items=2000): 
        bf._data = []
        bf._max_num_items = max_num_items
    def get_samp(bf): 
        ps = np.array([x.num_accesses**-2 for x in bf._data])
        index = np.random.choice(len(bf._data), p=ps/sum(ps))[-3]
        bf._data[index].num_accesses += 1
        return bf._data[index]

    ''' does not check for copies. '''
    def add_episode(bf, board, replay_or_update, episode, ):
        bf._data.append('stub')
        while len(bf._data)>bf._max_num_items:
            bf._data.pop(0)


replay_buffer, update_buffer = Buffer(), Buffer()

# so... looks like I'm just going to make it play itself
# With the update delay buffer, should maybe work ok


if __name__=='__main__':
  for e in range(C.episodes):
    print('e',e)

    ''' replay buffer '''
    B = Board(C.board_shape)
    if e>C.burn_in and random.random()<0.5:
        B.restore( replay_buffer.get_samp() )

    curr_state = B.view_padded_board()
    team_turn = np.random.choice([C.WHT,C.BLK])

    ''' action '''
    game_over = False
    for game_turn in range(C.max_actions):
        current_depth=0
        running_champion_action = -1 #?
        accept_move = True
        for d in range(C.max_search_iterations):
            q_actions, conf_ = net.get_predictions(curr_state, team_turn)
            print(np.argmax(q_actions), np.max(q_actions))
            print(np.argmax(conf_), np.max(conf_))
            print (q_actions, conf_)
            conf = conf_[0][0]
            accept_move = True if confidence > 0 else False
            if current_depth==max_search_depth: accept_move=False
            valid_moves = np.ma.masked_array(q_actions,\
                                 B.validate_moves())
            ''' eliminate pieces '''
            

            ''' if no valid moves, game over (or close enough) '''
            if sum(valid_moves)<=0: 
                game_over = True
                rew = B.evaluate_score()
                

            


            if game_over or not accept_move: break
        if not accept_move: break


                
    if C.human_mode: time.sleep(C.viz_delay)
    ''' update buffer '''
    pass




