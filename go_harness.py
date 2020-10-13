''' oct 4 2018: a test harness for testing our go player. '''

print('>>\t>> ',__name__)
import time
import numpy as np
import constants as C
from rv2_go import Board, Move, Episode, Buffer

b = Board(C.board_size)


b.display_board()
#b.player_moves( Move(C.WHT, (3,2)), display=True)

if C.harness_mode=='interactive':
    while(True):
        i = input('move >> ')
        break
elif C.harness_mode=='random':
    itr = 0
    move_optns = list(range(int(np.prod(C.board_size))))

    while(True):
        if not itr%5: print("CTRL-C to end")
        res=0
        while(res<=0):
            move = Move(team=[C.WHT,C.BLK][itr%2], loc=np.random.choice(move_optns), \
                        board_size=C.board_size)
            print('>>>>>>>>',move.loc())
            res = b.virtual_move( move )
        #if move.loc_id()<np.prod(boardsize):
        if move.lx()<C.board_size[C.x] and move.ly()<C.board_size[C.y]:
            b.player_moves(move, display=True)
        else:
            print('\n\n<pass>\n\n')
        itr += 1
        if C.harness_mode=='auto':
            time.sleep(C.viz_delay)
        elif C.harness_mode=='random':
            if (input('enter to continue...') in ['Q','q']): break

