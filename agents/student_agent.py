# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math
import random
import copy
dir_map_new = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
class Node:
    def __init__(self, state = None ,parent=None, move=None):
        self.head = None
        self.state = state
        self.parent = parent  
        self.move = move  
        self.children = []  
        self.wins = 0  
        self.visits = 0 
        self.untried_moves = [] 

    def best_child_uct(self): #choosing the best child of the node using upper confidence trees
        value = -1
        if self.children == []:
            best_child = random_move(self.state[0], self.state[1], self.state[2], self.state[3])
            (x, y), d = best_child
            new_board = move(self.state[0], (x, y, d))
            bestt = Node(state = (new_board, (x, y), self.state[2], self.state[3]), move = (x, y, d))
            self.children.append(bestt)
            return bestt

        else:
            best_child = None
        
        for child in self.children:
            
            
            v =  (child.wins / child.visits) + math.sqrt(2)*math.sqrt(math.log(self.visits)/child.visits)
            
            if v > value:
                best_child = child
                value = v
        return best_child

    def best_child(self):
        best = 0
        bc = None
        for child in self.children:
            score = child.wins/child.visits
            if score >= best:
                best = score
                bc = child
        return bc

    


def check_endgame(chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find 
        board_size = chess_board.shape[0]
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie

        return True, p0_score, p1_score


def check_valid_step(chess_board, start_pos, end_pos, barrier_dir, adv_pos, max_step):
        
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is border
        r, c = end_pos
        
        if r >= chess_board.shape[0] or c >= chess_board.shape[0]:
            return False
        
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True
        
        # Get position of the adversary
        

        # BFS
        state_queue = [(start_pos, 0)]
        
        visited = {tuple(start_pos)}
        is_reached = False
        
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        
        return is_reached

def reachable_coordinates(my_pos, max_step):
    xm, ym = my_pos
    reachable = set()
    reachable_2 = []
    for i in range(max_step +1):
        if i == 0:
            reachable.add((xm, ym))
        else:
            for s in range(i + 1):
                reachable.add((xm + s, ym + (i-s)))
                reachable.add((xm - s, ym - (i-s)))
                reachable.add((xm + s, ym - (i-s)))
                reachable.add((xm - s, ym +  (i-s)))
        
    for e in reachable:
        x, y = e
        reachable_2.append((x, y, 'u'))
        reachable_2.append((x, y, 'd'))
        reachable_2.append((x, y, 'r'))
        reachable_2.append((x, y, 'l'))


    return reachable_2

def move(chess_board, next_move, s = 1):
    if s == 1:
        x, y, d = next_move
        chess_board_next = copy.deepcopy(chess_board)
        chess_board_next[x, y, d] = True
        return chess_board_next
    elif s == 0:
        x, y, d = next_move
        chess_board[x, y, d] = True
        return chess_board


def valid_moves(chess_board, my_pos, reachable, adv_pos, max_step):
    valid_moves = []
    for i in range(len(reachable)):
        x, y, d= reachable[i]
        start_pos = np.array(my_pos)
        lol = (x,y)
        end_pos = np.array(lol)
        d = dir_map_new[d]
        if check_valid_step(chess_board, start_pos, end_pos, d, adv_pos, max_step):
            valid_moves.append((x, y, d))
    return valid_moves

def get_valid_moves(my_pos, max_step, chess_board, adv_pos):
    reachable = reachable_coordinates(my_pos, max_step)
    valid = valid_moves(chess_board, my_pos, reachable, adv_pos, max_step)
    return valid
def surrounded(chess_board, next):
    
    
    ((x, y), dir) = next
    
    walls = 0
    for i in range(4):
        if chess_board[x, y, i] == True:
            walls += 1
    if walls >= 2:
        return True
        
def random_move(chess_board, my_pos, adv_pos, max_step):
    old_my_pos = my_pos
    old_adv_pos = adv_pos
    steps = np.random.randint(0, max_step + 1)

        # Pick steps random but allowable moves
    for _ in range(steps):
        r, c = my_pos

            # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary

        if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
            break

        random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

            # This is how to update a row,col by the entries in moves 
            # to be consistent with game logic
        m_r, m_c = moves[random_dir]
        my_pos = (r + m_r, c + m_c)

        # Final portion, pick where to put our new barrier, at random
    r, c = my_pos
        # Possibilities, any direction such that chess_board is False
    allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
    if len(allowed_barriers) < 1:
        
        return my_pos, -1
    assert len(allowed_barriers)>=1 
    dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
    #if surrounded(chess_board, (my_pos, dir)):
    #    return random_move(chess_board, old_my_pos, old_adv_pos, max_step)

    return my_pos, dir

def simulate(node):
    f = 0
    chess_board, my_pos, adv_pos, max_step = node.state
    chess_board_2 = copy.deepcopy(chess_board)
    i = 0
    ended = check_endgame(chess_board_2, my_pos, adv_pos)[0]
    while not ended:
        if f == 1000:
            break
        f +=1
        if i == 0: #opponent move
            
            adv_pos , dra = random_move(chess_board_2, adv_pos, my_pos, max_step)
            
            xa, ya = adv_pos
            if dra == -1: 
                break
            chess_board_2 = move(chess_board_2, (xa, ya, dra), 0)
            i = 1
            ended = (check_endgame(chess_board_2, my_pos, adv_pos)[0])
            continue
        elif i == 1: #my move
            
            my_pos , drm  = random_move(chess_board_2, my_pos, adv_pos, max_step)
            
            xm, ym = my_pos
            #check_valid_step(chess_board_2, np.array(my_pos_old), np.array(my_pos), dr, np.array(adv_pos), max_step)
            if drm == -1 :
                score_my = 0
                break
            chess_board_2 = move(chess_board_2, (xm, ym, drm), 0)
            i = 0
            ended = (check_endgame(chess_board_2, my_pos, adv_pos)[0])
            continue
    end_bool, my_score, adv_score = check_endgame(chess_board_2, my_pos, adv_pos)
    if my_score > adv_score:
        return(1, 0)
    elif adv_score > my_score:
        return (0, 1)
    elif end_bool and my_score == adv_score:
        return (0.5, 0.5)
    else:
        return (10, 10)
    


    
    
def mcts( chess_board, my_pos, adv_pos, max_step, valid, start_time):
    head = Node(state = (chess_board, my_pos, adv_pos, max_step))
    head.untried_moves = valid
    i = 0
    factorss = False
    while (time.time() - start_time) < 1.98:
        
        while head.untried_moves != [] and (time.time() - start_time) < 1.9:
            i+=1
            
            random_move = random.choice(head.untried_moves)
            
            head.untried_moves.remove(random_move)
            
            x, y, dir = random_move
            while surrounded(chess_board, ((x, y), dir)):
                if head.untried_moves == []:
                    factorss = True
                    break
                random_move = random.choice(head.untried_moves)
            
                head.untried_moves.remove(random_move)
            
                (x, y, dir) = random_move
            if factorss == True:
                break
            child_pos = (x, y)
            
            new_chess_board = move(chess_board, random_move)
            
            child = Node(state = (new_chess_board, child_pos, adv_pos, max_step), move = random_move)
            
            
            head.children.append(child)
            
            my_score, adv_score = simulate(child)
            while my_score == 10:
                my_score, adv_score = simulate(child)
            
            head.visits +=1
            head.wins += my_score
            child.visits += 1
            child.wins += my_score
            

        while (time.time() - start_time) < 1.9:
            
            new_child = head.best_child_uct()

            my_score, adv_score = simulate(new_child)
            while my_score == 10:
                my_score, adv_score = simulate(new_child)
            head.visits +=1
            head.wins += my_score
            new_child.visits += 1
            new_child.wins += my_score

    
    best = head.best_child()
    print(best.visits)
    x, y, d = best.move
    return ((x, y), d)




@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.mcts_tree = None
    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        
        valid = get_valid_moves(my_pos, max_step, chess_board, adv_pos)
        print('got valid')
        best_move = mcts(chess_board, my_pos, adv_pos, max_step, valid, start_time)
        print('got_best')
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        print(best_move)
        return best_move
