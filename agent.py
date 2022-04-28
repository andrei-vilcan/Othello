"""
An AI player for Othello.
"""

import random
import sys
import time
import math
# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move


inf = float("inf")


def eprint(*args, **kwargs):  #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)


cached = {}


# Method to compute utility value of terminal state
def compute_utility(board, color):
    score = get_score(board)
    if color == 1:
        return score[0] - score[1]
    else:
        return score[1] - score[0]


# Better heuristic value of board
def compute_heuristic(board, color):  #not implemented, optional
    moves = get_possible_moves(board, color)
    if not moves:
        return None, compute_utility(board, color)

    if color == 1:
        opponent = 2
    else:
        opponent = 1

    potential_moves = {}

    for move in moves:
        next_board = play_move(board, color, move[0], move[1])
        movability = len(get_possible_moves(next_board, opponent))
        if movability == 0:
            if abs(compute_utility(next_board, opponent)) < abs(compute_utility(board, color)):
                continue

        util_diff = abs(compute_utility(board, color)) - abs(compute_utility(next_board, opponent))
        if util_diff < 0:
            continue
        potential_moves[move] = util_diff

    best_move = None
    value = float("-inf")
    for move in potential_moves.keys():
        if value < potential_moves[move]:
            value = potential_moves[move]
            best_move = move

    return best_move



############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching=0):
    moves = get_possible_moves(board, color)
    if not moves or limit == 0:
        return None, compute_utility(board, color)

    if color == 1:
        opponent = 2
    else:
        opponent = 1

    value = inf
    best_move = None

    for move in moves:
        next_board = play_move(board, color, move[0], move[1])
        # caching
        if next_board in cached:
            next_move, next_value = cached[next_board]
        else:
            next_move, next_value = minimax_max_node(next_board, opponent, limit - 1, caching)
            cached[next_board] = (next_move, next_value)

        if value > next_value:
            value = next_value
            best_move = move

    return best_move, value


def minimax_max_node(board, color, limit, caching=0):  #returns highest possible utility
    moves = get_possible_moves(board, color)
    if not moves or limit == 0:
        return None, compute_utility(board, color)

    if color == 1:
        opponent = 2
    else:
        opponent = 1

    value = -inf
    best_move = None

    for move in moves:
        next_board = play_move(board, color, move[0], move[1])
        # caching
        if next_board in cached:
            next_move, next_value = cached[next_board]
        else:
            next_move, next_value = minimax_min_node(next_board, opponent, limit - 1, caching)
            cached[next_board] = (next_move, next_value)

        if value < next_value:
            value = next_value
            best_move = move

    return best_move, value


def select_move_minimax(board, color, limit, caching=0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    """
    return minimax_max_node(board, color, limit, caching)[0]


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    moves = get_possible_moves(board, color)
    if not moves or limit == 0:
        return None, compute_utility(board, color)

    if color == 1:
        opponent = 2
    else:
        opponent = 1
    best_move = None
    value = float("inf")

    # ordering
    if ordering:
        ordered_moves = []
        for move in moves:
            ordered_moves.append([(move[0], move[1]), compute_utility(play_move(board, color, move[0], move[1]), color)])
        moves = sorted(ordered_moves, key=lambda x: x[1], reverse=True)

    for move in moves:
        next_board = play_move(board, color, move[0], move[1])

        # caching
        if next_board in cached:
            next_move, next_value = cached[next_board]
        else:
            next_move, next_value = alphabeta_max_node(next_board, opponent, alpha, beta, limit - 1, caching, ordering)
            cached[next_board] = (next_move, next_value)

        if next_value < value:
            value = next_value
            best_move = move

        if value <= alpha:
            return best_move, value

        beta = min(beta, value)

    return best_move, value


def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    moves = get_possible_moves(board, color)
    if not moves or limit == 0:
        return None, compute_utility(board, color)

    if color == 1:
        opponent = 2
    else:
        opponent = 1
    best_move = None
    value = float("-inf")

    # ordering
    if ordering:
        ordered_moves = []
        for move in moves:
            ordered_moves.append([(move[0], move[1]), compute_utility(play_move(board, color, move[0], move[1]), color)])
        moves = sorted(ordered_moves, key=lambda x: x[1], reverse=True)

    for move in moves:
        next_board = play_move(board, color, move[0], move[1])

        # caching
        if next_board in cached:
            next_move, next_value = cached[next_board]
        else:
            next_move, next_value = alphabeta_min_node(next_board, opponent, alpha, beta, limit - 1, caching, ordering)
            cached[next_board] = (next_move, next_value)

        if next_value > value:
            value = next_value
            best_move = move

        if value >= beta:
            return best_move, value

        alpha = max(alpha, value)

    return best_move, value


def select_move_alphabeta(board, color, limit, caching=0, ordering=0):
    return alphabeta_max_node(board, color, -inf, inf, limit, caching, ordering)[0]


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
