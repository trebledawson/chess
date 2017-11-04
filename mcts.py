"""
A parallelized algorithm for Monte Carlo tree search
MCTS Algorithm adapted from Jeff Bradberry

TODO:
-Implement alpha-beta pruning
-Parallelization
"""
"""
From 'Parallel Go on CUDA with MCTS'
------------------------------------

function CUDA_MCTS(state_0)
{
    create root node node_0 with state state_0
    while (nodeCount <= max allowed nodes):
    {
        chosenNode = TreePolicy(node_0)
    {
    result = CUDASimuulationKernel(chosenNode)
    BackPropagate(result, chosenNode)
    return best scoring node
}
   
function CUDASimulationKernel(chosenNode)
{
    create root node node_0 with state state_0
    while (!gameEndingCondition):
    {
        SimulationPolicy(chosenNode)
    }
    resultArray[threadIdx] = result
    SynchronizeAllThreads()
    stats = inclusiveScan(resultArray)
    return stats
}
"""
from random import choice
from copy import deepcopy
from tools import features
from math import log, sqrt
import time

def mcts(board, **kwargs):
    # Initialize MCTS variables
    thinking_time = kwargs.get('thinking_time', 10)
    C = kwargs.get('C', 1.4)
    state = deepcopy(board)
    whi, bla, player, cas, enp = features(state)
    wins = {}
    plays = {}

    # Evaluate legal moves at current board position
    legal = [move for move in board.generate_legal_moves()]

    # If no leagl moves, return nothing
    if not legal:
        return

    # If only one legal move, return that move
    if len(legal) == 1:
        return legal[0]

    # MCTS algorithm; run while elapsed time < thinking time
    games = 0
    start = time.time()
    while time.time() - start < thinking_time:
        wins, plays = run_sim(state, wins, plays, C)
        games += 1

    # All possible legal moves and their respective board states
    moves_states = [(move, next_state(state, move).fen()) for move in legal]

    # Display number of games simulated and time elapsed
    print('Simulations:', games, '| Simulation time:', time.time() - start,
          'seconds')

    # Pick move with highest percentage of wins
    best_percent = 0
    move = 0
    for move, state in moves_states:
        percent_wins = wins.get((player, state), 0) / \
                       plays.get((player, state), 1)
        if percent_wins >= best_percent:
            best_percent = percent_wins
            move = move

    return best_percent, move


def run_sim(board, wins, plays, C):
    # Initialize simulation variables
    state = deepcopy(board)
    visited_states = set()
    is_winner = False
    player = 0
    if state.fen().split()[1] == 'b':
        player = 1
    expand = True

    # Simulate game
    while not is_winner:
        # All possible legal moves and their respective board states
        moves_states = [(_move, next_state(state, _move)) for _move in
                        state.generate_legal_moves()]

        # Check if there are stats for all legal moves.
        if all(plays.get((player, _state.fen())) \
               for _move, _state in moves_states):
            # If so, use them
            log_total = log(sum(plays[(player, _state)] for _move, _state in
                                moves_states))
            value, move, state = max(((wins[(player, _state)] /
                                       plays[(player, _state)]) +
                                      C * sqrt(log_total /
                                               plays[(player, _state)]),
                                      _move, _state)
                                     for _move, _state in moves_states)
        else:
            # If not, choose arbitrarily
            move, state = choice(moves_states)

        if expand and (player, state.fen()) not in plays:
            plays[(player, state.fen())] = 0
            wins[(player, state.fen())] = 0
            expand = False

        visited_states.add((player, state.fen()))
        player = not player
        is_winner = state.is_game_over()

    # Update wins, plays
    winner = not player
    for _player, _state in visited_states:
        if (_player, _state) not in plays:
            continue

        plays[(_player, _state)] += 1
        if _player == winner:
            wins[(_player, _state)] += 1

    return wins, plays


def next_state(board, move):
    state = deepcopy(board)
    state.push(move)
    return state
