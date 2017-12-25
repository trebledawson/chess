"""
Cythonized versions of certain functions
"""

import chess
import numpy as np
cimport numpy as np
import time
from multiprocessing import Process, Queue
from copy import deepcopy
from tools import features, SearchTree
cimport cython

# From "tools.py"
@cython.boundscheck(False)
def get_move(edges_, int C):
    cdef unsigned int num_edges = len(edges_)
    cdef Py_ssize_t move
    cdef double frac
    cdef double P
    cdef np.ndarray edges = np.array(edges_)
    cdef np.ndarray U = np.zeros(num_edges)
    cdef unsigned int rows = edges.shape[0]
    cdef Py_ssize_t row_
    cdef sum_ = 0
    cdef np.ndarray a = np.zeros(num_edges)

    for move in range(num_edges):       # PUCT algorithm
        for row_ in range(rows):
            sum_ += edges[row_, 0]
        frac = sqrt(sum_) / (1 + edges[move, 0])
        P = edges[move, 3]
        U[move] = C * P * frac

    for row_ in range(rows):
        a[row_] = edges[row_, 2] + U[row_]     # First term is Q-value

    return np.random.choice(np.flatnonzero(a == a.max()))

@cython.boundscheck(False)
def get_pi(visits, float T):
    cdef unsigned int num_visits = len(visits)
    cdef Py_ssize_t move
    cdef double pi_m
    cdef double temp_pow = 1. / T
    cdef np.ndarray pi = np.zeros(num_visits)
    cdef float vpsum = 0.

    visits = np.array(np.ravel(visits))

    for move in range(num_visits):
        vpsum += visits[move] ** temp_pow
    for move in range(num_visits):
        pi_m = (visits[move] ** temp_pow) / vpsum
        if pi_m == np.inf:
            pi_m = 1
        pi[move] = pi_m

    return pi


cdef extern from "math.h":
    double sqrt(double m)

# From "parallel_mcts.py"
@cython.boundscheck(False)
def mcts(board, poss_moves, pipes_sim, float T=0.0001, float C=1.4,
         unsigned int thinking_time=10, tree=SearchTree()):
    cdef double start = time.time()
    cdef unsigned int workers = len(pipes_sim)
    cdef float epsilon = 0.25
    cdef Py_ssize_t len_priors = 1968
    cdef double sleep = 0.000000001
    cdef str uci
    cdef Py_ssize_t move_ = 0
    cdef Py_ssize_t index
    cdef double prior
    cdef double probability
    cdef unsigned int len_legal
    cdef np.ndarray w, b, p
    cdef double value
    cdef np.ndarray noise, noise_
    cdef int num_sims
    cdef unsigned int num_nodes
    cdef unsigned int num_trees
    cdef Py_ssize_t node_
    cdef Py_ssize_t tree_
    cdef np.ndarray Q_values
    cdef Py_ssize_t best
    cdef np.ndarray probs, pi
    cdef str fen
    cdef double Q

    state = deepcopy(board)

    legal = sorted([move.uci() for move in board.generate_legal_moves()])
    len_legal = len(legal)

    fen = state.fen()
    w, b, p = features(fen)
    w = w.reshape(1, 8, 8, 1)
    b = b.reshape(1, 8, 8, 1)
    p = p.reshape(1, 1)
    pipes_sim[0].send([w, b, p])
    pipes_sim[0].poll(timeout=None)
    priors, value = pipes_sim[0].recv()

    # Add Dirichlet noise to priors in root node
    noise = np.ravel(np.random.dirichlet([0.03, 0.03],
                                         size=len(priors)).reshape(1, -1))
    noise_ = noise[:len_priors]
    priors = ((1 - epsilon) * priors) + (epsilon * noise_)

    # Create node for each legal move, ignoring if node already exists for move
    indices = [poss_moves.index(uci) for uci in legal]
    for uci in legal:
        for child_node in tree.nodes:
            if child_node.name == uci:
                # Add Dirichlet noise to prior for node that already exists
                # for move
                prior = child_node.data[3]
                noise = np.random.dirichlet([0.03, 0.03])
                child_node.data[3] = ((1 - epsilon) * prior) + \
                                     (epsilon * noise[0])
                move_ += 1
                continue
        tree.create_node(P=priors[indices[move_]], name=uci)
        move_ += 1

    ####### While elapsed time < thinking time, search tree: ########
    # While simulations < 10 per process, search tree:
    tree_queue = Queue()
    sim_queue = Queue()
    sims = []
    for worker in range(workers):
        pipe_sim = pipes_sim[worker]
        sim = Process(target=parallel_simulation,
                      args=(tree, state, C, poss_moves, pipe_sim, start,
                            thinking_time, tree_queue, sim_queue))
        sims.append(sim)
        sim.start()
    trees = []
    simulations = []

    while len(simulations) < workers:
        trees.append(tree_queue.get())
        num_sims = sim_queue.get()
        simulations.append(num_sims)
        time.sleep(sleep)

    print('Simulations:', sum(simulations), '| Thinking time:', time.time() -
          start, 'seconds.')

    # Update master tree nodes based on best Q-values of simulated tree nodes
    num_nodes = len(tree.nodes)
    num_trees = len(trees)
    for node_ in range(num_nodes):
        Q_values = np.zeros(num_trees)
        for tree_ in range(num_trees):
            Q_values[tree_] = trees[tree_].nodes[node_].data[2]
        best = np.random.choice(np.where(Q_values == Q_values.max())[0])
        tree.nodes[node_] = trees[best].nodes[node_]

    # Select move
    visits = [tree.nodes[move_].data[0] for move_ in range(len_legal)]
    if sum(visits) > 0:
        probs = get_pi(visits, T)
        pi = np.zeros(priors.shape)
        for index, probability in zip(indices, probs):
            pi[index] = probability
        uci_ = np.random.choice(legal, p=probs)
    else:
        pi = priors
        uci_ = np.random.choice(legal)
    index = legal.index(uci_)

    # Prune tree for reuse in future searches
    tree = tree.nodes[index]
    Q = tree.data[2]

    # If chosen move's branch has not been explored, seed new nodes
    if Q == 0:
        state.push(chess.Move.from_uci(uci_))

        fen = state.fen()
        w, b, p = features(fen)
        w = w.reshape(1, 8, 8, 1)
        b = b.reshape(1, 8, 8, 1)
        p = p.reshape(1, 1)
        pipes_sim[0].send([w, b, p])
        pipes_sim[0].poll(timeout=None)
        priors, value = pipes_sim[0].recv()

        legal = [move_.uci() for move_ in state.generate_legal_moves()]
        indices = [poss_moves.index(uci) for uci in legal]
        move_ = 0
        for uci in legal:
            nodes_update(tree, priors, indices, move_, uci_)
            move_ += 1

    return uci_, pi, tree, index, Q


def parallel_simulation(tree, state, float C, poss_moves, pipe_sim, double start,
                        unsigned int thinking_time, tree_queue, sim_queue):
    cdef unsigned int simulations = 0
    tree_p = deepcopy(tree)

    while simulations < 10 and time.time() - start < thinking_time:
        tree_p = iteration(tree_p, state, C, poss_moves, pipe_sim)
        simulations += 1
    tree_queue.put(tree_p)
    sim_queue.put(simulations)


@cython.boundscheck(False)
def iteration(tree, board, float C, poss_moves, pipe_sim,
              unsigned int search_depth=50):
    cdef unsigned int state_ = 0
    cdef unsigned int depth = 0
    cdef Py_ssize_t index
    cdef double sleep = 0.000000001
    cdef Py_ssize_t i
    cdef unsigned int len_legal
    cdef unsigned int move_ = 0
    cdef str uci
    cdef str fen

    node = tree
    state = deepcopy(board)
    state_indices = []
    nodes = [node]
    is_winner = False

    # Traverse tree until end of game or search depth is reached
    while not is_winner and depth < search_depth:
        # Generate legal moves in position
        legal = sorted([move.uci() for move in state.generate_legal_moves()])
        len_legal = len(legal)
        # Select move
        if len_legal == 1:
            # If only one legal move, select that move
            index = 0
            uci = legal[index]

        else:
            # Select move using PUCT equation. See: tools.get_move()
            edges = node_edges(node, legal)
            index = get_move(edges, C)

        uci = legal[index]
        state.push(chess.Move.from_uci(uci))

        # Update evaluation node
        node = node.nodes[index]
        nodes.append(node)

        # Evaluate and expand
        legal = sorted([move.uci() for move in state.generate_legal_moves()])
        indices = [poss_moves.index(uci) for uci in legal]
        fen = state.fen()
        w, b, p = features(fen)
        w = w.reshape(1, 8, 8, 1)
        b = b.reshape(1, 8, 8, 1)
        p = p.reshape(1, 1)
        pipe_sim.send([w, b, p])
        pipe_sim.poll(timeout=None)
        priors, value = pipe_sim.recv()
        for uci in legal:
            nodes_update(node, priors, indices, move_, uci)
            move_ += 1
        move_ = 0

        # Bookkeeping
        is_winner = state.is_game_over()
        state_indices.append(index)
        state_ += 1
        depth += 1

    # Backup
    while state_ > 0:
        nodes = nodes[:-1]
        node = nodes[-1]
        i = state_indices[state_ - 1]
        nodes_backup(node, i, value)
        state_ -= 1

    return tree

@cython.boundscheck(False)
def node_edges(node, legal):
    cdef Py_ssize_t move
    cdef unsigned int len_legal = len(legal)
    edges = []
    for move in range(len_legal):
        edges.append(node.nodes[move].data)

    return edges

@cython.boundscheck(False)
def nodes_update(node, np.ndarray priors, indices, Py_ssize_t move, str uci):
    for child_node in node.nodes:
        if child_node.name == uci:
            return

    node.create_node(P=priors[indices[move]], name=uci)

@cython.boundscheck(False)
def nodes_backup(node, Py_ssize_t i, double value):
    node.nodes[i].data[0] += 1
    node.nodes[i].data[1] += value
    node.nodes[i].data[2] = node.nodes[i].data[1] / node.nodes[i].data[0]
