"""
Monte Carlo tree search algorithm adapted from DeepMind's AlphaGo Zero
architecture. Utilizes a neural network to evaluate positions and selects
moves based on node visits.
"""

import chess
import numpy as np
import time
from multiprocessing import Process, Queue
from copy import deepcopy
from tools import features, get_move, SearchTree, get_pi
from sklearn.metrics import mean_squared_error as mse


def mcts(board, poss_moves, pipes_sim, **kwargs):
    start = time.time()
    C = kwargs.get('C', 1.4)
    thinking_time = kwargs.get('thinking_time', 10)
    T = kwargs.get('T', 0.0001)
    tree = kwargs.get('tree', SearchTree())
    state = deepcopy(board)
    legal = sorted([move.uci() for move in board.generate_legal_moves()])
    sleep = 0.000000001

    feats = features(board.fen())
    feats = feats.reshape(1, 14, 8, 8)
    pipes_sim[0].send(feats)
    pipes_sim[0].poll(timeout=None)
    priors, value = pipes_sim[0].recv()

    # Add Dirichlet noise to priors in root node
    noise = np.ravel(np.random.dirichlet([0.03, 0.03],
                                         size=len(priors)).reshape(1, -1))
    noise = noise[:len(priors)]
    epsilon = 0.25
    priors = ((1 - epsilon) * priors) + (epsilon * noise)

    # Create node for each legal move, ignoring if node already exists for move
    indices = [poss_moves.index(move) for move in legal]
    for move, san in zip(range(len(legal)), legal):
        for child_node in tree.nodes:
            if child_node.name == san:
                # Add Dirichlet noise to prior for node that already exists
                # for move
                prior = child_node.data[3]
                noise = np.random.dirichlet([0.03, 0.03])
                child_node.data[3] = ((1 - epsilon) * prior) + \
                                     (epsilon * noise[0])
                continue

        tree.create_node(P=priors[indices[move]], name=san)

    # While elapsed time < thinking time, search tree:
    tree_queue = Queue()
    sim_queue = Queue()
    sims = []
    for worker in range(len(pipes_sim)):
        pipe_sim = pipes_sim[worker]
        sim = Process(target=parallel_simulation,
                      args=(tree, state, C, poss_moves, pipe_sim, start,
                            thinking_time, tree_queue, sim_queue))
        sims.append(sim)
        sim.start()
    trees = []
    simulations = []

    while len(simulations) < len(sims):
        trees.append(tree_queue.get())
        simulations.append(sim_queue.get())
        time.sleep(sleep)

    print('Simulations:', sum(simulations), '| Thinking time:', time.time() -
          start, 'seconds.')

    # Update master tree nodes based on best Q-values of simulated tree nodes
    for node in range(len(tree.nodes)):
        Q_values = np.zeros(len(trees))
        for tree_ in range(len(trees)):
            Q_values[tree_] = trees[tree_].nodes[node].data[2]
        best = np.random.choice(np.where(Q_values == Q_values.max())[0])
        tree.nodes[node] = trees[best].nodes[node]

    # Select move
    visits = [tree.nodes[move].data[0] for move in range(len(legal))]
    if sum(visits) > 0:
        probs = get_pi(visits, T)
        pi = np.zeros(priors.shape)
        for index, probability in zip(indices, probs):
            pi[index] = probability
        move = np.random.choice(legal, p=probs)
    else:
        pi = priors
        move = np.random.choice(legal)

    pi_move = pi[poss_moves.index(move)]
    index = legal.index(move)

    print('MSE:', mse(pi, priors))
    # Prune tree for reuse in future searches
    tree = tree.nodes[index]

    # If all simulated Q values were negative, seed leaves for new position
    if tree.data[2] == 0:
        state.push(chess.Move.from_uci(move))

        feats = features(board.fen())
        feats = feats.reshape(1, 14, 8, 8)
        pipes_sim[0].send(feats)
        pipes_sim[0].poll(timeout=None)
        priors, value = pipes_sim[0].recv()

        legal = [move_.uci() for move_ in state.generate_legal_moves()]
        indices = [poss_moves.index(move_) for move_ in legal]
        for move_, san in zip(range(len(legal)), legal):
            nodes_update(tree, priors, indices, move_, san)

    print('N:', tree.data[0], '| P:', tree.data[3], '| Pi:', pi_move)
    return move, pi, tree, index, tree.data[2]


def parallel_simulation(tree, state, C, poss_moves, pipe_sim, start,
                        thinking_time, tree_queue, sim_queue):
    tree_p = deepcopy(tree)
    simulations = 0
    while simulations < 5: #and time.time() - start < thinking_time:
        tree_p = iteration(tree_p, state, C, poss_moves, pipe_sim)
        simulations += 1
    tree_queue.put(tree_p)
    sim_queue.put(simulations)


def iteration(tree, board, C, poss_moves, pipe_sim, **kwargs):
    node = tree
    search_depth = kwargs.get('search_depth', 50)      # 25 fullmoves
    state = deepcopy(board)
    is_winner = False
    state_ = 0
    state_indices = []
    nodes = [node]
    depth = 0
    sleep = 0.000000001

    # Traverse tree until end of game or search depth is reached
    while not is_winner and depth < search_depth:
        # Generate legal moves in position
        legal = sorted([move.uci() for move in state.generate_legal_moves()])

        # Select move
        if len(legal) == 1:
            # If only one legal move, select that move
            index = 0
            state.push(chess.Move.from_uci(legal[0]))
        else:
            # Select move using PUCT equation. See: tools.get_move()
            edges = node_edges(node, legal)
            index = get_move(edges, C)
            move = legal[index]
            state.push(chess.Move.from_uci(move))

        # Update evaluation node
        node = node.nodes[index]
        nodes.append(node)

        # Evaluate and expand
        legal = sorted([move.uci() for move in state.generate_legal_moves()])
        indices = [poss_moves.index(move) for move in legal]
        feats = features(board.fen())
        feats = feats.reshape(1, 14, 8, 8)
        pipe_sim.send(feats)
        pipe_sim.poll(timeout=None)
        priors, value = pipe_sim.recv()

        for move, san in zip(range(len(legal)), legal):
            nodes_update(node, priors, indices, move, san)

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


def node_edges(node, legal):
    edges = []
    for move in range(len(legal)):
        edges.append(node.nodes[move].data)

    return edges


def nodes_update(node, priors, indices, move, san):
    for child_node in node.nodes:
        if child_node.name == san:
            return

    node.create_node(P=priors[indices[move]], name=san)


def nodes_backup(node, i, value):
    node.nodes[i].data[0] += 1
    node.nodes[i].data[1] += value
    node.nodes[i].data[2] = node.nodes[i].data[1] / node.nodes[i].data[0]
