"""
Monte Carlo tree search algorithm adapted from DeepMind's AlphaGo Zero
architecture. Utilizes a neural network to evaluate positions and selects
moves based on node visits.
"""

import chess
import numpy as np
import time
from copy import deepcopy
from tools import features, get_move, SearchTree, get_pi

# TODO: Parallelize iteration() to run multiple simulations simultaneously

def mcts(board, model, poss_moves, **kwargs):
    C = kwargs.get('C', 1.4)
    thinking_time = kwargs.get('thinking_time', 3)
    T = kwargs.get('T', 0.0001)
    tree = kwargs.get('tree', SearchTree())
    state = deepcopy(board)
    legal = sorted([move.uci() for move in board.generate_legal_moves()])

    # Perform MCTS only if there is more than one legal move
    start = time.time()
    w, b, p = features(board)
    w = w.reshape(1, 8, 8, 1)
    b = b.reshape(1, 8, 8, 1)
    p = p.reshape(1, 1)
    priors, _ = model.predict([w, b, p], verbose=0)
    priors = np.ravel(priors)

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
    simulations = 0
    while simulations < 1 or time.time() - start < thinking_time:
        tree = iteration(tree, model, state, C, poss_moves)
        simulations += 1
    print('Simulations:', simulations, '| Thinking time:', time.time() -
          start, 'seconds.')

    # Select move
    visits = [tree.nodes[move].data[0] for move in range(len(legal))]
    probs = get_pi(visits, T)
    pi = np.zeros(priors.shape)
    for index, probability in zip(indices, probs):
        pi[index] = probability
    index = np.random.choice(np.flatnonzero(probs == probs.max()))
    move = legal[index]

    # Prune tree for reuse in future searches
    tree = tree.nodes[index]

    return move, pi, tree, index

def iteration(tree, model, board, C, poss_moves, **kwargs):
    search_depth = kwargs.get('search_depth', 50)      # 25 fullmoves
    state = deepcopy(board)
    is_winner = False
    state_ = 0
    state_indices = []
    node = tree
    nodes = [node]
    depth = 0

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
        w, b, p = features(state)
        w = w.reshape(1, 8, 8, 1)
        b = b.reshape(1, 8, 8, 1)
        p = p.reshape(1, 1)
        priors, value = model.predict([w, b, p], verbose=0)
        priors = np.ravel(priors)
        value = np.ravel(value)

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

