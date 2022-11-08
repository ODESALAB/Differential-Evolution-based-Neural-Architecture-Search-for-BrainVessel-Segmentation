import copy
import numpy as np
from cell_module.ops import OPS as ops_dict
from utils.encodings import *

OPS = ['conv_2d_1x1',
        'conv_2d_3x3',
        'conv_2d_5x5',
        'conv_2d_7x7',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'sep_conv_7x7',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'dil_conv_7x7',
        'asym_conv_3x3',
        'asym_conv_5x5',
        'asym_conv_7x7']

def path_distance(cell_1, cell_2):
    """ 
    compute the distance between two architectures
    by comparing their path encodings
    """
    return np.sum(np.array(cell_1.encode('path') != np.array(cell_2.encode('path'))))

def caz_encoding_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their in-out edges and path encodings
    """
    return np.sum(cell_1.encode('caz') != cell_2.encode('caz')) + path_distance(cell_1, cell_2)

def jackard_distance_caz(cell_1, cell_2):
    """
    compute the jackard distance between two architectures
    by comparing their caz encodings (in-out edges + path encoding - Tanimoto Index)
    """

    # Cell 1 - Path encoding, Tanimoto Index (Vector with 364 elements)
    cell1_path_vct = np.array(encode('path', cell_1['org_matrix'], cell_1['org_ops']))
    # Cell 2 - Path encoding
    cell2_path_vct = np.array(encode('path', cell_2['org_matrix'], cell_2['org_ops']))
    # Cell 1 - In-out edges encoding - Can Hoca'nın Path Encoding üzerine önerdiği encoding
    cell1_caz_vct = np.array(encode('caz', cell_1['org_matrix'], cell_1['org_ops']))
    # Cell 2 - In-out edges encoding
    cell2_caz_vct = np.array(encode('caz', cell_2['org_matrix'], cell_2['org_ops']))

    # Compute the jackard distance
    jk_dist = np.sum(cell1_path_vct * cell2_path_vct) + np.sum(cell1_caz_vct * cell2_caz_vct)
    total_hamming_dist = np.sum(cell1_caz_vct != cell2_caz_vct) + np.sum(cell1_path_vct != cell2_path_vct)
    return total_hamming_dist / (total_hamming_dist + jk_dist)

def encode(type, org_matrix, org_ops):
    if type == 'path':
            return encode_paths(get_path_indices(org_matrix, org_ops))
    elif type == 'caz':
        ops = copy.deepcopy(org_ops)
        ops = [OPS[i] for i in ops]
        ops.insert(0, 'input')
        ops.append('output')
        return encode_caz(org_matrix, ops)

def get_paths(org_matrix, org_ops):
        """ 
        return all paths from input to output
        """
        ops = copy.deepcopy(org_ops)
        ops = [OPS[i] for i in ops]
        ops.insert(0, 'input')
        ops.append('output')
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if org_matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if org_matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, ops[i]])
        return paths[-1]

def get_path_indices(org_matrix, org_ops):
    """
    compute the index of each path
    There are 9^0 + ... + 9^5 paths total.
    (Paths can be length 0 to 5, and for each path, for each node, there
    are nine choices for the operation.)
    """
    paths = get_paths(org_matrix, org_ops)
    ops = OPS
    mapping = {op: idx for idx, op in enumerate(OPS)}

    path_indices = []

    for path in paths:
        index = 0
        for i in range(NUM_VERTICES - 1):
            if i == len(path):
                path_indices.append(index)
                break
            else:
                index += len(ops) ** i * (mapping[path[i]] + 1)

    path_indices.sort()
    return tuple(path_indices)