import numpy as np
from cell_module.ops import OPS as ops_dict

INPUT = 'input'
OUTPUT = 'output'
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

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def encode_paths(path_indices):
    """ output one-hot encoding of paths """
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding

def encode_caz(matrix, ops):
    """Can Hoca'nın önerdiği encoding"""
    encoding = {f"{op}-{in_out}-{i}":0 for in_out in ["in","out"] for op in OPS for i in range(1, 7)}
    encoding.update({f"in-out-{i}":0 for i in range(1, 7)})
    encoding.update({f"out-in-{i}":0 for i in range(1, 7)})

    for i in range(7):
        op = ops[i].split("-")[0]
        out_edges = int(matrix[i,:].sum())
        in_edges = int(matrix[:,i].sum())
        
        if ops[i] == INPUT and out_edges != 0:
            encoding[f"in-out-{out_edges}"] = 1
        elif ops[i] == OUTPUT and in_edges != 0:
            encoding[f"out-in-{in_edges}"] = 1
        else:
            if in_edges !=  0:
                encoding[f"{op}-in-{in_edges}"] = 1
            if out_edges != 0:
                encoding[f"{op}-out-{out_edges}"] = 1

    return np.array(list(encoding.values()))