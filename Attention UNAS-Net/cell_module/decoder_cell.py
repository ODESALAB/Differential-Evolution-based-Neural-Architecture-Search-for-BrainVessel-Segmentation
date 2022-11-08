import numpy as np
import torch
import torch.nn as nn
from cell_module.ops import OPS as ops_dict
from cell_module.attention_blocks import *

# Decoder operations
ENC_OPS = ['conv_2d_1x1',
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

class DecoderCell(nn.Module):
    def __init__(self, matrix, ops, prev_C, currrent_C):
        super(DecoderCell, self).__init__()
        
        self.ops = ops # Discrete values
        self.matrix = matrix
        self.prev_C = prev_C
        self.current_C = currrent_C # Number of filters

        self.NBR_OP = self.matrix.shape[0] - 1
        self.stem_conv = nn.Conv2d(self.current_C + self.current_C, self.current_C, kernel_size=1, padding='same')
        self.up = UpConv(self.prev_C, self.current_C)
        self.attention = AttentionBlock(self.current_C, self.current_C, self.current_C // 2)
        self.compile()

    def compile(self):
        self.ops_list = nn.ModuleList([self.up, self.attention, self.stem_conv])        

        # Iterate each operation in cell structure
        for op_idx in range(1, self.NBR_OP):
            op = ENC_OPS[self.ops[op_idx - 1]]
            self.ops_list.append(ops_dict[op](self.current_C, self.current_C))

    def forward(self, inputs, skip):

        outputs = [0] * len(self.ops_list)
        output_upsampling = self.ops_list[0](inputs) # Upsampling
        output_attention_gate = self.ops_list[1](output_upsampling, skip) # Attention Gate
        outputs[0] = self.ops_list[2](torch.cat([output_attention_gate, output_upsampling], axis=1)) # Stem Convolution - Bottleneck

        # Feed forward - Input to output ; Iterate each operation in the Cell
        for op_idx in range(2, self.NBR_OP + 1):
            op = self.ops_list[op_idx + 1]
            # Get input nodes/edges to the operation
            in_nodes = list(np.where(self.matrix[:, op_idx - 1] == 1)[0])
            
            # Sum and process if there is more than one input node/edge
            if len(in_nodes) > 1:
                _input = sum([outputs[i] for i in in_nodes])
                outputs[op_idx - 1] = op(_input)
            else:
                outputs[op_idx - 1] = op(outputs[in_nodes[0]])
        
        # Get input nodes/edges to the output node
        in_nodes = list(np.where(self.matrix[:, self.NBR_OP] == 1)[0])
        return sum([outputs[out] for out in in_nodes]) #/ len(in_nodes) # Output - Average Operation




        
            