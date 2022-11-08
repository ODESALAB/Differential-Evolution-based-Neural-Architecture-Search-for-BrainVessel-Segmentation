import torch.nn as nn

class Conv2D(nn.Module):
    '''
        Conv2D + Batch Normalization + GELU
    '''

    def __init__(self, in_c, out_c, kernel_size):
        super(Conv2D, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
    
    def forward(self, inputs):
        return self.op(inputs)

class SepConv2D(nn.Module):
    '''
        2D Seperable Convolution -> Ref: DARTS: https://github.com/quark0/darts/blob/master/cnn/operations.py
    '''
    def __init__(self, C_in, C_out, kernel_size, padding, affine=True):
        super(SepConv2D, self).__init__()
        self.op = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding='same', stride=1, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.GELU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding='same', stride=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class DilConv2D(nn.Module):
  """
    2D Dilated Convolution -> Ref: DARTS: https://github.com/quark0/darts/blob/master/cnn/operations.py
  """
  def __init__(self, C_in, C_out, kernel_size, dilation, affine=True):
    super(DilConv2D, self).__init__()
    self.op = nn.Sequential(
      nn.GELU(),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding='same', dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class ASymmetricConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size):
        super(ASymmetricConv, self).__init__()
        
        self.conv_1_d = nn.Conv2d(C_in, C_out, kernel_size=(1, kernel_size), padding='same', stride=1)
        self.conv_d_d = nn.Conv2d(C_in, C_out, kernel_size=(kernel_size, kernel_size), padding='same', stride=1)
        self.conv_d_1 = nn.Conv2d(C_in, C_out, kernel_size=(kernel_size, 1), padding='same', stride=1)
        self.bn = nn.BatchNorm2d(C_out)
        self.gelu = nn.GELU()
    
    def forward(self, inputs):
        output_1_d = self.conv_1_d(inputs)
        output_1_d = self.bn(output_1_d)

        output_d_d = self.conv_d_d(inputs)
        output_d_d = self.bn(output_d_d)

        output_d_1 = self.conv_d_1(inputs)
        output_d_1 = self.bn(output_d_1)

        output = self.gelu(output_1_d + output_d_d + output_d_1)
        return output



class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_c)

        self.GELU = nn.GELU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.GELU(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU(x)

        return x

OPS = {
    'skip_connect': lambda in_c, out_c: nn.Identity(),
    'conv_2d_1x1' : lambda in_c, out_c: Conv2D(in_c, out_c, 1),
    'conv_2d_3x3' : lambda in_c, out_c: Conv2D(in_c, out_c, 3),
    'conv_2d_5x5' : lambda in_c, out_c: Conv2D(in_c, out_c, 5),
    'conv_2d_7x7' : lambda in_c, out_c: Conv2D(in_c, out_c, 7),
    'sep_conv_3x3': lambda in_c, out_c: SepConv2D(in_c, out_c, 3, 'same'),
    'sep_conv_5x5': lambda in_c, out_c: SepConv2D(in_c, out_c, 5, 'same'),
    'sep_conv_7x7': lambda in_c, out_c: SepConv2D(in_c, out_c, 7, 'same'),
    'dil_conv_3x3': lambda in_c, out_c: DilConv2D(in_c, out_c, 3, 2),
    'dil_conv_5x5': lambda in_c, out_c: DilConv2D(in_c, out_c, 5, 2),
    'dil_conv_7x7': lambda in_c, out_c: DilConv2D(in_c, out_c, 7, 2),
    'asym_conv_3x3': lambda in_c, out_c: ASymmetricConv(in_c, out_c, 3),
    'asym_conv_5x5': lambda in_c, out_c: ASymmetricConv(in_c, out_c, 5),
    'asym_conv_7x7': lambda in_c, out_c: ASymmetricConv(in_c, out_c, 7),
    'bottleneck'  : lambda in_c, out_c: BottleNeck(in_c, out_c),
}
