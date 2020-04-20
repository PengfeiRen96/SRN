from __future__ import division, print_function
import torch
from torch import nn
import torch.nn.functional as F


class AddCoordinates(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).
    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.
    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.
    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`
    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        if image_height == 1:
            y_coords = torch.zeros([1]).unsqueeze(1)
        else:
            y_coords = 2.0 * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width).float() / (image_height - 1.0) - 1.0
        if image_width == 1:
            x_coords = torch.zeros([1]).unsqueeze(1)
        else:
            x_coords = 2.0 * torch.arange(image_width).unsqueeze(
                0).expand(image_height, image_width).float() / (image_width - 1.0) - 1.0


        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs.float() / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        image = torch.cat((coords.to(image.device), image), dim=1)

        return image


class CoordConv(nn.Module):

    r"""2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).
    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`
    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x


class CoordConvTranspose(nn.Module):

    r"""2D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).
    Args:
        Same as `torch.nn.ConvTranspose2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`
    Examples:
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv_tr(input)
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv_tr(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv_tr(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, with_r=False):
        super(CoordConvTranspose, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_tr_layer(x)

        return x


class CoordConvNet(nn.Module):

    r"""Improves 2D Convolutions inside a ConvNet by processing extra
    coordinate information as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).
    This module adds coordinate information to inputs of each 2D convolution
    module (`torch.nn.Conv2d`).
    Assumption: ConvNet Model must contain single `Sequential` container
    (`torch.nn.modules.container.Sequential`).
    Args:
        cnn_model: A ConvNet model that must contain single `Sequential`
            container (`torch.nn.modules.container.Sequential`).
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: Same as the input of the model.
        - Output: A list that contains all outputs (including
            intermediate outputs) of the model.
    Examples:
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = cnn_model(input)
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = cnn_model(input)
        >>> device = torch.device("cuda:0")
        >>> cnn_model = ...
        >>> cnn_model = CoordConvNet(cnn_model, True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> outputs = cnn_model(input)
    """

    def __init__(self, cnn_model, with_r=False):
        super(CoordConvNet, self).__init__()

        self.with_r = with_r

        self.cnn_model = cnn_model
        self.__get_model()
        self.__update_weights()

        self.coord_adder = AddCoordinates(self.with_r)

    def __get_model(self):
        for module in list(self.cnn_model.modules()):
            if module.__class__ == torch.nn.modules.container.Sequential:
                self.cnn_model = module
                break

    def __update_weights(self):
        coord_channels = 2
        if self.with_r:
            coord_channels += 1

        for l in list(self.cnn_model.modules()):
            if l.__str__().startswith('Conv2d'):
                weights = l.weight.data

                out_channels, in_channels, k_height, k_width = weights.size()

                coord_weights = torch.zeros(out_channels, coord_channels,
                                            k_height, k_width)

                weights = torch.cat((coord_weights.to(weights.device),
                                     weights), dim=1)
                weights = nn.Parameter(weights)

                l.weight = weights
                l.in_channels += coord_channels

    def __get_outputs(self, x):
        outputs = []
        for layer_name, layer in self.cnn_model._modules.items():
            if layer.__str__().startswith('Conv2d'):
                x = self.coord_adder(x)
            x = layer(x)
            outputs.append(x)

        return outputs

    def forward(self, x):
        return self.__get_outputs(x)


class AddHeatmap(object):
    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, image, heatmap):
        batch_size, _, image_height, image_width = image.size()
        _, _, heatmap_height, heatmap_width = heatmap.size()
        stride_height = int(heatmap_height/image_height)
        stride_width = int(heatmap_width/image_width)
        if stride_width != 1:
            heatmap_down = F.max_pool2d(heatmap, kernel_size=(stride_height, stride_width), stride=(stride_height, stride_width))
        else:
            heatmap_down = heatmap
        image = torch.cat((heatmap_down.to(image.device), image), dim=1)
        return image


class HeatmapConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 heatmap_dim=1):
        super(HeatmapConv, self).__init__()

        in_channels += heatmap_dim
        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        # self.heatmap_adder = AddHeatmap()

    def forward(self, x, heatmap):
        # x = self.heatmap_adder(x, heatmap)
        x = self.conv_layer(torch.cat((x,heatmap),dim=1))
        return x