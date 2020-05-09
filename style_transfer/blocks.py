import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_pad(kernel_size, stride, padding=nn.ReflectionPad1d):
    pad_l = (kernel_size - stride) // 2
    pad_r = (kernel_size - stride) - pad_l
    return padding((pad_l, pad_r))


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode = self.mode)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def ZeroPad1d(sizes):
    return nn.ConstantPad1d(sizes, 0)


def ConvLayers(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', use_bias=True):

    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)),
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride, bias=use_bias)]


def get_acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        # return [nn.InstanceNorm1d(norm_dim, affine=False)]  # for rt42!
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvBlock(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', dropout=None,
              norm='none', acti='lrelu', acti_first=False, use_bias=True, inplace=True):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = ConvLayers(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


class ResBlock(nn.Module):

    def __init__(self, kernel_size, channels, stride=1, pad_type='zero', norm='none', acti='relu'):
        super(ResBlock, self).__init__()
        layers = []
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti)
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti='none')

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ShallowResBlock(nn.Module):

    def __init__(self, kernel_size, channels, stride=1, pad_type='zero', norm='none', acti='relu', inplace=True):
        super(ShallowResBlock, self).__init__()
        layers = []
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, inplace=inplace)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActiFirstResBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', norm='none', acti='lrelu'):
        super(ActiFirstResBlock, self).__init__()

        self.learned_shortcut = (in_channels != out_channels)
        self.c_in = in_channels
        self.c_out = out_channels
        self.c_mid = min(in_channels, out_channels)

        layers = []
        layers += ConvBlock(kernel_size, self.c_in, self.c_mid,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, acti_first=True)
        layers += ConvBlock(kernel_size, self.c_mid, self.c_out,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti, acti_first=True)
        self.conv_model = nn.Sequential(*layers)

        if self.learned_shortcut:
            self.conv_s = nn.Sequential(*ConvBlock(kernel_size, self.c_in, self.c_out,
                                        stride=stride, pad_type='zero',
                                        norm='none', acti='none', use_bias=False))

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_model(x)
        out = x_s + dx
        return out


class BottleNeckResBlock(nn.Module):
    def __init__(self, kernel_size, c_in, c_mid, c_out, stride=1, pad_type='reflect', norm='none', acti='lrelu'):
        super(BottleNeckResBlock, self).__init__()

        self.learned_shortcut = (c_in != c_out)
        self.c_in = c_in
        self.c_out = c_out
        self.c_mid = c_mid

        layers = []
        layers += ConvBlock(kernel_size, self.c_in, self.c_mid,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti)
        layers += ConvBlock(kernel_size, self.c_mid, self.c_out,
                            stride=stride, pad_type=pad_type,
                            norm='none', acti='none') # !! no norm here
        self.conv_model = nn.Sequential(*layers)

        if self.learned_shortcut:
            self.conv_s = nn.Sequential(*ConvBlock(kernel_size, self.c_in, self.c_out,
                                        stride=stride, pad_type='zero',
                                        norm='none', acti='none', use_bias=False))

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_model(x)
        out = x_s + dx
        return out


