import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class Generator(nn.Module):
    """
    Generator network. If elliptical == False,
        G(z|b) = z + b,
    If elliptical == True,
        G(z,\\xi|b) = \\xi z + b.
    """
    def __init__(self, p, elliptical=False):
        """
        Args:
            p: number. p is the dimension of samples.
            elliptical: boolean.
        """
        super(Generator, self).__init__()
        self.p = p
        self.bias = nn.Parameter(torch.zeros(self.p))
        self.elliptical = elliptical

    def forward(self, z, xi=None):
        if self.elliptical:
            z = xi * z
        x = z + self.bias
        x = x.view(-1, self.p)
        return x


class GeneratorXi(nn.Module):
    """
    Generator using for elliptical distribution. An Elliptical distribution admits a 
    representation
        X = \\theta + \\xiAU,
    where U is uniformly distributed on the unit sphere {u\\in\\mathbb{R}^p:\\|u\\|_2 = 1}
    and \\xi\\geq 0 is a r.v. indenpendent of U that determines the shape of elliptical
    distribution. The center and the scatter matrix are \theta and \\Sigma = AA^T.

    GeneratorXi takes a random input and outputs samples with distribution \\xi.
    """
    def __init__(self, input_dim, hidden_units):
        """
        Args:
            input_dim: Number. Input dimension for \\xi network.
            hidden_units: A list of hidden units.
                          e.g. g_hidden_units = [24, 12, 8], then \\xi network has 
                          structure: input_dim - 24 - 12 - 8 - 1.
        """
        super(GeneratorXi, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.layers = len(self.hidden_units)
        self.map = self._make_layers()

    def _make_layers(self):
        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                layer_list += [
                    ('lyr%d'%(lyr+1), 
                        nn.Linear(self.input_dim, self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), nn.LeakyReLU(0.2))
                    ]
            else:
                layer_list += [
                    ('lyr%d'%(lyr+1), 
                        nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), nn.LeakyReLU(0.2))
                    ]
        layer_list += [('lyr%d'%(self.layers+1), nn.Linear(self.hidden_units[-1], 1))]
        return nn.Sequential(OrderedDict(layer_list))

    def forward(self, z):
        xi = self.map(z.view(-1, self.input_dim))
        xi = torch.abs(xi)
        return xi  


class Discriminator(nn.Module):
    """
    Discriminator network.
    """
    def __init__(self, p, hidden_units, activation_1):
        """
        Args:
            p: sample's dimension.
            hidden_units: a list of hidden units for Discriminator, 
                          e.g. d_hidden_units=[10, 5], then the discrimintor has
                          structure p (input) - 10 - 5 - 1 (output).
            activation_1: 'Sigmoid', 'ReLU' or 'LeakyReLU'. The first activation 
                          function after the input layer. Especially when 
                          true_type == 'Cauchy', Sigmoid activation is preferred.
        """
        super(Discriminator, self).__init__()
        self.p = p
        self.arg_1 = {'negative_slope':0.2} if (activation_1 == 'LeakyReLU') else {}
        self.activation_1 = activation_1
        self.layers = len(hidden_units)
        self.hidden_units = hidden_units
        self.feature = self._make_layers()
        self.map_last = nn.Linear(self.hidden_units[-1], 1)
        
    def forward(self, x):
        x = self.feature(x.view(-1,self.p)) 
        d = self.map_last(x).squeeze()
        return x, d

    def _make_layers(self):
        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                layer_list += [
                    ('lyr%d'%(lyr+1), nn.Linear(self.p, self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), getattr(nn, self.activation_1)(**self.arg_1))
                    ]
            else:
                layer_list += [
                    ('lyr%d'%(lyr+1), 
                        nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), nn.LeakyReLU(0.2))]
        return nn.Sequential(OrderedDict(layer_list))


