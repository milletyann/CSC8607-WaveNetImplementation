import torch
from torch import nn
import numpy as np
from classes.CausalDilatedConv import DilatedConv
from classes.GatedActivationUnit import GatedActivationUnit
import torch.nn.functional as F
    
class WaveNet(nn.Module):
    def __init__(self, num_layers, num_blocks, conditioning_channels, classes=256, kernel_size=2, dilation_channels=32, residual_channels=32, skip_channels=256, bias=False):
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.classes = classes
        self.kernel_size = kernel_size
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.conditioning_channels = conditioning_channels
        self.bias = bias
        
        
        self.start_conv = nn.Conv1d(in_channels=1,
                                    out_channels=self.residual_channels,
                                    kernel_size=1,
                                    bias=self.bias)
        
        self.stack = nn.ModuleList()
        for _ in range(self.num_blocks):
            dilation = 1
            for _ in range(self.num_layers):
                layer = WaveNetLayer(residual_channels=self.residual_channels,
                                     dilation_channels=self.dilation_channels,
                                     kernel_size=self.kernel_size,
                                     dilation=dilation,
                                     conditioning_channels=self.conditioning_channels,
                                     bias=self.bias)
                
                self.stack.append(layer)
                
        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=self.residual_channels, 
                      out_channels=self.skip_channels, 
                      kernel_size=1, 
                      bias=self.bias),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.skip_channels, 
                      out_channels=self.classes, 
                      kernel_size=1, 
                      bias=self.bias)
        )
    
    def forward(self, x, h):
        residual = self.start_conv(x.float())
        skips = torch.zeros_like(residual)
        
        for layer in self.stack:
            layer_out = layer(residual, h)
            
            residual = residual + layer_out
            skips = skips + layer_out
        
            
        logit = self.end_conv(skips)
        
        return logit
    
    
    
class WaveNetLayer(nn.Module):
    def __init__(self, residual_channels, dilation_channels, kernel_size, dilation, conditioning_channels, bias):
        super(WaveNetLayer, self).__init__()
        self.dilated_conv = DilatedConv(in_channels=residual_channels,
                                        out_channels=dilation_channels, 
                                        kernel_size=kernel_size,
                                        dilation=dilation)
        self.gated_activation_unit = GatedActivationUnit(residual_channels, dilation_channels, conditioning_channels)
        self.conv1d = nn.Conv1d(in_channels=dilation_channels, 
                                out_channels=residual_channels, 
                                kernel_size=1, 
                                bias=bias)

    def forward(self, x, h):
        x = self.dilated_conv(x)
        x = self.gated_activation_unit(x, h)
        x = self.conv1d(x)
        return x