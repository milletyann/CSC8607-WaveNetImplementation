import torch
import torch.nn as nn
import torch.nn.init as init

class GatedActivationUnit(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim):
        super(GatedActivationUnit, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1, bias=False)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1, bias=False)
        self.v_f = nn.Parameter(torch.randn(out_channels, conditioning_dim))
        self.v_g = nn.Parameter(torch.randn(out_channels, conditioning_dim))
        
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.conv_filter.weight)
        init.xavier_uniform_(self.conv_gate.weight)

    def forward(self, x, h):
        conv_f = self.conv_filter(x)[:, :, :-1]
        conv_g = self.conv_gate(x)[:, :, :-1]
        v_f = (h @ self.v_f.T).unsqueeze(-1)
        v_g = (h @ self.v_g.T).unsqueeze(-1)
        
        f = conv_f + v_f
        g = conv_g + v_g
        return self.tanh(f) * self.sigmoid(g)
    