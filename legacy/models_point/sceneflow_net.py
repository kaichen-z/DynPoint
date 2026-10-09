import torch
from torch import nn
from .blocks import PeriodicEmbed, Conv2dBlock

class SceneFlowFieldNet(nn.Module):
    def __init__(self, N_freq_xyz=0, N_freq_t=0, output_dim=3, net_width=32, n_layers=3, activation='lrelu', norm='none'):
        super(SceneFlowFieldNet, self).__init__()
        N_input_channel_xyz = 3 + 3 * 2 * N_freq_xyz
        N_input_channel_t = 1 + 1 * 2 * N_freq_t
        N_input_channel = N_input_channel_xyz + N_input_channel_t 
        if N_freq_xyz == 0:
            xyz_embed = nn.Identity()
        else:
            xyz_embed = PeriodicEmbed(max_freq=N_freq_xyz, N_freq=N_freq_xyz)
        if N_freq_t == 0:
            t_embed = nn.Identity()
        else:
            t_embed = PeriodicEmbed(max_freq=N_freq_t, N_freq=N_freq_t)
        convs = [Conv2dBlock(N_input_channel, net_width, 1, 1, norm=norm, activation=activation)]
        for i in range(n_layers):
            convs.append(Conv2dBlock(net_width, net_width, 1, 1, norm=norm, activation=activation))
        self.forward_cnn = Conv2dBlock(net_width, output_dim, 1, 1, norm='none', activation='none')
        self.backwarf_cnn = Conv2dBlock(net_width, output_dim, 1, 1, norm='none', activation='none')
        self.convs = nn.Sequential(*convs)
        self.t_embed = t_embed
        self.xyz_embed = xyz_embed

    def forward(self, x, t=None):
        x = x.contiguous()
        xyz_embedded = self.xyz_embed(x)
        t_embedded = self.t_embed(t)
        input_feat = torch.cat([t_embedded, xyz_embedded], 1)
        output_feat = self.convs(input_feat)
        return self.forward_cnn(output_feat), self.backwarf_cnn(output_feat)
