import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch3d.loss as py3d

class PcdDecoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 S


    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3
        
        '''
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3) # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1) # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine) # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # BG C N
    
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N

        fine = self.final_conv(feat) + center   # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine



    def recon_loss(self, ret, gt):
        coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs*g, -1, 3).contiguous()
        fine = fine.reshape(bs*g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs*g, -1, 3).contiguous()

        # loss_coarse_block = chamfer_distance(coarse, group_gt, norm=1)
        loss_fine_block = py3d.chamfer_distance(fine, group_gt, norm=1)[0]

        loss_recon =  loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):

        # reconstruction loss
        loss_recon = self.recon_loss(ret, gt)
        # kl divergence
        logits = ret[-1] # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device = gt.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean', log_target = True)

        return loss_recon, loss_klv
