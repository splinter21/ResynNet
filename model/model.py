import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.ResynNet import *
import torch.nn.functional as F
# from model.vgg import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.resynnet = ResynNet()
        degrad_params = list(map(id, self.resynnet.degrad.parameters()))
        base_params = filter(lambda p: id(p) not in degrad_params, self.resynnet.parameters())
        params = [
            {"params": base_params, 'name':'flow', "lr": 0, "weight_decay": 1e-3},
            {"params": self.resynnet.degrad.parameters(), 'name':'degrad', "lr": 0, "weight_decay":0}
        ]
        self.optimG = AdamW(params)
        self.device()
        # self.vgg = VGGPerceptualLoss().to(device)
        if local_rank != -1:
            self.resynnet = DDP(self.resynnet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.resynnet.train()

    def eval(self):
        self.resynnet.eval()

    def device(self):
        self.resynnet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        if rank <= 0:
            if torch.cuda.is_available():
                self.resynnet.load_state_dict(convert(torch.load('{}/resynnet.pkl'.format(path))))
            self.resynnet.load_state_dict(convert(torch.load('{}/resynnet.pkl'.format(path), map_location='cpu')))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.resynnet.state_dict(),'{}/resynnet.pkl'.format(path))

    def inference(self, imgs, deg, scale=[4, 2, 1]):
        self.eval()
        merged, loss_cons = self.resynnet(imgs, deg=deg, scale=scale)
        return merged

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, lowres=None, blend=False):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        if np.random.uniform(0, 1) < 0.5 and training:
            scale = [1, 1, 1]
        else:
            scale = [4, 2, 1]
        merged, loss_cons = self.resynnet(imgs, lowres, scale=scale, training=training, blend=blend)
        alpha = 0.5
        loss_l1 = (merged - gt).abs().mean() * (1 - alpha)
        # loss_l1 += self.vgg(merged, gt).mean() * alpha
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons
            loss_G.backward()
            self.optimG.step()
        return merged, {
            'lowres': lowres,
            'flow': flow[1][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            }
