import torch
import torch.nn as nn


def softsign(x):
    return x/(1+torch.abs(x))


class SPRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(45, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )

    def forward(self, pair_feat):
        return self.layers(pair_feat)


class DeepDDG(nn.Module):
    def __init__(self, neighbor_num=15) -> None:
        super().__init__()
        self.spr_net = SPRNet()
        self.after_spr = nn.Sequential(
            nn.Linear(neighbor_num*20, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, pair_feat):
        # B, N, D -> B, N, 20
        B, N, _ = pair_feat.shape
        x = self.spr_net(pair_feat)
        x = x.view(B, N*20)
        y = softsign(self.after_spr(x))
        return y


if __name__ == '__main__':
    deepddg = DeepDDG()
    batch_size = 32
    neighbour_number = 15
    feat_d = 45
    inp = torch.rand((batch_size, neighbour_number, feat_d))
    target = torch.rand((batch_size, 1))
    cal_loss = nn.MSELoss()

    pred = deepddg(inp)

    loss = cal_loss(pred, target)

    print(loss)
