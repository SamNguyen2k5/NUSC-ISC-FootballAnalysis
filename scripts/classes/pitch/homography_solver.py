import torch
import torch.nn as nn

class HomographySolver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, old_pts, new_pts):
        n_batch = old_pts.shape[0]
        # assert old_pts.shape[2:] == 3
        # assert new_pts.shape[2:] == 3

        new_xs, new_ys, new_zs = new_pts[:, :, 0], new_pts[:, :, 1], new_pts[:, :, 2]
        pz = new_zs.unsqueeze(-1) * old_pts
        px = new_xs.unsqueeze(-1) * old_pts
        py = new_ys.unsqueeze(-1) * old_pts

        zeros = torch.zeros_like(px)

        # Ah = 0
        A = torch.cat((
            torch.cat((pz, zeros, -px), dim=2),
            torch.cat((zeros, pz, -py), dim=2)
        ), dim=1)

        # Solution to Ah = 0 <-> last column of the V matrix (in the SVD decomposition)
        _, _, VT = torch.linalg.svd(A)
        h = VT[:, -1, :].unsqueeze(-1)                      
        H = h.reshape(n_batch, 3, 3)

        # print(H, H.shape)

        return H
