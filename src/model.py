"""
Todo:
- finish implementing the MoPeA module for multi-head
- consider giving mem linear bias by appending 1 to x
- do jit compilation
- make serial inference
- pray to the weight normalization gods that the matrix states do not aim for the heavens
"""


import torch as pt
from torch import nn


def add_last_x(x, last_x=0):
    zeros_shape = x.shape[:-2] + (1,) + (x.shape[-1],)
    zeros_padding = x.new_zeros(zeros_shape)
    cat_x = pt.cat((zeros_padding, x), dim=-2)
    cat_x[..., 0, :] += last_x
    return cat_x


def mem_scan(all_dA, all_dB, all_R, all_W, last_A=0, last_B=0):
    A = pt.zeros_like(all_dA)
    B = pt.zeros_like(all_dB)

    for i, (dA, dB, R, W) in enumerate(zip(all_dA, all_dB, all_R, all_W)):
        last_A = last_A * R + dA * W
        last_B = last_B * R + dB * W
        
        A[i] = last_A
        B[i] = last_B

    return A, B


def s(x):
    return pt.where(x >= 0, x + 1, 1 / (1 - x))


class LerpLinear(nn.Module):
    def __init__(self, in_features, out_features, branches=4):
        super().__init__()

        self.lerp_weight = nn.Parameter(pt.rand(branches, in_features))
        self.qkv_weight = nn.Parameter(pt.rand(branches, out_features, in_features))

    def forward(self, x, last_x):
        # shape (branch, batch, time, in_features)
        x = x[None, :, :, :]
        last_x = last_x[None, :, :, :]

        dx = x - last_x
        lerp_w = nn.functional.sigmoid(self.lerp_weight)[:, None, None, :]
        linear_x = last_x + dx * lerp_w

        # i = branch
        # j = batch
        # k = time
        # l = out_features
        # m = in_features
        out = pt.einsum("ilm,ijkm->ijkl", self.qkv_weight, linear_x)
        return out


class MoPeAttention(nn.Module):
    def __init__(self, in_features, mem_features=None, heads=1, min_r=0.99):
        super().__init__()
        
        if mem_features is None:
            mem_features = in_features
        self.min_r = min_r

        self.layer_norm = nn.LayerNorm(in_features)
        self.qkv_layer = LerpLinear(in_features, mem_features, 3)
        self.rw_layer = LerpLinear(in_features, 2 * mem_features, 2)
        self.out_linear = nn.Linear(mem_features, in_features)

    def forward(self, x, last_x=0, last_A=0, last_B=0):
        cat_x = add_last_x(x, last_x)
        norm_x = self.layer_norm(cat_x)

        norm_last_x = norm_x[:, :-1]
        norm_curr_x = norm_x[:, 1:]

        q, k, v = self.qkv_layer(norm_curr_x, norm_last_x)
        r, w = self.rw_layer(norm_curr_x, norm_last_x)
        r = self.min_r + (1 - self.min_r) * pt.nn.functional.sigmoid(r)
        w = s(w)

        r_k, r_v = pt.split(r, r.shape[-1] // 2, dim=-1)
        w_k, w_v = pt.split(w, w.shape[-1] // 2, dim=-1)

        # i = batch
        # j = time
        # k = key_features
        # l = value_features
        dA = pt.einsum("ijk, ijl -> ijkl", k, k)
        dB = pt.einsum("ijk, ijl -> ijkl", v, k)
        R = pt.einsum("ijk, ijl -> ijkl", r_v, r_k)
        W = pt.einsum("ijk, ijl -> ijkl", w_v, w_k)
        
        A, B = mem_scan(dA, dB, R, W, last_A, last_B)

        # i = batch
        # j = time
        # k = B_out_features
        # l = common
        # m = A_in_features
        memory = pt.einsum("ijkl, ijlm -> ijkm", B, pt.linalg.inv(A))
        out_x = pt.einsum("ijkm, ijm -> ijk", memory, q)

        dx = self.out_linear(out_x)
        y = x + dx

        return y, A[:, -1], B[:, -1]


class Block(nn.Module):
    def __init__(self, in_features, mem_features=None, heads=1, min_r=0.99):
        super().__init__()
        
        self.mopea_layer = MoPeAttention(in_features, mem_features, heads, min_r)

        self.linear_block = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x, last_x=0, last_A=0, last_B=0):
        
        x, A, B = self.mopea_layer(x, last_x, last_A, last_B)

        dx = self.linear_block(x)
        y = x + dx
        
        return y, A[:, -1], B[:, -1]


class MoPeAModel(nn.Module):
    def __init__(self, in_features, mid_features, out_features, mem_features=None, heads=1, min_r=0.99, depth=1):
        super().__init__()

        self.in_linear = nn.Linear(in_features, mid_features)
        self.layer_norm1 = nn.LayerNorm(mid_features)
        self.layer_norm2 = nn.LayerNorm(mid_features)
        self.out_linear = nn.Linear(mid_features, out_features)

        self.blocks = nn.ModuleList([
            Block(mid_features, mem_features, heads, min_r)
            for _ in range(depth)
        ])

    def forward(self, x):
        
        x = self.in_linear(x)
        x = self.layer_norm1(x)

        for block in self.blocks:
            x, _, _ = block(x)
        
        x = self.layer_norm2(x)
        y = self.out_linear(x)

        return y

        

if __name__ == "__main__":
    x = pt.randn(32, 16, 8)
    last_x = pt.randn(32, 16, 8)

    qkv_layer = LerpLinear(8, 4, 3)
    qkv = qkv_layer(x, last_x)
    print(f"LerpLinear\n\toutput shape: {qkv.shape}\n")

    mopea = MoPeAttention(8, 4)
    y, A, B = mopea(x)
    print(f"MoPeAttention\n\toutput shape: {y.shape}\n\tA shape: {A.shape}\n\tB shape: {B.shape}\n")
    
    block = Block(8, 4)
    y, A, B = block(x)
    print(f"Block\n\toutput shape: {y.shape}\n\tA shape: {A.shape}\n\tB shape: {B.shape}\n")

    mopea_model = MoPeAModel(8, 16, 8, 4, depth=2)
    y = mopea_model(x)
    print(f"MoPeA Model\n\toutput shape: {y.shape}\n")
    