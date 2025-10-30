"""
Todo:
- add lerp to the linear block
- finish implementing the MoPeA module for multi-head
- consider giving mem linear bias by appending 1 to x
- do jit compilation
- make serial inference
- pray to the weight normalization gods that the matrix states do not aim for the heavens
"""


import torch as pt
from torch import nn


def mean_trace(tensor):
    diagonal = pt.diagonal(tensor, dim1=-2, dim2=-1)
    traces = pt.sum(diagonal, dim=-1)
    return traces.mean()


def sum_trace(tensor):
    diagonal = pt.diagonal(tensor, dim1=-2, dim2=-1)
    return diagonal.sum()


def s(x):
    return pt.where(x >= 0, x + 1, 1 / (1 - x))


def stable_max(x, dim=-1):
    x = s(x)
    y = x / pt.sum(x, dim=dim, keepdim=True)
    return y


def add_last_x(x, last_x=0):
    zeros_shape = x.shape[:-2] + (1,) + (x.shape[-1],)
    zeros_padding = x.new_zeros(zeros_shape)
    cat_x = pt.cat((zeros_padding, x), dim=-2)
    cat_x[..., 0, :] += last_x
    return cat_x


def mem_scan(all_dC_kk, all_dC_vk, all_dC_vv, all_R_kk, all_R_vk, all_R_vv, last_C_kk=0, last_C_vk=0, last_C_vv=0):
    C_kk = pt.zeros_like(all_dC_kk)
    C_vk = pt.zeros_like(all_dC_vk)
    C_vv = pt.zeros_like(all_dC_vv)

    for i in range(all_dC_kk.shape[1]):
        dC_kk = all_dC_kk[:, i]
        dC_vk = all_dC_vk[:, i]
        dC_vv = all_dC_vv[:, i]
        R_kk = all_R_kk[:, i]
        R_vk = all_R_vk[:, i]
        R_vv = all_R_vv[:, i]
        #R = 1  # temporarily let's ignore R for now

        last_C_kk = last_C_kk + (dC_kk - last_C_kk) * R_kk
        last_C_vk = last_C_vk + (dC_vk - last_C_vk) * R_vk
        last_C_vv = last_C_vv + (dC_vv - last_C_vv) * R_vv
        
        C_kk[:, i] = last_C_kk
        C_vk[:, i] = last_C_vk
        C_vv[:, i] = last_C_vv
    
    return C_kk, C_vk, C_vv


def get_mem(k, v, r, last_C_kk=0, last_C_vk=0, last_C_vv=0):
    r_k, r_v = pt.split(r, r.shape[-1] // 2, dim=-1)

    # i = batch
    # j = time
    # k = features_1
    # l = features_2
    dC_kk = pt.einsum("ijk, ijl -> ijkl", k, k)
    dC_vk = pt.einsum("ijk, ijl -> ijkl", v, k)
    dC_vv = pt.einsum("ijk, ijl -> ijkl", v, v)
    R_kk = pt.einsum("ijk, ijl -> ijkl", r_k, r_k)
    R_vk = pt.einsum("ijk, ijl -> ijkl", r_v, r_k)
    R_vv = pt.einsum("ijk, ijl -> ijkl", r_v, r_v)
    
    C_kk, C_vk, C_vv = mem_scan(dC_kk, dC_vk, dC_vv, R_kk, R_vk, R_vv, last_C_kk, last_C_vk, last_C_vv)

    # i = batch
    # j = time
    # k = B_out_features
    # l = common
    # m = A_in_features
    memory = pt.einsum("ijkl, ijlm -> ijkm", C_vk, pt.linalg.inv(C_kk))
    
    # i = batch
    # j = time
    # k = memory cols
    # l = memory rows
    # m = value features
    mpi_loss = pt.abs(mean_trace(C_vv - pt.einsum("ijkl, ijml -> ijkm", memory, C_vk)))
    #mpi_loss = sum_trace(C_vv) - 2 * sum_trace(pt.einsum("ijkl, ijmn -> ijkm", memory, C_vk)) + sum_trace(pt.einsum("ijkl, ijlm, ijnm -> ij", memory, C_kk, memory))
    
    return memory, C_kk, C_vk, C_vv, mpi_loss


def get_mopea(q, k, v, r, last_C_kk=0, last_C_vk=0, last_C_vv=0):
    memory, C_kk, C_vk, C_vv, mpi_loss = get_mem(k, v, r, last_C_kk, last_C_vk, last_C_vv)
    out_x = pt.einsum("ijkm, ijm -> ijk", memory, q)
    return out_x, C_kk, C_vk, C_vv, mpi_loss


def get_qkva(q, k, v, m=0.1):
    q_len = q.shape[1]
    cols = pt.arange(q_len, device=q.device)[None, :]
    rows = pt.arange(q_len, device=q.device)[:, None]
    pos_bias = (cols - rows) * m

    # i = batch
    # j = q time
    # k = k time
    # l = features
    attention_matrix = pt.einsum("ijl, ikl -> ijk", q, k)
    attention_matrix = attention_matrix / q.shape[-1] ** 0.5 + pos_bias
    attention_matrix = pt.tril(attention_matrix)
    attention_matrix = stable_max(attention_matrix)
    
    # i = batch
    # j = q time
    # k = k time
    # l = v features
    retrieved_v = pt.einsum("ijk, ikl -> ijl", attention_matrix, v)
    return retrieved_v


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
    def __init__(self, in_features, mem_features=None, heads=1, r_range=(0.1, 0.99)):
        super().__init__()
        
        if mem_features is None:
            mem_features = in_features
        r_min, r_max = r_range
        self.r_scale = lambda x: r_min + (r_max - r_min) * x

        self.layer_norm = nn.LayerNorm(in_features)
        self.qkv_layer = LerpLinear(in_features, mem_features, 3)
        self.r_layer = LerpLinear(in_features, 2 * mem_features, 1)
        self.out_linear = nn.Linear(mem_features, in_features)

    def forward(self, x, last_x=0, last_C_kk=0, last_C_vk=0, last_C_vv=0):
        cat_x = add_last_x(x, last_x)
        norm_x = self.layer_norm(cat_x)

        norm_last_x = norm_x[:, :-1]
        norm_curr_x = norm_x[:, 1:]

        q, k, v = self.qkv_layer(norm_curr_x, norm_last_x)
        (r,) = self.r_layer(norm_curr_x, norm_last_x)
        r = self.r_scale(r)
        
        mope_out_x, C_kk, C_vk, C_vv, mpi_loss = get_mopea(q, k, v, r, last_C_kk, last_C_vk, last_C_vv)
        """
        qkv_out_x = get_qkva(q, k, v)
        mem_loss = pt.mean((mope_out_x - qkv_out_x) ** 2)
        out_x = qkv_out_x
        mem_loss = 0
        """
        out_x = mope_out_x
        mem_loss = mpi_loss

        dx = self.out_linear(out_x)
        y = x + dx

        return y, C_kk[:, -1], C_vk[:, -1], C_vv[:, -1], mem_loss


class Block(nn.Module):
    def __init__(self, in_features, mem_features=None, heads=1, r_range=(0.1, 0.99)):
        super().__init__()
        
        self.mopea_layer = MoPeAttention(in_features, mem_features, heads, r_range)

        self.linear_block = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x, last_x=0, last_C_kk=0, last_C_vk=0, last_C_vv=0):
        
        x, C_kk, C_vk, C_vv, mem_loss = self.mopea_layer(x, last_x, last_C_kk, last_C_vk, last_C_vv)

        dx = self.linear_block(x)
        y = x + dx
        
        return y, C_kk, C_vk, C_vv, mem_loss


class MoPeAModel(nn.Module):
    def __init__(self, in_features, mid_features, out_features, mem_features=None, heads=1, r_range=(0.1, 0.99), depth=1):
        super().__init__()

        self.in_linear = nn.Linear(in_features, mid_features)
        self.layer_norm1 = nn.LayerNorm(mid_features)
        self.layer_norm2 = nn.LayerNorm(mid_features)
        self.out_linear = nn.Linear(mid_features, out_features)

        self.blocks = nn.ModuleList([
            Block(mid_features, mem_features, heads, r_range)
            for _ in range(depth)
        ])

    def forward(self, x):
        
        x = self.in_linear(x)
        x = self.layer_norm1(x)

        C_kk = []
        C_vk = []
        C_vv = []
        for block in self.blocks:
            x, block_C_kk, block_C_vk, block_C_vv, mem_loss = block(x)
            C_kk.append(block_C_kk)
            C_vk.append(block_C_vk)
            C_vv.append(block_C_vv)
        
        x = self.layer_norm2(x)
        y = self.out_linear(x)
        y = stable_max(y)

        return y, C_kk, C_vk, C_vv, mem_loss
        

if __name__ == "__main__":
    x = pt.randn(32, 16, 8)
    last_x = pt.randn(32, 16, 8)

    qkv_layer = LerpLinear(8, 4, 3)
    qkv = qkv_layer(x, last_x)
    print(f"LerpLinear\n\toutput shape: {qkv.shape}\n")

    mopea = MoPeAttention(8, 4)
    y, C_kk, C_vk, C_vv, mem_loss = mopea(x)
    print(f"MoPeAttention\n\toutput shape: {y.shape}\n\tC_kk shape: {C_kk.shape}\n\tC_vk shape: {C_vk.shape}\n\tC_vv shape: {C_vv.shape}\n")
    
    block = Block(8, 4)
    y, C_kk, C_vk, C_vv, mem_loss = block(x)
    print(f"Block\n\toutput shape: {y.shape}\n\tC_kk shape: {C_kk.shape}\n\tC_vk shape: {C_vk.shape}\n\tC_vv shape: {C_vv.shape}\n")

    mopea_model = MoPeAModel(8, 16, 8, 4, depth=2)
    y, C_kk, C_vk, C_vv, mem_loss = mopea_model(x)
    print(f"MoPeA Model\n\toutput shape: {y.shape}\n")