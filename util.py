"""Fixed-point activation decomposition used by CS-ADS.

The decomposition estimates a more sanitary direction ``s`` and a more
backdoored direction ``b`` from the original/relieved activation pair."""

import torch
from copy import deepcopy


# -----------------------------------------------------------------------------
# Decomposition diagnostics
# -----------------------------------------------------------------------------
sim_all = []
#pos_all = []

# -----------------------------------------------------------------------------
# Public decomposition interface
# -----------------------------------------------------------------------------
def decompose_multi(h_vm, h_vr):

    """Apply CS-ADS decomposition to 2-D activations or batched 3-D sample-by-layer activations."""
    if len(h_vm.shape) == 2 and len(h_vr.shape) == 2: return _decompose_multi(h_vm, h_vr)
    if len(h_vm.shape) == 3 and len(h_vr.shape) == 3:
        num_sample = h_vm.shape[0]
        num_vec = h_vm.shape[1]
        h_vm = torch.reshape(h_vm, (-1, h_vm.shape[-1]))
        h_vr = torch.reshape(h_vr, (-1, h_vr.shape[-1]))
        #_decompose_multi(h_vm, h_vr)
        s, b, lamda_m, lamda_r, (l, l_m, l_r, sim1) = _decompose_multi(h_vm, h_vr)

        s = torch.reshape(s, (num_sample, num_vec, -1))
        b = torch.reshape(b, (num_sample, num_vec, -1))
        lamda_m = torch.reshape(lamda_m, (num_sample, num_vec, -1))
        lamda_r = torch.reshape(lamda_r, (num_sample, num_vec, -1))
        pass
        return s, b, lamda_m, lamda_r, (l, l_m, l_r, sim1)
    raise Exception("Not Effective Input")
    return _decompose_multi(h_vm, h_vr)

# -----------------------------------------------------------------------------
# Core fixed-point decomposition
# -----------------------------------------------------------------------------
def _decompose_multi(h_vm, h_vr):
    """Normalize paired activations and run the core fixed-point decomposition."""
    s_mat = torch.stack([h_vm, h_vr], dim=1).squeeze(dim=1)  # .transpose(0,1) (32,2, 4096)
    s_mat = s_mat#.to(torch.float32)

    s_mat_norm = s_mat.norm(dim=(1, 2)).unsqueeze(1).expand(-1, 2).unsqueeze(2)  # (32,2,4096) -> (32)
    s_mat = s_mat / s_mat_norm

    s_mat = s_mat.transpose(0, 1)
    s1 = s_mat[0]
    s2 = s_mat[1]

    s, b, lamda_m, lamda_r,sim1 = safe_decomp(s1, s2, imax=199) # 199

    l = lamda_r - lamda_m
    l_m = 1 - lamda_m
    l_r = 1 - lamda_r

    return s, b, lamda_m, lamda_r, (l, l_m, l_r, sim1)


def safe_decomp(vm, vr, imax=50):
    """Estimate sanitary/backdoored directions and mixture coefficients by fixed-point iteration."""
    def compute_s(vr, b, lamda_r):
        s = vr - (1 - lamda_r) * b
        s = s / lamda_r
        return s

    def compute_b(vm, s, lamda_m):
        b = vm - lamda_m * s
        b = b / (1 - lamda_m)
        return b

    def compute_lamda(v, s, b):
        comp_1 = s - b
        c = comp_1 * comp_1
        c = c.sum(dim=-1)
        lamda_out = (comp_1 * (v - b)).sum(dim=-1) / c
        lamda_out = lamda_out.unsqueeze(1)
        return lamda_out

    s = None
    b = None
    i = 0
    losses = []
    while i <= imax:
        # Initialization
        if i == 0:
            vm = vm.to(torch.float32)
            vr = vr.to(torch.float32)
            lamda_m = torch.ones(vm.shape[0], 1).to(vm) * 0.55 # 0.5 or 0.55 for Initialization.
            lamda_r = torch.ones(vr.shape[0], 1).to(vr) * 1 # 1 for Initialization. lamda_r > lamda_m must be met.
            s = deepcopy(vr)

        # Calculate cleaner s
        b = compute_b(vm, s, lamda_m)


        norm_b = b.norm(p=2, dim=-1).unsqueeze(-1)
        b = b / norm_b

        # Calculate poisoned s
        s = compute_s(vr, b, lamda_r)
        #s = torch.relu(s)
        norm_s = s.norm(p=2, dim=-1).unsqueeze(-1)
        #print(1 / norm_s)
        s = s / norm_s

        # calculate terms for
        lamda_r = compute_lamda(vr, s, b)
        lamda_m = compute_lamda(vm, s, b)

        sim1 = torch.cosine_similarity(s,b) #s,b

        sim_all.append(float(sim1.mean()))

        #test code
        loss = (vm - lamda_m * s - (1 - lamda_m) * b).norm(p=2) + (vr - lamda_r * s - (1 - lamda_r) * b).norm(p=2)
        losses.append(float(loss))

        i += 1

    s = s.to(torch.float16)
    b = b.to(torch.float16)



    return s, b, lamda_m, lamda_r, sim1
