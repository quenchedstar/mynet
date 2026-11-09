import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import sparse as sp
except ImportError:
    sp = None

class Vssi_Ard_ADMMLayer_edge(nn.Module):
    """
    VSSI-ARD edge-domain network (vssi_ard_net/version_10_75).

    version_9_75 U-step; S-step: plain CG(8) in all stages, gated conditional
    correction only in FinalLayer (CG12 + SourceCorrectionNetV9).
    SUZ + baseline ARD (gamma/xi) unchanged.
    """

    def __init__(self, L, V, k_features, num_stages=7, num_admm=1, balance_scaler=1.0):
        super().__init__()
        self.L = L
        L_float = L.float()
        self.register_buffer("LT_L", torch.matmul(L_float.T, L_float))
        self.k_features = k_features
        self.num_stages = num_stages
        self.original_layer = OriginalLayer(self.L, k_features)

        self.shared_prox_net = EdgeProxNetCondV7(
            channels=k_features, num_stages=num_stages, hidden_channels=64, embed_dim=16
        )
        self.shared_lambda_scale_net = EdgeLambdaScaleNet(
            num_stages=num_stages, hidden_channels=32, embed_dim=16
        )
        self.final_s_corr_net = SourceCorrectionNetV9(
            k_features=k_features, num_stages=1, hidden_channels=64, embed_dim=16
        )

        self.admm_blocks = nn.ModuleList()
        self.ard_blocks = nn.ModuleList()

        for stage_idx in range(num_stages):
            v_operator_stage = VariationEdgeOperator(V)
            rho_stage = nn.Parameter(torch.tensor(100.0), requires_grad=True)
            lambda_scaler_stage = nn.Parameter(torch.tensor(10.0), requires_grad=True)
            stage_admm_layers = nn.ModuleList()
            for _ in range(num_admm):
                stage_admm_layers.append(
                    S_UpdateADMM(
                        L,
                        v_operator_stage,
                        rho_stage,
                        cg_iters=8,
                        balance_scaler=balance_scaler,
                        lt_l=self.LT_L,
                    )
                )
                stage_admm_layers.append(
                    U_UpdateADMM_Learnable(
                        v_operator_stage,
                        rho_stage,
                        lambda_scaler_stage,
                        k_features,
                        prox_net=self.shared_prox_net,
                        lambda_scale_net=self.shared_lambda_scale_net,
                        stage_index=stage_idx,
                    )
                )
                stage_admm_layers.append(Z_UpdateADMM(v_operator_stage, rho_stage))
            self.admm_blocks.append(stage_admm_layers)
            self.ard_blocks.append(
                nn.ModuleList(
                    [
                        GammaUpdateLayer(v_operator_stage),
                        XiUpdateLayer(L, v_operator_stage, k_features, lt_l=self.LT_L),
                    ]
                )
            )

        final_v_op = self.admm_blocks[-1][0].V
        final_rho = self.admm_blocks[-1][0].rho
        self.final_layer = FinalLayer(
            L,
            final_v_op,
            final_rho,
            balance_scaler=balance_scaler,
            lt_l=self.LT_L,
            corr_net=self.final_s_corr_net,
            stage_index=0,
            cg_iters=12,
        )

    def forward(self, x, return_all_stages=True):
        b_trans = x["B_trans"]
        eeg_data = self.original_layer(b_trans, self.admm_blocks[0][0].V.num_edges)
        stage_snapshots = [] if return_all_stages else None

        for i in range(self.num_stages):
            for layer in self.admm_blocks[i]:
                eeg_data = layer(eeg_data)
            for layer in self.ard_blocks[i]:
                eeg_data = layer(eeg_data)

            if return_all_stages:
                stage_snapshots.append({"S": eeg_data["S"].clone()})

        if return_all_stages:
            return stage_snapshots

        return {"S": eeg_data["S"]}

class OriginalLayer(nn.Module):
    def __init__(self, L, k_features):
        super().__init__()
        self.L = L.float()
        self.k_features = k_features

    def forward(self, b_trans, num_edges):
        b = b_trans.float().to(self.L.device)
        batch_size = b.shape[0]
        _, ns = self.L.shape
        k = self.k_features

        eeg_data = dict()
        eeg_data["K"] = k
        eeg_data["B"] = b
        eeg_data["S"] = torch.zeros((batch_size, ns, k), device=b.device)
        eeg_data["U"] = torch.zeros((batch_size, num_edges, k), device=b.device)
        eeg_data["Z"] = torch.zeros((batch_size, num_edges, k), device=b.device)
        eeg_data["xi"] = torch.ones((batch_size, num_edges, 1), device=b.device)
        eeg_data["g"] = torch.ones((batch_size, num_edges, 1), device=b.device)
        return eeg_data

class GammaUpdateLayer(nn.Module):
    def __init__(self, v_operator):
        super().__init__()
        self.V = v_operator

    def forward(self, x):
        s = x["S"]
        xi = x["xi"]
        k = x["K"]
        output1 = self.V(s)
        u_norm2 = torch.norm(output1, p=2, dim=2)
        output2 = torch.sqrt(k * xi)
        denominator = output2.squeeze(-1)
        output3 = torch.div(u_norm2, denominator + 1e-8)
        x["g"] = output3.unsqueeze(-1).clamp_(min=1e-5, max=1e6)
        return x

class XiUpdateLayer(nn.Module):
    def __init__(self, L, v_operator, k_features, cg_max_iter=20, cg_tolerance=1e-6, lt_l=None, num_random_probes=4):
        super().__init__()
        self.L = L
        self.V = v_operator
        self.k_features = k_features
        self.cg_max_iter = cg_max_iter
        self.cg_tolerance = cg_tolerance
        self.num_random_probes = num_random_probes
        self.LT_L = lt_l

    def forward(self, x):
        gamma = x["g"]
        xi = x["xi"]
        batch_size, num_edges, _ = gamma.shape
        ns = self.L.shape[1]
        device = gamma.device
        gamma_inv_values = 1.0 / (gamma + 1e-8)

        def A_matmul_closure(vec):
            term1 = torch.matmul(self.LT_L, vec)
            v_eff_vec = self.V(vec)
            g_v_vec = gamma_inv_values * v_eff_vec
            term2 = self.V.forward_transpose(g_v_vec)
            return term1 + term2

        probes = torch.randint(
            low=0,
            high=2,
            size=(batch_size, num_edges, self.num_random_probes),
            device=device,
        ).float() * 2 - 1

        y = self.V.forward_transpose(probes)
        y_reshaped = y.permute(0, 2, 1).reshape(batch_size * self.num_random_probes, ns)

        def batched_A_closure(vec_reshaped):
            original_shape_vec = vec_reshaped.view(batch_size, self.num_random_probes, ns).permute(0, 2, 1)
            result = A_matmul_closure(original_shape_vec)
            return result.permute(0, 2, 1).reshape(batch_size * self.num_random_probes, ns)

        x_reshaped = conjugate_gradient(
            batched_A_closure,
            y_reshaped,
            max_iter=self.cg_max_iter,
            tolerance=self.cg_tolerance,
        )

        cg_solution = x_reshaped.view(batch_size, self.num_random_probes, ns).permute(0, 2, 1)

        if torch.isnan(cg_solution).any():
            cg_solution = torch.nan_to_num(cg_solution, nan=0.0, posinf=0.0, neginf=0.0)

        vx = self.V(cg_solution)
        diag_estimate = torch.mean(vx * probes, dim=-1)
        update_term = gamma * (diag_estimate.unsqueeze(-1) - 1.0)
        new_xi = xi + 0.1 * update_term
        x["xi"] = new_xi.clamp_(min=1e-8)
        return x

class S_UpdateADMM(nn.Module):
    def __init__(self, L, v_operator, rho, cg_iters=20, cg_tolerance=1e-6, balance_scaler=1.0, lt_l=None):
        super().__init__()
        self.L = L
        self.V = v_operator
        self.rho = rho
        self.cg_iters = cg_iters
        self.cg_tolerance = cg_tolerance
        self.scaler = balance_scaler
        self.LT_L = lt_l

    def forward(self, x):
        b = x["B"]
        u = x["U"]
        z = x["Z"]

        def A_matmul_closure(s_vec):
            term1 = torch.matmul(self.LT_L, s_vec)
            vs_vec = self.V(s_vec)
            vtv_s_vec = self.V.forward_transpose(vs_vec)
            regularization_term = self.scaler * self.rho * vtv_s_vec
            return term1 + regularization_term

        lt_b = torch.matmul(self.L.T, b)
        vt_res = self.V.forward_transpose(u - z / self.rho)
        b_vec = lt_b + self.scaler * self.rho * vt_res

        new_s = conjugate_gradient(
            A_matmul_closure,
            b_vec,
            max_iter=self.cg_iters,
            tolerance=self.cg_tolerance,
        )
        x["S"] = new_s
        return x

class SourceCorrectionNetV9(nn.Module):
    """Gated voxel-wise S correction conditioned on ARD state; zero-init -> S_cg."""

    def __init__(self, k_features, num_stages=7, hidden_channels=32, embed_dim=16):
        super().__init__()
        self.k_features = k_features
        self.stage_embed = nn.Embedding(num_stages, embed_dim)
        input_dim = 4 * k_features + 1 + embed_dim
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.to_res = nn.Linear(hidden_channels, k_features)
        self.to_gate = nn.Linear(hidden_channels, k_features)
        nn.init.zeros_(self.to_res.weight)
        nn.init.zeros_(self.to_res.bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, s_cg, b_vec, xi, gamma, rho, stage_idx, v_operator):
        batch_size = s_cg.shape[0]
        k = self.k_features
        if not torch.is_tensor(rho):
            rho_t = torch.tensor(float(rho), device=s_cg.device, dtype=s_cg.dtype)
        else:
            rho_t = rho.to(device=s_cg.device, dtype=s_cg.dtype)
        log_rho = torch.log(torch.relu(rho_t) + 1e-8).view(1, 1, 1).expand(batch_size, s_cg.shape[1], 1)
        log_xi_e = torch.log(xi.clamp(min=1e-8)).expand(-1, -1, k)
        log_gamma_e = torch.log(gamma.clamp(min=1e-8)).expand(-1, -1, k)
        xi_v = v_operator.forward_transpose(log_xi_e)
        gamma_v = v_operator.forward_transpose(log_gamma_e)
        stage_emb = self.stage_embed(stage_idx).unsqueeze(1).expand(-1, s_cg.shape[1], -1)
        feat = torch.cat([s_cg, b_vec, xi_v, gamma_v, log_rho, stage_emb], dim=-1)
        h = self.feat(feat)
        gate = torch.sigmoid(self.to_gate(h))
        res = self.to_res(h)
        return s_cg + gate * res

class S_UpdateADMM_Corrected(S_UpdateADMM):
    def __init__(
        self,
        L,
        v_operator,
        rho,
        balance_scaler=1.0,
        lt_l=None,
        corr_net=None,
        stage_index=0,
        cg_iters=8,
        cg_tolerance=1e-6,
    ):
        super().__init__(
            L,
            v_operator,
            rho,
            cg_iters=cg_iters,
            cg_tolerance=cg_tolerance,
            balance_scaler=balance_scaler,
            lt_l=lt_l,
        )
        self.corr_net = corr_net
        self.stage_index = stage_index

    def forward(self, x):
        b = x["B"]
        u = x["U"]
        z = x["Z"]
        batch_size = b.shape[0]

        def A_matmul_closure(s_vec):
            term1 = torch.matmul(self.LT_L, s_vec)
            vs_vec = self.V(s_vec)
            vtv_s_vec = self.V.forward_transpose(vs_vec)
            regularization_term = self.scaler * self.rho * vtv_s_vec
            return term1 + regularization_term

        lt_b = torch.matmul(self.L.T, b)
        vt_res = self.V.forward_transpose(u - z / self.rho)
        b_vec = lt_b + self.scaler * self.rho * vt_res

        s_cg = conjugate_gradient(
            A_matmul_closure,
            b_vec,
            max_iter=self.cg_iters,
            tolerance=self.cg_tolerance,
        )

        if self.corr_net is not None:
            stage_idx = torch.full(
                (batch_size,),
                self.stage_index,
                device=b.device,
                dtype=torch.long,
            )
            x["S"] = self.corr_net(
                s_cg,
                b_vec,
                x["xi"],
                x["g"],
                self.rho,
                stage_idx,
                self.V,
            )
        else:
            x["S"] = s_cg
        return x

class U_UpdateADMM_Learnable(nn.Module):
    def __init__(
        self,
        v_operator,
        rho,
        lambda_scaler,
        k_features,
        prox_net,
        lambda_scale_net,
        stage_index,
    ):
        super().__init__()
        self.V = v_operator
        self.rho = rho
        self.lambda_scaler = lambda_scaler
        self.shrink_net = prox_net
        self.lambda_scale_net = lambda_scale_net
        self.stage_index = stage_index

    def forward(self, x):
        s = x["S"]
        z = x["Z"]
        xi = x["xi"]
        k = x["K"]
        gamma = x["g"]
        batch_size = s.shape[0]
        stage_idx = torch.full(
            (batch_size,),
            self.stage_index,
            device=s.device,
            dtype=torch.long,
        )

        w = torch.sqrt(k * xi)
        output1 = self.V(s)
        output2 = (1.0 / self.rho) * z
        y = output1 + output2
        lam = torch.div(w, self.rho)
        scaler = torch.relu(self.lambda_scaler).to(device=lam.device, dtype=lam.dtype)
        edge_scale = self.lambda_scale_net(y, xi, gamma, self.rho, stage_idx)
        adjusted_lam = lam * scaler * edge_scale
        u_math = block_soft_thresholding_torch(y, adjusted_lam)
        u_delta = self.shrink_net(y, adjusted_lam, xi=xi, gamma=gamma, rho=self.rho, stage_idx=stage_idx)
        x["U"] = u_math + u_delta
        return x

class Z_UpdateADMM(nn.Module):
    def __init__(self, v_operator, rho):
        super().__init__()
        self.V = v_operator
        self.rho = rho

    def forward(self, x):
        s = x["S"]
        z = x["Z"]
        u = x["U"]
        output = torch.sub(self.V(s), u)
        x["Z"] = z + self.rho * output
        return x

class FinalLayer(S_UpdateADMM_Corrected):
    def __init__(
        self,
        L,
        v_operator,
        rho,
        balance_scaler=1.0,
        lt_l=None,
        corr_net=None,
        stage_index=0,
        cg_iters=8,
    ):
        super().__init__(
            L,
            v_operator,
            rho,
            balance_scaler=balance_scaler,
            lt_l=lt_l,
            corr_net=corr_net,
            stage_index=stage_index,
            cg_iters=cg_iters,
        )

class EdgeLambdaScaleNet(nn.Module):
    """Per-edge multiplicative scale on ADMM threshold lambda (init ~1)."""

    def __init__(self, num_stages=7, hidden_channels=32, embed_dim=16, scale_gain=0.15):
        super().__init__()
        self.num_stages = num_stages
        self.scale_gain = scale_gain
        self.stage_embed = nn.Embedding(num_stages, embed_dim)
        input_dim = 3 + embed_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Y, xi, gamma, rho, stage_idx):
        if not torch.is_tensor(rho):
            rho_t = torch.tensor(float(rho), device=Y.device, dtype=Y.dtype)
        else:
            rho_t = rho.to(device=Y.device, dtype=Y.dtype)
        log_rho = torch.log(torch.relu(rho_t) + 1e-8).view(1, 1, 1).expand(Y.shape[0], Y.shape[1], 1)
        log_xi = torch.log(xi.clamp(min=1e-8))
        log_gamma = torch.log(gamma.clamp(min=1e-8))
        y_norm = torch.norm(Y, p=2, dim=2, keepdim=True)
        stage_emb = self.stage_embed(stage_idx).unsqueeze(1).expand(-1, Y.shape[1], -1)
        feat = torch.cat([log_xi, log_gamma, log_rho, y_norm, stage_emb], dim=-1)
        delta = self.net(feat)
        return 1.0 + self.scale_gain * torch.tanh(delta)

class EdgeProxNetCondV7(nn.Module):
    """Shared conditional proximal correction with stage embedding."""

    def __init__(self, channels, num_stages=7, hidden_channels=64, embed_dim=16):
        super().__init__()
        self.channels = channels
        self.stage_embed = nn.Embedding(num_stages, embed_dim)
        input_dim = channels + 1 + 3 + embed_dim
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.to_res = nn.Linear(hidden_channels, channels)
        self.to_gate = nn.Linear(hidden_channels, channels)

        nn.init.zeros_(self.to_res.weight)
        nn.init.zeros_(self.to_res.bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, Y, threshold_map, *, xi, gamma, rho, stage_idx):
        if not torch.is_tensor(rho):
            rho_t = torch.tensor(float(rho), device=Y.device, dtype=Y.dtype)
        else:
            rho_t = rho.to(device=Y.device, dtype=Y.dtype)
        log_rho = torch.log(torch.relu(rho_t) + 1e-8).view(1, 1, 1).expand(Y.shape[0], Y.shape[1], 1)
        log_xi = torch.log(xi.clamp(min=1e-8))
        log_gamma = torch.log(gamma.clamp(min=1e-8))
        stage_emb = self.stage_embed(stage_idx).unsqueeze(1).expand(-1, Y.shape[1], -1)

        x = torch.cat([Y, threshold_map, log_xi, log_gamma, log_rho, stage_emb], dim=-1)
        h = self.feat(x)
        gate = torch.sigmoid(self.to_gate(h))
        res = self.to_res(h)
        return gate * res

def conjugate_gradient(A_matmul_closure, B, max_iter=20, tolerance=1e-6):
    dims_to_sum = list(range(1, B.dim()))

    x = torch.zeros_like(B)
    r = B - A_matmul_closure(x)
    p = r.clone()
    rs_old = torch.sum(r * r, dim=dims_to_sum, keepdim=True)

    for _ in range(max_iter):
        Ap = A_matmul_closure(p)
        p_Ap = torch.sum(p * Ap, dim=dims_to_sum, keepdim=True)
        alpha = rs_old / (p_Ap + 1e-8)

        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = torch.sum(r_new * r_new, dim=dims_to_sum, keepdim=True)

        if torch.isnan(rs_new).any() or torch.isinf(rs_new).any():
            return x

        x = x_new
        r = r_new

        if torch.all(rs_new < tolerance):
            break

        beta = rs_new / (rs_old + 1e-8)
        p = r + beta * p
        rs_old = rs_new

    return x

def block_soft_thresholding_torch(y_matrix, threshold_per_row):
    epsilon_norm = 1e-12
    norm_y_rows = torch.norm(y_matrix, p=2, dim=2, keepdim=True)
    scale_factors = 1.0 - threshold_per_row / (norm_y_rows + epsilon_norm)
    scale_factors_positive = torch.relu(scale_factors)
    output_matrix = y_matrix * scale_factors_positive
    return output_matrix

class VariationEdgeOperator(nn.Module):
    def __init__(self, variation_matrix):
        super().__init__()
        src_idx, dst_idx, num_sources = build_edge_index_from_variation_matrix(variation_matrix)
        self.num_edges = int(src_idx.numel())
        self.num_sources = int(num_sources)
        self.register_buffer("src_idx", src_idx)
        self.register_buffer("dst_idx", dst_idx)

    def forward(self, S):
        src = torch.index_select(S, 1, self.src_idx)
        dst = torch.index_select(S, 1, self.dst_idx)
        return src - dst

    def forward_transpose(self, U):
        batch_size, _, k = U.shape
        S = torch.zeros(batch_size, self.num_sources, k, device=U.device, dtype=U.dtype)
        src_idx = self.src_idx.view(1, -1, 1).expand(batch_size, -1, k)
        dst_idx = self.dst_idx.view(1, -1, 1).expand(batch_size, -1, k)
        S.scatter_add_(1, src_idx, U)
        S.scatter_add_(1, dst_idx, -U)
        return S

def build_edge_index_from_variation_matrix(variation_matrix):
    rows, cols, values, shape = _to_coo_triplets(variation_matrix)
    num_edges, num_sources = shape

    src = np.full(num_edges, -1, dtype=np.int64)
    dst = np.full(num_edges, -1, dtype=np.int64)

    for r, c, v in zip(rows, cols, values):
        if v > 0:
            src[r] = c
        elif v < 0:
            dst[r] = c

    if np.any(src < 0) or np.any(dst < 0):
        bad_rows = np.where((src < 0) | (dst < 0))[0][:10]
        raise ValueError(f"Invalid VariationEdge matrix rows detected: {bad_rows.tolist()}")

    return (
        torch.from_numpy(src.astype(np.int64)),
        torch.from_numpy(dst.astype(np.int64)),
        int(num_sources),
    )

def _to_coo_triplets(matrix):
    if sp is not None and sp.issparse(matrix):
        coo = matrix.tocoo()
        return coo.row, coo.col, coo.data, coo.shape

    if isinstance(matrix, torch.Tensor):
        if matrix.is_sparse:
            coo = matrix.coalesce()
            indices = coo.indices().cpu().numpy()
            values = coo.values().cpu().numpy()
            return indices[0], indices[1], values, tuple(coo.shape)
        dense = matrix.detach().cpu().numpy()
        rows, cols = np.nonzero(dense)
        values = dense[rows, cols]
        return rows, cols, values, dense.shape

    dense = np.asarray(matrix)
    rows, cols = np.nonzero(dense)
    values = dense[rows, cols]
    return rows, cols, values, dense.shape
