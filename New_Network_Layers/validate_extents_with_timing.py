"""Validate version_10_75 on various Extents with per-sample forward timing."""

from __future__ import print_function, division

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

_VERSION_DIR = Path(__file__).resolve().parent
_NET_ROOT = _VERSION_DIR.parent
for _p in (str(_VERSION_DIR), str(_NET_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from scipy.io import loadmat, savemat
from tqdm import tqdm

from Vssi_Ard_Network_edge import Vssi_Ard_ADMMLayer_edge
from utils.data_loader import get_data_validation

def sync_forward(model, x):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, return_all_stages=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, t1 - t0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_data_dir",
        default="/home16T/why/net/Data/ns_6003_miter_10000_simple/test/various Extents",
    )
    parser.add_argument(
        "--result_dir",
        default="/home16T/why/net/vssi-validation-output/mymethod/ns_6000_2026_version_10_75/various Extents",
    )
    parser.add_argument(
        "--model_params",
        default="/home16T/why/ns_5003_miter_10000/Vssi_Ard_outputs_6003_2026/version_10_75/various_conditions/6000_7.784194827079773_model_20260607_134405_76.pth",
    )
    parser.add_argument("--extent_list", default="2,5,10,18,32")
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument(
        "--timing_log",
        default="/home16T/why/net/Output/validate_version10_75_extents_timing.csv",
    )
    args = parser.parse_args()

    extent_values = [s.strip() for s in args.extent_list.split(",") if s.strip()]
    val_root = args.validation_data_dir.rstrip(os.sep)
    res_root = args.result_dir.rstrip(os.sep)
    timing_rows = []

    for extent in extent_values:
        val_dir = os.path.join(val_root, extent)
        out_dir = os.path.join(res_root, extent)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Extent: {extent}")

        valid_loader = get_data_validation(val_dir, "", "-10", 1)
        L = torch.tensor(loadmat(os.path.join(val_dir, "L.mat"))["L"]).to("cuda").float()
        V = loadmat(os.path.join(val_dir, "V.mat"))["V"]
        _, _, sample_tbfs, _, _ = next(iter(valid_loader))
        k_features = int(sample_tbfs.shape[1])

        model = Vssi_Ard_ADMMLayer_edge(L, V, k_features).to("cuda")
        model.load_state_dict(torch.load(args.model_params, map_location="cuda"))
        model.eval()

        warmup_x = None
        for batch_idx, (B_trans, s_real_trans, TBFs, ActiveVoxSeed, seedvox) in enumerate(valid_loader):
            if batch_idx >= args.test_limit:
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_h2d0 = time.perf_counter()
            s_real_trans = s_real_trans.to("cuda").float().squeeze(dim=0)
            B_trans = B_trans.to("cuda").float().squeeze(dim=0)
            TBFs = TBFs.to("cuda").float().squeeze(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            h2d_s = time.perf_counter() - t_h2d0

            x = {"B_trans": B_trans.unsqueeze(0)}
            if warmup_x is None:
                warmup_x = x
                for _ in range(args.warmup_iters):
                    sync_forward(model, warmup_x)

            final_snapshot, forward_s = sync_forward(model, x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_post0 = time.perf_counter()
            s_gen_trans = final_snapshot["S"]
            s_gen = torch.matmul(s_gen_trans, TBFs)
            s_real = torch.matmul(s_real_trans, TBFs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            post_s = time.perf_counter() - t_post0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_save0 = time.perf_counter()
            savemat(
                os.path.join(out_dir, f"result_{batch_idx}_stage_{7}.mat"),
                {
                    "reco_S": s_gen.cpu().numpy(),
                    "real_S": s_real.cpu().numpy(),
                    "B_trans": B_trans.cpu().numpy(),
                    "s_real_trans": s_real_trans.cpu().numpy(),
                    "seedvox": seedvox.cpu().numpy(),
                    "TBFs": TBFs.cpu().numpy(),
                    "ActiveVoxSeed": ActiveVoxSeed.cpu().numpy(),
                },
            )
            save_s = time.perf_counter() - t_save0

            timing_rows.append(
                {
                    "extent": extent,
                    "sample_idx": batch_idx,
                    "h2d_s": h2d_s,
                    "forward_sync_s": forward_s,
                    "post_matmul_s": post_s,
                    "save_mat_s": save_s,
                    "total_s": h2d_s + forward_s + post_s + save_s,
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.timing_log), exist_ok=True)
    with open(args.timing_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)

    forwards = [r["forward_sync_s"] for r in timing_rows]
    forwards_after_warmup = [
        r["forward_sync_s"]
        for r in timing_rows
        if r["sample_idx"] >= args.warmup_iters
    ]
    min_all = min(forwards)
    min_warm = min(forwards_after_warmup) if forwards_after_warmup else min_all
    med_warm = sorted(forwards_after_warmup)[len(forwards_after_warmup) // 2] if forwards_after_warmup else min_all

    print("\n=== Forward timing summary (CUDA-synchronized) ===")
    print(f"Samples: {len(timing_rows)}")
    print(f"Shortest forward (all samples): {min_all * 1000:.3f} ms")
    print(f"Shortest forward (after warmup, idx>={args.warmup_iters}): {min_warm * 1000:.3f} ms")
    print(f"Median forward (after warmup): {med_warm * 1000:.3f} ms")
    print(f"Timing CSV: {args.timing_log}")

if __name__ == "__main__":
    main()
