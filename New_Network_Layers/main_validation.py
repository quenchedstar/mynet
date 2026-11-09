from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from pathlib import Path

_VERSION_DIR = Path(__file__).resolve().parent
_NET_ROOT = _VERSION_DIR.parent
for _p in (str(_VERSION_DIR), str(_NET_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch
import argparse
from Vssi_Ard_Network_edge import Vssi_Ard_ADMMLayer_edge
from utils.data_loader import get_data_validation
import torch.utils.data as data
from scipy.io import loadmat, savemat
from os.path import join
import torch.nn as nn
from tqdm import tqdm
import time

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.abs(input - target).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" main ")
    parser.add_argument("--data_dir", default="Data", type=str, help="directory of data")
    parser.add_argument(
        "--validation_data_dir",
        default="/home16T/why/net/Data/ns_6003_miter_10000_simple/test/various SNRs",
    )
    parser.add_argument("--brainstorm_epilepsy", default="Data\\brainstorm_ecilepsy_data")
    parser.add_argument("--yokagawa", default="Data\\Yokagawa\\real_data\\")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--num_epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--outf", type=str, default="./logs_csnet", help="path of log files")
    parser.add_argument("--SNRs", type=str, default="-10", help="signal noise ratio")
    parser.add_argument("--extents", type=str, default="", help="area of the source")
    parser.add_argument("--patchs", type=str, default="1", help="area of the source")
    parser.add_argument("--channels", default="62", help="choose the number of channel(Data volume)")
    parser.add_argument("--cond", type=str, default="", help="Conditions for selecting research")
    parser.add_argument("--V", type=str, default="V.mat", help="VariationEdge operator file")
    parser.add_argument(
        "--result_dir",
        default="/home16T/why/net/vssi-validation-output/mymethod/ns_6000_2026_version_10_75/various SNRs",
    )
    parser.add_argument("--balance_scaler", default=1.0, type=float, help="scale factor for V^T V term")
    parser.add_argument("--model_params", type=str,
                        default="/home16T/why/ns_5003_miter_10000/Vssi_Ard_outputs_6003_2026/5-26-3.pth",
                        help="path to trained .pth checkpoint")
    parser.add_argument("--deep_supervision", type=int, default=0, choices=[0, 1],
                        help="1=forward all stages (for export_mode=0); 0=final stage only (faster)")
    parser.add_argument("--export_mode", type=int, default=1, choices=[0, 1], help="0=all stages, 1=final stage only")
    parser.add_argument("--test_limit", type=int,
                        default=100,
                        help="max samples to export; <=0 means all")
    parser.add_argument("--snr_list", type=str, default="-5,0,5,10", help="SNR subfolders under validation/result dir")
    args = parser.parse_args()

    snr_values = [s.strip() for s in args.snr_list.split(",") if s.strip()]
    val_root = args.validation_data_dir.rstrip(os.sep)
    res_root = args.result_dir.rstrip(os.sep)

    def adjust_learning_rate(opt, epo, lr):
        lr = lr * (0.5 ** (epo // 25))
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    for snr in snr_values:
        args.validation_data_dir = os.path.join(val_root, snr)
        args.result_dir = os.path.join(res_root, snr)
        print("SNR:", snr)

        valid_loader = get_data_validation(args.validation_data_dir, args.cond, args.SNRs, args.batch_size)

        subject01_L_path = os.path.join(args.validation_data_dir, "L.mat")
        L = torch.tensor(loadmat(subject01_L_path)["L"]).to("cuda").float()

        subject01_V_path = os.path.join(args.validation_data_dir, args.V)
        V = loadmat(subject01_V_path)["V"]
        print(f"Loaded VariationEdge operator from: {subject01_V_path}")

        _, _, sample_tbfs, _, _ = next(iter(valid_loader))
        k_features = sample_tbfs.shape[1]
        print(f"Detected k_features (number of TBFs): {k_features}")

        lambda1 = 10
        lambda2 = 1e5
        delta = 0.1
        model = Vssi_Ard_ADMMLayer_edge(
            L,
            V,
            k_features,
            balance_scaler=args.balance_scaler,
        ).to("cuda")

        num_params = count_parameters(model)
        print(f"Number of parameters:{num_params}")
        MAE = MAELoss()
        MSE = torch.nn.MSELoss(reduction="mean").cuda()

        model_params = args.model_params
        if model_params and os.path.exists(model_params):
            params_load = torch.load(model_params)
            model.load_state_dict(params_load)
            print(f"Loaded checkpoint: {model_params}")
        else:
            print("No valid --model_params provided; using current model weights as-is.")

        print("==================validation======================")
        print("deep_supervision=%d export_mode=%d" % (args.deep_supervision, args.export_mode))

        export_mode = args.export_mode
        if export_mode == 0 and not args.deep_supervision:
            print("export_mode=0 needs deep_supervision=1; using final stage only.")
            export_mode = 1
        test_limit = None if args.test_limit <= 0 else args.test_limit

        os.makedirs(args.result_dir, exist_ok=True)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (B_trans, s_real_trans, TBFs, ActiveVoxSeed, seedvox) in tqdm(
                enumerate(valid_loader),
                desc="valid",
                unit="file",
            ):
                if test_limit is not None and batch_idx >= test_limit:
                    print(f"\nReached requested test_limit ({test_limit}), stopping validation export.")
                    break

                s_real_trans = s_real_trans.to("cuda").float().squeeze(dim=0)
                B_trans = B_trans.to("cuda").float().squeeze(dim=0)
                TBFs = TBFs.to("cuda").float().squeeze(dim=0)

                x = dict()
                x["B_trans"] = B_trans.unsqueeze(0)

                start_time = time.time()

                if export_mode == 0 and args.deep_supervision:
                    stage_snapshots = model(x, return_all_stages=True)
                else:
                    final_snapshot = model(x, return_all_stages=False)

                end_time = time.time()
                print(f"\nModel forward pass took: {end_time - start_time:.4f} seconds")

                if export_mode == 0:
                    iterable_stages = list(enumerate(stage_snapshots))
                    print("\n" + "=" * 20 + f" Analysis for Sample {batch_idx} (All Stages) " + "=" * 20)
                else:
                    iterable_stages = [(model.num_stages - 1, final_snapshot)]

                for i, snapshot in iterable_stages:
                    stage_num = i + 1

                    s_gen_trans_stage = snapshot["S"]

                    s_gen = torch.matmul(s_gen_trans_stage, TBFs)
                    s_real = torch.matmul(s_real_trans, TBFs)

                    filename = os.path.join(args.result_dir, f"result_{batch_idx}_stage_{stage_num}.mat")

                    savemat(
                        filename,
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

                    if export_mode == 0:
                        print(f"Saved intermediate result for Stage {stage_num} to: {filename}")

                if export_mode == 0:
                    print("=" * 50 + "\n")
