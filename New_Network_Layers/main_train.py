from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from utils.data_loader import get_data_train
import time
from datetime import datetime
from scipy.io import loadmat
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(False)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" main ")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--localize_mi",
        default="/home16T/why/net/Data/ns_6003_miter_10000_simple",
    )
    parser.add_argument("--num_epoch", default=75, type=int, help="number of epochs")
    parser.add_argument("--lr", default=5e-5, type=float, help="initial learning rate")
    parser.add_argument("--SNRs", type=str, default="5", help="signal noise ratio")
    parser.add_argument("--cond", type=str, default="various conditions", help="Conditions for selecting research")
    parser.add_argument("--V", type=str, default="V.mat", help="VariationEdge operator file")
    parser.add_argument(
        "--result_dir",
        default="/home16T/why/ns_5003_miter_10000/Vssi_Ard_outputs_6003_2026/version_10_75",
        type=str,
        help="training checkpoints and logs",
    )
    parser.add_argument("--balance_scaler", default=1.0, type=float, help="scale factor for V^T V term")
    args = parser.parse_args()

    log_text_dir = os.path.join(args.result_dir, "text_logs")
    os.makedirs(log_text_dir, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file_path = os.path.join(log_text_dir, f"train_diagnostics_{current_time}.txt")
    sys.stdout = Logger(log_file_path)
    print(f"[system] Console logs will also be saved to: {log_file_path}\n")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader, valid_loader = get_data_train(args.localize_mi, args.cond, args.SNRs, args.batch_size)

    subject01_L_path = os.path.join(args.localize_mi, "train", "various conditions", "L.mat")
    L = torch.tensor(loadmat(subject01_L_path)["L"]).to("cuda").float()

    subject01_V_path = os.path.join(args.localize_mi, "train", "various conditions", args.V)
    V = loadmat(subject01_V_path)["V"]
    print(f"Loaded VariationEdge operator from: {subject01_V_path}")

    _, _, sample_tbfs = next(iter(train_loader))
    k_features = sample_tbfs.shape[1]
    print(f"Detected k_features (number of TBFs): {k_features}")
    print(
        "version_10_75: v9 U-step + FinalLayer-only S correction (CG12), num_epoch=%d (0..%d), CosineAnnealing T_max=num_epoch"
        % (args.num_epoch, args.num_epoch)
    )

    lambda2 = 3e5

    model = Vssi_Ard_ADMMLayer_edge(
        L,
        V,
        k_features,
        balance_scaler=args.balance_scaler,
    ).to("cuda")

    num_params = count_parameters(model)
    print(f"Number of parameters:{num_params}")

    MSE = torch.nn.MSELoss(reduction="mean").cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=1e-6)

    print("start training... (loss=stage_mse*lambda2, lambda2=%g, final stage only)" % lambda2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_nan_skips = 0
    train_losses = []
    valid_losses = []

    for epoch in range(0, args.num_epoch + 1):
        print("EPOCH {}:".format(epoch + 1))
        model.train(True)

        running_loss = 0.0
        last_loss = 0.0

        for batch_idx, (B, s_real, TBFs) in tqdm(enumerate(train_loader), desc="Training", unit="file"):
            s_real_trans = s_real.to("cuda").float()
            B_trans = B.to("cuda").float()

            optimizer.zero_grad()
            x = dict()
            x["B_trans"] = B_trans

            final_snapshot = model(x, return_all_stages=False)
            stage_mse = MSE(final_snapshot["S"], s_real_trans)
            total_loss = stage_mse * lambda2

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss at Epoch {epoch} Batch {batch_idx}. Skipping step.")
                total_nan_skips += 1
                optimizer.zero_grad()
                continue

            running_loss += total_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        last_loss = running_loss / len(train_loader)
        print(f"train loss avg (stage_mse*lambda2): {last_loss:.4e}")
        train_losses.append(last_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (B, s_real, TBFs) in tqdm(enumerate(valid_loader), desc="valid", unit="file"):
                s_real_trans = s_real.to("cuda").float()
                B_trans = B.to("cuda").float()
                x = dict()
                x["B_trans"] = B_trans
                final_snapshot = model(x, return_all_stages=False)
                vloss = MSE(final_snapshot["S"], s_real_trans) * lambda2
                running_val_loss += vloss.item()

        avg_val_loss = running_val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)
        print("m_LOSS train {} valid {}".format(last_loss, avg_val_loss))

        model_path = "6000_{}_model_{}_{}.pth".format(avg_val_loss, timestamp, epoch + 1)
        save_path = os.path.join(args.result_dir, args.cond.replace(" ", "_"), model_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

        plt.clf()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.title("Training and Validation Loss Curve (version_10_75)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plot_save_path = os.path.join(args.result_dir, f"loss_curve_{timestamp}.png")
        os.makedirs(args.result_dir, exist_ok=True)
        plt.savefig(plot_save_path)
        print(f"Loss curve plot saved to: {plot_save_path}")
        scheduler.step()

    print("-" * 50)
    print("Training Complete.")
    print(f"Total batches skipped due to NaN/Inf: {total_nan_skips}")
    print("-" * 50)
