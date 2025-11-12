#!/usr/bin/env python

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sbi.neural_nets.net_builders import build_nsf
import torch.optim as optim

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.latent_file, map_location="cpu")
    theta = data["latents"].float()   # [N, S]
    A     = data["A"].float()         # [N, 1]

    print("Loaded latents:", theta.shape)
    print("Loaded A:", A.shape)

    # standarise
    theta_mean = theta.mean(0, keepdim=True)
    theta_std  = theta.std(0, keepdim=True) + 1e-6
    theta = (theta - theta_mean) / theta_std

    A_mean = A.mean(0, keepdim=True)
    A_std  = A.std(0, keepdim=True) + 1e-6
    A = (A - A_mean) / A_std

    stats = {
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "A_mean": A_mean,
        "A_std": A_std,
    }

    # dataset: (theta | A)
    dataset = TensorDataset(theta, A)
    N = len(dataset)
    val_frac = 0.1
    N_val = int(N * val_frac)
    N_train = N - N_val

    indices = torch.randperm(N)
    train_idx = indices[:N_train]
    val_idx = indices[N_train:]

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )

    # p(theta | A) flow model
    dummy_theta, dummy_A = next(iter(train_loader))
    density_estimator = build_nsf(
        dummy_theta, dummy_A
    ).to(device) # (theta | A)

    optimizer = optim.Adam(density_estimator.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epoch_no_improve = 0

    for epoch in range(args.epochs):
        density_estimator.train()
        train_loss = 0.0
        n_train = 0

        for batch_theta, batch_A in train_loader:
            batch_theta = batch_theta.to(device)
            batch_A = batch_A.to(device)

            optimizer.zero_grad()
            
            loss_vec = density_estimator.loss(batch_theta, batch_A)
            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_theta.size(0)
            n_train += batch_theta.size(0)

        train_loss /= max(n_train, 1)

        density_estimator.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_theta, batch_A in val_loader:
                batch_theta = batch_theta.to(device)
                batch_A = batch_A.to(device)

                loss_vec = density_estimator.loss(batch_theta, batch_A)
                loss = loss_vec.mean()
                val_loss += loss.item() * batch_theta.size(0)
                n_val += batch_theta.size(0)

        val_loss /= max(n_val, 1)

        print(f"Epoch {epoch:3d}: train NLL={train_loss:.4f}, val NLL={val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                "flow_state": density_estimator.state_dict(),
                "stats": stats,
            }
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= args.patience:
                print("Early stopping triggered.")
                break
        
    # save best model
    if best_state is None:
        best_state = {
            "flow_state": density_estimator.state_dict(),
            "stats": stats,
        }
    
    torch.save(best_state, args.outfile)
    print(f"Saved trained flow model to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train normalizing flow on noise latents"
    )
    parser.add_argument(
        "latent_file",
        type=str,
        help="Path to latent space file (output of get_noise_latent_space.py)",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output file for trained flow model",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "-r",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )

    args = parser.parse_args()
    main(args)