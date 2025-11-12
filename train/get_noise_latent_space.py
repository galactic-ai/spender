#!/usr/bin/env python

import argparse
import torch

from spender import SpectrumAutoencoder
from spender.data import desi

def build_wave_rest(instrument, z_max):
    # Match the logic in train_DESI_noise.py
    if z_max > 0.01:  # BGS-style
        lmbda_min = instrument.wave_obs[0] / (1.0 + z_max)
        lmbda_max = instrument.wave_obs[-1]
        bins = 9780
    else:
        lmbda_min = instrument.wave_obs[0] / (1.0 + z_max)
        lmbda_max = instrument.wave_obs[-1]
        bins = int((lmbda_max-lmbda_min).item()/0.8)
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
    return wave_rest


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # instrument and data loaders
    inst = desi.DESI()
    wave_rest = build_wave_rest(inst, 0)

    # Same arch as in train_DESI_noise.py
    n_hidden = (64, 256, 1024)
    model = SpectrumAutoencoder(
        inst,
        wave_rest,
        n_latent=args.latents,
        n_hidden=n_hidden,
        act=[torch.nn.LeakyReLU()] * (len(n_hidden) + 1),
    ).to(device)

    # load the model checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model"][0]  # first (and only) encoder/decoder
    model.load_state_dict(state_dict)
    model.eval()

    # data loader
    loader = inst.get_data_loader(
        args.datadir,
        tag="chunk1024",
        which="train",
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_instance=False,
    )

    all_latents = []
    all_A = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spec, w, z = batch

            spec = spec.to(device)
            w    = w.to(device)
            z    = z.to(device)

            w = torch.clamp(w, min=1e-8)
            sigma = torch.sqrt(1.0 / w)

            s = model.encode(sigma)
            all_latents.append(s.cpu())

            wave_obs = inst._wave_obs.to(device)

            # restframe wavelengths
            wave_rest = wave_obs[None, :] / (1.0 + z[:, None])

            # your normalization window (same as training)
            mask = (w > 0) & (wave_rest > 5300) & (wave_rest < 5850)

            A_batch = torch.zeros(spec.size(0), 1, device=device)

            for i in range(spec.size(0)):
                sel = mask[i]
                if sel.any():
                    A_batch[i, 0] = spec[i, sel].median()
                else:
                    A_batch[i, 0] = 0.0  # or mark as bad and filter later

            # save A_batch alongside latents
            all_A.append(A_batch.cpu())

            if (i + 1) % 50 == 0:
                print(f"Processed { (i+1) * args.batch_size } spectra")
    
    latents = torch.cat(all_latents, dim=0)
    A = torch.cat(all_A, dim=0)

    print("Latents shape:", latents.shape)
    print("A shape:", A.shape)

    # 4) Save for flow training
    out = {
        "latents": latents,
        "A": A,
    }
    torch.save(out, args.outfile)
    print(f"Saved latents to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate noise latent space from trained autoencoder"
    )
    parser.add_argument(
        "datadir",
        type=str,
        help="Directory containing DESI training data",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the trained autoencoder checkpoint",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output file to save the latent representations",
    )
    parser.add_argument(
        "-n", "--latents",
        type=int,
        default=6,
        help="Number of latent dimensions in the autoencoder",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for processing spectra",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=0.0,
        help="Maximum redshift for rest-frame wavelength calculation",
    )

    args = parser.parse_args()
    main(args)
