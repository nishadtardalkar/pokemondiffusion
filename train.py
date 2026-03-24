"""
Train a single fully-convolutional denoising diffusion model on a multi-resolution
Pokémon dataset: resolutions 1, 2, 4, …, 128 (powers of two).

Each training step picks a scale H×H, diffuses noise on that image, and predicts
noise with the same CNN at arbitrary H. Optional conditioning is the parent
(H/2)×(H/2) image nearest-neighbor upsampled to H×H (zeros at 1×1).

Sampling runs DDPM at 1×1, then nearest upsample + noise and repeat up to 128×128.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SameResolutionBatchSampler:
    """
    Yields index lists whose samples all share the same H×W (same entry in resolution chain).
    Index layout must match MultiResSpriteDataset: idx = id_slot * n_res + res_slot.
    """

    def __init__(self, n_ids: int, n_res: int, batch_size: int, shuffle: bool = True) -> None:
        self.n_ids = n_ids
        self.n_res = n_res
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.n_res * (self.n_ids // self.batch_size)

    def __iter__(self):
        order = list(range(self.n_res))
        if self.shuffle:
            random.shuffle(order)
        for r in order:
            pool = [i * self.n_res + r for i in range(self.n_ids)]
            if self.shuffle:
                random.shuffle(pool)
            for k in range(0, len(pool), self.batch_size):
                batch = pool[k : k + self.batch_size]
                if len(batch) < self.batch_size:
                    continue
                yield batch


# Chain used for training and sampling (must match dataset folders).
RESOLUTIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


def rgb_from_rgba_arr(arr: np.ndarray) -> np.ndarray:
    """RGBA or RGB uint8 -> float32 RGB [0, 1], shape (H, W, 3)."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32) / 255.0
        a = arr[..., 3:4].astype(np.float32) / 255.0
        return (rgb * a + (1.0 - a)).clip(0.0, 1.0)
    return (arr[..., :3].astype(np.float32) / 255.0).clip(0.0, 1.0)


class MultiResSpriteDataset(Dataset):
    """
    Loads dataset/<res>/<id>.png.
    Item idx maps to one (pokémon id, resolution); use SameResolutionBatchSampler so each
    batch has a single spatial size (default collate stacks tensors).
    """

    def __init__(self, root: Path, resolution_chain: tuple[int, ...] = RESOLUTIONS):
        super().__init__()
        self.root = Path(root)
        self.chain = resolution_chain
        if not self.chain or self.chain[0] != 1:
            raise ValueError("resolution_chain must start with 1")

        self._by_res: dict[int, list[Path]] = {}
        for h in self.chain:
            folder = self.root / str(h)
            if not folder.is_dir():
                continue
            self._by_res[h] = sorted(folder.glob("*.png"))

        # IDs that exist at every resolution we need
        ids_sets = []
        for h in self.chain:
            if h not in self._by_res:
                raise FileNotFoundError(f"Missing dataset folder: {self.root / str(h)}")
            ids_sets.append({p.stem for p in self._by_res[h]})
        self.ids = sorted(set.intersection(*ids_sets))
        if not self.ids:
            raise RuntimeError(f"No Pokémon IDs with images for all resolutions in {self.root}")

    def __len__(self) -> int:
        return max(len(self.ids) * len(self.chain), 1)

    def _load_rgb_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        rgb = rgb_from_rgba_arr(arr)
        # (H, W, 3) -> (3, H, W)
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        n_res = len(self.chain)
        pid = self.ids[idx // n_res]
        h = self.chain[idx % n_res]

        x_path = self.root / str(h) / f"{pid}.png"
        x0 = self._load_rgb_tensor(x_path)

        if h == 1:
            cond = torch.zeros_like(x0)
        else:
            h_half = h // 2
            c_path = self.root / str(h_half) / f"{pid}.png"
            cond_low = self._load_rgb_tensor(c_path)
            cond = F.interpolate(
                cond_low.unsqueeze(0),
                size=(h, h),
                mode="nearest",
            ).squeeze(0)

        return {"x0": x0, "cond": cond, "res": h}


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) long or float in [0, T); returns (B, dim)."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Same-resolution conv block; time injected as channel bias."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return x + h


class FullyConvDenoiser(nn.Module):
    """
    ε-predictor: same spatial size as noisy x. Always 6 input channels:
    [noisy RGB | condition RGB]; condition is zeros at 1×1.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base: int = 64,
        time_dim: int = 256,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(base, time_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(min(8, base), base)
        self.out_conv = nn.Conv2d(base, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) noisy
        cond: (B, 3, H, W) parent upsampled (or zeros)
        t: (B,) int64 timesteps
        """
        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x, cond], dim=1)
        h = self.in_conv(h)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


class GaussianDDPM:
    def __init__(self, timesteps: int, device: torch.device) -> None:
        self.T = timesteps
        self.device = device
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_inv_alpha = torch.sqrt(1.0 / alpha)
        self.posterior_var = beta * (1.0 - torch.cat([alpha_bar[:1] * 0 + 1, alpha_bar[:-1]])) / (1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_alpha_bar[t][:, None, None, None] * x0 + self.sqrt_one_minus_alpha_bar[t][
            :, None, None, None
        ] * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            x_t - self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * eps
        ) / self.sqrt_alpha_bar[t][:, None, None, None].clamp(min=1e-8)

    def p_sample_step(self, model: nn.Module, x: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        """One DDPM reverse step at scalar time t for all batch elements."""
        b = x.shape[0]
        t_b = torch.full((b,), t, device=x.device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t_b, cond)
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x - coef2 * eps)

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_var[t])
            return mean + sigma * noise
        return mean


def train_step(
    model: nn.Module,
    ddpm: GaussianDDPM,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    x0 = batch["x0"].to(ddpm.device)
    cond = batch["cond"].to(ddpm.device)
    b = x0.shape[0]
    t = torch.randint(0, ddpm.T, (b,), device=ddpm.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise)
    pred = model(x_t, t, cond)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def sample_progressive(
    model: nn.Module,
    ddpm: GaussianDDPM,
    batch_size: int,
    noise_scale_next: float = 0.2,
) -> torch.Tensor:
    """
    Generate (B, 3, 128, 128): DDPM at each scale, then nearest ×2 between scales.
    Initial 2×2 etc. latents mix prior mean (upsampled parent) with Gaussian noise.
    """
    model.eval()
    device = ddpm.device
    chain = list(RESOLUTIONS)
    prev_clean: torch.Tensor | None = None

    for si, h in enumerate(chain):
        if si == 0:
            x = torch.randn(batch_size, 3, h, h, device=device)
            cond = torch.zeros_like(x)
        else:
            assert prev_clean is not None
            mean_up = F.interpolate(prev_clean, size=(h, h), mode="nearest")
            x = mean_up + noise_scale_next * torch.randn(batch_size, 3, h, h, device=device)
            cond = mean_up

        for t_ in range(ddpm.T - 1, -1, -1):
            x = ddpm.p_sample_step(model, x, t_, cond)
        prev_clean = x

    return prev_clean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiscale DDPM (1→128, one CNN).")
    p.add_argument("--dataset", type=Path, default=Path("dataset"), help="dataset root (contains 1/, 2/, …, 128/)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--save-path", type=Path, default=Path("checkpoint_denoiser.pt"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-every", type=int, default=0, help="if >0, save sample grid every N epochs")
    p.add_argument("--noise-next", type=float, default=0.2, help="noise multiplier after upsample between scales")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ds = MultiResSpriteDataset(args.dataset)
    batch_sampler = SameResolutionBatchSampler(
        n_ids=len(ds.ids),
        n_res=len(ds.chain),
        batch_size=args.batch_size,
        shuffle=True,
    )
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = FullyConvDenoiser(base=args.base_ch, num_blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ddpm = GaussianDDPM(args.timesteps, device)

    print(f"Dataset: {len(ds.ids)} IDs, resolutions {RESOLUTIONS}. Device: {device}")

    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for batch in loader:
            losses.append(train_step(model, ddpm, batch, optimizer))
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.5f}")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            imgs = sample_progressive(model, ddpm, batch_size=min(4, args.batch_size), noise_scale_next=args.noise_next)
            # simple debug grid: first image only saved as PNG
            out = (imgs[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(out).save(f"sample_epoch_{epoch}.png")
            print(f"  wrote sample_epoch_{epoch}.png")

    payload = {
        "model": model.state_dict(),
        "resolutions": list(RESOLUTIONS),
        "args": vars(args),
    }
    torch.save(payload, args.save_path)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
