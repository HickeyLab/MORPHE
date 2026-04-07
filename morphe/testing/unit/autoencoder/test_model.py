from __future__ import annotations

import torch

from core.autoencoder.model import Autoencoder


def test_forward_returns_latent_and_reconstruction() -> None:
    model = Autoencoder(in_dim=10, bottleneck_dim=3, hidden_dim=16)
    x = torch.randn(4, 10)

    z, x_recon = model(x)

    assert z.shape == (4, 3)
    assert x_recon.shape == (4, 10)


def test_encode_returns_correct_shape() -> None:
    model = Autoencoder(in_dim=8, bottleneck_dim=3, hidden_dim=16)
    x = torch.randn(5, 8)

    z = model.encode(x)

    assert z.shape == (5, 3)


def test_decode_returns_correct_shape() -> None:
    model = Autoencoder(in_dim=8, bottleneck_dim=3, hidden_dim=16)
    z = torch.randn(5, 3)

    x_recon = model.decode(z)

    assert x_recon.shape == (5, 8)


def test_encode_decode_roundtrip_preserves_shape() -> None:
    model = Autoencoder(in_dim=12, bottleneck_dim=4, hidden_dim=32)
    x = torch.randn(3, 12)

    z = model.encode(x)
    x_recon = model.decode(z)

    assert x_recon.shape == x.shape


def test_forward_matches_encode_decode() -> None:
    model = Autoencoder(in_dim=6, bottleneck_dim=3, hidden_dim=16)
    model.eval()
    x = torch.randn(2, 6)

    with torch.no_grad():
        z_fwd, x_recon_fwd = model(x)
        z_enc = model.encode(x)
        x_recon_dec = model.decode(z_enc)

    assert torch.equal(z_fwd, z_enc)
    assert torch.equal(x_recon_fwd, x_recon_dec)


def test_model_respects_custom_dimensions() -> None:
    model = Autoencoder(in_dim=20, bottleneck_dim=5, hidden_dim=64)
    x = torch.randn(2, 20)

    z, x_recon = model(x)

    assert z.shape == (2, 5)
    assert x_recon.shape == (2, 20)


def test_model_eval_disables_dropout() -> None:
    model = Autoencoder(in_dim=10, bottleneck_dim=3, hidden_dim=16)
    model.eval()
    x = torch.randn(4, 10)

    with torch.no_grad():
        z1, recon1 = model(x)
        z2, recon2 = model(x)

    assert torch.equal(z1, z2)
    assert torch.equal(recon1, recon2)


def test_single_sample_batch() -> None:
    model = Autoencoder(in_dim=5, bottleneck_dim=3, hidden_dim=16)
    x = torch.randn(1, 5)

    z, x_recon = model(x)

    assert z.shape == (1, 3)
    assert x_recon.shape == (1, 5)
