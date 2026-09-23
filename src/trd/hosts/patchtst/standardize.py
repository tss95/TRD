"""Training-split channel standardization."""

from typing import Any, Iterable
import logging
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _first_view(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        return batch[0]
    if not isinstance(batch, (tuple, list)) or len(batch) != 6 or not isinstance(batch[2], torch.Tensor):
        raise ValueError("PatchTST preparation requires a single-source pair or six-field SSL batch")
    return batch[2]


@torch.no_grad()
def fit_training_standardizer(method: Any, train_loader: Iterable[Any]) -> None:
    """Stream population statistics over TRAIN rows only, retaining no waveforms.

    A sequential loader includes the incomplete final batch even when the
    optimization loader drops it. Normalization follows existing domain input
    preprocessing; it never sees probe/validation/test loaders.
    """
    encoder = method.encoder
    if bool(encoder.standard_fitted):
        return
    if isinstance(train_loader, DataLoader):
        if train_loader.batch_size is None:
            raise ValueError("PatchTST standard scaling requires a DataLoader with an explicit batch_size")
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=train_loader.collate_fn,
        )
    count = 0
    mean = torch.zeros(encoder.channels, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for batch in train_loader:
        values = _first_view(batch).transpose(1, 2).detach().to(device="cpu", dtype=torch.float64)
        encoder.validate_input(values)
        values = values.reshape(-1, encoder.channels)
        n = values.shape[0]
        variance, batch_mean = torch.var_mean(values, dim=0, unbiased=False)
        delta = batch_mean - mean
        total = count + n
        m2 += variance * n + delta.square() * (count * n / total)
        mean += delta * (n / total)
        count = total
    if count == 0:
        raise ValueError("PatchTST cannot fit standard scaling on an empty TRAIN loader")
    std = (m2 / count).sqrt()
    # Source sklearn StandardScaler uses unit scale for constant channels.
    std = torch.where(std == 0, torch.ones_like(std), std)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise FloatingPointError("PatchTST training standard-scaling statistics are not finite")
    encoder.standard_mean.copy_(mean)
    encoder.standard_std.copy_(std)
    encoder.standard_fitted.fill_(True)
    logger.info("PatchTST standard scaler fitted: training_samples=%d channels=%d", count, encoder.channels)
