"""
Continuous / incremental fine-tuning of the CLIP+LoRA deepfake detector.

Strategy: freeze the LoRA backbone, retrain only the classification head
on user-corrected feedback samples. This avoids catastrophic forgetting
while still adapting the decision boundary to new data.
"""
import threading
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

MIN_SAMPLES  = 4
EPOCHS       = 5
LR           = 1e-4
BATCH        = 8

_status = {
    "state":        "idle",   # idle | training | done | error
    "epochs_done":  0,
    "total_epochs": EPOCHS,
    "loss":         None,
    "last_trained": None,
    "message":      "",
}
_lock = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def get_status() -> dict:
    from .models import FeedbackSample
    with _lock:
        s = dict(_status)
    s['pending_samples'] = FeedbackSample.objects.filter(used_in_train=False).count()
    s['total_samples']   = FeedbackSample.objects.count()
    return s


def start_retraining(checkpoint_path: str) -> bool:
    """Launch background fine-tuning. Returns False if already running."""
    with _lock:
        if _status['state'] == 'training':
            return False
        _status.update(state='training', epochs_done=0, loss=None, message='Starting…')
    threading.Thread(target=_run, args=(str(checkpoint_path),), daemon=True).start()
    return True


# ── Internal ──────────────────────────────────────────────────────────────────

class _FeedbackDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.image.path).convert('RGB')
        return self.transform(img), torch.tensor(1.0 if s.true_label == 'deepfake' else 0.0)


def _set(**kwargs):
    with _lock:
        _status.update(kwargs)


def _run(checkpoint_path: str):
    from .models import FeedbackSample
    from .clip_detector import DeepfakeDetector, _PREPROCESS

    try:
        samples = list(FeedbackSample.objects.filter(used_in_train=False))
        n = len(samples)
        if n < MIN_SAMPLES:
            _set(state='error', message=f'Need ≥{MIN_SAMPLES} feedback samples, have {n}')
            return

        _set(message=f'Retraining on {n} samples — freezing backbone, training head only…')

        detector = DeepfakeDetector(checkpoint_path)
        model    = detector.model
        device   = detector.device

        # Freeze LoRA backbone; only classification head gets gradient
        for p in model.bb.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

        loader = DataLoader(
            _FeedbackDataset(samples, _PREPROCESS),
            batch_size=min(BATCH, n),
            shuffle=True,
        )
        opt  = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)
        crit = torch.nn.BCEWithLogitsLoss()

        model.train()
        final_loss = None
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            final_loss = round(epoch_loss / len(loader), 4)
            _set(epochs_done=epoch + 1, loss=final_loss,
                 message=f'Epoch {epoch + 1}/{EPOCHS} · loss {final_loss}')

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Save updated checkpoint (only model_state changes)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt['model_state']   = model.state_dict()
        ckpt['retrain_count'] = ckpt.get('retrain_count', 0) + 1
        torch.save(ckpt, checkpoint_path)

        # Mark samples consumed and reset singleton → next predict reloads weights
        FeedbackSample.objects.filter(pk__in=[s.pk for s in samples]).update(used_in_train=True)
        DeepfakeDetector._instance = None

        _set(
            state='done',
            last_trained=time.strftime('%Y-%m-%d %H:%M'),
            message=f'Done — {n} samples · final loss {final_loss}',
        )

    except Exception as exc:
        _set(state='error', message=str(exc))
