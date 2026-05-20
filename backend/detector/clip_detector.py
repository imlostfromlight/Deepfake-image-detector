"""
CLIP ViT-B/16 + LoRA deepfake detector.
Replaces the three-ViT ensemble with a single fine-tuned model.
"""
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Same preprocessing pipeline used during training validation.
# Order is mandatory: resize → tensor (uint8→float, HWC→CHW) → ImageNet normalise.
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class _Detector(nn.Module):
    """Mirrors the training-time Detector class exactly."""
    def __init__(self, backbone, hidden=768):
        super().__init__()
        self.bb = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        out = self.bb(pixel_values=x)
        feat = out.pooler_output
        return self.head(feat).squeeze(-1)


class DeepfakeDetector:
    """Singleton — loads the checkpoint once, reused for all requests."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path=None, device=None, threshold=0.918):
        if self._initialized:
            return
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self._load(checkpoint_path)
        self._initialized = True

    def _load(self, checkpoint_path):
        logger.info(f"Loading CLIP backbone on {self.device}…")
        backbone = AutoModel.from_pretrained(
            'openai/clip-vit-base-patch16'
        ).vision_model

        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=['q_proj', 'v_proj'],
            bias='none',
        )
        backbone = get_peft_model(backbone, lora_cfg)

        self.model = _Detector(backbone).to(self.device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        logger.info(
            f"Detector loaded — epoch {ckpt.get('epoch', '?')}, "
            f"AUC {ckpt.get('auc', 0):.4f}, threshold {self.threshold}"
        )

    @torch.no_grad()
    def predict(self, image_input):
        """
        image_input: bytes, PIL.Image, or filesystem path.
        Returns dict with fake_prob, prediction, confidence, threshold.
        """
        if isinstance(image_input, (bytes, bytearray)):
            img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        img = img.convert('RGB')
        x = _PREPROCESS(img).unsqueeze(0).to(self.device)

        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                logit = self.model(x)
            prob = torch.sigmoid(logit.float()).item()
        else:
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        verdict = 'fake' if prob > self.threshold else 'real'
        if verdict == 'fake':
            confidence = (prob - self.threshold) / (1.0 - self.threshold) * 100
        else:
            confidence = (self.threshold - prob) / self.threshold * 100

        return {
            'prediction': verdict,
            'fake_prob':  float(prob),
            'confidence': float(max(0.0, min(100.0, confidence))),
            'threshold':  self.threshold,
        }
