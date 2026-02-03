"""
Nicheformer Model Wrapper - Milestone 3 (stabilized pretrained fine-tuning)

Key features:
1) Always create an input adapter in __init__ so it is part of model.parameters()
   BEFORE the optimizer is created (no "created during forward" params).
2) Infer backbone embedding dim from the loaded Nicheformer encoder and build the
   classifier to match it (prevents 256 vs 512 mismatch).
3) Provide parameter groups so backbone LR can be smaller than head/adapter LR.
4) Add basic spatial scaling to avoid huge coordinate magnitudes destabilizing training.
5) Requires Nicheformer package - no placeholder fallback.

See training.md (project root) for model contract summary.
"""

from __future__ import annotations

import inspect
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# One-time log per process (avoids spam when federated creates many client models)
_pretrained_load_logged = False
_new_model_logged = False


def _log_pretrained_loaded() -> None:
    global _pretrained_load_logged
    _pretrained_load_logged = True


def _log_new_model_created() -> None:
    global _new_model_logged
    _new_model_logged = True


class NicheformerWrapper(nn.Module):
    """
    Wrapper for Nicheformer model that implements the Model Contract interface.

    forward(batch_dict) -> logits
    compute_loss(logits, labels) -> loss
    get_weights() / set_weights()
    """

    def __init__(
        self,
        num_genes: int,
        num_labels: int,
        pretrained_path: Optional[str] = None,
        fine_tune_mode: str = "head_only",
        include_spatial: bool = True,
        hidden_dim: int = 256,           # used for new Nicheformer model (when no pretrained_path)
        num_layers: int = 4,             # used for new Nicheformer model (when no pretrained_path)
        dropout: float = 0.1,
        spatial_scale: float = 1000.0,   # divide x,y by this when include_spatial=True
        backbone_lr_mult: float = 0.1,   # backbone lr = base_lr * this (for partial/full)
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_labels = num_labels
        self.fine_tune_mode = fine_tune_mode
        self.include_spatial = include_spatial
        self.spatial_scale = spatial_scale
        self.backbone_lr_mult = backbone_lr_mult

        # Input dimension: genes (+ optional x,y)
        self.input_dim = num_genes + (2 if include_spatial else 0)

        # Load backbone (Nicheformer encoder - required)
        self.backbone = self._load_backbone(
            pretrained_path=pretrained_path,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Infer backbone embedding dimension
        self.backbone_dim = self._infer_backbone_dim(self.backbone, default=hidden_dim)

        # Adapter maps our continuous features -> backbone embedding space
        # (created up-front so optimizer sees it)
        self.input_adapter = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Optional stabilization after backbone
        self.post_backbone_norm = nn.LayerNorm(self.backbone_dim)

        # Classification head (matches backbone_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_dim // 2, num_labels),
        )

        self.criterion = nn.CrossEntropyLoss()

        # Configure which params are trainable
        self._configure_fine_tuning()

    # -------------------------
    # Backbone loading / creation
    # -------------------------

    def _load_backbone(
        self,
        pretrained_path: Optional[str],
        hidden_dim: int,
        num_layers: int,
    ) -> nn.Module:
        """
        Load the Nicheformer backbone.
        Requires nicheformer package to be installed.
        """
        try:
            import nicheformer  # noqa: F401
            from nicheformer.models import Nicheformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Nicheformer package is required but not installed. "
                "Please install it: cd temp_nicheformer && pip install -e ."
            ) from e

        if pretrained_path and os.path.exists(pretrained_path):
            # Only print once per process (federated creates one model per client per round)
            if not _pretrained_load_logged:
                print(f"Loading pretrained Nicheformer from {pretrained_path}")
            try:
                # Lightning checkpoint
                # Note: weights_only=False is needed for PyTorch 2.6+ due to numpy objects in checkpoint
                # This is safe since the checkpoint is from the official Nicheformer repository
                import torch.serialization
                torch.serialization.add_safe_globals([type(torch.tensor(0).numpy().item())])
                nf = Nicheformer.load_from_checkpoint(
                    pretrained_path,
                    map_location="cpu",
                    weights_only=False  # Required for checkpoints with numpy objects
                )
                if not _pretrained_load_logged:
                    print("Successfully loaded Nicheformer backbone")
                    _log_pretrained_loaded()
                return nf.encoder
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained Nicheformer from {pretrained_path}: {e}"
                ) from e

        # No checkpoint: create a fresh Nicheformer (still uses its encoder)
        if not _new_model_logged:
            print("Creating new Nicheformer model (no pretrained weights)")
        try:
            nf = Nicheformer(
                dim_model=hidden_dim,
                nheads=8,
                dim_feedforward=hidden_dim * 4,
                nlayers=num_layers,
                dropout=0.1,
                batch_first=True,
                masking_p=0.0,
                n_tokens=1000,
                context_length=512,
                lr=1e-4,
                warmup=1000,
                batch_size=32,
                max_epochs=10,
            )
            if not _new_model_logged:
                print("Created new Nicheformer model")
                _log_new_model_created()
            return nf.encoder
        except Exception as e:
            raise RuntimeError(f"Failed to create Nicheformer model: {e}") from e

    def _infer_backbone_dim(self, backbone: nn.Module, default: int) -> int:
        """
        Best-effort extraction of embedding dimension from a transformer encoder.
        """
        # Common patterns:
        # - backbone.layers[0].self_attn.embed_dim  (nn.TransformerEncoder)
        if hasattr(backbone, "layers") and len(getattr(backbone, "layers")) > 0:
            layer0 = backbone.layers[0]
            if hasattr(layer0, "self_attn") and hasattr(layer0.self_attn, "embed_dim"):
                return int(layer0.self_attn.embed_dim)

        # Some models store d_model / dim_model
        for attr in ("d_model", "dim_model", "model_dim", "hidden_size"):
            if hasattr(backbone, attr):
                val = getattr(backbone, attr)
                if isinstance(val, int):
                    return val

        return int(default)

    # -------------------------
    # Fine-tuning configuration
    # -------------------------

    def _configure_fine_tuning(self) -> None:
        """
        head_only:
          - freeze backbone
          - train adapter + classifier
        partial:
          - unfreeze last half of backbone layers (if available)
          - train adapter + classifier
        full:
          - train everything
        """
        # First freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # Adapter + classifier always trainable (otherwise you're not learning a mapping into the backbone space)
        for p in self.input_adapter.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True
        for p in self.post_backbone_norm.parameters():
            p.requires_grad = True

        if self.fine_tune_mode == "head_only":
            # Backbone stays frozen
            return

        if self.fine_tune_mode == "partial":
            # Unfreeze later layers if backbone exposes .layers
            if hasattr(self.backbone, "layers") and len(getattr(self.backbone, "layers")) > 0:
                layers = list(self.backbone.layers)
                n = len(layers)
                start = n // 2
                for i in range(start, n):
                    for p in layers[i].parameters():
                        p.requires_grad = True
            else:
                # If we can't identify layers, unfreeze whole backbone (still stabilized by LR mult + clipping)
                for p in self.backbone.parameters():
                    p.requires_grad = True
            return

        if self.fine_tune_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        raise ValueError("fine_tune_mode must be one of: head_only, partial, full")

    # -------------------------
    # Optimizer helpers
    # -------------------------

    def parameter_groups(self, base_lr: float, weight_decay: float) -> List[Dict]:
        """
        Return optimizer param groups:
          - adapter/head at base_lr
          - backbone at base_lr * backbone_lr_mult (if trainable)
        """
        groups: List[Dict] = []

        head_params = []
        for m in (self.input_adapter, self.post_backbone_norm, self.classifier):
            head_params.extend([p for p in m.parameters() if p.requires_grad])

        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay})

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": base_lr * float(self.backbone_lr_mult), "weight_decay": weight_decay}
            )

        return groups

    # -------------------------
    # Forward / loss / contract
    # -------------------------

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone with some signature flexibility.
        x is (B, S, D)
        """
        # Try the simplest first
        try:
            return self.backbone(x)
        except TypeError:
            pass

        # Try is_causal=False (some implementations accept it)
        try:
            return self.backbone(x, is_causal=False)
        except TypeError:
            pass

        # Fallback: try passing a None mask
        try:
            return self.backbone(x, src_key_padding_mask=None)
        except TypeError as e:
            raise RuntimeError(f"Backbone forward signature not supported. Original error: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]  # (B, input_dim)

        # Basic spatial scaling to avoid huge coords dominating
        if self.include_spatial and features.shape[1] == self.num_genes + 2 and self.spatial_scale is not None:
            genes = features[:, : self.num_genes]
            coords = features[:, self.num_genes :]
            coords = coords / float(self.spatial_scale)
            features = torch.cat([genes, coords], dim=1)

        # Adapt into backbone embedding space
        x = self.input_adapter(features)  # (B, D)

        # Treat each cell as a "sequence" of length 1
        x = x.unsqueeze(1)  # (B, 1, D)

        # Backbone
        x = self._run_backbone(x)  # (B, 1, D) in most encoder implementations

        # Pool + norm
        x = x.mean(dim=1)  # (B, D)
        x = self.post_backbone_norm(x)

        # Classifier
        logits = self.classifier(x)  # (B, num_labels)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()

    def set_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.load_state_dict(state_dict, strict=False)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    num_genes: int,
    num_labels: int,
    pretrained_path: Optional[str] = None,
    fine_tune_mode: str = "head_only",
    include_spatial: bool = True,
    **kwargs,
) -> NicheformerWrapper:
    return NicheformerWrapper(
        num_genes=num_genes,
        num_labels=num_labels,
        pretrained_path=pretrained_path,
        fine_tune_mode=fine_tune_mode,
        include_spatial=include_spatial,
        **kwargs,
    )
