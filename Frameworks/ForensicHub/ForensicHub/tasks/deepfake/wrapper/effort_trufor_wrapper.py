
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ForensicHub.registry import register_model
from ForensicHub.tasks.deepfake.wrapper.wrappers import Deepfake2ForensicWrapper

sys.path.insert(0, "/content/drive/MyDrive/Capstone_Hripsime/Frameworks/DeepfakeBench")
from training.detectors.effort_detector import EffortDetector

sys.path.insert(0, "/content/drive/MyDrive/Capstone_Hripsime/Frameworks/IMDLBenCo")
from IMDLBenCo.model_zoo.trufor.trufor import Trufor


@register_model("EffortTruforReweight")
class EffortTruforReweight(Deepfake2ForensicWrapper):
    """
    Effort with per-sample loss reweighting using the frozen, pretrained TruFor.
    For each sample i:
        w_i = 1 + alpha * p_TruFor(x_i)            # alpha >= 0
        L   = mean( w_i * CE(z_Effort_i, y_i) )
    """

    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    _IN_MEAN   = (0.485, 0.456, 0.406)
    _IN_STD    = (0.229, 0.224, 0.225)

    def __init__(
        self,
        # Effort
        yaml_config_path,
        # TruFor architecture
        trufor_config_path,
        np_pretrain_weights,
        mit_b2_pretrain_weights,
        # The OFFICIAL TruFor pretrained checkpoint (e.g. trufor.pth.tar)
        trufor_official_weights,
        # Reweighting hyperparameter
        alpha: float = 1.0,
    ):
        super().__init__(EffortDetector, yaml_config_path)

        # Build TruFor and load OFFICIAL weights — no training of TruFor
        teacher = Trufor(
            phase=2,
            np_pretrain_weights=np_pretrain_weights,
            mit_b2_pretrain_weights=mit_b2_pretrain_weights,
            config_path=trufor_config_path,
        )
        ckpt  = torch.load(trufor_official_weights, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt

        # Official ckpt's keys are flat: "backbone.X", "dncnn.X" — strip optional prefixes just in case
        cleaned = {}
        for k, v in state.items():
            new_k = k
            for pref in ("module.", "_base_model.", "base_model.", "model."):
                if new_k.startswith(pref):
                    new_k = new_k[len(pref):]
            cleaned[new_k] = v

        missing, unexpected = teacher.model.load_state_dict(cleaned, strict=False)
        print(f"[EffortTruforReweight] TruFor (official) loaded: "
              f"missing={len(missing)}  unexpected={len(unexpected)}")

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher

        self.alpha = float(alpha)

        self.register_buffer("_clip_mean", torch.tensor(self._CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_clip_std",  torch.tensor(self._CLIP_STD ).view(1, 3, 1, 1))
        self.register_buffer("_in_mean",   torch.tensor(self._IN_MEAN  ).view(1, 3, 1, 1))
        self.register_buffer("_in_std",    torch.tensor(self._IN_STD   ).view(1, 3, 1, 1))

    def _effort_input_to_trufor(self, x):
        """Effort feeds 224×224 CLIP-norm; TruFor wants 256×256 ImageNet-norm."""
        x = x * self._clip_std + self._clip_mean
        x = F.interpolate(x, size=256, mode="bilinear", align_corners=False)
        x = (x - self._in_mean) / self._in_std
        return x

    @torch.no_grad()
    def _trufor_prob(self, x):
        _, _, det, _ = self.teacher.model(x)
        return torch.sigmoid(det).view(-1)

    def forward(self, image, label, *args, **kwargs):
        # Effort forward
        data_dict   = {"image": image, "label": label.long()}
        predictions = self._base_model(data_dict)

        # Frozen TruFor inference → P(fake)
        with torch.no_grad():
            p_trufor = self._trufor_prob(self._effort_input_to_trufor(image))

        # Per-sample weights: heavier where TruFor thinks "fake"
        weights = 1.0 + self.alpha * p_trufor          # shape [B], range [1, 1+alpha]

        # Per-sample CE on Effort's logits (no reduction)
        pred_cls = predictions["cls"]                  # [B, 2]
        label_long = label.long().view(-1)             # [B]
        ce_per = F.cross_entropy(pred_cls, label_long, reduction="none")   # [B]

        # Weighted mean — backward goes through Effort only
        loss_weighted = (weights * ce_per).mean()

        # Logging-only, unweighted CE for comparison
        loss_plain = ce_per.mean()

        return {
            "backward_loss": loss_weighted,
            "pred_label":    predictions["prob"],
            "visual_loss": {
                "ce_plain":      loss_plain.detach(),
                "ce_weighted":   loss_weighted.detach(),
                "mean_w":        weights.mean().detach(),
                "mean_p_trufor": p_trufor.mean().detach(),
            },
        }
