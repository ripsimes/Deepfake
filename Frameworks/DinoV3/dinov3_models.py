import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3LinearClassifier(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone=True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        feat = outputs.pooler_output
        logits = self.classifier(feat).squeeze(1)
        return logits


class DINOv3MACClassifier(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone=True,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden * 6, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pixel_values, return_features=False):
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state

        cls_token = tokens[:, 0]
        reg_tokens = tokens[:, 1:5]
        patch_tokens = tokens[:, 5:]
        avg_patch = patch_tokens.mean(dim=1)

        mac_feature = torch.cat(
            [
                cls_token,
                reg_tokens[:, 0],
                reg_tokens[:, 1],
                reg_tokens[:, 2],
                reg_tokens[:, 3],
                avg_patch,
            ],
            dim=1,
        )

        logits = self.head(mac_feature).squeeze(1)

        if return_features:
            return logits, mac_feature

        return logits


from peft import LoraConfig, get_peft_model


class DINOv3LoRAMACClassifier(nn.Module):
    def __init__(
        self,
        model_name="/content/drive/MyDrive/Capstone_Hripsime_Working/Frameworks/DinoV3/hf_dinov3_vitb16",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        dropout=0.2,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["query", "value"],
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        hidden = self.backbone.base_model.model.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden * 6, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pixel_values, return_features=False):
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state

        cls_token = tokens[:, 0]
        reg_tokens = tokens[:, 1:5]
        patch_tokens = tokens[:, 5:]
        avg_patch = patch_tokens.mean(dim=1)

        mac_feature = torch.cat(
            [
                cls_token,
                reg_tokens[:, 0],
                reg_tokens[:, 1],
                reg_tokens[:, 2],
                reg_tokens[:, 3],
                avg_patch,
            ],
            dim=1,
        )

        logits = self.head(mac_feature).squeeze(1)

        if return_features:
            return logits, mac_feature

        return logits
