"""SmolVLA encoder → cadenza-lab LoRA action head (chase-mission only).

This is chase-mission's base/VLA. It wraps the FROZEN ``lerobot/smolvla_base``
checkpoint and exposes the exact contract cadenza-lab's ``LoRAActionDecoder``
expects from an encoder::

    forward(text) -> Tensor[batch, max_seq_len, hidden_dim]

So the architecture the LoRA head sees is the real thing:

    SmolVLA (frozen)
      VLM text tower            ->  goal hidden states [T_text, 960]
      VLM vision tower (RGB)    ->  image hidden states (pooled) [n_img, 960]   (optional)
                                ->  fused [B, T, 960]
                                ->  cadenza-lab LoRAActionHead (trainable A/B)
                                ->  go1 ActionCall sequence

Vision: ``forward(text, images=...)`` runs cadenza's RGB camera frame through
SmolVLA's OWN vision tower (``embed_image`` = vision_model + connector) and
prepends the pooled image tokens to the goal tokens — i.e. the camera IS fed
into the VLA. IMPORTANT: to benefit from vision, the LoRA must be FINE-TUNED
with those image tokens present (image-paired data). The current
``env lora finetune`` is text-goal driven, so it produces a text-conditioned
head; passing images only helps once fine-tuning includes them. Default
``forward(text)`` stays text-only so train/inference slot layout matches.

Only cadenza-lab's LoRA is trained; SmolVLA stays frozen. lerobot lives here in
chase-mission, never in cadenza-api (which is VLA-agnostic). The cadenza-api
fine-tuning pipeline is handed this encoder via its ``encoder=`` hook.

Needs: ``pip install 'lerobot[smolvla]'`` and the ``smolvla_base`` checkpoint
(cached on first use). Runs on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SMOLVLA_ID = "lerobot/smolvla_base"
_PROCESSOR_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class SmolVLAEncoder(nn.Module):
    """Frozen SmolVLA text-tower encoder producing [B, max_seq_len, 960] states.

    The hidden states are SmolVLA's *contextualised* token representations of
    the goal (its VLM language model's ``last_hidden_state``), so the LoRA head
    adapts a genuinely grounded representation rather than a random hash.
    """

    def __init__(self, max_seq_len: int = 16, device: str = "cpu",
                 model_id: str = _SMOLVLA_ID, processor_id: str = _PROCESSOR_ID):
        super().__init__()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        policy = SmolVLAPolicy.from_pretrained(model_id).to(self.device).eval()
        self._vlm_we = policy.model.vlm_with_expert
        self._text_model = self._vlm_we.get_vlm_model().text_model
        self.tokenizer = AutoTokenizer.from_pretrained(processor_id)
        self._image_processor = None   # lazily loaded only if images are used
        self._processor_id = processor_id
        # SmolVLM2-500M text hidden size — the dim the LoRA head must match.
        self.hidden_dim = int(policy.model.vlm_with_expert.vlm.config.text_config.hidden_size)

        # Freeze the base entirely: only cadenza-lab's LoRA A/B will train.
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_text(self, goal: str) -> torch.Tensor:
        ids = self.tokenizer(goal, return_tensors="pt").input_ids.to(self.device)
        embeds = self._vlm_we.embed_language_tokens(ids)
        out = self._text_model(inputs_embeds=embeds)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return h[0].float()   # [T_text, 960]

    @torch.no_grad()
    def _encode_image(self, image) -> torch.Tensor:
        """Run an RGB frame through SmolVLA's vision tower → 1 pooled token [1, 960].

        ``image`` may be a path (str/Path), a PIL image, or a numpy array.
        """
        if self._image_processor is None:
            from transformers import AutoImageProcessor
            self._image_processor = AutoImageProcessor.from_pretrained(self._processor_id)
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        pv = self._image_processor(images=image, return_tensors="pt")["pixel_values"]
        pv = pv.reshape(-1, *pv.shape[-3:]).to(self.device, dtype=torch.float32)
        ih = self._vlm_we.embed_image(pv)            # [tiles, 64, 960]
        return ih.reshape(-1, ih.shape[-1]).mean(0, keepdim=True).float()  # [1, 960]

    @torch.no_grad()
    def forward(self, text, images=None) -> torch.Tensor:
        """text -> [B, max_seq_len, 960]. If ``images`` given (one RGB frame per
        item, or one shared frame), a pooled SmolVLA vision token is PREPENDED."""
        batch = [text] if isinstance(text, str) else list(text)
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images] * len(batch)
        slots = torch.zeros(len(batch), self.max_seq_len, self.hidden_dim,
                            dtype=torch.float32)
        for b, goal in enumerate(batch):
            parts = []
            if images is not None and images[b] is not None:
                parts.append(self._encode_image(images[b]))   # [1, 960] vision token
            parts.append(self._encode_text(goal))             # [T_text, 960]
            h = torch.cat(parts, dim=0)
            t = min(h.shape[0], self.max_seq_len)
            slots[b, :t] = h[:t]
        return slots
