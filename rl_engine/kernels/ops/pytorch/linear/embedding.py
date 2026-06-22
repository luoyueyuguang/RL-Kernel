# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
import torch.nn.functional as F


class NativeEmbeddingOp:
    """
    Pure PyTorch native token-embedding reference.
    out = weight[token_ids]   (a plain row gather, no accumulation)

    Maps integer token ids to their hidden-state rows. For Qwen3-8B the
    weight is the input embedding table ``[vocab=151936, hidden=4096]`` and
    is *independent* from the lm_head weight (``tie_word_embeddings=false``).
    """

    def __init__(self) -> None:
        pass

    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids, weight)

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Canonical entry: gather in fp32, cast the result back to weight.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._embedding(token_ids, weight, output_dtype=weight.dtype)

    def forward_fp32(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Ground-truth: gather in fp32 and force fp32 output."""
        return self._embedding(token_ids, weight, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _embedding(
        token_ids: torch.Tensor,
        weight: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        out = F.embedding(token_ids.long(), weight.float())
        return out.to(output_dtype)
