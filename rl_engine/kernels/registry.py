# SPDX-License-Identifier: Apache-2.0  
# Copyright (c) 2026 Kernel-Align Contributors

import importlib
from enum import Enum, EnumMeta
from typing import Optional, Dict, Any, Type
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

class _KernelEnumMeta(EnumMeta):
    """Metaclass to provide enhanced error messaging for backend lookups."""
    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid_ops = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Operator '{name}' not found. Supported backends: {valid_ops}"
            )

class OpBackend(Enum, metaclass=_KernelEnumMeta):
    # NVIDIA optimized stack
    FLASH_ATTN = "rl_engine.kernels.cuda.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.cuda.flashinfer.FlashInferOp"
    
    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.rocm.composable_kernel.CKOp"
    
    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.triton.generic.TritonOp"
    PYTORCH_NATIVE = "rl_engine.kernels.native.pytorch_op.NativeOp"

class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between ROCm and CUDA backends at runtime.
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._priority_map = {
            "cuda": {
                "logp": [OpBackend.FLASHINFER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.FLASH_ATTN, OpBackend.TRITON_GENERIC],
            },
            "rocm": {
                "logp": [OpBackend.ROCM_AITER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.TRITON_GENERIC],
            }
        }
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")

    def get_op(self, op_type: str) -> Any:
        """
        Core distribution logic: Automatically select the best operator based on hardware and priority.
        """
        platform = "rocm" if device_ctx.is_rocm else "cuda"
        candidates = self._priority_map.get(platform, {}).get(op_type, [OpBackend.PYTORCH_NATIVE])

        for backend in candidates:
            op_class = self._load_backend(backend)
            if op_class:
                return op_class()
        
        raise RuntimeError(f"No functional backend found for {op_type} on {platform}")

    def _load_backend(self, backend: OpBackend) -> Optional[Type]:
        """
        Dynamic loading technique: Import modules only when needed and check environment dependencies.
        """
        if backend.name in self._cache:
            return self._cache[backend.name]

        module_path, class_name = backend.value.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            op_cls = getattr(module, class_name)
            self._cache[backend.name] = op_cls
            return op_cls
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            logger.warning(f"Backend {backend.name} unavailable: {e}. Falling back...")
            return None


kernel_registry = KernelRegistry()