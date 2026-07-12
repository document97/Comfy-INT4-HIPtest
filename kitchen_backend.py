"""comfy-kitchen registration for the gfx1103 native Triton W4A4 path."""

from __future__ import annotations

import logging
import sys

import torch

from comfy_kitchen.constraints import (
    DivisibleBy,
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
)
from comfy_kitchen.registry import registry

from .native_int4 import is_gfx1103_target, triton_convrot_w4a4_linear


BACKEND_NAME = "gfx1103_int4_triton"


class _Int4LinearDtypeConstraint(ParamConstraint):
    """Accept the W4A4 backend only for an explicitly native INT4 call.

    comfy-kitchen also uses ``linear_dtype='int8'`` for a deliberate mixed
    precision tier.  Returning a failed constraint for that value lets the
    registry continue to an INT8-capable backend instead of silently treating
    it as native W4A4.
    """

    def check_dtype(self, value) -> bool:
        return value == "int4"


class _ConvRotGroupConstraint(ParamConstraint):
    def check_dtype(self, value) -> bool:
        return type(value) is int and value in (16, 64, 256)


class _QuantGroupConstraint(ParamConstraint):
    def check_dtype(self, value) -> bool:
        return type(value) is int and value == 64


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    return triton_convrot_w4a4_linear(
        x,
        qweight,
        wscales,
        bias,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype=linear_dtype,
        output_dtype=x.dtype,
    )


def _constraints() -> dict[str, FunctionConstraints]:
    floats = frozenset({torch.float16, torch.bfloat16})
    return {
        "convrot_w4a4_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats, shape_rules=(MinDims(2),)),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=32)),
                ),
                "wscales": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
                    shape_rules=(ExactDims(1),),
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "convrot_groupsize": _ConvRotGroupConstraint(dtypes=frozenset({int})),
                "quant_group_size": _QuantGroupConstraint(dtypes=frozenset({int})),
                "linear_dtype": _Int4LinearDtypeConstraint(dtypes=frozenset({str})),
            },
            # ZLUDA exposes the AMD device through torch's CUDA device type.
            default_devices=frozenset({"cuda"}),
        )
    }


def register_kitchen_backend() -> bool:
    """Register ahead of stock backends without replacing any of them."""
    if not is_gfx1103_target():
        logging.info("INT4 Fast: gfx1103 Triton target not active; kitchen backend not registered.")
        return False

    registry.register(
        name=BACKEND_NAME,
        module=sys.modules[__name__],
        capabilities=_constraints(),
    )
    # comfy-kitchen 0.2.x has a setter but no public priority getter.  Preserve
    # every existing entry and prepend only our narrowly-scoped backend.
    current_priority = list(registry._priority)
    registry.set_priority(
        [BACKEND_NAME] + [name for name in current_priority if name != BACKEND_NAME]
    )
    logging.info(
        "INT4 Fast: registered comfy-kitchen backend %s for native gfx1103 W4A4.",
        BACKEND_NAME,
    )
    return True
