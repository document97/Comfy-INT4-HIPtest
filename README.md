# Comfy-INT4-HIPtest

An AMD/HIP-oriented ComfyUI custom node for INT4 W4A4 ConvRot diffusion
models. The primary validation target is the Radeon 780M (`gfx1103`) running
Windows, HIP SDK 5.7, ZLUDA and a Triton build with native INT4 support.

This plugin does not contain a Triton wheel, LLVM build or model weights. It
expects the native-INT4 Triton wheel from
[triton-windows-for-radeon780m](https://github.com/document97/triton-windows-for-radeon780m)
and a ComfyUI/comfy-kitchen installation with W4A4 ConvRot support.

## What it provides

- W4A4 `convrot_w4a4` model loading and saving nodes.
- On-the-fly INT4 quantization for supported floating-point checkpoints.
- Mixed INT4/INT8 quantization metadata handling.
- Native gfx1103 Triton W4A4 dispatch through comfy-kitchen.
- Hadamard ConvRot handling and LoRA-compatible model patching.
- A narrow backend registration that only claims explicit INT4 calls and
  supported ConvRot/quantization group sizes.

The native path uses `tl.dot` with INT4 operands and INT32 accumulation. The
matching compiler lowers it to AMD's `v_wmma_i32_16x16x16_iu4` instruction. It
does not intentionally fall back to eager execution for the native INT4 path.

## Nodes

- **Load Diffusion Model INT4 (W4A4)** (`OTUNetLoaderW4A4`)
- **Save Int4 Model** (`INT4ModelSave`)

## Installation

Clone this repository into ComfyUI's `custom_nodes` directory:

```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/document97/Comfy-INT4-HIPtest.git
```

Install the native Triton wheel first. For the build and verification steps,
see the [Triton compiler repository](https://github.com/document97/triton-windows-for-radeon780m).
Use the same Python executable that launches ComfyUI.

Restart ComfyUI after installing or updating the node.

## Requirements

- ComfyUI dev branch with W4A4 ConvRot support.
- `comfy-kitchen` with the required quantized tensor APIs.
- PyTorch exposed through `torch.cuda` by the working ZLUDA environment.
- A Triton build that reports a HIP `gfx1103` target and provides `tl.int4`
  and `tl.uint4`.
- For the tested setup: Radeon 780M, HIP SDK 5.7, ZLUDA and Python 3.12.

The plugin is not a general CUDA INT4 implementation. Other GPUs may require
different Triton kernels and backend constraints.

## Verified model

The primary validation model is
[Krea2 Turbo INT4](https://huggingface.co/comfyanonymous/int4_tests/blob/main/split_files/diffusion_models/krea2_turbo_convrot_int4_fast.safetensors).
Place model files in ComfyUI's normal diffusion-model directory and load them
with the W4A4 node.

The first run may compile Triton kernels. Subsequent runs reuse the Triton
cache. If ComfyUI reports the CUDA backend instead of
`gfx1103_int4_triton`, check that the native Triton wheel is installed in the
same Python environment as ComfyUI.

## Runtime diagnostics

The backend name is:

```text
gfx1103_int4_triton
```

Enable optional timing output with:

```powershell
$env:COMFY_INT4_PROFILE = "1"
```

Only use profiling while diagnosing performance; it adds synchronization and
can make interactive ComfyUI use feel slower.

## Credits

This project is based on and adapted from
[viralvfx/ComfyUI-INT4-Fast](https://github.com/viralvfx/ComfyUI-INT4-Fast).
That project provided the W4A4/ConvRot layout, quantized checkpoint handling,
model patching and ComfyUI integration foundation. This repository's changes
focus on the AMD/HIP gfx1103 backend registration and native Triton execution
path.

Additional upstream credit goes to
[BobJohnson24/ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)
for the INT8 quantization and ConvRot architecture that informed the broader
implementation.

## License

See [`LICENSE`](LICENSE). Upstream licenses and attribution notices are
retained where applicable.
