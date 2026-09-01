#!/usr/bin/env python3
"""Generate images from a base Stable Diffusion model, optionally with a
LoRA adapter produced by train_lora.py applied on top.

Standalone, same spirit as train_lora.py:

    python inference.py \
        --base_model runwayml/stable-diffusion-v1-5 \
        --lora_dir ./out/my_concept \
        --prompt "a photo of sks person on the beach" \
        --output_dir ./generated --num_images 4

With --mock, no ML libraries are imported; solid-color placeholder PNGs are
written instead so the rest of the app can be developed/tested without a GPU.
"""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_dir", default=None, help="Path to a trained LoRA adapter dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def run_mock(args: argparse.Namespace) -> list[str]:
    from PIL import Image, ImageDraw

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i in range(args.num_images):
        digest = hashlib.sha256(f"{args.prompt}-{args.seed}-{i}".encode()).hexdigest()
        color = tuple(int(digest[j : j + 2], 16) for j in (0, 2, 4))
        img = Image.new("RGB", (args.resolution, args.resolution), color=color)
        draw = ImageDraw.Draw(img)
        label = args.prompt[:60] + ("..." if len(args.prompt) > 60 else "")
        draw.rectangle([0, 0, args.resolution - 1, args.resolution - 1], outline="white", width=4)
        draw.text((16, 16), "MOCK GENERATION", fill="white")
        draw.text((16, 40), label, fill="white")
        path = output_dir / f"{digest[:16]}.png"
        img.save(path)
        paths.append(str(path))
    print(json.dumps({"images": paths}))
    return paths


def run_real(args: argparse.Namespace) -> list[str]:
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Safety checker disabled: this is a private, personal instance generating
    # from the operator's own trained concepts, not a public-facing product.
    # SD1.5's default checker is also prone to false positives on face
    # close-ups, which is exactly this app's typical use case, and a flagged
    # image is silently replaced with solid black with no way to inspect it.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.lora_dir:
        pipe.load_lora_weights(args.lora_dir)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        generator=generator,
    )

    paths: list[str] = []
    for i, image in enumerate(result.images):
        digest = hashlib.sha256(f"{args.prompt}-{args.seed}-{i}".encode()).hexdigest()
        path = output_dir / f"{digest[:16]}.png"
        image.save(path)
        paths.append(str(path))

    print(json.dumps({"images": paths}))
    return paths


def main() -> None:
    args = parse_args()
    if args.mock:
        run_mock(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
