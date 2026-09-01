#!/usr/bin/env python3
"""DreamBooth-style LoRA fine-tuning for Stable Diffusion.

Standalone by design: this script only depends on its own CLI args, never on
the FastAPI app or its database, so it can be run directly on a cloud GPU box
(Colab, RunPod, a rented A100, ...) exactly the way the backend invokes it:

    python train_lora.py \
        --instance_data_dir ./my_photos \
        --output_dir ./out/my_concept \
        --instance_prompt "a photo of sks person" \
        --max_train_steps 800

Progress is reported by printing lines of the form
``IMUSE_PROGRESS step=<n> total=<total>`` to stdout, which the backend's job
runner tail-parses. When invoked with --mock, no ML libraries are imported at
all and a tiny placeholder adapter is produced instead -- this is what lets
the rest of the app (uploads, job tracking, UI) be developed and tested on a
machine with no GPU and none of torch/diffusers/peft installed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip real training; write a tiny placeholder adapter instead. "
        "Used for GPU-less development of the surrounding app.",
    )
    return parser.parse_args()


def load_captions(instance_data_dir: Path, fallback_prompt: str) -> list[tuple[Path, str]]:
    images = sorted(
        p for p in instance_data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    if not images:
        raise SystemExit(f"no training images found in {instance_data_dir}")

    captions_path = instance_data_dir / "captions.json"
    captions: dict[str, str] = {}
    if captions_path.exists():
        captions = json.loads(captions_path.read_text())

    return [(img, captions.get(img.name, fallback_prompt)) for img in images]


def write_metadata(output_dir: Path, args: argparse.Namespace, num_images: int) -> None:
    metadata = {
        "base_model": args.base_model,
        "instance_prompt": args.instance_prompt,
        "resolution": args.resolution,
        "lora_rank": args.lora_rank,
        "max_train_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "num_training_images": num_images,
    }
    (output_dir / "imuse_metadata.json").write_text(json.dumps(metadata, indent=2))


def run_mock(args: argparse.Namespace) -> None:
    """Fake training loop: no torch/diffusers import, just simulates progress
    and produces a placeholder adapter file so downstream code (inference,
    the UI) has something real to point at."""
    instance_data_dir = Path(args.instance_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_captions(instance_data_dir, args.instance_prompt)

    steps = max(1, min(args.max_train_steps, 50))  # keep mock runs fast
    for step in range(1, steps + 1):
        time.sleep(0.02)
        print(f"IMUSE_PROGRESS step={step} total={steps}", flush=True)

    (output_dir / "adapter_model.mock").write_text(
        f"mock LoRA adapter for prompt: {args.instance_prompt}\n"
        f"trained on {len(samples)} image(s) from {instance_data_dir}\n"
    )
    write_metadata(output_dir, args, len(samples))
    print(f"IMUSE_PROGRESS step={steps} total={steps}", flush=True)
    print("training complete (mock mode)")


def run_real(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no CUDA GPU detected -- real LoRA training on CPU will be "
            "extremely slow. Run this on a cloud GPU instance (Colab/RunPod/etc).",
            file=sys.stderr,
        )
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    instance_data_dir = Path(args.instance_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_captions(instance_data_dir, args.instance_prompt)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class InstanceDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, idx: int):
            path, caption = samples[idx]
            image = Image.open(path).convert("RGB")
            pixel_values = image_transforms(image)
            input_ids = tokenizer(
                caption,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            return {"pixel_values": pixel_values, "input_ids": input_ids}

    dataloader = DataLoader(
        InstanceDataset(), batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    unet.train()
    global_step = 0
    data_iter = iter(dataloader)
    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        print(f"IMUSE_PROGRESS step={global_step} total={args.max_train_steps}", flush=True)
        if global_step % 20 == 0:
            print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}")

    # Save in diffusers' own LoRA state-dict format (component-prefixed keys),
    # not a raw PEFT save -- StableDiffusionPipeline.load_lora_weights() only
    # recognizes the former. A bare unet.save_pretrained() here silently
    # produces an adapter that load_lora_weights() can't match against any
    # component, so it loads nothing and generation quietly falls back to the
    # unmodified base model.
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    write_metadata(output_dir, args, len(samples))
    print("training complete")


def main() -> None:
    args = parse_args()
    if args.mock:
        run_mock(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
