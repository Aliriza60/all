
# python inpainting.py --model_name runwayml/stable-diffusion-inpainting --image overture-creations-5sI6fQgYIuo.png --mask_image overture-creations-5sI6fQgYIuo_mask.png --output output --prompt "Face of a yellow cat, high resolution, sitting on a park bench" --single_file_or_not 2

# python inpainting.py --model_path sd-v1-5-inpainting.ckpt --image overture-creations-5sI6fQgYIuo.png --mask_image overture-creations-5sI6fQgYIuo_mask.png --output output --prompt "Face of a yellow cat, high resolution, sitting on a park bench" --single_file_or_not 1

# runwayml/stable-diffusion-v1-5

import torch
import argparse
import os
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from torchvision.transforms import ToPILImage, Resize, Compose
from diffusers import LMSDiscreteScheduler
from diffusers import DDIMScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerDiscreteScheduler
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler
from diffusers import EulerAncestralDiscreteScheduler



parser = argparse.ArgumentParser()

parser.add_argument(
    "--workerTaskId",
    type=int,
    default=0,
    help="Worker Task Id"
)

parser.add_argument(
    "--model_path",
    type=str,
    default="", 
    help="Model Path",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="",
    help="Model Name",
)

parser.add_argument(
    "--image",
    type=str,
    default="",
    help="Image Path",
)

parser.add_argument(
    "--mask_image",
    type=str,
    default="",
    help="Mask Image Path",
)

parser.add_argument(
    "--output",
    type=str,
    default="",
    help="Output Folder",
)

parser.add_argument(
    "--prompt",
    type=str,
    default="",
    help="Positive Prompt",
)

parser.add_argument(
    "--negative_prompt",
    type=str,
    default="",
    help="Negative Prompt",
)

parser.add_argument(
    "--strength",
    type=float,
    default=0.8,
    help="Strength"
)
parser.add_argument(
    "--samples",
    type=int,
    default=1,
    help="Samples"
)
parser.add_argument(
    "--inference_steps",
    type=int,
    default=30,
    help="Inference_Steps"
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="Guidance Scale"
)

parser.add_argument(
   "--scheduler",
    type=str,
    default="EulerDiscreteScheduler",
    help="",  
)

parser.add_argument(
    "--seed",
    type=int,
    default=3466454,
    help="Seed"
)

parser.add_argument(
    "--width",
    type=int,
    default=512,
    help=""
)

parser.add_argument(
    "--height",
    type=int,
    default=512,
    help=""
)

parser.add_argument(
    "--single_file_or_not",
    type=int,
    default=2,
    help="Check single file or not"
)

ap = parser.parse_args()

image_org = load_image(ap.image)

image_mask = load_image(ap.mask_image)

if ap.single_file_or_not == 1:

    pipeline = StableDiffusionInpaintPipeline.from_single_file(ap.model_path, safety_checker=None).to("cuda")

elif ap.single_file_or_not == 2:

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(ap.model_name, safety_checker=None).to("cuda")

    
if ap.scheduler == "DDIMScheduler":
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "LMSDiscreteScheduler":
    pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "DPMSolverMultistepScheduler":
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "EulerDiscreteScheduler":
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "PNDMScheduler":
    pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "DDPMScheduler":
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "EulerAncestralDiscreteScheduler":
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
else:
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)


pipeline.enable_xformers_memory_efficient_attention()

generator = torch.Generator().manual_seed(ap.seed)

images = pipeline(
    prompt=ap.prompt,
    image=image_org,
    mask_image = image_mask,
    width = ap.width,
    height = ap.height,
    generator = generator,
    negative_prompt=ap.negative_prompt,
    num_images_per_prompt=ap.samples,
    num_inference_steps=ap.inference_steps,
    strength=ap.strength,
    guidance_scale = ap.guidance_scale,
).images

os.makedirs(ap.output, exist_ok=True)
for i, image in enumerate(images):
    image.save(f""+ap.output+"/"+str(i)+".png")


