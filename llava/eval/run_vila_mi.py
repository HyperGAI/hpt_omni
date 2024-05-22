import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

query = '''<image>
Generate a concise and appealing category name for a set of mobile phone theme images. The criteria are as follows:

Brevity: The category name must be under 6 words.
Accuracy: It should be a true reflection of the images' content and style.
Attractiveness: The name should be engaging to potential users looking for phone themes.
Uniqueness: Ensure the category is distinctive and highlights the images' unique features.
To create the category, analyze key aspects of the images: colors, subjects, and artistic style. Then, craft a name that captures these elements in a marketable way.

Examples: 'Abyssal Oceans', 'Alpine Meadow', 'Astral Horizons'.
'''

# query = '''<image>
# "I am providing you with a mobile phone theme image. Your task is to generate a detailed and engaging caption for this image. The caption should be concise and capture the essence of the image in the following aspects:
# Content: Describe the main subject or scene depicted in the image.
# Style: Characterize the artistic or design style of the image (e.g., minimalist, abstract, realistic, retro, etc.).
# Color: Mention the dominant or most noticeable colors and how they contribute to the overall mood or theme of the image.
# The caption should be crafted to not only describe but also to evoke interest and set the right tone for potential users who are browsing through phone themes. Aim for a caption that is both informative and enticing, suitable for use in a marketing context or a theme description."
# Some examples: aesthetic black, green kawaii, avengers clauncher
# '''


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):
    # Model
    disable_torch_init()
    if args.video_file is not None:
        if args.video_file.startswith("http") or args.video_file.startswith("https"):
            print("downloading video from url", args.video_file)
            response = requests.get(args.video_file)
            video_file = BytesIO(response.content)
        else:
            assert osp.exists(args.video_file), "video file not found"
            video_file = args.video_file
        from llava.mm_utils import opencv_extract_frames
        images = opencv_extract_frames(video_file, args.num_video_frames)
    elif args.image_file is not None:
        image_files = image_parser(args)
        images = load_images(image_files)
    elif args.image_dir is not None:
        images = os.listdir(args.image_dir)
        images = [osp.join(args.image_dir, item) for item in images]
        images = load_images(images)
        
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    # qs = args.query
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    print("input: ", qs)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)