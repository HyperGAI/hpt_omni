# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import re
from io import BytesIO
import os, os.path as osp
import gradio as gr

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


MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b/stage3'
MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/llama3_8b/0614_fuse_interleaved_ocr_data/stage3'
MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/llama3_8b/0614_fuse_interleaved_ocr_data/stage3_with_text'
# MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b/stage3_hpto_v0'
MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b_internvit6b/stage3_with_text_06m'
MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/hpt_llama3_8b_internvit6b_fix_lr/stage3_with_text_06m'
conv_mode = 'llama_3'
conv_mode = 'llama_3_fix'

# MODEL_PATH = '/export/share/yucheng/hpt/hpt_omni/checkpoints/vila_3b/stage3'
# conv_mode = 'v1'

# MODEL_PATH = '/export/share/models/VILA1.5-3b'
# conv_mode = 'v1'

args = argparse.Namespace(
    model_path = MODEL_PATH,
    conv_mode = conv_mode,
    sep = ',',
    temperature = 0.2,
    top_p = None,
    num_beams = 1,
    max_new_tokens = 512,
    model_base = None
)

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

def inference(image=None, question=None):
    qs = question
    images = [image]
    
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
    print(prompt)
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
    
    return outputs


def gradio_demo(text, image):
    out_image_path = ''
    try:
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        question = text
        answer = inference(image=image, question=question)
    except Exception as e:
        print (e)
        answer = 'There is some error with your input, Please try again!'
    
    return answer


interface = gr.Interface(fn=gradio_demo, inputs=['text', 'image'], outputs='text')
interface.launch(share=True)