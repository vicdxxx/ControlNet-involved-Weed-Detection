from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = create_model('./models/cldm_v15.yaml').cpu()

model_dict = {
    # 'Lambsquarters': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Lambsquarters\version_0\checkpoints\epoch=22-step=22585.ckpt',
    # 'Lambsquarters': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize640_Lambsquarters\version_0\checkpoints\epoch=49-step=13349.ckpt',
    # 'Lambsquarters': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_Lambsquarters_bbox\version_1\checkpoints\epoch=49-step=17649.ckpt',
    
    # 'Purslane': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Purslane_25_to_50_epochs\version_0\checkpoints\epoch=24-step=6474.ckpt',
    # 'PalmerAmaranth': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_PalmerAmaranth\version_0\checkpoints\epoch=49-step=6499.ckpt',
    # 'Waterhemp': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Waterhemp\version_0\checkpoints\epoch=29-step=28229.ckpt',
    # 'Waterhemp': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize640_Waterhemp\version_0\checkpoints\epoch=49-step=18749.ckpt',
    # 'MorningGlory': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize640_MorningGlory\version_0\checkpoints\epoch=29-step=8729.ckpt',
    # 'Goosegrass': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Goosegrass\version_0\checkpoints\epoch=49-step=4749.ckpt',
    # 'Carpetweed': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Carpetweed\version_0\checkpoints\epoch=49-step=8949.ckpt',
    
    # 'SpottedSpurge': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_SpottedSpurge\version_0\checkpoints\epoch=49-step=8499.ckpt',
    'SpottedSpurge': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_SpottedSpurge_bbox\version_0\checkpoints\epoch=49-step=10699.ckpt',
    
    # 'Ragweed': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Ragweed\version_0\checkpoints\epoch=49-step=5849.ckpt',
    # 'Eclipta': r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Eclipta\version_0\checkpoints\epoch=49-step=2049.ckpt',
}

# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Lambsquarters\version_0\checkpoints\epoch=22-step=22585.ckpt', location='cuda'))

# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Purslane\version_0\checkpoints\epoch=29-step=10139.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Purslane_block\version_0\checkpoints\epoch=48-step=16561.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Purslane_25_to_50_epochs\version_0\checkpoints\epoch=24-step=6474.ckpt', location='cuda'))

# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Pigweed\version_0\checkpoints\epoch=29-step=3329.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_PalmerAmaranth\version_0\checkpoints\epoch=49-step=6499.ckpt', location='cuda'))

# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_crop1500_resize640_Waterhemp\version_0\checkpoints\epoch=29-step=28229.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize640_MorningGlory\version_0\checkpoints\epoch=29-step=8729.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Goosegrass\version_0\checkpoints\epoch=49-step=4749.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Carpetweed\version_0\checkpoints\epoch=49-step=8949.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_SpottedSpurge\version_0\checkpoints\epoch=49-step=8499.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Ragweed\version_0\checkpoints\epoch=49-step=5849.ckpt', location='cuda'))
# model.load_state_dict(load_state_dict(r'D:\BoyangDeng\StableDiffusion\ControlNet\lightning_logs_origin_resize_640_Eclipta\version_0\checkpoints\epoch=49-step=2049.ckpt', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        # detected_map[np.min(img, axis=2) < 127] = 255
        detected_map = img

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=True, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

prompt = 'high details, plants in the field'
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
eta = 0.0
# seed = 1310373017
# use same seed will generate similar style images?
seed = -1
scale = 9.0
ddim_steps =50
guess_mode =False
strength=1.0
# strength=1.5
image_resolution=640
num_samples = 4
input_images =[]

def list_dir(path, list_name, extension, return_names=False):
    import os
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                if return_names:
                    list_name.append(file)
                else:
                    list_name.append(file_path)
    try:
        list_name = sorted(list_name, key=lambda k: int(os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name

# sku='Lambsquarters'
# sku='Purslane'

# sku='Pigweed'
# sku='PalmerAmaranth'

# sku='Waterhemp'
# sku='MorningGlory'
# sku='Goosegrass'
# sku='Carpetweed'
# sku='SpottedSpurge'
# sku='Ragweed'
# sku='Eclipta'

skus = [
    # 'Lambsquarters',
    # 'PalmerAmaranth',
    # 'Waterhemp',
    # 'MorningGlory',
    # 'Purslane',
    # 'Goosegrass',
    # 'Carpetweed',
    'SpottedSpurge',
    # 'Ragweed',
    # 'Eclipta',
]

# src_dir = r'D:\Dataset\WeedData\weed_10_species\train_image_for_controlnet'
# root_dir = r'D:\Dataset\WeedData\weed_all\train_image_for_controlnet'
src_dir = r'D:\Dataset\WeedData\weed_10_species\train_image_for_controlnet_no_block'
dst_dir = r'D:\test'
# im_dir = os.path.join(src_dir, f'instances_train2017_{sku}_images_origin_block_hint_detection')
print(skus)
for sku in skus:
    # if sku == 'Eclipta':
    #     continue
    im_dir = os.path.join(src_dir, f'{sku}_hint_for_synthetic')
    im_paths = list_dir(im_dir, [],'.jpg')
    save_dir = os.path.join(dst_dir, f'{sku}_synthetic')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    from tqdm import tqdm
    model.load_state_dict(load_state_dict(model_dict[sku], location='cuda'))
    for i_im, im_path in tqdm(enumerate(im_paths)):
        im_name = os.path.basename(im_path)
        base_name = im_name.split('.jpg')[0]
        im_idx = base_name.split('_')[-1]
        print(base_name)
        if int(im_idx) < 0:
            continue
        # if offset+i_im<105:
        #     continue
        offset = 0
        seed = offset+i_im
        print('seed:', seed)
        input_image = cv2.imread(im_path)
        results = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
        for i_res, result in enumerate(results):
            cv2.imwrite(os.path.join(save_dir, base_name + '_' + str(i_res) + '.jpg'), result)
