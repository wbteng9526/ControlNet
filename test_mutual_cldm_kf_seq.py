import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

from tutorial_dataset_test import MyDataset
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def get_data_info(data_dict):
    cond_txt = data_dict['txt']
    control = torch.from_numpy(data_dict['hint']).float().cuda()
    control = control.unsqueeze(0)
    control = einops.rearrange(control, 'b h w c -> b c h w')
    _, C, H, W = control.shape

    reference_location = torch.from_numpy(data_dict['loc']).float().cuda()
    reference_location = reference_location.unsqueeze(0)

    shape = (1, 4, H // 8, W // 8)
    return cond_txt, control, reference_location, shape


if __name__ == "__main__":
    single_model = create_model('./models/cldm_v15.yaml')
    single_model.load_state_dict(load_state_dict('./models/single_control_epoch12.ckpt', location='cuda'))
    single_model = single_model.cuda()

    mutual_model = create_model('./models/mutual_cldm_v15.yaml')
    mutual_model.load_state_dict(load_state_dict('./models/mutual_fintune_control_epoch19.ckpt', location='cuda'))
    mutual_model = mutual_model.cuda()

    ddim_sampler = DDIMSampler(single_model)

    # TODO: create argparse
    for seq in tqdm([1,7,9]):
        seq_name = f'seq_{seq:04d}'
        save_image = True
        save_video = True
        save_image_dir = f'test/mutual_kf_omega0.1/{seq_name}/image'
        save_video_dir = f'test/mutual_kf_omega0.1/{seq_name}/video'
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_video_dir, exist_ok=True)

        dataset = MyDataset(seq_name)
        reference = None
        x_samples = []
        length = 6

        start_frame = 0
        end_frame = start_frame + length

        # set up start and end key frame
        with torch.no_grad():
            start_cond_txt, start_control, start_ref_location, shape = get_data_info(dataset.__getitem__(start_frame))
            end_cond_txt, end_control, end_ref_location, _ = get_data_info(dataset.__getitem__(end_frame))

            start_sample = single_model.sample(
                cond={"c_crossattn": [single_model.get_learned_conditioning(start_cond_txt)], "c_concat": [start_control]}, 
                batch_size=1, return_intermediates=False, verbose=True, shape=shape)
            start_sample = single_model.decode_first_stage(start_sample)

            encoder_posterior = mutual_model.encode_first_stage(start_sample)
            reference = mutual_model.get_first_stage_encoding(encoder_posterior)

            reference_start = mutual_model.get_first_stage_encoding(encoder_posterior)

            end_sample = mutual_model.sample(
                cond={"c_t_crossattn": [mutual_model.get_learned_conditioning(end_cond_txt)], "c_t_concat": [end_control]}, 
                reference=reference_start, reference_location=start_ref_location - end_ref_location,
                batch_size=1, return_intermediates=False, verbose=True, shape=shape)
            end_sample = single_model.decode_first_stage(end_sample)

            encoder_posterior = mutual_model.encode_first_stage(end_sample)
            reference_end = mutual_model.get_first_stage_encoding(encoder_posterior)

            anchor_location = start_ref_location

            for i in tqdm(range(len(dataset))):
                if i == start_frame:
                    x_sample = start_sample
                elif i == end_frame:
                    x_sample = end_sample
                    start_frame = end_frame
                    end_frame += length
                    if end_frame >= len(dataset):
                        end_frame = len(dataset) - 1
                        if end_frame == start_frame:
                            continue

                    start_cond_txt, start_control, start_ref_location, shape = get_data_info(dataset.__getitem__(start_frame))
                    end_cond_txt, end_control, end_ref_location, _ = get_data_info(dataset.__getitem__(end_frame))

                    start_sample = end_sample

                    encoder_posterior = mutual_model.encode_first_stage(start_sample)
                    reference = mutual_model.get_first_stage_encoding(encoder_posterior)

                    reference_start = mutual_model.get_first_stage_encoding(encoder_posterior)
         
                    anchor_location = start_ref_location

                    end_sample = mutual_model.sample(
                        cond={"c_t_crossattn": [single_model.get_learned_conditioning(end_cond_txt)], "c_t_concat": [end_control]}, 
                        reference=reference_start, reference_location=start_ref_location - end_ref_location,
                        batch_size=1, return_intermediates=False, verbose=True, shape=shape)
                    end_sample = single_model.decode_first_stage(end_sample)

                    
                    encoder_posterior = mutual_model.encode_first_stage(end_sample)
                    reference_end = mutual_model.get_first_stage_encoding(encoder_posterior)

                else:
                    data_dict = dataset.__getitem__(i)

                    cond_txt, control, reference_location, _ = get_data_info(data_dict)                
                
                    x_sample = mutual_model.sample(
                        cond={"c_t_crossattn": [mutual_model.get_learned_conditioning(cond_txt)], "c_t_concat": [control]}, 
                        reference=reference, reference_location=anchor_location - reference_location, 
                        loc0_r=start_ref_location - reference_location, loc1_r=end_ref_location - reference_location,
                        x0_r=reference_start, x1_r=reference_end,
                        batch_size=1, return_intermediates=False, verbose=True, shape=shape)
                    x_sample = mutual_model.decode_first_stage(x_sample)
                    
                    encoder_posterior = mutual_model.encode_first_stage(x_sample)
                    reference = mutual_model.get_first_stage_encoding(encoder_posterior)
                    anchor_location = reference_location

                x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                x_samples.append(x_sample[0])

                if save_image:
                    img = Image.fromarray(x_sample[0])
                    img.save(os.path.join(save_image_dir, f'{i:03d}.png'))
            
            if save_video:
                imageio.mimwrite(os.path.join(save_video_dir, 'video.mp4'), x_samples, fps=4)