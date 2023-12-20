import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import argparse

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

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--test_seqs", default=[1,7])
    args.add_argument("--exp_name", default="multidiff", required=True, help="name of the test experiment")
    args.add_argument("--no_save_image", action="store_true")
    args.add_argument("--no_save_video", action="store_true")
    args.add_argument("--clip_size", default=6, help="length of sub-video")
    args.add_argument("--window_size", default=2, help="window size")
    args = args.parse_args()

    model = create_model('./models/mutual_cldm_v15_infer.yaml')
    model.load_state_dict(load_state_dict('./models/mutual_finetune_control_epoch40.ckpt', location="cuda"))
    model = model.cuda()

    single_control_model = create_model('./models/mutual_cldm_v15.yaml')
    single_control_model.load_state_dict(load_state_dict('./models/mutual_finetune_control_epoch40.ckpt', location="cuda"))
    single_control_model = single_control_model.cuda()

    if not args.no_save_image:
        save_image_dir = os.path.join("./exp", args.exp_name, "image")
        os.makedirs(save_image_dir, exist_ok=True)
    
    if not args.no_save_video:
        save_video_dir = os.path.join("./exp", args.exp_name, "video")
        os.makedirs(save_video_dir, exist_ok=True)
    
    seq_names = args.test_seqs
    if isinstance(seq_names, int):
        seq_names = [seq_names]
    
    if isinstance(seq_names[0], str):
        raise ValueError("Type of sequence name must be an integer")
    
    for seq in seq_names:
        seq_name = f'seq_{seq:04d}'
        print("====================")
        print(f"Sampling Sequence {seq}")
        print("====================")

        if not args.no_save_image:
            os.makedirs(os.path.join(save_image_dir, seq_name), exist_ok=True)
        
        if not args.no_save_video:
            os.makedirs(os.path.join(save_video_dir, seq_name), exist_ok=True)
        
        print("Loading data ...")
        dataset = MyDataset(seq_name)
        x_samples = dict()

        start_frame = 0
        end_frame = start_frame + args.clip_size

        with torch.no_grad():
            print("Sampling initial start end pair ...")
            start_cond_txt, start_control, start_ref_loc, shape = get_data_info(dataset.__getitem__(start_frame))
            end_cond_txt, end_control, end_ref_loc, _ = get_data_info(dataset.__getitem__(end_frame))

            start_sample = single_control_model.sample(
                cond={"c_t_crossattn": [model.get_learned_conditioning(start_cond_txt)], "c_t_concat": [start_control]},
                reference=None,
                reference_location=None,
                batch_size=1,
                return_intermediates=False,
                verbose=True,
                shape=shape
            )
            start_sample = model.decode_first_stage(start_sample)
            encoder_posterior = model.encode_first_stage(start_sample)
            reference_prev = model.get_first_stage_encoding(encoder_posterior)

            reference_start = reference_prev.clone()

            x_samples[0] = [start_sample]

            end_sample = model.sample(
                cond={"c_t_crossattn": [model.get_learned_conditioning(end_cond_txt)], "c_t_concat": [end_control]},
                reference=reference_start,
                reference_location=start_ref_loc - end_ref_loc,
                batch_size=1,
                return_intermediates=False,
                verbose=True,
                shape=shape
            )
            end_sample = model.decode_first_stage(end_sample)
            encoder_posterior = model.encode_first_stage(end_sample)
            reference_end = model.get_first_stage_encoding(encoder_posterior)

            prev_ref_loc = start_ref_loc

            while end_frame < len(dataset) - 1:
                print(f"Sampling from frame {start_frame + 1} to frame {end_frame}")
                for i in range(start_frame + 1, end_frame + 1):
                    data_dict = dataset.__getitem__(i)

                    cond_txt, control, ref_loc, _ = get_data_info(data_dict)

                    cur_sample = model.sample(
                        cond={"c_t_crossattn": [model.get_learned_conditioning(cond_txt)], "c_t_concat": [control]},
                        reference=reference_prev,
                        reference_location=prev_ref_loc - ref_loc,
                        loc0_r=start_ref_loc - ref_loc,
                        loc1_r=end_ref_loc - ref_loc,
                        x0_r=reference_start,
                        x1_r=reference_end,
                        batch_size=1,
                        return_intermediates=False,
                        verbose=True,
                        shape=shape
                    )
                    cur_sample = model.decode_first_stage(cur_sample)
                    encoder_posterior = model.encode_first_stage(cur_sample)
                    reference_prev = model.get_first_stage_encoding(encoder_posterior)
                    prev_ref_loc = ref_loc

                    if i in x_samples.keys():
                        x_samples[i].append(cur_sample)
                    else:
                        x_samples[i] = [cur_sample]
                
                print("Calculating the average of frames", [i for i in range(start_frame + 1, start_frame + args.window_size + 1)])
                for i in range(start_frame + 1, start_frame + args.window_size + 1):
                    done_sample = x_samples[i]
                    done_tensor = torch.stack(done_sample, dim=0)
                    done_tensor = done_tensor.mean(dim=0)
                    x_samples[i] = done_tensor
                    x_sample = (einops.rearrange(done_tensor, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    if not args.no_save_image:
                        img = Image.fromarray(x_sample[0])
                        img.save(os.path.join(save_image_dir, seq_name, f"{i:03d}.png"))
                
                start_frame += args.window_size
                end_frame = start_frame + args.clip_size

                if end_frame >= len(dataset):
                    end_frame = len(dataset) - 1

                start_cond_txt, start_control, start_ref_loc, shape = get_data_info(dataset.__getitem__(start_frame))
                end_cond_txt, end_control, end_ref_loc, _ = get_data_info(dataset.__getitem__(end_frame))

                start_sample = x_samples[start_frame]
                encoder_posterior = model.encode_first_stage(start_sample)
                reference_prev = model.get_first_stage_encoding(encoder_posterior)

                reference_start = reference_prev.clone()

                end_sample = model.sample(
                    cond={"c_t_crossattn": [model.get_learned_conditioning(end_cond_txt)], "c_t_concat": [end_control]},
                    reference=reference_start,
                    reference_location=start_ref_loc - end_ref_loc,
                    batch_size=1,
                    return_intermediates=False,
                    verbose=True,
                    shape=shape
                )

                end_sample = model.decode_first_stage(end_sample)
                encoder_posterior = model.encode_first_stage(end_sample)
                reference_end = model.get_first_stage_encoding(encoder_posterior)

                prev_ref_loc = start_ref_loc

            for i in sorted(x_samples.keys()):
                x_sample = x_samples[i]
                if isinstance(x_sample, list):
                    x_sample = torch.stack(x_sample)
                    x_sample = x_sample.mean(dim=0)
                
                    x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    if not args.no_save_image:
                        img = Image.fromarray(x_sample[0])
                        img.save(os.path.join(save_image_dir, seq_name, f"{i:03d}.png"))
            
            # for i in range(sorted(x_samples.keys())):
            #         video.append(x_sample[0])
            
            # if not args.no_save_video:
            #     imageio.mimwrite(os.path.join(save_video_dir, seq_name, 'video.mp4'), video, fps=4)

if __name__ == "__main__":
    main()