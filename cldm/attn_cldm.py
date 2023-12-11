import einops
import torch
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.diffusionmodules.openaimodel import MutualAttentionUNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion, MutualDiffusionWrapper
from ldm.util import log_txt_as_img, exists, default, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler


class MutualAttentionControlledUNetModel(MutualAttentionUNetModel):
    def forward(self, x_t, x_r, timesteps=None, context_t=None, context_r=None, location=None, control=None, locs=None, ks_r=None, vs_r=None, only_mid_control=False, **kwargs):
        hs_t, hs_r = [], []
        t_emb = timestep_embedding(timesteps, self.unet_target.model_channels, repeat_only=False)
        emb_t = self.unet_target.time_embed(t_emb)
        emb_r = self.unet_reference.time_embed(t_emb)

        if location is not None:
            emb_loc = self.loc_embed(location)
            if locs is not None:
                emb_loc = (1. - self.omega_end - self.omega_start) * emb_loc + \
                    self.omega_start * self.loc_embed(locs[0]) + \
                    self.omega_end * self.loc_embed(locs[1])
            emb_r = th.cat([emb_r, emb_loc], dim=1)
        
        if ks_r is not None:
            ks_s, ks_e = ks_r
        if vs_r is not None:
            vs_s, vs_e = vs_r

        h_t, h_r = x_t.type(self.unet_target.dtype), x_r.type(self.unet_reference.dtype)

        k_r, v_r = None, None
        for m_t, m_r in zip(self.unet_target.input_blocks, self.unet_reference.input_blocks):
            h_r = m_r(h_r, emb_r, context=context_r)
            if isinstance(h_r, tuple):
                h_r, k_r, v_r = h_r
                if ks_r is not None and vs_r is not None:
                    k_r = self.omega_start * ks_s.pop(0) + self.omega_end * ks_e.pop(0) + (1. - self.omega_end - self.omega_start) * k_r
                    v_r = self.omega_start * vs_s.pop(0) + self.omega_end * vs_e.pop(0) + (1. - self.omega_end - self.omega_start) * v_r
            h_t = m_t(h_t, emb_t, context=context_t, shared_k=k_r, shared_v=v_r)
            hs_t.append(h_t)
            hs_r.append(h_r)
            
        h_r = self.unet_reference.middle_block(h_r, emb_r, context=context_r)
        if isinstance(h_r, tuple):
            h_r, k_r, v_r = h_r
            if ks_r is not None and vs_r is not None:
                k_r = self.omega_start * ks_s.pop(0) + self.omega_end * ks_e.pop(0) + (1. - self.omega_end - self.omega_start) * k_r
                v_r = self.omega_start * vs_s.pop(0) + self.omega_end * vs_e.pop(0) + (1. - self.omega_end - self.omega_start) * v_r
        h_t = self.unet_target.middle_block(h_t, emb_t, context=context_t, shared_k=k_r, shared_v=v_r)

        if control is not None:
            h_t += control.pop()

        for m_t, m_r in zip(self.unet_target.output_blocks, self.unet_reference.output_blocks):
            h_r = m_r(th.cat([h_r, hs_r.pop()], dim=1), emb_r, context=context_r)
            if isinstance(h_r, tuple):
                h_r, k_r, v_r = h_r
                if ks_r is not None and vs_r is not None:
                    k_r = self.omega_start * ks_s.pop(0) + self.omega_end * ks_e.pop(0) + (1. - self.omega_end - self.omega_start) * k_r
                    v_r = self.omega_start * vs_s.pop(0) + self.omega_end * vs_e.pop(0) + (1. - self.omega_end - self.omega_start) * v_r
            if only_mid_control or control is None:
                h_t = m_t(th.cat([h_t, hs_t.pop()], dim=1), emb_t, context=context_t, shared_k=k_r, shared_v=v_r)
            else:
                h_t = m_t(th.cat([h_t, hs_t.pop() + control.pop()], dim=1), emb_t, context=context_t, shared_k=k_r, shared_v=v_r)

        del k_r, v_r
        h_r, h_t = h_r.type(x_r.dtype), h_t.type(x_t.dtype)

        if self.unet_target.predict_codebook_ids:
            return self.unet_target.id_predictor(h_t)
        else:
            return self.unet_target.out(h_t)



class MutualAttentionControlledLDM(LatentDiffusion):

    def __init__(self, mutual_key, mutual_cond_stage_key, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.model = MutualDiffusionWrapper(kwargs['unet_config'], kwargs['conditioning_key'])
        self.mutual_key = mutual_key
        self.mutual_cond_stage_key = mutual_cond_stage_key
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, bs=None, *args, **kwargs):
        x_t, c_t = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        x_r = batch[self.mutual_key]
        x_r = rearrange(x_r, 'b h w c -> b c h w')
        if bs is not None:
            x_r = x_r[:bs]
        x_r = x_r.to(self.device)
        encoder_posterior = self.encode_first_stage(x_r)
        x_r = self.get_first_stage_encoding(encoder_posterior).detach()

        l_r = None
        if self.mutual_cond_stage_key is not None:
            l_r = batch[self.mutual_cond_stage_key]
            if bs is not None:
                l_r = l_r[:bs]
            l_r = l_r.to(self.device)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        return x_t, x_r, dict(c_t_crossattn=[c_t], c_t_concat=[control]), l_r#, dict(c_r_crossattn=[c_r])
    
    def shared_step(self, batch, *args, **kwargs):
        x_t, x_r, c_t, l_r = self.get_input(batch, *args, *kwargs)
        loss = self(x_t, x_r, c_t, l_r)
        return loss
    
    def forward(self, x_t, x_r, c_t, l_r, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_t.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c_t is not None
            if self.cond_stage_trainable:
                c_t = self.get_learned_conditioning(c_t)
        return self.p_losses(x_t, x_r, c_t, t, l_r, *args, **kwargs)
    
    def apply_model(self, x_t_noisy, x_r, t, cond_t, l_r, locs=None, ks=None, vs=None, *args, **kwargs):
        assert isinstance(cond_t, dict)
        diffusion_model = self.model.diffusion_model

        cond_t_txt = torch.cat(cond_t['c_t_crossattn'], 1)

        if cond_t['c_t_concat'] is None:
            eps = diffusion_model(x_t=x_t_noisy, x_r=x_r, timesteps=t, context_t=cond_t_txt, context_r=None, location=l_r, control=None, locs=locs, ks=ks, vs=vs, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_t_noisy, hint=torch.cat(cond_t['c_t_concat'], 1), timesteps=t, context=cond_t_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x_t=x_t_noisy, x_r=x_r, timesteps=t, context_t=cond_t_txt, context_r=None, location=l_r, control=control, locs=locs, ks=ks, vs=vs, only_mid_control=self.only_mid_control)
        
        return eps
    
    def p_losses(self, x_t_start, x_r, cond_t, t, l_r, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_t_start))
        x_t_noisy = self.q_sample(x_start=x_t_start, t=t, noise=noise)
        model_output = self.apply_model(x_t_noisy, x_r, t, cond_t, l_r)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_t_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_t_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def p_mean_variance(self, x_t, x_r, c_t, t, l_r, clip_denoised: bool, locs=None, ks=None, vs=None, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x_t, x_r, t_in, c_t, l_r, locs=locs, ks=ks, vs=vs, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x_t, t, c_t, **corrector_kwargs)
        
        if return_codebook_ids:
            model_out, logits = model_out
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x_t, x_r, c_t, t, l_r, locs=None, ks=None, vs=None, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x_t.shape, x_t.device
        outputs = self.p_mean_variance(x_t=x_t, x_r=x_r, c_t=c_t, t=t, l_r=l_r, clip_denoised=clip_denoised,
                                       locs=locs, ks=ks, vs=vs,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Supported dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x_t.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, cond, reference, reference_location, shape, locs=None, ks=None, vs=None, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, reference, cond, ts, reference_location, locs=locs, ks=ks, vs=vs,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def sample(self, cond, reference, reference_location, locs=None, ks=None, vs=None, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  reference,
                                  reference_location,
                                  shape,
                                  locs=locs, ks=ks, vs=vs,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, reference, reference_location, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, reference=reference, reference_location=reference_location, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0, return_keys=None, 
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, 
                   plot_diffusion_rows=True, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, 
                   use_ema_scope=True, 
                   **kwargs):
        use_ddim = False #ddim_steps is not None

        log = dict()
        z_t, z_r, c_t, l_r = self.get_input(batch, bs=N)
        c_t_cat, c_t = c_t["c_t_concat"][0][:N], c_t["c_t_crossattn"][0][:N]
        N = min(z_t.shape[0], N)
        n_row = min(z_t.shape[0], n_row)

        log["target_reconstruction"] = self.decode_first_stage(z_t)
        log["reference_reconstruction"] = self.decode_first_stage(z_r)
        log["control"] = c_t_cat * 2.0 - 1.0
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z_t[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid
        
        if sample:
            samples, z_denoise_row = self.sample_log(cond={"c_t_concat": [c_t_cat], "c_t_crossattn": [c_t]}, 
                                                     reference=z_r, 
                                                     reference_location=l_r, 
                                                     batch_size=N, 
                                                     ddim=use_ddim, 
                                                     ddim_steps=ddim_steps, 
                                                     eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log['samples'] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        
        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_t_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_t_concat": [uc_cat], "c_t_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_t_concat": [c_t_cat], "c_t_crossattn": [c_t]},
                                             reference=z_r,
                                             reference_location=l_r,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
        
    def configure_optimizers(self):
        lr = self.learning_rate
        params = self.model.diffusion_model.unet_reference.parameters()
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt