import einops
import torch
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, DiffusionWrapper
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler



class MutualAttentionLDM(LatentDiffusion):

    def __init__(self, mutual_key, mutual_cond_stage_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutual_key = mutual_key
        self.mutual_cond_stage_key = mutual_cond_stage_key

    @torch.no_grad()
    def get_input(self, batch, *args, **kwargs):
        x_t, c_t = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        x_r, c_r = super().get_input(batch, self.mutual_key, cond_key=self.mutual_cond_stage_key, *args, **kwargs)
        c_r = None # TODO add camera delta R and delta T like zero-123
        return x_t, x_r, dict(c_crossattn=[c_t]), dict(c_crossattn=[c_r])
    
    def shared_step(self, batch, *args, **kwargs):
        x_t, x_r, c_t, c_r = self.get_input(batch, *args, *kwargs)
        loss = self(x_t, x_r, c_t, c_r)
        return loss
    
    def forward(self, x_t, x_r, c_t, c_r, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_t.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x_t, x_r, c_t, c_r, t, *args, **kwargs)
    
    def apply_model(self, x_t_noisy, x_r, t, cond_t, cond_r, return_ids=False):
        assert isinstance(cond_t, dict)
        assert isinstance(cond_r, dict)
        x_recon = self.model(x_t_noisy, x_r, t, **cond_t, **cond_r)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    
    def p_losses(self, x_t_start, x_r, cond_t, cond_r, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_t_start))
        x_t_noisy = self.q_sample(x_start=x_t_start, t=t, noise=noise)
        model_output = self.apply_model(x_t_noisy, x_r, t, cond_t, cond_r)

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
    
    def p_mean_variance(self, x_t, x_r, c_t, c_r, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x_t, x_r, t_in, c_t, c_r, return_ids=return_codebook_ids)

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
    def p_sample(self, x_t, x_r, c_t, c_r, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantized_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x_t.shape, x_t.device
        outputs = self.p_mean_variance(x_t=x_t, x_r=x_r, c_t=c_t, c_r=c_r, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantized_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Supported dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0, return_keys=None, 
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, 
                   plot_diffusion_rows=True, unconditional_guidance_scale=1, unconditional_guidance_label=None, 
                   use_ema_scope=True, 
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z_t, z_r, c_t, c_r = self.get_input(batch, bs=N)
        N = min(z_t.shape[0], N)
        n_row = min(z_t.shape[0], n_row)

        log["target_reconstruction"] = self.decode_first_stage(z_t)
        log["reference_reconstruction"] = self.decode_first_stage(z_r)
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc_t = self.cond_stage_model.decode(c_t)
                log['target_conditioning'] = xc_t
            elif isimage(xc_t):
                log['target_conditioning'] = xc_t
            if ismap(xc_t):
                log["original_conditioning"] = xc_t
        
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
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c_t, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log['samples'] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
            
            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples
        
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c_t,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
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