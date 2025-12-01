import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # Pass LLM parameters if using text conditioning
        model_kwargs = {
            'input_size': args.img_size,
            'in_channels': 3,
            'num_classes': args.class_num,
            'attn_drop': args.attn_dropout,
            'proj_drop': args.proj_dropout,
        }
        if args.use_text_conditioning or '-Text' in args.model:
            model_kwargs['llm_model_name'] = args.llm_model_name
            model_kwargs['freeze_llm'] = args.freeze_llm

        self.net = JiT_models[args.model](**model_kwargs)
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def drop_text(self, input_ids, attention_mask):
        """Drop text conditioning for classifier-free guidance"""
        bsz = input_ids.shape[0]
        drop = torch.rand(bsz, device=input_ids.device) < self.label_drop_prob
        # Create empty text tokens (all zeros) for unconditional
        empty_input_ids = torch.zeros_like(input_ids)
        empty_attention_mask = torch.zeros_like(attention_mask)
        # Apply dropout
        out_input_ids = torch.where(drop.unsqueeze(1), empty_input_ids, input_ids)
        out_attention_mask = torch.where(drop.unsqueeze(1), empty_attention_mask, attention_mask)
        return out_input_ids, out_attention_mask

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels=None, input_ids=None, attention_mask=None):
        # Handle text or label dropout for classifier-free guidance
        if self.net.use_text_conditioning:
            assert input_ids is not None and attention_mask is not None
            if self.training:
                input_ids_dropped, attention_mask_dropped = self.drop_text(input_ids, attention_mask)
            else:
                input_ids_dropped, attention_mask_dropped = input_ids, attention_mask
        else:
            assert labels is not None
            labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        if self.net.use_text_conditioning:
            x_pred = self.net(z, t.flatten(), input_ids=input_ids_dropped, attention_mask=attention_mask_dropped)
        else:
            x_pred = self.net(z, t.flatten(), y=labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels=None, input_ids=None, attention_mask=None):
        if self.net.use_text_conditioning:
            assert input_ids is not None and attention_mask is not None
            device = input_ids.device
            bsz = input_ids.size(0)
        else:
            assert labels is not None
            device = labels.device
            bsz = labels.size(0)

        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            if self.net.use_text_conditioning:
                z = stepper(z, t, t_next, labels=None, input_ids=input_ids, attention_mask=attention_mask)
            else:
                z = stepper(z, t, t_next, labels=labels)
        # last step euler
        if self.net.use_text_conditioning:
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels=None, input_ids=input_ids, attention_mask=attention_mask)
        else:
            z = self._euler_step(z, timesteps[-2], timesteps[-1], labels=labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels=None, input_ids=None, attention_mask=None):
        if self.net.use_text_conditioning:
            # conditional
            x_cond = self.net(z, t.flatten(), input_ids=input_ids, attention_mask=attention_mask)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

            # unconditional (empty text)
            empty_input_ids = torch.zeros_like(input_ids)
            empty_attention_mask = torch.zeros_like(attention_mask)
            x_uncond = self.net(z, t.flatten(), input_ids=empty_input_ids, attention_mask=empty_attention_mask)
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        else:
            # conditional
            x_cond = self.net(z, t.flatten(), y=labels)
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

            # unconditional
            x_uncond = self.net(z, t.flatten(), y=torch.full_like(labels, self.num_classes))
            v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels=None, input_ids=None, attention_mask=None):
        v_pred = self._forward_sample(z, t, labels=labels, input_ids=input_ids, attention_mask=attention_mask)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels=None, input_ids=None, attention_mask=None):
        v_pred_t = self._forward_sample(z, t, labels=labels, input_ids=input_ids, attention_mask=attention_mask)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels=labels, input_ids=input_ids, attention_mask=attention_mask)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
