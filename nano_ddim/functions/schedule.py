import numpy as np
import torch

"""
    diffusion_schedule tells us the noise levels and signal levels of the noisy image corresponding to the
    actual diffusion time at each point in the diffusion process.
    noisy_images = x_{t} = sqrt(alpha_{t}) * x_{0} + sqrt(1 - alpha_{t}) * noise
    
    cosine schedule: as introduced in section 3.2 of "improved diffusion model"
        it is symmetric, slow towards the start and end of the diffusion process.
    noise_rates: sqrt(1 - alpha_bar)
    signal_rate: sqrt(alpha_bar)
    Let us say, that a diffusion process starts at time = 0, and ends at time = 1(or 1000). 
    params:
        t: time step between [0, 1]
            can have shape [batch_size] or [batch_size, seq_len]
            can be either discrete (common in diffusion models) or continuous (common in score-based model)
            the latter allows the number of sampling steps can be changed at inference time
    return:
        alpha_bar: have same shape as t
    Note that:
        alpha_{t} = 1 - beta_{t}
        alpha_bar_{t} = alpha_{1} * ... * alpha_{t}
        DDIM's alpha_{t} = DDPM's alpha_bar_{t}
"""
def cosine_schedule(t : torch.FloatTensor,
                    alpha_bar_start : torch.Tensor, 
                    alpha_bar_end : torch.Tensor):
    # diffusion times -> angles
    start_angle = torch.acos(alpha_bar_end)
    end_angle = torch.acos(alpha_bar_start)

    # angles -> signal and noise rates
    diffusion_angles = start_angle + t * (end_angle - start_angle)
    # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
    alpha_bar = torch.cos(diffusion_angles) ** 2
    return alpha_bar

def get_alpha_bar(t, alphas_cumprod, beta_schedule):
    if not isinstance(t, torch.Tensor):
        if not isinstance(t, np.ndarray):
            if type(t) in [list, tuple]:
                t = np.array(t)
            else:
                raise TypeError(
                    "expect type of param `t` is one of `np.ndarray`, `list`, or `tuple`"
                    f"but receive {type(t)}"
                    )
        t = torch.from_numpy(t).to(alphas_cumprod.device)
    if beta_schedule != "cosine":
        alpha_bar = alphas_cumprod.index_select(0, t.long())
    else:
        alpha_bar = cosine_schedule(
                t.float(), alphas_cumprod[0],
                alphas_cumprod[-1])
    return alpha_bar

"""
    for discrete time step
"""
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas