import torch

"""
    to make alpha_bar_{0} = 1
    betas = [0, beta_{1}, ...., beta_{T}]
    alphas = [1, alpha_{1}, ...., alpha_{T}]
    so when sampling step at t = 1, note that indice in `index_select` is i + 1
    `+1` is to skip first 1 in alphas
        alpha_bar_{t} = alpha_{1}
        alpha_bar_{t-1} = 1
"""
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

"""
    params:
        x : random noise
        seq : sampling sub sequence
        a : alpha_bar, has same length as seq
        eta : factor of controling randomness
    a = torch.concat([a, torch.ones(1).to(a.device)], dim=0), then get
    alpha_bars = [1, alpha_bar_{1}, ...., alpha_bar_{T}]
    so when sampling step at i = 0
        at = alpha_bar_{1}
        at_next = 1
    this is corresponding to make alpha_bar_{0}(i.e. alpha_{0} in DDIM) = 1 as describing in paper DDIM
"""
def generalized_steps(x, seq, model, a, **kwargs):
    assert len(seq) == len(a)
    a = torch.concat([torch.ones(1).to(x.device), a], dim=0)
    emb_type = kwargs.get("emb_type", "noise")
    with torch.no_grad():
        n = x.size(0)
        x0_preds = []
        xs = [x]
        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (see min_signal_rate in cosine schedule)
        for i in reversed(range(1, len(seq)+1)):
            at = a[i] * torch.ones(n).to(x.device)
            at_next = a[i - 1] * torch.ones(n, 1, 1, 1).to(x.device)
            if emb_type == "noise":
                emb = 1 - at
            else:
                emb = seq[i-1] * torch.ones(n).to(x.device)
            at = at.view(n, 1, 1, 1)
            xt = xs[-1].to('cuda')
            et = model(xt, emb)
            # x0' = (x_{τ_i} - sqrt(1 - alpha_bar_{τ_i}) * e_theta) / sqrt(alpha_bar_{τ_i})
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            # σ_{τ_i}(eta) = eta * sqrt((1 - alpha_bar_{τ_i} / alpha_bar_{τ_{i-1}}) * (1 - alpha_bar_{τ_{i-1}}) / (1 - alpha_bar_{τ_i}))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            # Eq(12): sqrt(1 - alpha_bar_{τ_{i-1}} - σ_{τ_i}(eta)^2)
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # x_{τ_{i-1}} = sqrt(alpha_bar_{τ_{i-1}}) * x0' + c2 * e_theta + σ_{τ_i}(eta) * e
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds

"""
    DDPM's alpha_bar_{t} = DDIM's alpha_{t}
    DDPM's alpha_{t} = DDIM's alpha_{t} / alpha_{t-1}
    DDPM's beta_{t} = 1 - DDIM's alpha_{t} / alpha_{t-1}
"""
def ddpm_steps(x, seq, model, a, **kwargs):
    assert len(seq) == len(a)
    a = torch.concat([torch.ones(1).to(x.device), a], dim=0)
    emb_type = kwargs.get("emb_type", "noise")
    with torch.no_grad():
        n = x.size(0)
        xs = [x]
        x0_preds = []
        for i in reversed(range(1, len(seq)+1)):
            at = a[i] * torch.ones(n).to(x.device)
            at_next = a[i - 1] * torch.ones(n, 1, 1, 1).to(x.device)
            if emb_type == "noise":
                emb = 1 - at
            else:
                emb = seq[i-1] * torch.ones(n).to(x.device)
            at = at.view(n, 1, 1, 1)
            beta_t = 1 - at / at_next
            x = xs[-1].to('cuda')

            output = model(x, emb)
            e = output
            # x0' = (x_t - sqrt(1 - alpha_bar_t) * e_theta) / sqrt(alpha_bar_t)
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            # Eq(6), miu_t = [sqrt(alpha_bar_{t-1}) * beta_t * x0 + sqrt(alpha_t) * (1 - alpha_bar_{t-1}) * xt] / (1 - alpha_bar_t)
            mean_eps = (
                (at_next.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (emb == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            # x_{t-1} = miu_t + sqrt(beta_t) * e
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
