import torch
from torch.nn import functional as F

"""
    params:
        x0 : image with shape [batch_size, 3, h, w]
        t : time step with shape [batch_size]
        e : noise with shape same as x0
        a : sqrt(alpha_bar_{t}) with shape [batch_size]
    Note that:
        alpha_{t} = 1 - beta_{t}
        alpha_bar_{t} = alpha_{1} * alpha_{2} * ... * alpha_{t}
"""
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.Tensor,
                          e: torch.Tensor,
                          a: torch.Tensor,
                          keepdim=False,
                          emb_type="noise",
                          loss_type="mse"):
    if emb_type == "noise":
        emb = 1 - a
    else:
        emb = t.float()
    b = torch.sqrt(1 - a).view(-1, 1, 1, 1)
    a = a.sqrt().view(-1, 1, 1, 1)
    # xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * e
    xt = x0 * a + e * b
    # e_theta = model(xt, t)
    output = model(xt, emb)
    # x0' = (xt - sqrt(1 - alpha_bar_t) * e_theta) / ssqrt(alpha_bar_t)
    pred_x0 = (xt - b * output) / a
    reduction = "mean" if not keepdim else "none"
    # Lt = (|| e_theta - e ||_2)^2
    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif loss_type == "mae":
        loss_fn = torch.nn.L1Loss(reduction=reduction)
    else:
        raise ValueError("Only support loss type either `mse` for Mean Square"
                          f"Error or `mae` for Mean Average Error, but receive {loss_type}")
    return loss_fn(e, output), loss_fn(x0, pred_x0)


loss_registry = {
    'simple': noise_estimation_loss,
}
