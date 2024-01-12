import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

# from torchmetrics.image.kid import KernelInceptionDistance
# from models.diffusion import Model
from models.keras_diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.schedule import get_alpha_bar
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.diffusion.beta_schedule == "cosine":
            self.alphas_cumprod = torch.from_numpy(
                np.array([config.diffusion.alpha_bar_start,
                          config.diffusion.alpha_bar_end])).float().to(self.device)
        else:
            from functions.schedule import get_beta_schedule
            self.model_var_type = config.model.var_type
            betas = get_beta_schedule(
                beta_schedule=config.diffusion.beta_schedule,
                beta_start=config.diffusion.beta_start,
                beta_end=config.diffusion.beta_end,
                num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
            )
            betas = self.betas = torch.from_numpy(betas).float().to(self.device)

            alphas = 1.0 - betas
            alphas_cumprod = self.alphas_cumprod = alphas.cumprod(dim=0)
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
            )
            posterior_variance = (
                    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            if self.model_var_type == "fixedlarge":
                self.logvar = betas.log()
                # torch.cat(
                # [posterior_variance[1:2], betas[1:]], dim=0).log()
            elif self.model_var_type == "fixedsmall":
                self.logvar = posterior_variance.clamp(min=1e-20).log()

    def diffusion_times(self, n):
        # antithetic sampling
        if self.config.diffusion.beta_schedule != "cosine":  # discrete
            t = torch.randint(
                low=0, high=self.num_timesteps, size=(n // 2 + 1,)
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        else:  # continuous, time step between 0 and 1
            t = torch.rand(n)
        return t.to(self.device)

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        tb_logger.add_graph(model, (torch.zeros(
            (config.training.batch_size, config.data.channels,
             config.data.image_size, config.data.image_size),
            torch.zeros(config.training.batch_size))))

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            eval_model = ema_helper.model
        else:
            ema_helper = None
            eval_model = model

        start_epoch, step = 0, 0
        if self.args.resume_training:
            ckpt = self.args.ckpt_fname if self.args.ckpt_fname != "" else "ckpt.pth"
            ckpt = os.path.join(self.args.log_path, ckpt)
            states = torch.load(ckpt)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resume model training from file {ckpt} with epoch {start_epoch}, step {step}")
        time_start = time.time()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            if args.plot_epoch > 0 and epoch % args.plot_epoch == 0:
                eval_model.eval()
                self.sample_once(eval_model, epoch=epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)  # x0 ~ q(x0)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)  # e ~ N(0, I)
                t = self.diffusion_times(n)
                a = get_alpha_bar(  # alpha_bar_t
                    t, self.alphas_cumprod, config.diffusion.beta_schedule)

                n_loss, i_loss = loss_registry[config.model.type](model, x, t, e, a,
                                                                  loss_type=config.training.loss_type)

                tb_logger.add_scalar("noise_loss", n_loss, global_step=step)
                tb_logger.add_scalar("image_loss", i_loss, global_step=step)
                logging.info(
                    "epoch: {}, step: {}, n_loss: {}, i_loss: {}, {:.1f}ms, {:.1f}ms/step".format(
                        epoch + 1,
                        step,
                        n_loss.item(),
                        i_loss.item(),
                        data_time / (i + 1) * 1000,
                        (time.time() - time_start) / step * 1000,
                    )
                )

                optimizer.zero_grad()
                n_loss.backward()
                if config.optim.grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                data_start = time.time()
            # save last epoch
            if config.training.save_last and epoch + 1 == config.training.n_epochs:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                )
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        logging.info("Finish training in {:.3f}s".format(time.time() - time_start))

    def sample(self):
        model = Model(self.config)
        if not self.args.debug:
            if not self.args.use_pretrained:
                ckpt = self.args.ckpt_fname if self.args.ckpt_fname != "" else "ckpt.pth"
                ckpt = os.path.join(self.args.log_path, ckpt)
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1], strict=False)
                    ema_helper.ema(model)
                    logging.info("Using ema model")
                else:
                    ema_helper = None
                logging.info(f"Successfully load model from file {ckpt} with epoch {states[2]}, step {states[3]}")
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                logging.info("Successfully load checkpoint from {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        elif self.args.once:
            self.sample_once(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_once(self, model, epoch=None, plot_image_size=128):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        x = torch.randn(
            self.args.sample_num,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        with torch.no_grad():
            x = self.sample_image(x, model, last=True)
        x = inverse_data_transform(config, x)
        if plot_image_size > self.config.data.image_size:
            from torchvision.transforms.functional import resize
            with torch.no_grad():
                x = resize(x, plot_image_size)
        image_dir = self.args.image_folder
        # monitoring training
        if epoch is not None:
            image_dir = os.path.join(image_dir, f"epoch_{epoch}")
            os.mkdir(image_dir)
        for i in range(self.args.sample_num):
            tvu.save_image(
                x[i], os.path.join(image_dir, f"{img_id}.png")
            )
            img_id += 1

    def sample_sequence(self, model, plot_image_size=128):
        config = self.config

        x = torch.randn(
            self.args.sample_num,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]
        if plot_image_size > self.config.data.image_size:
            from torchvision.transforms.functional import resize
            with torch.no_grad():
                x = [resize(y, plot_image_size) for y in x]
        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                    torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i: i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1  # sampling cross step

        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = np.linspace(
                0, self.num_timesteps - skip, self.args.timesteps)
        elif self.args.skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    ) ** 2
            )
        else:
            raise NotImplementedError
        if self.config.model.emb_type == "noise":
            seq = seq / self.num_timesteps
            seq = seq + 1.0 - seq[-1]
        # seq=[τ_1, ..., τ_s], a = [alpha_bar_{τ_1}, ..., alpha_bar_{τ_s}]
        a = get_alpha_bar(
            seq, self.alphas_cumprod, self.config.diffusion.beta_schedule)
        if self.args.sample_type == "generalized":
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, a,
                                   eta=self.args.eta,
                                   emb_type=self.config.model.emb_type)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, a,
                           emb_type=self.config.model.emb_type)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        # self.test_data_pipeline()
        # self.test_schedule()
        # self.test_model_embedding()
        # self.test_model_output()
        # self.test_seq_schedule()
        # self.test_compute_alpha()
        self.test_ema()

    def test_data_pipeline(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        print(len(dataset))
        for i in range(8):
            x, y = dataset[i]
            x = data_transform(config, x)
            x = inverse_data_transform(config, x)
            tvu.save_image(
                x, os.path.join(args.image_folder, f"{i}.png")
            )

    def test_schedule(self):
        args, config = self.args, self.config
        # t = self.diffusion_times(2)
        # print(t)
        t = torch.as_tensor([0.5176873, 0.39660096], dtype=torch.float32).to(self.device)
        a = get_alpha_bar(
            t, self.alphas_cumprod, config.diffusion.beta_schedule)
        print(a.sqrt())  # expect around [0.57680005, 0.69191194]

    def test_seq_schedule(self):
        args, config = self.args, self.config
        try:
            skip = self.args.skip
        except Exception:
            skip = 1  # sampling cross step

        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = np.linspace(
                0, self.num_timesteps - skip, self.args.timesteps)
        elif self.args.skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
            )
        else:
            raise NotImplementedError
        if self.config.diffusion.beta_schedule == "cosine":
            step_size = 1.0 / self.args.timesteps
            step_range = self.num_timesteps
        else:
            step_size = 1
            step_range = 1
        seq = seq / step_range
        print(seq, seq.shape)
        a = get_alpha_bar(
            seq + step_size, self.alphas_cumprod, self.config.diffusion.beta_schedule)
        print(a, a.shape)

    def test_model_embedding(self):
        args, config = self.args, self.config
        from models.diffusion import get_timestep_embedding, sinusoidal_embedding
        t = torch.as_tensor([0.78180575, 0.9576472], dtype=torch.float32).to(self.device)
        print("diffusion_times============================>")
        print(t, t.shape)
        temb = get_timestep_embedding(t, config.model.ch)
        a = get_alpha_bar(
            t, self.alphas_cumprod, config.diffusion.beta_schedule)
        print("diffusion_schedule============================>")
        print(a.sqrt(), a.shape)
        nemb = sinusoidal_embedding(1 - a, config.model.ch)
        print("time_embedding============================>")
        print(temb, temb.shape)
        print("noise_embedding============================>")
        print(nemb, nemb.shape)
        # expect nemb with value
        """
            [[-0.4887232   0.27240002  0.9348897  -0.8359303  -0.9570708
                0.92343205 -0.37343788  0.46788004 -0.45598125 -0.19946295
                -0.71530944 -0.63316613 -0.98782384 -0.99976027 -0.9060526
                -0.99322826  0.8724389  -0.9621841  -0.35493854 -0.54883564
                0.28985417  0.3837619  -0.92765516  0.883792   -0.8899894
                0.97990537  0.69880784 -0.7740159   0.15557654  0.02189598
                -0.42316508 -0.11617941]
             [-0.03271979 -0.46313846  0.00751802 -0.2466418   0.9859499
                -0.3214462  -0.9947069  -0.07510632 -0.60475785 -0.9942272
                0.1306044  -0.85699904 -0.68311197  0.20935303 -0.87977546
                -0.9660352   0.9994646  -0.88628596 -0.99997175  0.96910673
                -0.1670416   0.94692785  0.10275246  0.9971755  -0.7964094
                0.10729541 -0.9914346  -0.51531804  0.7303137   0.9778401
                -0.47538945  0.25841066]
            ]
        """

    def test_model_output(self):
        args, config = self.args, self.config
        t = torch.as_tensor([0.14723265, 0.02082253], dtype=torch.float32).to(self.device)
        a = get_alpha_bar(
            t, self.alphas_cumprod, config.diffusion.beta_schedule)
        model = Model(self.config)
        model.to(self.device)
        model.eval
        emb = 1 - a
        b = torch.sqrt(1 - a).view(-1, 1, 1, 1)
        a = a.sqrt().view(-1, 1, 1, 1)
        x = np.load("./images.npy")
        x = np.transpose(x, axes=[0, 3, 1, 2])
        x = torch.from_numpy(x).to(dtype=torch.float32)
        # x = torch.randn((1, 3, config.data.image_size, config.data.image_size))
        x = x.to(self.device)
        # x = data_transform(self.config, x)
        e = x
        xt = x * a + e * b
        output = model(xt, emb)
        print(output, output.shape)

    def test_compute_alpha(self):
        args, config = self.args, self.config
        from functions.schedule import get_beta_schedule
        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        print(betas, betas.shape)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
        a = (1 - betas).cumprod(dim=0)
        print(a, a.shape)

    def test_ema(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)
        model = model.to(self.device)
        if self.config.model.ema:
            ema_helper = EMAHelper(model, mu=self.config.model.ema_rate)
            eval_model = ema_helper.model
        else:
            ema_helper = None
            eval_model = model
        optimizer = get_optimizer(self.config, model.parameters())
        for i in range(10):
            (x, y) = next(iter(train_loader))
            n = x.size(0)
            model.train()

            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            t = self.diffusion_times(n)
            a = get_alpha_bar(
                t, self.alphas_cumprod, config.diffusion.beta_schedule)

            loss = loss_registry[config.model.type](model, x, t, e, a,
                                                    loss_type=config.training.loss_type)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if config.optim.grad_clip > 0:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
            optimizer.step()
            if self.config.model.ema:
                ema_helper.update(model, debug=(i == 9))
            eval_model.eval()
            self.sample_once(eval_model, epoch=i)
