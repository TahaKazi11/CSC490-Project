import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from ..evaluators.loss import AdversarialLoss, PerceptualLoss, StyleLoss

"""
https://github.com/AndreFagereng/polyp-GAN/tree/main/edge-connect/src/models.py
"""


class BaseModel(nn.Module):
    def __init__(self, name, cfg):
        super(BaseModel, self).__init__()

        self.name = name
        self.cfg = cfg
        self.iteration = 0

        self.gen_weights_path = os.path.join(cfg.PATH, name + "_425000_432000_gen.pth")
        self.dis_weights_path = os.path.join(cfg.PATH, name + "_425000_dis.pth")
        print(self.gen_weights_path)

    def load(self):
        print("Here", self.gen_weights_path)
        if os.path.exists(self.gen_weights_path):
            print("Loading %s generator..." % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(
                    self.gen_weights_path, map_location=lambda storage, loc: storage
                )

            self.generator.load_state_dict(data["generator"])
            self.iteration = data["iteration"]

        # load discriminator only when training
        if self.cfg.MODE == 1 and os.path.exists(self.dis_weights_path):
            print("Loading %s discriminator..." % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(
                    self.dis_weights_path, map_location=lambda storage, loc: storage
                )

            self.discriminator.load_state_dict(data["discriminator"])

    def save(self):
        _gen_weight_path = self.gen_weights_path
        _dis_weight_path = self.dis_weights_path

        _gen_weight_path = (
            _gen_weight_path.replace("_gen.pth", "")
            + "_"
            + str(self.iteration)
            + "_gen.pth"
        )
        _dis_weight_path = (
            _dis_weight_path.replace("_dis.pth", "")
            + "_"
            + str(self.iteration)
            + "_dis.pth"
        )

        print("\nsaving %s...\n" % self.name)
        torch.save(
            {"iteration": self.iteration, "generator": self.generator.state_dict()},
            _gen_weight_path,
        )

        torch.save({"discriminator": self.discriminator.state_dict()}, _dis_weight_path)


class EdgeModel(BaseModel):
    def __init__(self, cfg):
        super(EdgeModel, self).__init__("EdgeModel", cfg)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(
            in_channels=2, use_sigmoid=cfg.GAN_LOSS != "hinge"
        )
        if len(cfg.GPU) > 1:
            generator = nn.DataParallel(generator, cfg.GPU)
            discriminator = nn.DataParallel(discriminator, cfg.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=cfg.GAN_LOSS)

        self.add_module("generator", generator)
        self.add_module("discriminator", discriminator)

        self.add_module("l1_loss", l1_loss)
        self.add_module("adversarial_loss", adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(cfg.LR),
            betas=(cfg.BETA1, cfg.BETA2),
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(cfg.LR) * float(cfg.D2G_LR),
            betas=(cfg.BETA1, cfg.BETA2),
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(
            dis_input_real
        )  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(
            dis_input_fake
        )  # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(
            gen_input_fake
        )  # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.cfg.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = edges * (1 - masks)
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)  # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward(retain_graph=True)

        if gen_loss is not None:
            gen_loss.backward()
        self.dis_optimizer.step()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, cfg):
        super(InpaintingModel, self).__init__("InpaintingModel", cfg)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(
            in_channels=3, use_sigmoid=cfg.GAN_LOSS != "hinge"
        )
        if len(cfg.GPU) > 1:
            generator = nn.DataParallel(generator, cfg.GPU)
            discriminator = nn.DataParallel(discriminator, cfg.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=cfg.GAN_LOSS)

        self.add_module("generator", generator)
        self.add_module("discriminator", discriminator)

        self.add_module("l1_loss", l1_loss)
        self.add_module("perceptual_loss", perceptual_loss)
        self.add_module("style_loss", style_loss)
        self.add_module("adversarial_loss", adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(cfg.LR),
            betas=(cfg.BETA1, cfg.BETA2),
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(cfg.LR) * float(cfg.D2G_LR),
            betas=(cfg.BETA1, cfg.BETA2),
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = (
            self.adversarial_loss(gen_fake, True, False)
            * self.cfg.INPAINT_ADV_LOSS_WEIGHT
        )
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = (
            self.l1_loss(outputs, images) * self.cfg.L1_LOSS_WEIGHT / torch.mean(masks)
        )
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.cfg.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.cfg.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward(retain_graph=True)

        gen_loss.backward()
        self.dis_optimizer.step()
        self.gen_optimizer.step()
