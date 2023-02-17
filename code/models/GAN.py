import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm


import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Upsale the spatial size by a factor of 2.

        Args:
            in_channels (int): Number of in channels.
            out_channels (int): Number of out channels.

        Returns:
            torch.tensor: The feature map after upBlock.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=1, 
                      padding=1, bias=False), 
            nn.BatchNorm2d(out_channels * 2), 
            nn.GLU(1)
        )
    
    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num * 2, kernel_size=3, stride=1, 
                      padding=1, bias=False), 
            nn.BatchNorm2d(channel_num * 2),
            nn.GLU(1),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, 
                      padding=1, bias=False), 
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class CA(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, t_dim, c_dim):
        super(CA, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = nn.GLU(1)

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class GEN_Module_0(nn.Module):
    def __init__(self, z_dim, ngf, ncf):
        super(GEN_Module_0, self).__init__()
        self.gf_dim = ngf
        self.in_dim = z_dim + ncf
        
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.GLU(1)
        )

        self.upsample1 = UpBlock(ngf, ngf // 2)
        self.upsample2 = UpBlock(ngf // 2, ngf // 4)
        self.upsample3 = UpBlock(ngf // 4, ngf // 8)
        self.upsample4 = UpBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class GEN_Module_1(nn.Module):
    def __init__(self, ngf, nef, ncf, size, res_num):
        super(GEN_Module_1, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = res_num  # cfg.GAN.RES_NUM
        self.size = size
        
        ngf = self.gf_dim
        self.W_s = nn.Linear(self.ef_dim, self.size*self.size, bias=False)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = UpBlock(ngf * 2, ngf)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, h_code, c_code, sent_emb, word_embs, mask, cap_lens):
        """
            h_code(image features):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(word features): batch x cdf x sourceL (sourceL=seq_len)
            c_code: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        bs, ih, iw = h_code.size(0), h_code.size(2), h_code.size(3)
        
        h_code_flat = torch.flatten(h_code, start_dim=2)  # (N, 128, 64*64)
        ws = self.W_s(sent_emb).unsqueeze(-1)  # (N, 64*64)
        R = torch.bmm(h_code_flat, ws).softmax(dim=1).repeat(1, 1, h_code_flat.shape[-1])
        h_star = torch.mul(R, h_code_flat)
        h_concat = torch.concat([h_star, h_code_flat], dim=1)
        h_concat = h_concat.view(bs, -1, ih, iw).contiguous()
        
        out_code = self.residual(h_concat)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        
        # print(f'h_code shape: {h_code.shape}')
        # print(f'h_code_flat shape: {h_code_flat.shape}')
        # print(f'sent_emb shape: {sent_emb.shape}')
        # print(f'ws shape: {ws.shape}')
        # print(f'R shape: {R.shape}')
        # print(f'h_star shape: {h_star.shape}')
        # print(f'h_concat shape: {h_concat.shape}')
        # print(f'out_code shape: {out_code.shape}\n')
        
        return out_code


class GEN_Module_2(nn.Module):
    def __init__(self, ngf, nef, ncf, size, res_num):
        super(GEN_Module_2, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = res_num  # cfg.GAN.RES_NUM
        self.size = size
        
        self.U = nn.Linear(self.ef_dim, self.gf_dim, bias=False)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = UpBlock(ngf * 2, ngf)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, h_code, c_code, sent_emb, word_embs, mask=None, cap_lens=None):
        """
            h_code(image features):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(word features): batch x cdf x sourceL (sourceL=seq_len)
            c_code: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        bs, ih, iw = h_code.shape[0], h_code.shape[2], h_code.shape[3]  # (N, 64, 128, 128)
        h_code_flat = torch.flatten(h_code, start_dim=2).contiguous()  # (N, 64, 128*128)
        e_prime = self.U(word_embs.permute(0, 2, 1)).permute(0, 2, 1)  # (N, 64, 18)
        
        scoring = torch.matmul(h_code_flat.permute(0, 2, 1), e_prime)  # (N, 128*128, 18)
        scoring = scoring.view(bs*ih*iw, word_embs.shape[2])  # (N*128*128, 18)
        if mask is not None:
            mask = mask.repeat(ih * iw, 1)
            scoring.data.masked_fill_(mask.data, -float('inf'))
        attn_weights = F.softmax(scoring, dim=1)  # (N*128*128, 18)
        attn_weights = attn_weights.view(bs, ih * iw, word_embs.shape[2])  # (N, 128*128, 18)

        c = torch.matmul(attn_weights, e_prime.permute(0, 2, 1)).permute(0, 2, 1)  # (N, 128, 128*128)
        c = c.view(bs, -1, ih, iw)  # (N, 128, 128, 128)
        im_feat_concat = torch.concat([c, h_code], dim=1)
        
        # print(f'h_code shape: {h_code.shape}')
        # print(f'h_code_flat shape: {h_code_flat.shape}')
        # print(f'word_embeds shape: {word_embeds.shape}')
        # print(f'word_embeds_prime shape: {e_prime.shape}')
        # print(f'scoring shape: {scoring.shape}')
        # print(f'attn_weights shape: {attn_weights.shape}')
        # print(f'c shape: {c.shape}')
        # print(f'im_feat_concat shape: {im_feat_concat.shape}')

        out_code = self.residual(im_feat_concat)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class NetG(nn.Module):
    def __init__(self, ngf, nef, ncf, z_dim):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.nef = nef
        self.ncf = ncf
        self.ca_net = CA()

        
        self.h_net1 = GEN_Module_0(z_dim, ngf * 16, ncf)
        # self.img_net1 = GET_IMAGE_G(ngf)

        # gf x 64 x 64
        self.h_net2 = GEN_Module_1(ngf, nef, ncf, 64)
        # self.img_net2 = GET_IMAGE_G(ngf)

        self.h_net3 = GEN_Module_2(ngf, nef, ncf, 128)
        self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask, cap_lens):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """

        c_code, mu, logvar = self.ca_net(sent_emb)

        h_code1 = self.h_net1(z_code, c_code)  # (10, 128, 64, 64)
        # print(h_code1.shape)
        # fake_img1 = self.img_net1(h_code1)
        # fake_imgs.append(fake_img1)
        
        h_code2 = self.h_net2(h_code1, c_code, sent_emb, word_embs, mask, cap_lens)  # (10, 64, 128, 128)
        # print(h_code2.shape)
        # fake_img2 = self.img_net2(h_code2)
        # fake_imgs.append(fake_img2)
        # if att1 is not None:
        #     att_maps.append(att1)
        
        h_code3 = self.h_net3(h_code2, c_code, sent_emb, word_embs, mask, cap_lens)  # (10, 128, 256, 256)
        # print(h_code3.shape)
        fake_img3 = self.img_net3(h_code3)  # (10, 3, 256, 256)
        # if att2 is not None:
        #     att_maps.append(att2)
        
        # print(f'h_code1 shape:\t{h_code1.shape}')
        # print(f'h_code2 shape:\t{h_code2.shape}')
        # print(f'h_code3 shape:\t{h_code3.shape}')
        # print(f'fake_imgs[0] shape:\t{fake_imgs[0].shape}')

        return fake_img3, mu, logvar





class Encode_16times(nn.Module):
    def __init__(self, ndf):
        """Downsale the spatial size by a factor of 16.

        Args:
            ndf (int): The dimension of discriminator in channels.
        """
        super().__init__()
        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1, bias=True)), 
            nn.LeakyReLU(0.2, inplace=True), 
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)), 
            nn.LeakyReLU(0.2, inplace=True), 
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)), 
            nn.LeakyReLU(0.2, inplace=True), 
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)), 
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)


class Block3x3_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Keep the spatial size"""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, stride=1, 
                      padding=1, bias=False), 
            nn.BatchNorm2d(out_channels * 2),
            nn.GLU(1)
        )
    
    def forward(self, x):
        return self.block(x)


class Block3x3_leakyrelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Downsale the spatial size by a factor of 2."""
        super().__init__()
        self.block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, 
                                   bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class DIS_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, s_condition=False, f_condition=False):
        super(DIS_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.s_condition = s_condition
        self.f_condition = f_condition
        if self.s_condition:
            self.s_jointConv = Block3x3_leakyrelu(ndf * 8 + nef, ndf * 8)
            self.f_jointConv = None
        if self.f_condition:
            self.s_jointConv = None
            self.f_jointConv = Block3x3_leakyrelu(ndf * 8 * 2, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code_s=None, c_code_f=None):
        if self.s_condition and c_code_s is not None:
            # conditioning output
            c_code_s = c_code_s.view(-1, self.ef_dim, 1, 1)
            c_code_s = c_code_s.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code_s), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.s_jointConv(h_c_code)
        elif self.f_condition and c_code_f is not None:
            # conditioning output
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code_f), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.f_jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


class NetD(nn.Module):
    def __init__(self, ndf, nef, uncon=True):
        """Discriminator for 256 x 256 images."""
        super(NetD, self).__init__()
        self.ndf = ndf
        self.nef = nef
        self.img_code_s16 = Encode_16times(ndf)
        self.img_code_s32 = DownBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = DownBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakyrelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakyrelu(ndf * 16, ndf * 8)
        if uncon:
            self.UNCOND_DNET = DIS_GET_LOGITS(ndf, nef)
        else:
            self.UNCOND_DNET = None
        self.S_COND_DNET = DIS_GET_LOGITS(ndf, nef, s_condition=True)
        self.F_COND_DNET = DIS_GET_LOGITS(ndf, nef, f_condition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4
