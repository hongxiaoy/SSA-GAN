import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from lib.utils import truncated_noise
from lib.utils import mkdir_p, get_rank
from lib.GlobalAttention import func_attention

from lib.datasets import TextImgDataset as Dataset
from lib.datasets import prepare_data, encode_tokens
from models.inception import InceptionV3

from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist


############   modules   ############
def train(dataloader, netG, netD, image_encoder, text_encoder, optimizerG, optimizerD, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD = netG.train(), netD.train()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        # prepare_data
        imgs, captions, sorted_cap_lens, class_ids, \
            sent_emb, words_embs, keys = prepare_data(data, text_encoder)
        imgs = imgs.to(device).requires_grad_()
        sent_emb = sent_emb.to(device).requires_grad_()
        words_embs = words_embs.to(device).requires_grad_()
        
        # =========================================================================== #
        # =========================================================================== #
        
        # synthesize fake images
        noise = torch.randn(batch_size, z_dim).to(device)
        fake, mu, logvar = netG(noise, sent_emb, words_embs, mask=None, cap_lens=None)
        
        # whole D loss
        optimizerD.zero_grad()
        errD = discriminator_loss(netD, imgs.detach(), fake.detach(), sent_emb)
        errD.backward()
        # update D
        optimizerD.step()
        
        # whole G loss
        match_labels = torch.LongTensor(range(batch_size)).cuda()
        errG, logs = generator_loss(netD, image_encoder, imgs, fake.detach(), words_embs, sent_emb, 
                                    match_labels=match_labels, cap_lens=sorted_cap_lens, class_ids=class_ids, args=args)
        # update G
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()
        
        # ========================================================================== #
        # ========================================================================== #
        
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp, truncation, trunc_rate, times):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        imgs, captions, sorted_cap_lens, class_ids, sent_emb, words_embs, keys \
                = prepare_data(data, text_encoder)
        sent_emb = sent_emb.to(device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            if truncation==True:
                noise = truncated_noise(batch_size, z_dim, trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb, words_embs, mask=None, cap_lens=None)
        for j in range(batch_size):
            s_tmp = '%s/single/%s' % (save_dir, keys[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            ######################################################
            # (3) Save fake images
            ######################################################            
            if multi_gpus==True:
                filename = 'd%d_s%s.png' % (get_rank(),times)
            else:
                filename = 's%s.png' % (stamp)
            fullpath = '%s_%s.png' % (s_tmp, filename)
            im.save(fullpath)


def test(args, dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    fid = calculate_fid(args, dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch, \
                        times, z_dim, batch_size, truncation, trunc_rate)
    return fid


def calculate_fid(args, dataloader, text_encoder, netG, device, m1, s1, epoch, max_epoch,
                    times=1, z_dim=100, batch_size=64, truncation=True, trunc_rate=0.8):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, sorted_cap_lens, class_ids, sent_emb, words_embs, keys \
                = prepare_data(data, text_encoder)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs, log, var = netG(noise, sent_emb, words_embs, mask=None, cap_lens=None)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluate Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def eval(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
                times, z_dim, batch_size, truncation=True, trunc_rate=0.86):
    """ Calculates the FID """
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()

    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
            sent_emb = sent_emb.to(device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb)
                if save_imgs==True:
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            #print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(time*dl_len*batch_size+batch_size*batch_n+j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def sample_one_batch(noise, sent, words, netG, multi_gpus, epoch, img_save_dir, writer):
    fixed_results = generate_samples(noise, sent, words, netG)
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        if writer!=None:
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)


def generate_samples(noise, caption, word_embs, model):
    with torch.no_grad():
        fake, mu, logvar = model(noise, caption, word_embs, mask=None, cap_lens=None)
    return fake


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def discriminator_loss(netD, real_imgs, fake_imgs, s_conditions):
    loss_list = []
    # Forward
    real_features = netD(real_imgs.detach())
    
        
    # s_cond_real_logits = netD.module.S_COND_DNET(real_features.detach(), c_code_s=s_conditions.detach())
    # f_cond_real_logits = netD.module.F_COND_DNET(real_features.detach(), c_code_f=real_features.detach())
        
    # batch_size = real_features.size(0)
    # s_cond_wrong_logits = netD.module.S_COND_DNET(real_features[:(batch_size - 1)].detach(), c_code_s=s_conditions[1:batch_size].detach())
    # f_cond_wrong_logits = netD.module.F_COND_DNET(real_features[:(batch_size - 1)].detach(), c_code_f=real_features[1:batch_size].detach())
        
    real_logits = netD.module.UNCOND_DNET(real_features.detach())
    
    # ==== Multi-gpu ========= #
    
    # s_cond_real_logits = netD.module.S_COND_DNET(real_features.detach(), c_code_s=s_conditions.detach())
    # f_cond_real_logits = netD.module.F_COND_DNET(real_features.detach(), c_code_f=real_features.detach())
        
    # batch_size = real_features.size(0)
    # s_cond_wrong_logits = netD.module.S_COND_DNET(real_features[:(batch_size - 1)].detach(), c_code_s=s_conditions[1:batch_size].detach())
    # f_cond_wrong_logits = netD.module.F_COND_DNET(real_features[:(batch_size - 1)].detach(), c_code_f=real_features[1:batch_size].detach())
        
    # real_logits = netD.module.UNCOND_DNET(real_features.detach())
    
    # ============================= #
    
    # s_cond_real_errD = logit_loss(s_cond_real_logits, False)
    # f_cond_real_errD = logit_loss(f_cond_real_logits, False)
    # s_cond_wrong_errD = logit_loss(s_cond_wrong_logits, True)
    # f_cond_wrong_errD = logit_loss(f_cond_wrong_logits, True)
    
    real_errD = logit_loss(real_logits, False)
    
    # loss_list.extend([real_errD, s_cond_real_errD, f_cond_real_errD, f_cond_wrong_errD, s_cond_wrong_errD])
    # errD_real = (real_errD + s_cond_real_errD + f_cond_real_errD + f_cond_wrong_errD + s_cond_wrong_errD) / 5.
    errD_real = real_errD
    

    fake_features = netD(fake_imgs.detach())
    
    # s_cond_fake_logits = netD.module.S_COND_DNET(fake_features.detach(), c_code_s=s_conditions)
    # f_cond_fake_logits = netD.module.F_COND_DNET(fake_features.detach(), c_code_f=real_features.detach())
    
    fake_logits = netD.module.UNCOND_DNET(fake_features.detach())
    
    # s_cond_fake_errD = logit_loss(s_cond_fake_logits, True)
    # f_cond_fake_errD = logit_loss(f_cond_fake_logits, True)
    
    fake_errD = logit_loss(fake_logits, True)
    # errD_fake = (fake_errD + s_cond_fake_errD + f_cond_fake_errD) / 3.
    errD_fake = fake_errD
    
    return errD_real + errD_fake


def generator_loss(netD, image_encoder, real_img, fake_img, 
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, args):

    batch_size = real_img.size(0)
    logs = ''
    errG_total = 0
    # Forward
    fake_features = netD(fake_img)  # (32, 256, 4, 4)
    real_features = netD(real_img)
    s_cond_logits = netD.module.S_COND_DNET(fake_features, sent_emb)
    s_cond_errG = logit_loss(s_cond_logits, False)
    f_cond_logits = netD.module.F_COND_DNET(fake_features, real_features)
    f_cond_errG = logit_loss(f_cond_logits, False)
    logits = netD.module.UNCOND_DNET(fake_features)
    errG = logit_loss(logits, False)
    g_loss = errG + s_cond_errG + f_cond_errG

    errG_total += g_loss
    logs += 'g_loss: %.2f ' % (g_loss.item())

    # Ranking loss
    # words_features: batch_size x nef x 17 x 17
    # sent_code: batch_size x nef
    region_features, cnn_code = image_encoder(fake_img)  # (32, 256, 17, 17), (32, 256)
    w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size, args=args)
    w_loss = (w_loss0 + w_loss1) * args.TRAIN.SMOOTH.LAMBDA

    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size, args=args)
    s_loss = (s_loss0 + s_loss1) * args.TRAIN.SMOOTH.LAMBDA
    # err_sent = err_sent + s_loss.data[0]

    errG_total += w_loss + s_loss
    logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())

        #
        # # Ranking loss
        # # words_features: batch_size x nef x 17 x 17
        # # sent_code: batch_size x nef
        # region_features, cnn_code = image_encoder(fake_imgs[i])
        # w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
        #                                  match_labels, cap_lens,
        #                                  class_ids, batch_size)
        # w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
        # # err_words = err_words + w_loss.data[0]
        #
        # s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
        #                              match_labels, class_ids, batch_size)
        # s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
        # # err_sent = err_sent + s_loss.data[0]
        #
        # errG_total += w_loss + s_loss
        # logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())

    return errG_total, logs


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size, args):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> (1, nef, words_num)
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> (bs, nef, words_num)
        word = word.repeat(batch_size, 1, 1)
        # (bs, nef, 17, 17)
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, args.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(args.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).cuda()

    similarities = similarities * args.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8, args=None):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * args.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()