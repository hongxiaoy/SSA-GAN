from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import time
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from .utils import truncated_noise


def get_one_batch_data_DF_GAN(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, sent_emb, words_embs = prepare_data_DF_GAN(data, text_encoder)
    return imgs, words_embs, sent_emb


def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    # imgs, captions, sorted_cap_lens, class_ids, sent_emb, words_embs, keys
    return prepare_data(data, text_encoder)


def get_fix_data_DF_GAN(train_dl, test_dl, text_encoder, args):
    fixed_image_train, fixed_word_train, fixed_sent_train = get_one_batch_data_DF_GAN(train_dl, text_encoder, args)
    fixed_image_test, fixed_word_test, fixed_sent_test = get_one_batch_data_DF_GAN(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    if args.truncation==True:
        noise = truncated_noise(fixed_image.size(0), args.z_dim, args.trunc_rate)
        fixed_noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_noise


def get_fix_data(train_dl, test_dl, text_encoder, args):
    # imgs, captions, sorted_cap_lens, class_ids, sent_emb, words_embs, keys
    train_batch_data = get_one_batch_data(train_dl, text_encoder, args)
    test_batch_data = get_one_batch_data(test_dl, text_encoder, args)
    
    fixed_image_train = train_batch_data[0]
    fixed_word_train = train_batch_data[-2]
    fixed_sent_train = train_batch_data[-3]
    
    fixed_image_test = test_batch_data[0]
    fixed_word_test = test_batch_data[-2]
    fixed_sent_test = test_batch_data[-3]
    
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_words = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    if args.truncation==True:
        noise = truncated_noise(fixed_image.size(0), args.z_dim, args.trunc_rate)
        fixed_noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_words, fixed_noise


def prepare_data_DF_GAN(data, text_encoder):
    imgs, captions, caption_lens, class_ids, keys = data
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    imgs = imgs.cuda()
    return imgs, sent_emb, words_embs


def prepare_data(data, text_encoder):
    imgs, captions, caption_lens, class_ids, keys = data
    # imgs: (bs, 3, 256, 256)
    # captions: (bs, 18, 1)
    # captions_lens: (bs,)
    # class_ids: (bs,)
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    imgs = imgs[sorted_cap_idxs]
    class_ids = class_ids[sorted_cap_idxs].numpy()
    keys = [keys[i] for i in sorted_cap_idxs.numpy()]
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    # sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    # words_embs = rm_sort(words_embs, sorted_cap_idxs)
    imgs = imgs.cuda()
    return imgs, captions, sorted_cap_lens, class_ids, sent_emb, words_embs, keys


def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = captions.cuda()
    sorted_cap_lens = sorted_cap_lens.cuda()
    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))
        # words_embs: bs x nef x seq_len
        # sent_emb: bs x nef
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img


# ############################################################### #
#                             Dataset                             #
# ############################################################### #
class TextImgDataset(data.Dataset):
    def __init__(self, split='train', transform=None, args=None):
        self.transform = transform
        self.word_num = args.TEXT.WORDS_NUM
        self.embeddings_num = args.TEXT.CAPTIONS_PER_IMAGE
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(self.data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(self.data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_DAMSM.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.word_num
        return x, x_len

    def get_mis_caption(self, cls_id):
        """Get 99 unmatched captions.

        Args:
            cls_id (int): The class id of the image, not find its class captions.

        Returns:
            tuple: 99 unmatched captions, and its cap lengths.
        """
        mis_match_captions_t = []
        mis_match_captions = torch.zeros(99, self.word_num)
        mis_match_captions_len = torch.zeros(99)
        i = 0
        while len(mis_match_captions_t) < 99:  # when not achieve 99
            # random get one idx
            idx = random.randint(0, self.number_example)
            if cls_id == self.class_id[idx]:
                continue
            # random get 1 caption of 10
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = idx * self.embeddings_num + sent_ix
            cap_t, cap_len_t = self.get_caption(new_sent_ix)
            # get 1 unmatched cap and its len
            mis_match_captions_t.append(torch.from_numpy(cap_t).squeeze())
            mis_match_captions_len[i] = cap_len_t
            i = i +1
        sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)

        for i in range(99):
            mis_match_captions[i,:] = mis_match_captions_t[sorted_cap_indices[i]]
        return mis_match_captions.type(torch.LongTensor).cuda(), sorted_cap_lens.type(torch.LongTensor).cuda()

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        if self.dataset_name.find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
        else:
            img_name = '%s/images/%s.jpg' % (data_dir, key)

        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key

    def __len__(self):
        return len(self.filenames)


