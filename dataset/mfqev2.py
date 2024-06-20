import glob
import random
import torch
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, FlipRotate, totensor, import_yuv,yuv_rgb

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class MFQEv2Dataset(data.Dataset):
    """MFQEv2 dataset.
    For training data: LMDB is adopted. See create_lmdb for details.

    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['lq_path']
        )
        self.ref_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['ref_path']
        )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root,
            self.opts_dict['meta_info_fp']
        )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root,
            self.gt_root,
            
        ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # 4      | 9   | 1, 2, 3, 4, 5, 6, 7, 8, 9
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')

        img_gt = _bytes2img(img_bytes)  # (H W 1)
        h,w,_=img_gt.shape
        # cv2.imwrite('gt.png', img_gt * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        ref_dir = op.join(self.opts_dict['root'], self.opts_dict['ref_folder'])
        ref_video_list = sorted(glob.glob(op.join(ref_dir, '*.yuv')))

        name =op.basename(ref_video_list[int(clip)-1]).split(".")[0]
        nfs = int(name.split('_')[2]) - 1
        start_frm = ((3 + (int(seq) - 1) * 7) // 4) * 4
        img_refs = []
        ref_num = [start_frm, start_frm + 4]
        ref_num = np.clip(ref_num, 0, nfs - 1)

        for i in range(len(ref_num)):
            img_ref = import_yuv(
                seq_path=ref_video_list[int(clip) - 1],
                yuv_type='420p',
                h=h,
                w=w,
                tot_frm=1,
                start_frm=ref_num[i],
                only_y=True
            )
            img_ref = np.transpose(img_ref.astype(np.float32) / 255., (1, 2, 0))
            img_refs.append(img_ref)
        # (H W 3)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            frm = ((int(seq) - 1) * 7)+neighbor-1
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            if frm%4==0:
                img_ref = import_yuv(
                    seq_path=ref_video_list[int(clip) - 1],
                    yuv_type='420p',
                    h=h,
                    w=w,
                    tot_frm=1,
                    start_frm=frm,
                    only_y=True
                )
                img_ref = np.transpose(img_ref.astype(np.float32) / 255., (1, 2, 0))
                img_lqs.append(np.expand_dims(cv2.resize(img_ref, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC),axis=2))
                continue
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        # randomly crop
        img_gt, img_lqs,img_refs = paired_random_crop(
            img_gt, img_lqs, img_refs,gt_size, img_gt_path,scale=2
        )

        # flip, rotate
        # img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_gt,img_lqs,img_refs = FlipRotate(
            img_gt, img_lqs,img_refs
        )

        # to tensor
        img_lqs = totensor(img_lqs)
        img_lqs = torch.stack(img_lqs,dim=0)
        img_refs = torch.stack(totensor(img_refs), dim=0)
        img_gt = torch.unsqueeze(totensor(img_gt),dim=0)


        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            'ref': img_refs
        }

    def __len__(self):
        return len(self.keys)


class VideoTestMFQEv2Dataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.
    For validation data: Disk IO is adopted.

    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['gt_path']
        )
        self.ref_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['ref_path']
        )
        self.lq_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['lq_path']
        )

        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'ref_path': [],
            'gt_path': [],
            'gt_index': [],
            'lq_indexes': [],
            'gt_h': [],
            'gt_w': [],
            'lq_h': [],
            'lq_w': [],
            'index_vid': [],
            'gt_name_vid': [],
        }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        lq_path_list = sorted(glob.glob(op.join(self.lq_root, '*.yuv')))
        ref_path_list = sorted(glob.glob(op.join(self.ref_root, '*.yuv')))
        self.vid_num = len(gt_path_list)

        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            ref_vid_path=ref_path_list[idx_vid]
            gt_name_vid = gt_vid_path.split('/')[-1]
            gt_w, gt_h = map(int, gt_name_vid.split('_')[-2].split('x'))
            nfs = int(gt_name_vid.split('.')[-2].split('_')[-1])

            lq_vid_path=op.join(self.lq_root,gt_name_vid.split('_')[0]+'_'+str(gt_w//2)+'x'+str(gt_h//2)+'_'+str(nfs)+'.yuv')
            lq_name_vid = lq_vid_path.split('/')[-1]
            lq_w, lq_h = map(int, lq_name_vid.split('_')[-2].split('x'))

            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['ref_path'].append(ref_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['gt_name_vid'].append(gt_name_vid)
                self.data_info['gt_w'].append(gt_w)
                self.data_info['gt_h'].append(gt_h)
                self.data_info['lq_w'].append(lq_w)
                self.data_info['lq_h'].append(lq_h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        nfs = int(self.data_info['gt_path'][index].split('/')[-1].split('.')[-2].split('_')[-1])
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['gt_h'][index],
            w=self.data_info['gt_w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
        )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
        ).astype(np.float32) / 255.  # (H W 1)

        h,w,_=img_gt.shape

        start_frm = (self.data_info['gt_index'][index] // 4) * 4
        img_refs = []
        ref_num =[start_frm,start_frm+4]
        ref_num =np.clip(ref_num , 0, nfs - 1)
        for i in range(len(ref_num)):
            img_ref = import_yuv(
                seq_path=self.data_info['ref_path'][index],
                h=self.data_info['gt_h'][index],
                w=self.data_info['gt_w'][index],
                tot_frm=1,
                start_frm=ref_num[i],
                only_y=True
            )
            img_ref =np.expand_dims(
                    np.squeeze(img_ref), 2
                ).astype(np.float32) / 255.
            img_refs.append(img_ref)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            if lq_index % 4 == 0:
                img = import_yuv(
                    seq_path=self.data_info['ref_path'][index],
                    h=self.data_info['gt_h'][index],
                    w=self.data_info['gt_w'][index],
                    tot_frm=1,
                    start_frm=lq_index,
                    only_y=True
                )
                img_lq = np.expand_dims(np.squeeze(img), 2).astype(np.float32) / 255.
                img_lqs.append(np.expand_dims(cv2.resize(img_lq, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC),2))
                continue

            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['lq_h'][index],
                w=self.data_info['lq_w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
            )
            img_lq = np.expand_dims(np.squeeze(img), 2).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor
        # img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results,dim=0)
        img_refs = totensor(img_refs)
        img_refs = torch.stack(img_refs, dim=0)
        img_gt = totensor(img_gt)


        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'ref': img_refs,
            'gt_name_vid': self.data_info['gt_name_vid'][index],
            'index_vid': self.data_info['index_vid'][index],
        }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num