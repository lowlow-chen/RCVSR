import torch
import numpy as np
import os
from collections import OrderedDict
from net_rcvsr import SR
import torch.nn.functional as F
import utils
from tqdm import tqdm
from datetime import datetime as dt


ckp_path = 'exp/LDP_QP32.pt' #trained at QP32, LDP, HM16.5

raw_yuv_path = '/home/***/mfqe/test_raw_yuv/BQSquare_416x240_600.yuv'
lq_yuv_path = '/home/***/mfqe/LDP/test_qp_down/QP32/BQSquare_208x120_600.yuv'
ref_yuv_path = '/home/***/mfqe/LDP/test_qp_yuv/QP37/BQSquare_416x240_600.yuv'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

h, w, nfs = 120,208,600

def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 48,
            'nf': 48,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 5,
            },
        'self_alig': {
            'in_nc': 1,
            'out_nc': 48,
            'nf': 48,
            'nb': 3,
            'deform_ks': 5,
        },
        'extractor': {
            'in_nc': 1,
            'nf': 48,
            'base_ks': 3,
        },
        'upnet': {
            'in_nc': 48,
            'nf': 48,
            'scale_factor': 2,
            'base_ks': 3,
        },
        'fusion': {
            'in_nc': 48,
            'out_nc': 1,
            'nf': 48,
            'base_ks': 3,
        },
        'attention': {
            'in_nc': 48,
            'nf':48,
            'base_ks': 3,
            'nb': 2,
        },

        }
    model = SR(opts_dict=opts_dict)
    print('%s Parameters in model: %d.' % (dt.now(), count_parameters(model)))
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # # ==========
    # # Load entire video
    # # ==========

    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h*2, w=w*2, tot_frm=nfs, start_frm=0, only_y=True
        )
    lq_y = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    ref_y = utils.import_yuv(
        seq_path=ref_yuv_path, h=2*h, w=2*w, tot_frm=nfs, start_frm=0, only_y=True
        )

    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    ref_y = ref_y.astype(np.float32) / 255.

    msg = '> yuv loaded.'
    print(msg)


    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========

    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    with torch.no_grad():
        for idx in range(nfs):
        # load lq
            idx_list = list(range(idx-3,idx+4))
            idx_list = np.clip(idx_list, 0, nfs-1)
            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()
            gt_frm = torch.from_numpy(raw_y[idx]).cuda()

            ref = []
            start_frm = (idx//4)*4
            ref_num = [start_frm,start_frm+4]
            ref_num = np.clip(ref_num, 0, nfs - 1)
            for frm in ref_num:
                ref.append(torch.from_numpy(ref_y[frm]).unsqueeze(dim=0))

            ref = torch.cat(ref,dim = 0).cuda()

            sr_frm = model(input_data, ref.unsqueeze(dim=0))

            # eval
            lq = F.interpolate(input_data[:, 3, ...].unsqueeze(dim=1), scale_factor=2, mode='bicubic', align_corners=True)
            batch_ori = criterion(lq.squeeze(dim=0).squeeze(dim=0), gt_frm)
            batch_perf = criterion(sr_frm[0, 0, ...], gt_frm)
            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)
            # display
            pbar.set_description(
                " [{:.3f}] {:s} -> [{:.3f}] {:s}"
                    .format( batch_ori, unit, batch_perf, unit)
            )
            pbar.update()

    pbar.close()

    enh_ = enh_psnr_counter.get_ave()
    ori_ = ori_psnr_counter.get_ave()

    msg = (
        f"{'> ori: [{:.3f}] {:s}'.format(ori_, unit)}\n"
        f"{'> ave: [{:.3f}] {:s}'.format(enh_, unit)}\n"
        f"{'> delta: [{:.3f}] {:s}'.format(enh_ - ori_, unit)}"
        )
    print(msg)
    print('> done.')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    main()
