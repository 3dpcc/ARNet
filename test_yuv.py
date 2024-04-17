import numpy as np
import torch
import os, glob, argparse, time
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_processing.data_utils import *
from extension.pc_error import pc_error
import MinkowskiEngine as ME
from shutil import copyfile


def test_enhance(input_rootdir, output_rootdir, input_format, ckptsdir_list):
    torch.cuda.empty_cache()

    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*' + input_format), recursive=True))
    for idx_file, input_filedir in enumerate(input_filedirs):
        # load data
        # label = label_list[idx_file]
        if input_filedir.endswith('ply'):
            filedir_ply = input_filedir
        elif input_filedir.endswith('h5'):
            # coords, feats, label = read_h5_label(input_filedir)
            filedir_ply = os.path.join(output_rootdir, 'ply', input_filedir[len(input_rootdir):].split('.')[0] + '.ply')
            ply_folder, _ = os.path.split(filedir_ply);
            os.makedirs(ply_folder, exist_ok=True)
            write_ply_ascii(filedir_ply, coords, feats)
        # gpcc post processing
        for idx_qp, ckpt_list in enumerate(ckptsdir_list):
            from model_arnet import ARNet
            # gpcc
            qp = ckpt_list[0]
            ckptsdir = ckpt_list[1]
            model = ARNet(ckpt_y=ckptsdir[0], ckpt_u=ckptsdir[1], ckpt_v=ckptsdir[2]).to(device)

            bin_dir = os.path.join(output_rootdir, 'bin', input_filedir[len(input_rootdir):].split('.')[0]
                                   + '_qp' + str(qp) + '.bin')
            rec_dir = os.path.join(output_rootdir, 'rec', input_filedir[len(input_rootdir):].split('.')[0]
                                   + '_qp' + str(qp) + '.ply')
            bin_dir = output_rootdir + 'bin' + bin_dir
            rec_dir = output_rootdir + 'rec' + rec_dir
            bin_folder, _ = os.path.split(bin_dir);
            os.makedirs(bin_folder, exist_ok=True)
            rec_folder, _ = os.path.split(rec_dir);
            os.makedirs(rec_folder, exist_ok=True)

            from test_gpcc import test
            # if not os.path.exists(rec_dir):
            results_one = test(filedir=filedir_ply, bin_dir=bin_dir, rec_dir=rec_dir, transformType=0, qp=qp, gpcc_version=14,
                            test_bpp=True, test_psnr=True)
            print('gpcc results:')
            print(results_one)
            # post processing
            gt = load_sparse_tensor(input_filedir)
            x = load_sparse_tensor(rec_dir, order='gbr')

            with torch.no_grad():
                out_set = model(x, gt)
            # ground truth coordinates same with x
            torch.cuda.empty_cache()
            #save point cloud as .ply file
            coords = out_set['out'].C.detach().cpu().numpy()[:, 1:]
            feats = (out_set['out'].F.detach().cpu().numpy() * 255).round()
            feats = np.clip(feats, 0, 255).astype('uint8')
            enh_dir = rec_dir[:-4] + '_enh.ply'
            write_ply_ascii(enh_dir, coords, feats)
            print(filedir_ply, enh_dir)
            pc_error_results = pc_error(filedir_ply, enh_dir, res=1023, normal=False, color=True, show=False)
            print('enhance results:')
            print(pc_error_results)
            for k, v in pc_error_results.items(): results_one['Enh_' + k] = v
            if idx_qp == 0:
                results_list = [results_one]
            else:
                results_list.append(results_one)
        # merge all qp
        results = {}
        one_keys = ['filename', 'num_points', 'num_points', 'bpp_geo', 'mseF,PSNR (p2point)', 'Enh_mseF,PSNR (p2point)']
        multi_keys = ['qp', 'bpp_att', '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF', 'Enh_  c[0],PSNRF', 'Enh_  c[1],PSNRF', 'Enh_  c[2],PSNRF']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items():
                if k in one_keys and idx_rate == 0: results[k] = v
                if k in multi_keys: results['R' + str(idx_rate) + '_' + k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file == 0:
            results_allfile = results.copy(deep=True)
        else:
            results_allfile = pd.concat([results_allfile,results], ignore_index=True)
        csvfile = os.path.join(output_rootdir, output_rootdir.split('/')[-2] + '.csv')
        results_allfile.to_csv(csvfile, index=False)
    print('save results to ', csvfile)

    return results_allfile


if __name__ == '__main__':
    """
    if you have Y, U and V trained models, you can use this code to test YUV three channels
    python test_yuv.py  --mode='enhancer' --output_rootdir='./output/enhancer/'
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default='')
    parser.add_argument("--input_rootdir", type=str, default='')
    parser.add_argument("--output_rootdir", type=str, default='output/tp/')
    parser.add_argument("--input_format", default='ply')
    args = parser.parse_args()
    os.makedirs(args.output_rootdir, exist_ok=True)
    print('dbg:\t output_rootdir:\t', args.output_rootdir)
    copyfile('test_yuv.py', args.output_rootdir + 'test_yuv.py')

    if args.mode == 'enhancer':
        ckptsdir_list = [[22, ['pretrained/qp22/qp22_y/epoch_last.pth',
                               'pretrained/qp22/qp22_u/epoch_last.pth',
                               'pretrained/qp22/qp22_v/epoch_last.pth']]]
        results = test_enhance(args.input_rootdir, args.output_rootdir, args.input_format, ckptsdir_list)


