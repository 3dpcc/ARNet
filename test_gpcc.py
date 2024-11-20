import numpy as np
import os, glob, argparse, random
from extension.gpcc import gpcc_encode, gpcc_decode, get_points_number
from extension.gpcc import gpccv6_encode, gpccv6_decode
from extension.gpcc import gpcc_encode_intra, gpcc_decode_intra
from tqdm import tqdm
from data_processing.data_utils import read_h5, write_h5_label, read_ply_ascii, write_ply_ascii

import pandas as pd

def test(filedir, bin_dir, rec_dir, transformType=0, qp=34, gpcc_version=14, test_bpp=True, test_psnr=True):
    # [22,28,34,40,46,51]
    if gpcc_version==14:
        results_enc = gpcc_encode(filedir=filedir, bin_dir=bin_dir, transformType=transformType, qp=qp)
        results_dec = gpcc_decode(bin_dir=bin_dir, rec_dir=rec_dir)
    elif gpcc_version==6:
        results_enc = gpccv6_encode(filedir, bin_dir, transformType=transformType, qp=qp)
        results_dec = gpccv6_decode(bin_dir, rec_dir)
    elif gpcc_version == 'ges':
        results_enc = gpcc_encode_intra(filedir=filedir, bin_dir=bin_dir, transformType=transformType, qp=qp)
        results_dec = gpcc_decode_intra(bin_dir, rec_dir)
    else: raise ValueError('gpcc_version'+str(gpcc_version))
    # record results
    results = {'filename':os.path.split(filedir)[-1].split('.')[0], 'qp':qp}
    for k, v in results_enc.items(): results['Enc_'+ k] = v
    for k, v in results_dec.items(): results['Dec_'+ k] = v
    if test_bpp:
        num_points = get_points_number(filedir)
        bpp_geo = results_enc['positions bitstream size']*8/num_points
        bpp_att = results_enc['colors bitstream size']*8/num_points
        results['num_points'] = num_points
        results['bpp_geo'] = bpp_geo
        results['bpp_att'] = bpp_att
        print('dbg: filedir:\t', filedir)
        print('dbg: bpp_att:\t', round(bpp_att, 2))
    if test_psnr:
        from extension.pc_error import pc_error
        pc_error_results = pc_error(filedir, rec_dir, res=1023, show=False)
        for k, v in pc_error_results.items(): results[k] = v
        print('dbg: PSNR:\t', round(pc_error_results['  c[0],PSNRF'], 2))

    return results


def make_dataset(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, qp, write_label=True):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    random.shuffle(input_filedirs)
    input_filedirs=input_filedirs[:input_length]
    for idx, input_filedir in enumerate(tqdm(input_filedirs)):
        # input
        if input_filedir.endswith('h5'):
            coords, feats = read_h5(input_filedir)
            filedir_ply = os.path.join(output_rootdir, 'ply', input_filedir[len(input_rootdir):].split('.')[0]+'.ply')
            ply_folder, _ = os.path.split(filedir_ply); os.makedirs(ply_folder, exist_ok=True)
            write_ply_ascii(filedir_ply, coords, feats)
        elif input_filedir.endswith('ply'):
            filedir_ply = input_filedir
        # test
        bin_dir = os.path.join(output_rootdir, 'bin', input_filedir[len(input_rootdir):].split('.')[0]+'.bin')
        rec_dir = os.path.join(output_rootdir, 'rec_qp'+str(qp), input_filedir[len(input_rootdir):].split('.')[0]+'.ply')
        bin_folder, _ = os.path.split(bin_dir); os.makedirs(bin_folder, exist_ok=True)
        rec_folder, _ = os.path.split(rec_dir); os.makedirs(rec_folder, exist_ok=True)
        results_one = test(filedir_ply, bin_dir, rec_dir, qp, test_bpp=False, test_psnr=False)
        # output
        if idx >= output_length: break
        if output_format=='h5':
            coords, feats = read_ply_ascii(rec_dir, order='gbr')
            h5_dir = os.path.join(output_rootdir, 'h5_rec_qp'+str(qp), input_filedir[len(input_rootdir):].split('.')[0]+'.h5')
            h5_folder, _ = os.path.split(h5_dir); os.makedirs(h5_folder, exist_ok=True)
            coords_label, feats_label = read_ply_ascii(filedir_ply)
            from data_processing.data_utils import sort_points
            coords_label, feats_label = sort_points(coords_label, feats_label)
            coords, feats = sort_points(coords, feats)
            if False:
                assert (coords == coords_label).all()
                mae = np.abs(feats.astype('float') - feats_label.astype('float')).mean().round(2)
                mse = ((feats.astype('float') - feats_label.astype('float'))**2).mean().round(4)
                PSNR = (20 * np.log10(255./np.sqrt(mse))).round(2)
                print('mae:', mae, '\tmse:', mse, '\trmse:', np.sqrt(mse).round(2), '\tPSNR:', PSNR)
            write_h5_label(h5_dir, coords, feats, feats_label)
        # results
        results_one = pd.DataFrame([results_one])
        if idx==0: all_results = results_one.copy(deep=True)
        else: all_results = all_results.append(results_one, ignore_index=True)
        csvfile = os.path.join(output_rootdir, 'qp'+str(qp)+'.csv')        
        all_results.to_csv(csvfile, index=False)
    print('save results to ', csvfile)
    
    return all_results


def test_dataset(input_rootdir, output_rootdir, input_format='ply', transformType=0, qp_list=[22,28,34,40,46,51], gpcc_version=14):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    for idx_file, input_filedir in enumerate(input_filedirs):
        from data_processing.data_utils import read_h5, write_ply_ascii
        if input_filedir.endswith('ply'):
            filedir_ply = input_filedir
        elif input_filedir.endswith('h5'):
            coords, feats = read_h5(input_filedir)
            filedir_ply = os.path.join(output_rootdir, 'ply', input_filedir[len(input_rootdir):].split('.')[0]+'.ply')
            ply_folder, _ = os.path.split(filedir_ply); os.makedirs(ply_folder, exist_ok=True)
            write_ply_ascii(filedir_ply, coords, feats)
        # test
        for idx_qp, qp in enumerate(qp_list):
            bin_dir = os.path.join(output_rootdir, 'bin', input_filedir[len(input_rootdir):].split('.')[0] 
                                + '_qp' + str(qp) + '.bin')
            rec_dir = os.path.join(output_rootdir, 'rec', input_filedir[len(input_rootdir):].split('.')[0] 
                                + '_qp' + str(qp) + '.ply')
            bin_folder, _ = os.path.split(bin_dir); os.makedirs(bin_folder, exist_ok=True)
            rec_folder, _ = os.path.split(rec_dir); os.makedirs(rec_folder, exist_ok=True)
            results_one = test(filedir_ply, bin_dir, rec_dir, transformType=transformType, qp=qp, gpcc_version=gpcc_version, test_bpp=True, test_psnr=True)
            if idx_qp==0: results_list = [results_one]
            else: results_list.append(results_one)
        # merge all qp
        results = {}
        one_keys = ['filename', 'num_points', 'num_points', 'bpp_geo', 'mseF,PSNR (p2point)']
        multi_keys = ['qp', 'bpp_att', '  c[0],PSNRF', '  c[1],PSNRF', '  c[2],PSNRF']
        for idx_rate, results_one in enumerate(results_list):
            for k, v in results_one.items(): 
                if k in one_keys and idx_rate==0: results[k] = v
                if k in multi_keys: results['R'+str(idx_rate)+'_'+k] = v
        results = pd.DataFrame([results])
        # merge all file
        if idx_file==0: results_allfile = results.copy(deep=True)
        else: results_allfile = results_allfile.append(results, ignore_index=True)
        csvfile = os.path.join(output_rootdir, 'gpcc.csv')        
        results_allfile.to_csv(csvfile, index=False)
    print('save results to ', csvfile)

    return results_allfile


if __name__ == '__main__':
    """
    python test_gpcc.py 
    python test_gpcc.py --transformType=2 --output_rootdir='./output/predlift/'
    python test_gpcc.py --transformType=2 --gpcc_version=6 --output_rootdir='./output/gpccv6_predlift/'
    python test_gpcc.py --transformType=1 --gpcc_version=6 --output_rootdir='./output/gpccv6_raht/'
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_rootdir", default='')
    parser.add_argument("--output_rootdir", default='')
    parser.add_argument("--input_format", default='ply')
    parser.add_argument("--output_format", default='ply')
    parser.add_argument("--input_length", type=int, default=int(1e6))
    parser.add_argument("--output_length", type=int, default=int(1e6))
    parser.add_argument("--transformType", type=int, default=2)
    parser.add_argument("--gpcc_version", type=int, default=14)
    # parser.add_argument("--qp", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_rootdir, exist_ok=True)
    

    if args.gpcc_version==14: qp_list=[51,46,40,34]
    elif args.gpcc_version==6: qp_list=[52,46,40,34,28,22]
    results = test_dataset(args.input_rootdir, args.output_rootdir, args.input_format, 
                            transformType=args.transformType, qp_list=qp_list, gpcc_version=args.gpcc_version)
    print(results.mean())
    print(results)