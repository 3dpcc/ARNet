import open3d as o3d
import os
import numpy as np
import h5py

####################################### io #######################################
def read_h5(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('uint8')

    return coords, feats



def read_h5_label(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('uint8')
    label = h5py.File(filedir, 'r')['label'][:].astype('uint8')
    if False: print('DBG read h5 mae:', np.abs(feats.astype('float') - label.astype('float')).mean().round(2))
    

    return coords, feats, label

def write_h5_label(filedir, coords, feats, label):
    coords = coords.astype('int16')
    feats = feats.astype('uint8')
    label = label.astype('uint8')
    if False: print('DBG write h5 mae:', np.abs(feats.astype('float') - label.astype('float')).mean().round(2))
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('coords', data=coords, shape=coords.shape)
        h.create_dataset('feats', data=feats, shape=feats.shape)
        h.create_dataset('label', data=label, shape=label.shape)
    # print(feats.mean().round(), feats.max(), feats.min())

    return

def read_ply_ascii(filedir, order='rgb'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int16')
    if data.shape[-1]==6: feats = data[:,3:6].astype('uint8')
    if data.shape[-1]>6: feats = data[:,6:9].astype('uint8')
    if order=='gbr': feats = np.hstack([feats[:,2:3], feats[:,0:2]])

    return coords, feats

def write_ply_ascii(filedir, coords, feats):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n', 
                'property uchar red\n','property uchar green\n','property uchar blue\n',])
    f.write('end_header\n')
    coords = coords.astype('int16')
    feats = feats.astype('uint8')
    for xyz, rgb in zip(coords, feats):
        f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                    str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    f.close() 

    return


####################################### reorder #######################################
def sort_points(coords, feats):
    indices_sort = np.argsort(array2vector(coords))

    return coords[indices_sort], feats[indices_sort]

def array2vector(array):
    # 3D -> 1D by sum each dimension
    array = array.astype('int64')
    step = array.max() + 1     
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])
    
    return vector



####################################### color space conversion #######################################
def rgb2yuv(rgb):
    """input: [0,1];    output: [0,1]
    """
    rgb = 255*rgb
    yuv = rgb.clone()
    yuv[:,0] = 0.257*rgb[:,0] + 0.504*rgb[:,1] + 0.098*rgb[:,2] + 16
    yuv[:,1] = -0.148*rgb[:,0] - 0.291*rgb[:,1] + 0.439*rgb[:,2] + 128
    yuv[:,2] = 0.439*rgb[:,0] - 0.368*rgb[:,1] - 0.071*rgb[:,2] + 128
    yuv[:,0] = (yuv[:,0]-16)/(235-16)
    yuv[:,1] = (yuv[:,1]-16)/(240-16)
    yuv[:,2] = (yuv[:,2]-16)/(240-16)
    
    return yuv

def yuv2rgb(yuv):
    """input: [0,1];    output: [0,1]
    """
    yuv[:,0] = (235-16)*yuv[:,0]+16
    yuv[:,1] = (240-16)*yuv[:,1]+16
    yuv[:,2] = (240-16)*yuv[:,2]+16
    rgb = yuv.clone()
    rgb[:,0] = 1.164*(yuv[:,0]-16) + 1.596*(yuv[:,2]-128)
    rgb[:,1] = 1.164*(yuv[:,0]-16) - 0.813*(yuv[:,2]-128) - 0.392*(yuv[:,1]-128)
    rgb[:,2] = 1.164*(yuv[:,0]-16) + 2.017*(yuv[:,1]-128)
    rgb = rgb/255
    
    return rgb

def mse_yuv(dataA, dataB, weight):
    MSELoss = torch.nn.MSELoss().to(dataA.device)
    dataA = rgb2yuv(dataA)
    dataB = rgb2yuv(dataB)
    mse = (weight*MSELoss(dataA[:,0], dataB[:,0]) + \
            MSELoss(dataA[:,1], dataB[:,1]) + \
            MSELoss(dataA[:,2], dataB[:,2]))/(weight+2)

    return mse

####################################### sparse tensor processing #######################################
import torch
import MinkowskiEngine as ME


def load_sparse_tensor(filedir, max_num_points=1e8, device='cuda', order='rgb'):
    if filedir.endswith('h5'): coords, feats = read_h5(filedir)
    if filedir.endswith('ply'): coords, feats = read_ply_ascii(filedir, order=order) 
    if coords.shape[0] <= max_num_points:      
        coords = torch.tensor(coords).int()
        feats = torch.tensor(feats).float()/255.
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

        return x

    else:
        points = np.hstack([coords.astype('int16'), feats.astype('int16')])
        points_list = kdtree_partition(points, max_num=max_num_points)
        x_list = []
        for points_part in points_list:
            # print('DBG!!! ', points_part.shape)
            coords_part = torch.tensor(points_part[:, 0:3]).int()
            feats_part = torch.tensor(points_part[:, 3:6]).float()/255.
            # print('DBG!!! ', coords_part.shape, feats_part.shape)
            coords_part, feats_part = ME.utils.sparse_collate([coords_part], [feats_part])
            x_part = ME.SparseTensor(features=feats_part, coordinates=coords_part, tensor_stride=1, device=device)
            x_list.append(x_part)
            
        return x_list

def sort_sparse_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu().numpy()))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)

    return sparse_tensor_sort