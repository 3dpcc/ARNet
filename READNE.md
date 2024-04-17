# ARNet: Attribute Artifact Reduction for G-PCC Compressed Point Clouds


Abstract —   A learning-based adaptive loop filter is developed for the geometry-based point-cloud compression (G-PCC) standard to reduce attribute compression artifacts. The proposed method first generates multiple most probable sample offsets (MPSOs) as potential compression distortion approximations, and then linearly weights them for artifact mitigation. Therefore, we drive the filtered reconstruction as closely to the uncompressed PCA as possible. To this end, we devise an attribute artifact reduction network (ARNet) consisting of two consecutive processing phases: MPSOs derivation and MPSOs combination. The MPSOs derivation uses a two-stream network to model local neighborhood variations from direct spatial embedding and frequency-dependent embedding, where sparse convolutions are utilized to best aggregate information from sparsely and irregularly distributed points. The MPSOs combination is guided by the least-squares error metric to derive weighting coefficients on the fly to further capture the content dynamics of the input PCAs. ARNet is implemented as an in-loop filtering tool for G-PCC, where the linear weighting coefficients are encapsulated into the bitstream with negligible bitrate overhead. The experimental results demonstrate significant improvements over the latest G-PCC both subjectively and objectively. For example, our method offers a 22.12% YUV BD-Rate (Bjontegaard Delta Rate) reduction compared to G-PCC across various commonly used test point clouds. Compared with a recent study showing state-of-the-art performance, our work not only gains 13.23% YUV BD-Rate but also provides a 30x processing speedup.

## News
- 2023.10 The paper was accpeted by CVM (Computational Visual Media). (Junzhe Zhang, Junteng Zhang, Dandan Ding, and Zhan Ma, "ARNet: Attribute Artifact Reduction for G-PCC Compressed Point Clouds")



## Requirments
- pytorch **1.10**
- MinkowskiEngine 0.54
- h5py
- open3d
- Training Dataset: ShapeNet 



## Testing
Testing:

You need write ckpts path and qp in the test.py files
```python
python test.py --mode='enhancer_y' --input_rootdir='dataset' --output_rootdir='output/enhancer_y/'
python test.py --mode='enhancer_u' --input_rootdir='dataset' --output_rootdir='output/enhancer_u/'
python test.py --mode='enhancer_v' --input_rootdir='dataset' --output_rootdir='output/enhancer_v/'
```
If you have all three models, you can use 'test_yuv.py' to enhance three channels.
```python
python test_yuv.py  --mode='enhancer' --output_rootdir='./output/enhancer/'
```

## Authors
These files are provided by Hangzhou Normal University [IVC Lab](https://github.com/3dpcc/3DPCC) and Nanjing University [Vision Lab](https://vision.nju.edu.cn/).  Please contact us (zhangjunzhe@stu.hznu.edu.cn, zhangjunteng@stu.hznu.edu.cn, DandanDing@hznu.edu.cn and mazhan@nju.edu.cn) if you have any questions.
