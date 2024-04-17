import torch
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
from data_processing.data_utils import rgb2yuv, yuv2rgb
import math


class ResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


def make_layer(block, block_layers, channels):
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)


class InceptionResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0_0(x))
        out0 = self.relu(self.conv0_2(self.relu(self.conv0_1(out))))
        out1 = self.relu(self.conv1_1(self.relu(self.conv1_0(x))))
        out = ME.cat(out0, out1)
        return out + x

#####################################################################
class Enhancer(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res0 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res1 = InceptionResNet(channels=128)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res2 = InceptionResNet(channels=128)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )
        # self.block3 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res3 = InceptionResNet(channels=64)
        # self.knn3 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv_out0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv_out1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.res0(self.relu(self.conv0(x)))
        # out = self.knn0(out)

        out = self.res1(self.relu(self.conv1(out)))
        # out = self.knn1(out)

        out = self.res2(self.relu(self.conv2(out)))
        # out = self.knn2(out)

        out = self.res3(self.relu(self.conv3(out)))

        # out = self.res4(self.relu(self.conv4(out)))

        out = self.conv_out0(out)
        out = out + self.conv_out1(x)

        return out


class Global_Enhancer(torch.nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res0 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res1 = InceptionResNet(channels=64)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res2 = InceptionResNet(channels=64)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv_out1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.res0(self.relu(self.conv0(x)))
        # out = self.knn0(out)

        out = self.res1(self.relu(self.conv1(out)))
        # out = self.knn1(out)

        out = self.res2(self.relu(self.conv2(out)))
        # out = self.knn2(out)

        out = out + self.conv_out1(x)

        return out


class Local_enhancer(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        # down sample 3 times
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.res0 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res1 = InceptionResNet(channels=64)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.res2 = InceptionResNet(channels=64)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.conv_mid = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.up_conv2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block0 = make_layer(block=ResNet, block_layers=3, channels=32)
        self.up_res2 = InceptionResNet(channels=64)
        # self.knn0 = Point_Transformer_Last(block=4, channels=64, head=1, k=16)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.up_conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block1 = make_layer(block=ResNet, block_layers=3, channels=64)
        self.up_res1 = InceptionResNet(channels=channels)
        # self.knn1 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.up_conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.block2 = make_layer(block=ResNet, block_layers=3, channels=64)
        # self.up_res1 = InceptionResNet(channels=32)
        # self.knn2 = Point_Transformer_Last(channels=128, head=1, k=16)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.res0(self.relu(self.down0(self.conv0(x))))

        out = self.res1(self.relu(self.down1(self.conv1(out))))

        out = self.res2(self.relu(self.down2(self.conv2(out))))

        out = self.conv_mid(out)

        out = self.up_res2(self.relu(self.up_conv2(self.up2(out))))

        out = self.up_res1(self.relu(self.up_conv1(self.up1(out))))

        out = self.up_conv0(self.up0(out))

        return out


class High_enhancer(torch.nn.Module):
    def __init__(self, k=3, channels=32):
        super().__init__()
        # self.res = InceptionResNet(channels=channels)
        self.avg = ME.MinkowskiAvgPooling(kernel_size=k, stride=2, dimension=3)

        self.upsample = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

    def forward(self, x):
        # out = self.res(x)
        out = self.avg(x)
        out = self.upsample(out)

        out = x - out
        return out

class conv_module(torch.nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )
        self.res = InceptionResNet(channels=channels)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        return x + self.res(self.relu(self.conv(x)))

class conv_out_v(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )
        self.res = InceptionResNet(channels=channels)

        self.relu = ME.MinkowskiReLU(inplace=True)

        self.conv_out = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3
        )

    def forward(self, x):
        out = x + self.res(self.relu(self.conv(x)))
        return self.conv_out(out)

class Module_v(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        # global feature extract
        self.glb_enhancer = Global_Enhancer(channels=32)

        # local feature extract
        self.loc_enhancer = Local_enhancer(channels=32)

        self.high_enhancer = High_enhancer(channels=32)

        self.conv_model = conv_module(channels=64)

        self.conv_out = conv_out_v(channels=128)

        self.conv_outx = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # input = self.conv_in(x)

        out_global = self.glb_enhancer(x)

        out_local = self.loc_enhancer(x)

        out_high = self.high_enhancer(x)

        out_LCH = ME.cat(out_local, out_high)
        out_LCH = self.conv_model(out_LCH)

        out = ME.cat(out_global, out_LCH)

        out = self.conv_out(out)

        return out + self.conv_outx(x)


class Enhancer_v(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in_y = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_in_u = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_in_v = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_in = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.model_v = Module_v()
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1,
                                           stride=1,
                                           dilation=1,
                                           dimension=3)
        self.scale = 128

    def guided(self, y, u, v):
        input_y = self.relu(self.conv_in_y(y))
        input_u = self.relu(self.conv_in_u(u))
        input_v = self.relu(self.conv_in_v(v))
        feats = torch.cat((input_y.F, input_u.F, input_v.F), dim=1)
        yuv = ME.SparseTensor(features=feats, coordinate_map_key=input_u.coordinate_map_key,
                              coordinate_manager=input_u.coordinate_manager)
        in_yuv = self.conv_in(yuv)
        out = self.model_v(in_yuv)

        return out

    def forward(self, x_ori, gt):
        gt = self.sort(gt, x_ori.C)
        feats = rgb2yuv(x_ori.F)
        label = rgb2yuv(gt.F)
        y = feats[:, :1]
        u = feats[:, 1:2]
        v = feats[:, 2:3]
        true_v = label[:, 2:3]
        x = ME.SparseTensor(features=v.float(), coordinates=x_ori.C, device=x_ori.device)
        y = ME.SparseTensor(features=y.float(), coordinates=x_ori.C, device=x_ori.device)
        u = ME.SparseTensor(features=u.float(), coordinates=x_ori.C, device=x_ori.device)
        q1, q2 = x.shape
        # forward
        out = self.guided(y, u, x)
        r_out = out.F

        true_v = true_v.cuda()
        b_feature = torch.subtract(true_v, x.F)

        A = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
            b_feature)
        b_feature = b_feature.permute(1, 0)
        loss = torch.sum(-(torch.matmul(torch.matmul(b_feature, r_out), A)))
        residual = torch.sum(torch.matmul(r_out, A), dim=1).reshape(q1, q2)
        final = residual + x.F
        feats[:,2:] = final
        rgb_final = yuv2rgb(feats)
        x_out = ME.SparseTensor(features=rgb_final.float(), coordinates=x_ori.C, device=x_ori.device)
        return {'out': x_out,
                'loss': loss,
                'A': A}

    @torch.no_grad()
    def encode(self, x_ori, gt):
        gt = self.sort(gt, x_ori.C)
        feats = rgb2yuv(x_ori.F)
        label = rgb2yuv(gt.F)
        y = feats[:, :1]
        u = feats[:, 1:2]
        v = feats[:, 2:3]
        true_v = label[:, 2:3]
        x = ME.SparseTensor(features=v.float(), coordinates=x_ori.C, device=x_ori.device)
        y = ME.SparseTensor(features=y.float(), coordinates=x_ori.C, device=x_ori.device)
        u = ME.SparseTensor(features=u.float(), coordinates=x_ori.C, device=x_ori.device)
        q1, q2 = x.shape
        # forward
        out = self.guided(y, u, x)
        r_out = out.F

        true_v = true_v.cuda()
        b_feature = torch.subtract(true_v, x.F)

        A = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(r_out.permute(1, 0), r_out)), r_out.permute(1, 0)),
            b_feature)
        A = A * self.scale
        np.savetxt('A.txt', A.short().detach().cpu().numpy())
        A_bytes = os.path.getsize('A.txt')
        return A, A_bytes


    @torch.no_grad()
    def decode(self, x_ori, A):
        A = A.float() / self.scale
        feats = rgb2yuv(x_ori.F)
        y = feats[:, :1]
        u = feats[:, 1:2]
        v = feats[:, 2:3]
        x = ME.SparseTensor(features=v.float(), coordinates=x_ori.C, device=x_ori.device)
        y = ME.SparseTensor(features=y.float(), coordinates=x_ori.C, device=x_ori.device)
        u = ME.SparseTensor(features=u.float(), coordinates=x_ori.C, device=x_ori.device)
        q1, q2 = x.shape
        # forward
        out = self.guided(y, u, x)
        r_out = out.F

        residual = torch.sum(torch.matmul(r_out, A), dim=1).reshape(q1, q2)
        final = residual + x.F
        feats[:,2:] = final
        rgb_final = yuv2rgb(feats)
        x_out = ME.SparseTensor(features=rgb_final.float(), coordinates=x_ori.C, device=x_ori.device)
        return {'out': x_out}



if __name__ == '__main__':
    # encoder = Encoder(128, 3)
    # print(encoder)
    # decoder = Decoder(128, 3)
    # print(decoder)
    #
    # hyperEncoder = HyperEncoder(128)
    # print(hyperEncoder)
    # hyperDecoder = HyperDecoder(128)
    # print(hyperDecoder)
    #
    # contextModelBase = ContextModelBase(128)
    # print(contextModelBase)
    #
    # contextModelHyper = ContextModelHyper(128)
    # print(contextModelHyper)
    enhance = v_module()
    print(enhance)
    print('params:', sum(param.numel() for param in enhance.parameters()))
