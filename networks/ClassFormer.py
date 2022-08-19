import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import einops
import math


class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class IntraClassAtten(nn.Module):
    def __init__(self, dim, img_size=28, rate=[2, 4, 7, 14]):
        super(IntraClassAtten, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** -0.5
        self.img_size = img_size
        self.find_kv = FindKV(kernel_size=3, dim=dim, rate= rate)

    def forward(self, x):

        B, N, C = x.shape  # (B, N, C)
        q = self.q_linear(x)
        q_kv = self.find_kv(q.contiguous().permute(0, 2, 1).reshape(B, C, self.img_size, self.img_size))
        q = q.view(B, N, self.num_head, C // self.num_head).permute(2, 0, 1, 3)

        kv = self.kv_linear(q_kv)
        n_kv = kv.shape[1]
        kv = kv.view(B, n_kv, 2, self.num_head, C // self.num_head).permute(2, 3, 0, 1, 4)  # (3, num_head, b, n, c)

        k, v = kv[0], kv[1]
        q = q * self.scale
        atten = q @ k.transpose(-1, -2).contiguous()

        atten = self.softmax(atten)
        atten_value = (atten @ v).contiguous().permute(1, 2, 0, 3).reshape(B, N, C)
        atten_value = self.proj(atten_value)
        return atten_value


class FindKV(nn.Module):
    def __init__(self, kernel_size=3, dim=256, rate=[2,4,7,14]):
        super(FindKV, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.rate = rate
        self.poolings = nn.ModuleList()
        self.conv_proj = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size, padding=padding, bias=False),
            LayerNormProxy(dim//2),
            nn.GELU(),
        )
        self.atten = nn.ModuleList()
        for i in range(len(self.rate)):
            self.poolings.append(nn.MaxPool2d(self.rate[i], stride=self.rate[i], ceil_mode=True, return_indices=True))
            self.atten.append(nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, C, H, W = x.shape
        x_proj = self.conv_proj(x)
        avg_out = torch.mean(x_proj, dim=1, keepdim=True)
        max_out, _ = torch.max(x_proj, dim=1, keepdim=True)
        attMap_ = torch.cat([avg_out, max_out], dim=1)

        for i in range(len(self.rate)):
            N_h = math.ceil(1.0 * H / self.rate[i])
            N_w = math.ceil(1.0 * W / self.rate[i])
            N = N_h * N_w

            attMap = self.sigmoid(self.atten[i](attMap_))
            _, indexes_original = self.poolings[i](attMap)
            x_kv = torch.gather(input=x.view(B,C,H*W),
                                dim=2,
                                index=indexes_original.view(B, 1, N).repeat(1, C, 1)).contiguous()
            x_kv = x_kv.permute(0, 2, 1)

            if i == 0:
                x_kv_cat = x_kv
            else:
                x_kv_cat = torch.cat([x_kv_cat, x_kv], dim =1)

        return x_kv_cat


class InterClassAtten(nn.Module):
    def __init__(self, dim):
        super(InterClassAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_linear(x).view(B, N, 3, self.num_head, C // self.num_head).permute(2, 3, 0, 1,
                                                                                          4)  # (3, num_head, b, n, c)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).contiguous().permute(1, 2, 0, 3).reshape(B, N, C)
        atten_value = self.proj(atten_value)
        return atten_value


class InterClassTrans(nn.Module):
    def __init__(self, dim):
        super(InterClassTrans, self).__init__()
        self.layerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.layerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = InterClassAtten(dim)
        self.ffn = MLP(dim)

    def forward(self, x_sample, x_global):
        n_ij = x_sample.shape[1]

        x = torch.cat([x_sample, x_global], dim = 1)
        h = x  # (B, N, H)
        x = self.layerNorm_1(x)
        x = self.Attention(x)  # padding 到right_size

        x = h + x

        h = x
        x = self.layerNorm_2(x)
        x = self.ffn(x)
        x = h + x

        x_sample = x[:, :n_ij, :]
        x_global = x[:, n_ij:, :]

        return x_sample, x_global


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class ResClassFormer(nn.Module):
    def __init__(self, in_ch, classes, train_state):
        super(ResClassFormer, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)

        self.class_former = ClassFormer(in_channels=512,
                                        size=28,
                                        classes=classes,
                                        train_state=train_state)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, classes, 1)

        self.conv4_seg = nn.Conv2d(512, classes, 1)
        self.upsample = nn.UpsamplingBilinear2d(224)

    def forward(self, x,  GT=None):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        c4_refined, loss_eu, loss_bce, c_mask = self.class_former(c4, GT)

        up_7 = self.up7(c4_refined)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        out = c10

        if self.training:
            return out, self.upsample(c_mask), loss_eu, loss_bce
        else:
            return out


class IntraClassTrans(nn.Module):
    def __init__(self, dim, img_size, rate = [2,4,7,14]):
        super(IntraClassTrans, self).__init__()
        self.layerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.layerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = IntraClassAtten(dim, img_size=img_size, rate=rate)
        self.ffn = MLP(dim)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.layerNorm_1(x)
        x = self.Attention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.layerNorm_2(x)

        x = self.ffn(x)
        x = h + x

        return x


class ClassFormer(nn.Module):
    def __init__(self, in_channels, size, classes, train_state):
        super(ClassFormer, self).__init__()
        self.training = train_state
        self.dim_inter = in_channels // 2
        self.dim_intra = in_channels // 2
        self.num_classes = classes
        self.size = size

        '''intra class transformer'''
        self.conv_intra = nn.Sequential(nn.Conv2d(in_channels, self.dim_intra, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(self.dim_intra),
                                        nn.LeakyReLU(inplace=True))
        self.class_groupconv = nn.Conv2d(self.num_classes * self.dim_intra, self.num_classes * self.dim_intra, 3, 1,
                                         1, groups=self.num_classes)

        self.intra_class_trans = nn.ModuleList()
        self.seghead = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(classes):
            self.intra_class_trans.append(IntraClassTrans(dim=self.dim_intra, img_size=28, rate=[2, 4, 7, 14]))
            self.seghead.append(nn.Conv2d(self.dim_intra, 1, 1))
            self.upsample.append(nn.UpsamplingBilinear2d(224))

        '''inter class transformer'''
        self.inter_class_trans = InterClassTrans(self.dim_inter)
        self.conv_inter_1 = nn.Sequential(
            nn.MaxPool2d(7),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True))

        self.conv_inter_2 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim_inter, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_inter),
            nn.LeakyReLU(inplace=True))

        '''fuse intra and inter'''
        self.up = nn.UpsamplingBilinear2d(scale_factor=7)
        self.conv4_seg = nn.Conv2d(in_channels, classes, 1)
        self.fuse_intra = nn.Sequential(nn.Conv2d(self.dim_intra + in_channels, in_channels, 1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.LeakyReLU(inplace=True))
        self.fuse_inter = nn.Sequential(nn.Conv2d(self.dim_inter + in_channels, in_channels, 1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.LeakyReLU(inplace=True))

        self.group_conv = nn.Conv2d(self.num_classes * self.dim_intra, self.num_classes * self.dim_intra, 3, 1, 1,
                                    groups=self.num_classes)

        self.compact = nn.Sequential(nn.Conv2d(self.num_classes * self.dim_intra, self.dim_intra, 1, bias=False),
                                     nn.BatchNorm2d(self.dim_intra),
                                     nn.LeakyReLU(inplace=True))

    def forward(self, feature, GT=None):

        feature_in = feature

        '''intra class transformer'''

        feature_in_intra = self.conv_intra(feature_in)
        feature_intra = feature_in_intra.repeat(self.num_classes, 1, 1, 1, 1)  # (num_classes, b, d, h, w)
        m, b, d, h, w = feature_intra.shape
        feature_intra = feature_intra.permute(1, 0, 2, 3, 4).reshape(b, m * d, h, w)
        feature_intra = self.class_groupconv(feature_intra)

        feature_loss_in = feature_intra.reshape(b, m, self.dim_intra, h, w).permute(1, 0, 2, 3, 4)  # (num_classes, b, d, h, w)
        feature_intra = feature_loss_in.view(m, b, self.dim_intra, -1)

        feature_intra_ = torch.zeros(self.num_classes, b, h * w, self.dim_intra).cuda()

        if self.training:
            gt = GT.type(torch.LongTensor).cuda()
            bce_loss = nn.BCEWithLogitsLoss()
            loss_bce = 0.0

        for i in range(self.num_classes):
            if self.training:
                gt_c = torch.where(gt == i, torch.tensor([1.0]).cuda(), torch.tensor([0.0]).cuda()).unsqueeze(1)  # (b, 28*28)
                mask_c = self.upsample[i](self.seghead[i](feature_loss_in[i]))  # [8, 1, 224, 224]
                loss_bce += bce_loss(mask_c, gt_c)

            feature_intra_[i] = self.intra_class_trans[i](feature_intra[i].transpose(-1, -2))

        feature_intra_ = feature_intra_.view(self.num_classes, b, h, w, self.dim_intra).permute(1, 2, 3, 0, 4).contiguous().view(b, h, w, -1)
        feature_intra_ = feature_intra_.permute(0, 3, 1, 2).contiguous()
        feature_intra_ = self.group_conv(feature_intra_)
        feature_intra_ = self.compact(feature_intra_)
        feature_intra = self.fuse_intra(torch.cat((feature_intra_, feature_in), dim=1))

        '''inter class transformer'''
        feature_in_inter = self.conv_inter_1(feature_intra)
        out_c = self.conv4_seg(feature_in_inter)
        feature_in_inter = self.conv_inter_2(feature_in_inter)

        b, d, h, w = feature_in_inter.shape
        prob_mask = nn.Softmax(dim=1)(out_c)

        feature_in_inter_ = feature_in_inter.view(b, -1, h * w).transpose(-1, -2).contiguous()  # (b, h*w, c)
        prob_mask = prob_mask.view(b, self.num_classes, -1)  # (b, classes, h*w)
        global_class_feature = torch.zeros((b, self.num_classes, d))

        for classes in range(self.num_classes):
            for i in range(b):
                global_class_feature[i, classes, :] = torch.sum(
                    feature_in_inter_[i, :, :] * nn.Softmax(dim=-1)(prob_mask[i, classes, :]).unsqueeze(1), dim=0)  # (1, 1,c)

        global_class_feature = global_class_feature.cuda()
        feature_inter, global_class_feature = self.inter_class_trans(feature_in_inter_, global_class_feature)  # (b, h*w, c)
        feature_inter = feature_inter.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # (m, b, h*w, c)
        feature_out = self.fuse_inter(torch.cat((self.up(feature_inter), feature_intra), dim=1))

        '''loss_eu 计算'''

        if self.training:
            BT = global_class_feature  # (B, Cl, C)
            A = BT.permute(0, 2, 1)  # (B, C, Cl)
            vecProd = torch.matmul(BT, A)  # (B, Cl, Cl)
            SqA = A ** 2
            sumSqA = torch.sum(SqA, dim=1).unsqueeze(dim=2).repeat(1, 1, self.num_classes)  # (B, Cl, Cl)
            Euclidean_dis = sumSqA * 2 - vecProd * 2
            Euclidean_dis = torch.sum(Euclidean_dis, dim=(1, 2))
            Euclidean_dis = torch.pow(Euclidean_dis, 1 / 2).sum()
            Euclidean_loss = 10.0 * torch.log((1e+2 / Euclidean_dis) + 1)

        if self.training:
            return feature_out, Euclidean_loss, loss_bce / self.num_classes, out_c
        else:
            return feature_out


# if __name__ == "__main__":
#     model = ResClassFormer(1, 9, True).cuda()
#     input = torch.zeros((1, 1, 224, 224)).cuda()
#     GT = torch.zeros((1, 224, 224)).cuda()
#     out = model(input, GT)
