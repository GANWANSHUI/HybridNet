from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


###---------------------------------------Depth Shift Module --------------------------------------------------------------###

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):

        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None



def temporal_pool(x, n_segment):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
    x = F.adaptive_avg_pool3d(x, (n_segment//2, h//2, w//2))
    x = x.transpose(1, 2).contiguous().view(nt // 2, c, h//2, w//2)
    return x


def temporal_upsample(x, n_segment):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
    x = nn.functional.interpolate(x, size = (n_segment*2, h*2, w*2), scale_factor=None, mode='trilinear', align_corners=False)
    x = x.transpose(1, 2).contiguous().view(nt*2, c, h*2, w*2)

    return x



class DSM(nn.Module):
    def __init__(self, in_planes, out_planes,stride, n_segment, n_div=4, inplace=True, Transpose = False):
        super(DSM, self).__init__()

        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.Transpose = Transpose

        # if inplace:
        #     print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))


        self.con2d_3x3 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )


    def forward(self, x):

        # 2D + DSM
        if self.stride == 1:
            x_shift = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)

        elif self.stride == 2 and self.Transpose == False:
            x = temporal_pool(x, self.n_segment)
            x_shift = self.shift(x, self.n_segment // 2, fold_div=self.fold_div, inplace=self.inplace)

        else:
            x = temporal_upsample(x, self.n_segment)
            x_shift = self.shift(x, self.n_segment * 2, fold_div=self.fold_div, inplace=self.inplace)

        x = x_shift +x


        return self.con2d_3x3(x)


    @staticmethod
    def shift(x, n_segment, fold_div=4, inplace=False):

        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:

            """
            Note that the naive implementation involves large data copying and increases memory consumption during training.
             It is suggested to use the in-place version of TSM to improve speed
            """
            out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)



class ONLY_2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride, n_segment, n_div=4, inplace=True, Transpose=False):
        super(ONLY_2D, self).__init__()

        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.Transpose = Transpose


        self.con2d_3x3 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )


    def forward(self, x):

        if self.stride == 1:

            return self.con2d_3x3(x)

        elif self.stride == 2 and self.Transpose == False:

            x = temporal_pool(x, self.n_segment)

            return self.con2d_3x3(x)

        else:
            x = temporal_upsample(x, self.n_segment)

            return self.con2d_3x3(x)




def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, n_segment=48, Transpose=False, conv_3d_types="3D"):

    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))


    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_planes))



    elif conv_3d_types == "DSM":

        return nn.Sequential(
            DSM(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment,
                             Transpose=Transpose))

    elif conv_3d_types == "2D":

        return nn.Sequential(
            ONLY_2D(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment,
                    Transpose=Transpose))


    else:

        AssertionError("please define conv_3d_types")





def conv_3d(in_planes, out_planes, kernel_size, stride, pad, n_segment=48, Transpose=False, conv_3d_types="3D"):


    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)
            )


    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),

            )

    elif conv_3d_types == "DSM":

        return nn.Sequential(
            DSM(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment,
                             Transpose=Transpose))

    elif conv_3d_types == "2D":

        return nn.Sequential(
            ONLY_2D(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment,
                    Transpose=Transpose))


    else:

        AssertionError("please define conv_3d_types")




def convTranspose3d(in_planes, out_planes, kernel_size, stride, padding=1, n_segment=24, Transpose=True, conv_3d_types="3D"):

    if conv_3d_types == '3D':
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size, padding = padding, output_padding=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))


    elif conv_3d_types == "DSM":

        return nn.Sequential(
            DSM(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment, Transpose=Transpose))

    elif conv_3d_types == "2D":

        return nn.Sequential(
            ONLY_2D(in_planes=in_planes, out_planes=out_planes, stride=stride, n_segment=n_segment, Transpose=Transpose,))



    else:
        AssertionError("please define conv_3d_types")




def activation_function(types = "ELU"):     # ELU or Relu


    if types == "ELU":

        return nn.Sequential(nn.ELU(inplace=True))

    elif types == "Mish":

        return nn.Sequential(Mish())

    elif types == "Relu":

        return nn.Sequential(nn.ReLU(inplace=True))

    else:

        AssertionError("please define the activate function types")




class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * torch.tanh(F.softplus(input))




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out




# PSMNet

class hourglass_DSM_PSMNet(nn.Module):
    def __init__(self, inplanes, conv_3d_types1, conv_3d_types2):
        super(hourglass_DSM_PSMNet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, n_segment=48, conv_3d_types = conv_3d_types1),

                                   nn.ReLU(inplace=True)
                                   )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, n_segment=24, conv_3d_types = conv_3d_types2)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1,n_segment=24, conv_3d_types = conv_3d_types1),

                                   nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, n_segment=12, conv_3d_types = conv_3d_types2),

                                   nn.ReLU(inplace=True)
                                   )

        self.conv5 = nn.Sequential(
            convTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, n_segment=12,
                            conv_3d_types=conv_3d_types1),

        )  # +conv2

        self.conv6 = nn.Sequential(
            convTranspose3d(inplanes * 2, inplanes, kernel_size=3, stride=2, n_segment=24,
                            conv_3d_types=conv_3d_types1),

        )  # +x


    def forward(self, x, presqu, postsqu):

            out = self.conv1(x)  # in:1/4 out:1/8
            pre = self.conv2(out)  # in:1/8 out:1/8
            if postsqu is not None:
                pre = F.relu(pre + postsqu, inplace=True)
            else:
                pre = F.relu(pre, inplace=True)

            out = self.conv3(pre)  # in:1/8 out:1/16
            #print("out3 :", out.shape)
            out = self.conv4(out)  # in:1/16 out:1/16

            if presqu is not None:
                post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
            else:
                post = F.relu(self.conv5(out) + pre, inplace=True)

            out = self.conv6(post)  # in:1/8 out:1/4

            return out, pre, post




class hourglass_PSMNet(nn.Module):
    def __init__(self, inplanes, conv_3d_types1):
        super(hourglass_PSMNet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   #activation_function(),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   #activation_function(),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1),
                                   #activation_function(),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post



# Hybird_Net

class hourglass_DSM_Hybird_Net(nn.Module):
    def __init__(self, inplanes, conv_3d_types1, conv_3d_types2, activation_types2 = "ELU"):
        super(hourglass_DSM_Hybird_Net, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, n_segment=48, conv_3d_types = conv_3d_types1),
                                   activation_function(types = activation_types2),

                                   )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, n_segment=24, conv_3d_types = conv_3d_types2)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, n_segment=24, conv_3d_types = conv_3d_types1),
                                   activation_function(types = activation_types2),

                                   )

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, n_segment=12, conv_3d_types = conv_3d_types2),
                                   activation_function(types = activation_types2),

                                   )


        self.conv5 = nn.Sequential(
            convTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, n_segment=12, conv_3d_types = conv_3d_types1),
            activation_function(types=activation_types2),
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, n_segment=24, conv_3d_types=conv_3d_types2)
        )  # +conv2

        self.conv6 = nn.Sequential(
            convTranspose3d(inplanes * 2, inplanes, kernel_size=3, stride=2, n_segment=24, conv_3d_types = conv_3d_types1),
            activation_function(types=activation_types2),
            convbn_3d(inplanes , inplanes , kernel_size=3, stride=1, pad=1, n_segment=48, conv_3d_types=conv_3d_types2)
        )  # +x


        self.activatefunction = nn.Sequential(

            activation_function(types=activation_types2)

        )


    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8

        # print("out size:", out.size()) # [24, 64, 32, 64])

        pre = self.conv2(out)  # in:1/8 out:1/8
        # print("pre size:", pre.size())  # [24, 64, 32, 64])

        if postsqu is not None:
            #pre = F.relu(pre + postsqu, inplace=True)
            pre = self.activatefunction(pre + postsqu)
        else:
            #pre = F.relu(pre, inplace=True)
            pre = self.activatefunction(pre)
        # print("pre size:", pre.size())  # [24, 64, 32, 64])

        out = self.conv3(pre)  # in:1/8 out:1/16
        # print('out size:', out.size()) # [12, 64, 16, 32])

        out = self.conv4(out)  # in:1/16 out:1/16

        # print("out size:", out.size()) # out size: torch.Size([12, 64, 16, 32])

        if presqu is not None:
            #post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
            post = self.activatefunction(self.conv5(out) + presqu)

        else:
            #post = F.relu(self.conv5(out) + pre, inplace=True)
            post = self.activatefunction(self.conv5(out) + pre)


        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post





class hourglass_Hybird_Net(nn.Module):
    def __init__(self, inplanes, conv_3d_types1, activation_types2="ELU"):
        super(hourglass_Hybird_Net, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activation_function(types = activation_types2),

                                   )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1, conv_3d_types = conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   #nn.ReLU(inplace=True)
                                   )

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1),
                                   activation_function(types = activation_types2),

                                   )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            activation_function(types=activation_types2),
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            activation_function(types=activation_types2),
            convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1, conv_3d_types = conv_3d_types1))  # +x

        self.activatefunction = nn.Sequential(

            activation_function(types=activation_types2)

        )

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            # pre = F.relu(pre + postsqu, inplace=True)
            pre = self.activatefunction(pre + postsqu)
        else:
            # pre = F.relu(pre, inplace=True)
            pre = self.activatefunction(pre)



        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            # post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
            post = self.activatefunction(self.conv5(out) + presqu)

        else:
            # post = F.relu(self.conv5(out) + pre, inplace=True)
            post = self.activatefunction(self.conv5(out) + pre)



        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post






class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out




class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out
