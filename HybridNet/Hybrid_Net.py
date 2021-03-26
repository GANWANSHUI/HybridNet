from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

from .cspn import  Affinity_Propagate
from .submodel import disparityregression, disparityregression2
from .feature_extraction import  PSM_feature_extraction, Hybrid_Net_feature
from .aggregation import  Costvolume_process_PSM, Costvolume_process_PSM_DSM, \
Hybrid_Net_Aggregation_1, Hybrid_Net_Aggregation_2
from .self_supervised_loss import self_supervised_loss
from .cost import _build_redidual_cost_volume, _build_cost_volume, _build_volume_2d_psmnet


class Hybrid_Net(nn.Module):
    def __init__(self,  args):
        super(Hybrid_Net, self).__init__()

        self.model_types = args.model_types     #  Hybrid_Net_DSM  OR Hybrid_Net
        self.cost_volume = args.cost_volume
        self.maxdisp = args.maxdisp
        self.with_residual_cost = args.with_residual_cost
        self.residual_disparity_range = args.residual_disparity_range
        self.with_cspn = args.with_cspn

        self.activation_types1 = args.activation_types1
        self.activation_types2 = args.activation_types2

        self.conv_3d_types1 = args.conv_3d_types1
        self.conv_3d_types2 = args.conv_3d_types2
        self.CSPN_step = args.CSPN_step
        self.supervise_types = args.supervise_types
        self.disparity_mask =  args.disparity_mask
        self.denormalization = args.denormalization

        if self.supervise_types == "self_supervised" or self.supervise_types == "semi_supervised":
            self.self_supervised_loss = self_supervised_loss(denormalization =self.denormalization, disparity_mask = self.disparity_mask)

        self.feature_extraction = Hybrid_Net_feature(activation_types1=self.activation_types1)    # channel = 32   16    8

        if self.cost_volume == "Concat":

            self.Hybrid_Net_Aggregation_1 = Hybrid_Net_Aggregation_1(input_planes=64, planes=8,
                                                                     model_types=self.model_types,
                                                                     conv_3d_types1=self.conv_3d_types1,
                                                                     conv_3d_types2=self.conv_3d_types2,
                                                                     activation_types2=self.activation_types2
                                                                     )

            if self.with_residual_cost:
                self.Hybrid_Net_Aggregation_2 = Hybrid_Net_Aggregation_2(input_planes=32, planes=16,
                                                                         model_types=self.model_types,
                                                                         conv_3d_types1=self.conv_3d_types1,
                                                                         conv_3d_types2=self.conv_3d_types2,
                                                                         n_segment=self.residual_disparity_range*2-1,
                                                                         activation_types2=self.activation_types2)

        elif self.cost_volume == "Difference":
            self.Hybrid_Net_Aggregation_1 = Hybrid_Net_Aggregation_1(input_planes=32, planes=8,
                                                                     model_types=self.model_types,
                                                                     conv_3d_types1=self.conv_3d_types1,
                                                                     conv_3d_types2=self.conv_3d_types2,
                                                                     activation_types2=self.activation_types2)

            if self.with_residual_cost:
                self.Hybrid_Net_Aggregation_2 = Hybrid_Net_Aggregation_2(input_planes=16, planes=16,
                                                                         model_types=self.model_types,
                                                                         conv_3d_types1=self.conv_3d_types1,
                                                                         conv_3d_types2=self.conv_3d_types2,
                                                                         n_segment=self.residual_disparity_range*2-1,
                                                                         activation_types2=self.activation_types2)


        elif self.cost_volume == "Distance_based":

            self.Hybrid_Net_Aggregation_1 = Hybrid_Net_Aggregation_1(input_planes=1, planes=8,
                                                                     model_types=self.model_types,
                                                                     conv_3d_types1=self.conv_3d_types1,
                                                                     conv_3d_types2=self.conv_3d_types2,
                                                                     activation_types2=self.activation_types2)

            if self.with_residual_cost:
                self.Hybrid_Net_Aggregation_2 = Hybrid_Net_Aggregation_2(input_planes=16, planes=16,
                                                                         model_types=self.model_types,
                                                                         conv_3d_types1=self.conv_3d_types1,
                                                                         conv_3d_types2=self.conv_3d_types2,
                                                                         n_segment=self.residual_disparity_range*2-1,
                                                                         activation_types2=self.activation_types2)
        else:
            AssertionError("please define cost_volume aggregation types")



        # CSPN
        if self.with_cspn:
            cspn_config_default = {'step': self.CSPN_step, 'kernel': 3, 'norm_type': '8sum'}
            self.post_process_layer = [self._make_post_process_layer(cspn_config_default)]
            self.post_process_layer = nn.ModuleList(self.post_process_layer)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()




    def _make_post_process_layer(self, cspn_config=None):
        return Affinity_Propagate(cspn_config['step'],
                                  cspn_config['kernel'],
                                  norm_type=cspn_config['norm_type'])


    def forward(self, left, right, dispL):


        img_size = left.size()

        refimg_fea, L2, L_CSPN = self.feature_extraction(left, image_left = True)
        targetimg_fea, R2 = self.feature_extraction(right, image_left = False)


        # build cost volume
        cost = _build_cost_volume(cost_volume_type=self.cost_volume, refimg_fea = refimg_fea, targetimg_fea = targetimg_fea, maxdisp = self.maxdisp)
        cost_size = cost.size()

        # reshape for DSM
        if self.model_types == "Hybrid_Net_DSM":

            cost = cost.permute(0, 2, 1, 3, 4).contiguous()   # (b, c, maxdisp, h, w)
            cost = cost.view(-1, cost_size[1], cost_size[3], cost_size[4])
            cost = cost.contiguous()

        pred = []

        # Aggregation
        if self.training:

            pred1, pred2, pred3 = self.Hybrid_Net_Aggregation_1(cost, img_size , cost_size)
            pred.append(pred1)
            pred.append(pred2)
            pred.append(pred3)

        else:
            pred3 = self.Hybrid_Net_Aggregation_1(cost, img_size, cost_size)
            pred.append(pred3)


        # for residual cost volume
        if self.with_residual_cost:

            wflow = F.upsample(pred[-1], (L2.size(2), L2.size(3)), mode='bilinear') * L2.size(2) / img_size[2]

            cost_residual = _build_redidual_cost_volume(cost_volume_type=self.cost_volume,L2 = L2, R2 = R2, wflow =wflow, maxdisp = self.residual_disparity_range)

            if self.model_types == "Hybrid_Net_DSM":
                # for 2D-TSM
                cost_residual_size = cost_residual.size()
                cost_residual = cost_residual.permute(0, 2, 1, 3, 4).contiguous()
                cost_residual = cost_residual.view(-1, cost_residual_size[1], cost_residual_size[3], cost_residual_size[4])
                cost_residual = cost_residual.contiguous()

                #print("cost_residual size:", cost_residual.shape)

                cost_residual = self.Hybrid_Net_Aggregation_2(cost_residual).squeeze(1)

                cost_residual = cost_residual.view(cost_residual_size[0], 1, cost_residual_size[2], cost_residual_size[3], cost_residual_size[4])
                cost_residual =cost_residual.squeeze(1)


            else:

                cost_residual = self.Hybrid_Net_Aggregation_2(cost_residual).squeeze(1)


            pred_low_res = disparityregression2(-self.residual_disparity_range +1, self.residual_disparity_range, stride=1)(F.softmax(cost_residual, dim=1))
            pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)

            pred4 = pred[-1] + F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')

            pred.append(pred4)

        # CSPN refinement
        pred5 = self.post_process_layer[0](L_CSPN, pred[-1])
        pred.append(pred5)

        # output

        if self.supervise_types == "self_supervised" or self.supervise_types == "semi_supervised":


            self_supervised_loss = self.self_supervised_loss(pred, left, right, dispL)

        else:


            self_supervised_loss = []

        return pred, self_supervised_loss



class PSMNet(nn.Module):
    def __init__(self, args):
        super(PSMNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.cost_volume =  args.cost_volume
        self.conv_3d_types1 = args.conv_3d_types1   # only for "normal" and "separate_only"

        self.feature_extraction = PSM_feature_extraction()     # parameters:3.34 M

        self.Costvolume_process1 = Costvolume_process_PSM(conv_3d_types1 = self.conv_3d_types1)

        self.supervise_types = args.supervise_types

        if self.supervise_types == "self_supervised":
            self.self_supervised_loss = self_supervised_loss()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        #time_start1 = time.perf_counter()

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)


        cost = _build_volume_2d_psmnet(refimg_fea, targetimg_fea,maxdisp =  self.maxdisp)

        #time_start2 = time.perf_counter()
        cost1, cost2, cost3 = self.Costvolume_process1(cost)

        #Aggregation_time = time.perf_counter() - time_start2

        pred = []


        if self.training:

            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
            pred.append(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
            pred.append(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')   # 有先上采样再softmax， 也有先softmax,后深度图上采样
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)


    #For your information: This formulation 'softmax(c)' learned "similarity"
    #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
    #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.


        pred3 = disparityregression(self.maxdisp)(pred3)
        pred.append(pred3)

        pred = [torch.unsqueeze(pred, 1) for pred in pred]
        if self.supervise_types == "self_supervised":

            self_supervised_loss = self.self_supervised_loss(pred, left, right)

        else:
            self_supervised_loss = []


        return pred, self_supervised_loss



class PSMNet_DSM(nn.Module):
    def __init__(self, args):
        super(PSMNet_DSM, self).__init__()
        self.maxdisp = args.maxdisp
        self.cost_volume =  args.cost_volume

        self.conv_3d_types1 = args.conv_3d_types1   # # only for "TemporalShift_2D" and "ONLY_2D"
        self.conv_3d_types2 = args.conv_3d_types2

        self.feature_extraction = PSM_feature_extraction()     # parameters:3.34 M

        self.Costvolume_process1 = Costvolume_process_PSM_DSM(conv_3d_types1 = self.conv_3d_types1,conv_3d_types2 = self.conv_3d_types2 )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        #start_time = time.time()
        img_size = left.size()
        print("img_size:", img_size)
        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)


        cost = _build_volume_2d_psmnet(refimg_fea, targetimg_fea, maxdisp =  self.maxdisp)
        #print("cost size:", cost.shape)

        cost_size = cost.size()
        # for 2D-TSM

        cost = cost.permute(0, 2, 1, 3, 4).contiguous()

        cost = cost.view(-1, cost_size[1], cost_size[3], cost_size[4])
        #print("cost size2:", cost.size())

        cost = cost.contiguous()




        cost1, cost2, cost3 = self.Costvolume_process1(cost, img_size, cost_size)

        pred = []
        if self.training:

            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
            pred.append(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
            pred.append(pred2)


        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')   # 有先上采样再softmax， 也有先softmax,后深度图上采样
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)


        #For your information: This formulation 'softmax(c)' learned "similarity"
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.


        pred3 = disparityregression(self.maxdisp)(pred3)
        pred.append(pred3)




        if self.training:
                # print("len pred:", pred[[0].shape])
                return pred
        else:
            return [pred[-1]]
