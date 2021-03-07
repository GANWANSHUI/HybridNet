from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

from .submodel import convbn_3d,conv_3d, disparityregression, hourglass_PSMNet, hourglass_Hybird_Net, \
hourglass_DSM_PSMNet, hourglass_DSM_Hybird_Net, activation_function


class Costvolume_process_PSM(nn.Module):
    def __init__(self,  input_planes = 64, maxdisp = 192, planes = 16, conv_3d_types1 = "normal"):
        super(Costvolume_process_PSM, self).__init__()
        self.maxdisp = maxdisp

        #self.conv_3d_types = conv_3d_types1

        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1))

        self.dres2 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1 )

        self.dres3 = hourglass_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1)

        self.dres4 = hourglass_PSMNet(planes*2 , conv_3d_types1 = conv_3d_types1)


        if conv_3d_types1 =="normal":

            self.classif1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))

            self.classif2 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))

            self.classif3 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))



        else:
            self.classif1 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          nn.ReLU(inplace=True),



                                          nn.Conv3d(planes * 2, 1, kernel_size=(1, 3, 3), stride=1,
                                                    padding=(0, 1, 1), bias=False),
                                          # nn.BatchNorm3d(out_planes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=1,
                                                    padding=(1, 0, 0), bias=False)


                                        )

            self.classif2 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(planes * 2, 1, kernel_size=(1, 3, 3), stride=1,
                                                    padding=(0, 1, 1), bias=False),
                                          # nn.BatchNorm3d(out_planes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=1,
                                                    padding=(1, 0, 0), bias=False))

            self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(planes * 2, 1, kernel_size=(1, 3, 3), stride=1,
                                                    padding=(0, 1, 1), bias=False),
                                          # nn.BatchNorm3d(out_planes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=1,
                                                    padding=(1, 0, 0), bias=False))



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




    def forward(self, cost):



        cost0 = self.dres0(cost)
        #print("cost0 size:", cost0.size())

        # [3, 32, 6, 8, 15] [3, 32, 48, 64, 128])

        cost0 = self.dres1(cost0) + cost0

        # print("cost0 size:", cost0.shape)

        out1, pre1, post1 = self.dres2(cost0, None, None)

        # print("out1 size", out1.size())
        # print("cost0 size:", cost0.size())
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        return cost1, cost2, cost3




class Costvolume_process_PSM_DSM(nn.Module):
    def __init__(self,  input_planes = 64, maxdisp = 192, planes = 16, conv_3d_types1 = "DSM", conv_3d_types2 = "2D"):
        super(Costvolume_process_PSM_DSM, self).__init__()
        self.maxdisp = maxdisp

        # self.conv_3d_types1 = conv_3d_types1
        # self.conv_3d_types2 = conv_3d_types2


        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types2),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types =  conv_3d_types2))




        self.dres2 = hourglass_DSM_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1 , conv_3d_types2 = conv_3d_types2)

        self.dres3 = hourglass_DSM_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1 , conv_3d_types2 = conv_3d_types2)

        self.dres4 = hourglass_DSM_PSMNet(planes*2, conv_3d_types1 = conv_3d_types1 , conv_3d_types2 = conv_3d_types2)

        self.classif1 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      nn.ReLU(inplace=True),
                                      conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1,
                                              conv_3d_types=conv_3d_types2))

        self.classif2 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      nn.ReLU(inplace=True),
                                      conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1,
                                              conv_3d_types=conv_3d_types2))

        self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                      nn.ReLU(inplace=True),
                                      conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1,
                                              conv_3d_types=conv_3d_types2))



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




    def forward(self, cost, img_size, cost_size = 1):

        #print("cost0 size:", cost.size())
        cost0 = self.dres0(cost)
        #print("cost0 size:", cost0.size())

        # [3, 32, 6, 8, 15] [3, 32, 48, 64, 128])

        cost0 = self.dres1(cost0) + cost0

        # print("cost0 size:", cost0.shape)

        out1, pre1, post1 = self.dres2(cost0, None, None)

        # print("out1 size", out1.size())
        #print("cost0 size:", cost0.size())
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = cost1.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
        cost2 = cost2.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
        cost3 = cost3.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])

        return cost1, cost2, cost3



#
# class Hybrid_Net_Aggregation_1(nn.Module):
#     def __init__(self,  input_planes = 64,  planes = 16, kernel_size = 3, maxdisp = 192, conv_3d_types1 = "normal", conv_3d_types2 = "normal", model_types = "Hybrid_Net", activation_types2 = "ELU"):
#         super(Hybrid_Net_Aggregation_1, self).__init__()
#         self.maxdisp = maxdisp
#
#         self.model_types = model_types
#
#
#         self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types1),
#                                      activation_function(types = activation_types2),
#                                      convbn_3d(planes*2, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types2),
#                                      activation_function(types = activation_types2))
#
#         self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes*2, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types2))
#
#
#         if self.model_types == "Hybrid_Net":
#
#             self.dres2 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)
#
#             self.dres3 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)
#
#             self.dres4 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)
#
#
#             self.classif1 = nn.Sequential(convbn_3d(planes * 2, planes * 2, kernel_size, 1, 1, conv_3d_types=conv_3d_types1),
#                                           activation_function(types=activation_types2),
#                                           conv_3d(planes * 2, 1, kernel_size=kernel_size, pad=1, stride=1, conv_3d_types=conv_3d_types1))
#
#             self.classif2 = nn.Sequential(convbn_3d(planes * 2, planes * 2, kernel_size, 1, 1, conv_3d_types=conv_3d_types1),
#                                           activation_function(types=activation_types2),
#                                           conv_3d(planes * 2, 1, kernel_size=kernel_size, pad=1, stride=1, conv_3d_types=conv_3d_types1))
#
#             self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, kernel_size, 1, 1, conv_3d_types=conv_3d_types1),
#                                           activation_function(types=activation_types2),
#                                           conv_3d(planes * 2, 1, kernel_size=kernel_size, pad=1, stride=1, conv_3d_types=conv_3d_types1))
#
#
#         elif self.model_types == "Hybrid_Net_DSM":
#
#             self.dres2 = hourglass_DSM_Hybird_Net(planes*2, conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)
#
#             self.dres3 = hourglass_DSM_Hybird_Net(planes*2,conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)
#
#             self.dres4 = hourglass_DSM_Hybird_Net(planes*2, conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)
#
#
#             self.classif1 = nn.Sequential(convbn_3d(planes*2, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types1),
#                                           activation_function(types = activation_types2),
#                                           nn.Conv2d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))
#
#             self.classif2 = nn.Sequential(convbn_3d(planes*2, planes*2, kernel_size, 1, 1,conv_3d_types =  conv_3d_types1),
#                                           activation_function(types = activation_types2),
#                                           nn.Conv2d(planes*2, 1, kernel_size=kernel_size, padding=1, stride=1,bias=False))
#
#             self.classif3 = nn.Sequential(convbn_3d(planes*2, planes*2, kernel_size, 1, 1, conv_3d_types = conv_3d_types1),
#                                           activation_function(types = activation_types2),
#                                           nn.Conv2d(planes*2, 1, kernel_size=kernel_size, padding=1, stride=1,bias=False))
#
#
#
#         else:
#
#             AssertionError
#
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#
#
#
#     def forward(self, cost, img_size, cost_size = 1):
#
#
#         cost0 = self.dres0(cost)
#
#
#         cost0 = self.dres1(cost0) + cost0
#
#
#         out1, pre1, post1 = self.dres2(cost0, None, None)
#
#
#         out1 = out1+cost0
#
#         out2, pre2, post2 = self.dres3(out1, pre1, post1)
#         out2 = out2+cost0
#
#         out3, pre3, post3 = self.dres4(out2, pre1, post2)
#         out3 = out3+cost0
#
#         cost1 = self.classif1(out1)
#         cost2 = self.classif2(out2) + cost1
#         cost3 = self.classif3(out3) + cost2
#
#
#         if self.model_types == "Hybrid_Net_DSM":
#
#             cost3 = cost3.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
#
#
#
#         cost3 = F.upsample(cost3, [self.maxdisp, img_size[2], img_size[3]],
#                            mode='trilinear')  # 有先上采样再softmax， 也有先softmax,后深度图上采样
#         cost3 = torch.squeeze(cost3, 1)
#
#         pred3 = F.softmax(cost3, dim=1)
#         pred3 = disparityregression(self.maxdisp)(pred3)
#
#
#
#
#         if self.training:
#
#             if self.model_types == "Hybrid_Net_DSM":
#
#                 cost1 = cost1.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
#                 cost2 = cost2.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
#
#             cost1 = F.upsample(cost1, [self.maxdisp,img_size[2],img_size[3]], mode='trilinear')
#             cost2 = F.upsample(cost2, [self.maxdisp,img_size[2],img_size[3]], mode='trilinear')
#
#
#             cost1 = torch.squeeze(cost1,1)
#             pred1 = F.softmax(cost1,dim=1)
#             pred1 = disparityregression(self.maxdisp)(pred1)
#
#             cost2 = torch.squeeze(cost2,1)
#             pred2 = F.softmax(cost2,dim=1)
#             pred2 = disparityregression(self.maxdisp)(pred2)
#
#
#
#         #single_inference_time = time.perf_counter() - time_start
#
#         #print("single_inference_time:", single_inference_time)
#
#
#             return  torch.unsqueeze(pred1,1), torch.unsqueeze(pred2,1), torch.unsqueeze(pred3,1)
#
#         else:
#
#             return torch.unsqueeze(pred3,1)
#
#
#
#
# class Hybrid_Net_Aggregation_2(nn.Module):  # base on PSMNet basic
#     def __init__(self, input_planes=16, planes=16, kernel_size = 3, maxdisp=192,  model_types = "Hybrid_Net", conv_3d_types1 = "normal", conv_3d_types2 = "normal", n_segment = 5, activation_types2 = "ELU"):
#         super(Hybrid_Net_Aggregation_2, self).__init__()
#         self.maxdisp = maxdisp
#
#         self.model_types = model_types
#
#
#         self.dres0 = nn.Sequential(convbn_3d(input_planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2),
#                                    activation_function(types = activation_types2))
#
#         self.dres1 = nn.Sequential(convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))
#
#         self.dres2 = nn.Sequential(convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment,conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment,conv_3d_types =  conv_3d_types2))
#
#         self.dres3 = nn.Sequential(convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))
#
#         self.dres4 = nn.Sequential(convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
#                                    activation_function(types = activation_types2),
#                                    convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))
#
#         if self.model_types == "Hybrid_Net":
#             self.classify = nn.Sequential(convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
#                                       activation_function(types = activation_types2),
#                                       conv_3d(planes , 1, kernel_size=kernel_size, pad=1, stride=1, conv_3d_types=conv_3d_types1))
#
#
#         elif self.model_types == "Hybrid_Net_DSM":
#             self.classify = nn.Sequential(
#                 convbn_3d(planes, planes, kernel_size, 1, 1, n_segment=n_segment, conv_3d_types=conv_3d_types1),
#                 activation_function(types = activation_types2),
#                 nn.Conv2d(planes, 1, kernel_size=kernel_size, padding=1, stride=1, bias=False))
#
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def forward(self, residual_cost):
#
#         #print("residual_cost size:", residual_cost.shape)
#
#         cost0 = self.dres0(residual_cost)
#         cost0 = self.dres1(cost0) + cost0
#         cost0 = self.dres2(cost0) + cost0
#         cost0 = self.dres3(cost0) + cost0
#         cost0 = self.dres4(cost0) + cost0
#
#         cost = self.classify(cost0)
#
#         return cost
#




class Hybrid_Net_Aggregation_1(nn.Module):
    def __init__(self,  input_planes = 64, maxdisp = 192, planes = 16, conv_3d_types1 = "normal", conv_3d_types2 = "normal", model_types = "Hybrid_Net", activation_types2 = "ELU"):
        super(Hybrid_Net_Aggregation_1, self).__init__()
        self.maxdisp = maxdisp

        self.model_types = model_types


        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types1),
                                     activation_function(types = activation_types2),
                                     convbn_3d(planes*2, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types2),
                                     activation_function(types = activation_types2))

        self.dres1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes*2, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types2))

        if self.model_types == "Hybrid_Net":

            self.dres2 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)

            self.dres3 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)

            self.dres4 = hourglass_Hybird_Net(planes * 2, conv_3d_types1=conv_3d_types1)


            self.classif1 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          activation_function(types=activation_types2),
                                          conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1, conv_3d_types=conv_3d_types1))

            self.classif2 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          activation_function(types=activation_types2),
                                          conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1, conv_3d_types=conv_3d_types1))

            self.classif3 = nn.Sequential(convbn_3d(planes * 2, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types1),
                                          activation_function(types=activation_types2),
                                          conv_3d(planes * 2, 1, kernel_size=3, pad=1, stride=1, conv_3d_types=conv_3d_types1))


        elif self.model_types == "Hybrid_Net_DSM":

            self.dres2 = hourglass_DSM_Hybird_Net(planes*2, conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)

            self.dres3 = hourglass_DSM_Hybird_Net(planes*2,conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)

            self.dres4 = hourglass_DSM_Hybird_Net(planes*2, conv_3d_types1 = conv_3d_types1, conv_3d_types2 = conv_3d_types2)


            self.classif1 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types1),
                                          activation_function(types = activation_types2),
                                          nn.Conv2d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))

            self.classif2 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1,conv_3d_types =  conv_3d_types1),
                                          activation_function(types = activation_types2),
                                          nn.Conv2d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))

            self.classif3 = nn.Sequential(convbn_3d(planes*2, planes*2, 3, 1, 1, conv_3d_types = conv_3d_types1),
                                          activation_function(types = activation_types2),
                                          nn.Conv2d(planes*2, 1, kernel_size=3, padding=1, stride=1,bias=False))



        else:

            AssertionError



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




    def forward(self, cost, img_size, cost_size = 1):


        cost0 = self.dres0(cost)


        cost0 = self.dres1(cost0) + cost0


        out1, pre1, post1 = self.dres2(cost0, None, None)


        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2


        if self.model_types == "Hybrid_Net_DSM":

            cost3 = cost3.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])



        cost3 = F.upsample(cost3, [self.maxdisp, img_size[2], img_size[3]],
                           mode='trilinear')  # 有先上采样再softmax， 也有先softmax,后深度图上采样
        cost3 = torch.squeeze(cost3, 1)

        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)




        if self.training:

            if self.model_types == "Hybrid_Net_DSM":

                cost1 = cost1.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])
                cost2 = cost2.view(cost_size[0], 1, cost_size[2], cost_size[3], cost_size[4])

            cost1 = F.upsample(cost1, [self.maxdisp,img_size[2],img_size[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,img_size[2],img_size[3]], mode='trilinear')


            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)



        #single_inference_time = time.perf_counter() - time_start

        #print("single_inference_time:", single_inference_time)


            return  torch.unsqueeze(pred1,1), torch.unsqueeze(pred2,1), torch.unsqueeze(pred3,1)

        else:

            return torch.unsqueeze(pred3,1)




class Hybrid_Net_Aggregation_2(nn.Module):  # base on PSMNet basic
    def __init__(self, input_planes=16, planes=16, maxdisp=192,  model_types = "Hybrid_Net", conv_3d_types1 = "normal", conv_3d_types2 = "normal", n_segment = 5, activation_types2 = "ELU"):
        super(Hybrid_Net_Aggregation_2, self).__init__()
        self.maxdisp = maxdisp

        self.model_types = model_types


        self.dres0 = nn.Sequential(convbn_3d(input_planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2),
                                   activation_function(types = activation_types2))

        self.dres1 = nn.Sequential(convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))

        self.dres2 = nn.Sequential(convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment,conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment,conv_3d_types =  conv_3d_types2))

        self.dres3 = nn.Sequential(convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))

        self.dres4 = nn.Sequential(convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
                                   activation_function(types = activation_types2),
                                   convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types2))

        if self.model_types == "Hybrid_Net":
            self.classify = nn.Sequential(convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types =  conv_3d_types1),
                                      activation_function(types = activation_types2),
                                      conv_3d(planes , 1, kernel_size=3, pad=1, stride=1, conv_3d_types=conv_3d_types1))


        elif self.model_types == "Hybrid_Net_DSM":
            self.classify = nn.Sequential(
                convbn_3d(planes, planes, 3, 1, 1, n_segment=n_segment, conv_3d_types=conv_3d_types1),
                activation_function(types = activation_types2),
                nn.Conv2d(planes, 1, kernel_size=3, padding=1, stride=1, bias=False))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, residual_cost):

        #print("residual_cost size:", residual_cost.shape)

        cost0 = self.dres0(residual_cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)

        return cost