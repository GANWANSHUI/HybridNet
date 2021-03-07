from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
#from torch.autograd import Variable
import torch.nn.functional as F

from .submodel import convbn, BasicBlock, activation_function


class PSM_feature_extraction(nn.Module):
    def __init__(self):
        super(PSM_feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature




class PSM_UNet_S_2_feature(nn.Module):
    def __init__(self):
        super(PSM_UNet_S_2_feature, self).__init__()
        self.inplanes = 16
        self.inplanes2 = 32
        self.inplanes4 = 64
        self.inplanes10 = 160

        self.firstconv = nn.Sequential(convbn(3, self.inplanes, 3, 2, 1, 1),

                                       activation_function(),
                                       #nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),

                                       activation_function(),
                                       #nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),
                                       activation_function(),
                                       #nn.ReLU(inplace=True)
                                       )

        self.layer1 = self._make_layer(BasicBlock, self.inplanes, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, self.inplanes2, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, self.inplanes4, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, self.inplanes4, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(),

                                     #nn.ReLU(inplace=True)
                                     )

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(),
                                     #nn.ReLU(inplace=True)
                                     )

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(),
                                     #nn.ReLU(inplace=True)
                                     )

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(),
                                     #nn.ReLU(inplace=True)
                                     )

        self.lastconv = nn.Sequential(convbn(320 // 2, 128 // 2, 3, 1, 1, 1),
                                      activation_function(),
                                      #nn.ReLU(inplace=True),
                                      nn.Conv2d(128 // 2, 32, kernel_size=1, padding=0, stride=1, bias=False))


        self.up_sample_1 =nn.Sequential( nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1, bias=False))
        self.up_sample_2 = nn.Sequential( nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1, bias=False))



        self.output_feature_2 = nn.Sequential(convbn(48, 16, 3, 1, 1, 1),
                                              activation_function(),
                                              #nn.ReLU(inplace=True),
                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(),

                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(),

                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(),

                                              #nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, kernel_size=1, padding=0, stride=1, bias=False),
                                              #nn.ReLU(inplace=True)

                                        )



        self.output_CSPN =  nn.Sequential(



                                convbn(16 , 16, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(),

                                convbn(16, 16, 3, 1, 1, 1),
                                # nn.ReLU(inplace=True),
                                activation_function(),

                                convbn(16, 16, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(),
                                convbn(16, 8, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(),

                                nn.Conv2d(8, 8, 1, 1, 0, bias=False),

                                    )



    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x, image_left):

        #feature_size = x.size()

        output = self.firstconv(x)


        output_residual = output
        #print("output size:", output.shape)
        output = self.layer1(output)


        #print("output size:", output.shape)
        output_raw = self.layer2(output)
        #print("output size:", output_raw.shape)


        output = self.layer3(output_raw)


        #print("output_residual size:", output_residual.shape)


        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)


        output_feature_1 = self.lastconv(output_feature)
        #print("output_feature_1 size:", output_feature_1.shape)


        if image_left:

            output_feature_2 =self.up_sample_1(output_feature_1)
            #print("output_feature_2 size:", output_feature_2.shape)

            output_feature_2 = torch.cat((output_feature_2, output_residual), 1 )
            #print("output_feature_2 size:", output_feature_2.shape)

            output_feature_2 = self.output_feature_2(output_feature_2)




            #output_CSPN = F.upsample(output_feature_2, (output_skip.size()[2]*4, output_skip.size()[3]*4), mode='bilinear')
            output_CSPN = self.up_sample_2(output_feature_2)


            output_CSPN = self.output_CSPN(output_CSPN)



            return output_feature_1, output_feature_2, output_CSPN

        else:
            return output_feature_1




class Hybrid_Net_feature(nn.Module):
    def __init__(self, activation_types1 = "ELU"):
        super(Hybrid_Net_feature, self).__init__()
        self.inplanes = 16
        self.inplanes2 = 32
        self.inplanes4 = 64
        self.inplanes10 = 160

        self.firstconv = nn.Sequential(convbn(3, self.inplanes, 3, 2, 1, 1),

                                       activation_function(types = activation_types1),
                                       #nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),

                                       activation_function(types = activation_types1),
                                       #nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),
                                       activation_function(types = activation_types1),
                                       #nn.ReLU(inplace=True)
                                       )

        self.layer1 = self._make_layer(BasicBlock, self.inplanes, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, self.inplanes2, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, self.inplanes4, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, self.inplanes4, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(types = activation_types1),

                                     #nn.ReLU(inplace=True)
                                     )

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(types = activation_types1),
                                     #nn.ReLU(inplace=True)
                                     )

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(types = activation_types1),
                                     #nn.ReLU(inplace=True)
                                     )

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128 // 2, 32 // 2, 1, 1, 0, 1),
                                     activation_function(types = activation_types1),
                                     #nn.ReLU(inplace=True)
                                     )

        self.lastconv = nn.Sequential(convbn(320 // 2, 128 // 2, 3, 1, 1, 1),
                                      activation_function(types = activation_types1),
                                      #nn.ReLU(inplace=True),
                                      nn.Conv2d(128 // 2, 32, kernel_size=1, padding=0, stride=1, bias=False))




        self.up_sample_1 =nn.Sequential( nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1, bias=False))
        self.up_sample_2 = nn.Sequential( nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1, bias=False))



        self.output_feature_2 = nn.Sequential(convbn(48, 16, 3, 1, 1, 1),
                                              activation_function(types = activation_types1),
                                              #nn.ReLU(inplace=True),
                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(types = activation_types1),

                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(),

                                              convbn(16, 16, 3, 1, 1, 1),
                                              activation_function(types = activation_types1),

                                              #nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, kernel_size=1, padding=0, stride=1, bias=False),
                                              #nn.ReLU(inplace=True)

                                        )



        self.output_CSPN =  nn.Sequential(

                                convbn(16 , 16, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(types = activation_types1),

                                convbn(16, 16, 3, 1, 1, 1),
                                # nn.ReLU(inplace=True),
                                activation_function(types = activation_types1),

                                convbn(16, 16, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(types = activation_types1),
                                convbn(16, 8, 3, 1, 1, 1),
                                #nn.ReLU(inplace=True),
                                activation_function(types = activation_types1),

                                nn.Conv2d(8, 8, 1, 1, 0, bias=False),

                                )



    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x, image_left):

        #feature_size = x.size()

        output = self.firstconv(x)


        output_residual = output   # 1/2

        output = self.layer1(output)

        #print("output size:", output.shape)
        output_raw = self.layer2(output)
        #print("output size:", output_raw.shape)


        output = self.layer3(output_raw)



        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)


        output_feature_1 = self.lastconv(output_feature)
        #print("output_feature_1 size:", output_feature_1.shape)


        output_feature_2 =self.up_sample_1(output_feature_1)
        #print("output_feature_2 size:", output_feature_2.shape)

        output_feature_2 = torch.cat((output_feature_2, output_residual), 1 )
        #print("output_feature_2 size:", output_feature_2.shape)

        output_feature_2 = self.output_feature_2(output_feature_2)


        if image_left:

            output_CSPN = self.up_sample_2(output_feature_2)
            output_CSPN = self.output_CSPN(output_CSPN)



            return output_feature_1, output_feature_2, output_CSPN

        else:
            return output_feature_1, output_feature_2

