from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from HybridNet.Hybrid_Net import Hybrid_Net, PSMNet, PSMNet_TSM
from PIL import Image
import utils.logger as logger

import cv2 as cv
import numpy as np
import pdb

# python3 -m pip install -i https://douban.com/sample torchvision

parser = argparse.ArgumentParser(description='HybridNet with KITTI Stereo datasets')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1., 1.,1.])
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--residual_disparity_range', type=int, default=3)
parser.add_argument('--datapath2015', default='/home/wsgan/KITTI_DATASET/KITTI2015/testing/', help='datapath')
parser.add_argument('--datapath2012', default='/home/wsgan/KITTI_DATASET/KITTI2012/testing/', help='datapath')
parser.add_argument('--datatype', default='2015', help='finetune dataset: 2012, 2015, Middlebury')
parser.add_argument('--split_for_val', type =bool, default=True,  help='finetune for submission or for validation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')

parser.add_argument('--train_bsize', type=int, default=16, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='/home/wsgan/Stereo_SOTA/HybridNet/result/md/',
                    help='the path of saving checkpoints and log')

parser.add_argument('--resume', type=str, default=None,#'/home/wsgan/Stereo_SOTA/PSMNet/PSMNet-master/results/HybridNet/senceflow/Hybrid_Net_DSM/TemporalShift_2D_ONLY_2D/test1/checkpoint_34.tar',
                    help='resume path')
parser.add_argument('--pretrained', type=str, default='/home/wsgan/HybridNet/results/HybridNet/KITTI2015/Submit/mix/bs_2/finetune2012/finetune_1000.tar',
                    help='pretrained model path')

parser.add_argument('--print_freq', type=int, default=10, help='print frequence')
parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')

parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--with_residual_cost', type =bool, default=True,  help='with residual cost network or not')

parser.add_argument('--with_cspn', type =bool, default=True,  help='with cspn network or not')

parser.add_argument('--model_types', type=str, default='Hybrid_Net_DSM', help='model_types :  '
               'PSMNet, PSMNet_TSM, Hybrid_Net, Hybrid_Net_DSM, StereoNet')

parser.add_argument('--augment_types', type=str, default='PSMNet', help='augment_types : PSMNet, HSMNet ')
parser.add_argument('--activation_types1', type=str, default='ELU', help='activation_function_types (for feature extraction) : ELU, Relu, Mish ')
parser.add_argument('--activation_types2', type=str, default='Relu', help='activation_function_types (for feature aggregation): ELU, Relu, Mish ')



#  when the model_types is defined as PSMNet or Hybrid_Net,
#  the conv_3d_types1 and conv_3d_types2 should be limited between normal and separate_only

parser.add_argument('--conv_3d_types1', type=str, default='TemporalShift_2D', help='model_types :  normal, separate_only, TemporalShift_2D, ONLY_2D,ONLY_Deform_2D ')
parser.add_argument('--conv_3d_types2', type=str, default='ONLY_2D', help='model_types :  normal, separate_only, TemporalShift_2D, ONLY_2D, ONLY_Deform_2D')

parser.add_argument('--train', type =bool, default=True,  help='train or test ')
parser.add_argument('--supervise_types', type=str, default='supervised', help='supervise_types :  supervised, self_supervised')
parser.add_argument('--CSPN_step', type=int, default=4, help='print frequence')


args = parser.parse_args()


# CUDA_VISIBLE_DEVICES=0 python prid_md.py



args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)


if args.cuda:
    torch.cuda.manual_seed(args.seed)


if args.model_types == "PSMNet":
    model = PSMNet(args)
    args.loss_weights = [0.5, 0.7, 1.]

elif args.model_types == "PSMNet_TSM":
    model = PSMNet_TSM(args)
    args.loss_weights = [0.5, 0.7, 1.]


elif args.model_types == "Hybrid_Net":
    model = Hybrid_Net(args)
    args.loss_weights = [0.5, 0.7, 1., 1., 1.]


elif args.model_types == "Hybrid_Net_DSM":
    model = Hybrid_Net(args)
    args.loss_weights = [0.5, 0.7, 1., 1., 1.]




else:

    AssertionError("model error")




if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()



log = logger.setup_logger(args.save_path + '/training.log')
for key, value in sorted(vars(args).items()):
    log.info(str(key) + ': ' + str(value))

if args.pretrained:
    if os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        log.info('=> loaded pretrained model {}'.format(args.pretrained))
    else:
        log.info('=> no pretrained model found at {}'.format(args.pretrained))
        log.info("=> Will start from scratch.")


else:
    log.info('Not Resume')





def test(imgL,imgR):
    model.eval()

    if args.cuda:
       imgL = imgL.cuda()
       imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp[-1])
    #print('disp size:', disp.shape)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp



def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])

    total_inference_time = 0
    test_path = '/home/wsgan/HybridNet/750*500/750MiddEval3-data-Q500/MiddEval3/trainingQ/'
    test_save_path = '/home/wsgan/Stereo_SOTA/HybridNet/result/md/'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    test_list = os.listdir(test_path)

    for item in test_list:

        ImgL = test_path + item + '/im0.png'
        ImgR = test_path + item + '/im1.png'
        imgL_o = Image.open(ImgL).convert('RGB')
        imgR_o = Image.open(ImgR).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
        print("IMGL:", imgL.shape)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        # pred_disp = pred_disp[-1].squeeze(1)

        total_inference_time += time.time() - start_time
        #print('time = %.2f' %(time.time() - start_time))

        if top_pad !=0 :
            img = pred_disp[top_pad:,:]

        if right_pad != 0:
            img = pred_disp[:, :-right_pad]

        # else:
        #     img = pred_disp

        #pdb.set_trace()
        # img = (img*256).astype('uint16')
        # img = Image.fromarray(img)
        # print("image size:", img)
        # print("image type",item)
        # img.save(test_save_path + item + '.png')


        pred_color = cv.applyColorMap(np.array(img * 2, dtype=np.uint8), cv.COLORMAP_JET)

        cv.imwrite(test_save_path + item + '.png', pred_color)








if __name__ == '__main__':
    main()






