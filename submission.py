from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from HybridNet.Hybrid_Net import Hybrid_Net, PSMNet, PSMNet_DSM
from PIL import Image
import utils.logger as logger


# python3 -m pip install -i https://douban.com/sample torchvision

parser = argparse.ArgumentParser(description='HybridNet with KITTI Stereo datasets')


parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1., 1.,1.])
parser.add_argument('--residual_disparity_range', type=int, default=3)
parser.add_argument('--CSPN_step', type=int, default=4, help='CSPN iteration times')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--with_residual_cost', type =bool, default=True,  help='with residual cost network or not')
parser.add_argument('--with_cspn', type =bool, default=True,  help='with cspn network or not')

parser.add_argument('--model_types', type=str, default='Hybrid_Net_DSM', help='model_types :PSMNet, PSMNet_DSM, Hybrid_Net, Hybrid_Net_DSM')
parser.add_argument('--activation_types1', type=str, default='ELU', help='activation_function_types (for feature extraction) : ELU, Relu, Mish ')
parser.add_argument('--activation_types2', type=str, default='Relu', help='activation_function_types (for feature aggregation): ELU, Relu, Mish ')
parser.add_argument('--conv_3d_types1', type=str, default='DSM', help='model_types: 3D, P3D, DSM, 2D')
parser.add_argument('--conv_3d_types2', type=str, default='2D', help='model_types: 3D, P3D, DSM, 2D')
parser.add_argument('--supervise_types', type=str, default='supervised', help='supervise_types :  supervised, self_supervised')

parser.add_argument('--save_path', type=str, default='./result/finetune/2015/disp_0/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--pretrained', type=str, default='./result/finetune/2015/finetune_1000.tar',
                    help='pretrained model path')
parser.add_argument('--datapath2015', default='/data6/wsgan/KITTI/KITTI2015/testing/', help='datapath')
parser.add_argument('--datapath2012', default='/data6/wsgan/KITTI/KITTI2012/testing/', help='datapath')


# 54
# /home/wsgan/KITTI_DATASET/KITTI2015/testing
# /home/wsgan/KITTI_DATASET/KITTI2012/testing

# 46
# /data6/wsgan/KITTI/KITTI2015/testing/
# /data6/wsgan/KITTI/KITTI2012/testing/


parser.add_argument('--datatype', default='2015', help='finetune dataset: 2012, 2015')

args = parser.parse_args()


# CUDA_VISIBLE_DEVICES=0 python submission.py



args.cuda =  torch.cuda.is_available()



if args.datatype == '2015':
   from dataloader import KITTI_submission_loader as DA

   test_left_img, test_right_img = DA.dataloader2015(args.datapath2015)

elif args.datatype == '2012':

   from dataloader import KITTI_submission_loader as DA
   test_left_img, test_right_img = DA.dataloader2012(args.datapath2012)

else:

    AssertionError("None found datatype")


if args.model_types == "PSMNet":
    model = PSMNet(args)
    args.loss_weights = [0.5, 0.7, 1.]

elif args.model_types == "PSMNet_TSM":
    model = PSMNet_DSM(args)
    args.loss_weights = [0.5, 0.7, 1.]

elif args.model_types == "Hybrid_Net_DSM" or "Hybrid_Net":
    model = Hybrid_Net(args)
    args.loss_weights = [0.5, 0.7, 1., 1., 1.]
else:
    AssertionError("model error")


if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


log = logger.setup_logger(args.save_path + '/submission.log')
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
        disp, self_supervised_loss = model(imgL,imgR)

    disp = torch.squeeze(disp[-1])

    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])

    total_inference_time = 0

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')


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

        start_time = time.time()
        pred_disp = test(imgL,imgR)

        total_inference_time += time.time() - start_time

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        print("inx:", inx)
        img.save(args.save_path  + test_left_img[inx].split('/')[-1])


    log.info("mean inference time:  %.3fs " % (total_inference_time/len(test_left_img)))

    log.info("finish {} images inference".format(len(test_left_img)))



if __name__ == '__main__':
    main()






