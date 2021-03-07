import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import time
import torch.optim as optim
import torchvision
import cv2 as cv
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
import numpy as np
from metric import d1_metric, thres_metric
from torch.autograd import Variable
#from thop import profile
from HybridNet.Hybrid_Net import Hybrid_Net, PSMNet, PSMNet_TSM


parser = argparse.ArgumentParser(description='HybridNet with Scene flow')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1., 1.,1.])
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--residual_disparity_range', type=int, default=3)


parser.add_argument('--datapath', default='/home/wsgan/KITTI_DATASET/SenceFlow/train/', help='datapath')
parser.add_argument('--datapath2015', default='/home/wsgan/KITTI_DATASET/KITTI2015/training/', help='datapath')
parser.add_argument('--datapath2012', default='/home/wsgan/KITTI_DATASET/KITTI2012/training/', help='datapath')


parser.add_argument('--datatype', default='2015', help='finetune dataset: 2012, 2015, mix')
parser.add_argument('--split_for_val', type =bool, default=True,  help='finetune for submission or for validation')

parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train')

parser.add_argument('--train_bsize', type=int, default=12, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='/home/wsgan/test/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='/home/wsgan/HybridNet/results/HybridNet/KITTI/Hybrid_Net_DSM/TemporalShift_2D_ONLY_2D/2015/finetune_1000.tar',
                    help='resume path')

parser.add_argument('--print_freq', type=int, default=400, help='print frequence')
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

parser.add_argument('--conv_3d_types1', type=str, default='TemporalShift_2D', help='model_types :  normal, P3D, separate_only, TemporalShift_2D, ONLY_2D,ONLY_Deform_2D ')
parser.add_argument('--conv_3d_types2', type=str, default='ONLY_2D', help='model_types :  normal, P3D, separate_only, TemporalShift_2D, ONLY_2D, ONLY_Deform_2D')

parser.add_argument('--testing', type =bool, default=False,  help='counting flops or testing ')
parser.add_argument('--visualization', type =bool, default=True,  help='train or test ')
parser.add_argument('--supervise_types', type=str, default='supervised', help='supervise_types :  supervised, self-supervised')

parser.add_argument('--CSPN_step', type=int, default=4, help='print frequence')


args = parser.parse_args()


# CUDA_VISIBLE_DEVICES=0 python FLOPs_reference_time.py

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls2015



def main():


    global args

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    if args.datatype == '2015':
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls2015.dataloader(
            args.datapath2015, split = args.split_for_val)
        from dataloader import KITTILoader as DA


    elif args.datatype == 'Sence Flow':
        train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)
        from dataloader import SecenFlowLoader as DA


    else:
        AssertionError


    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    log = logger.setup_logger(args.save_path + '/FLOPs_inference_time.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    if args.model_types == "PSMNet":
        model = PSMNet(args)
        args.loss_weights = [0.5, 0.7, 1.]
        #from dataloader import SecenFlowLoader as DA

    elif args.model_types == "PSMNet_TSM":
        model = PSMNet_TSM(args)
        args.loss_weights = [0.5, 0.7, 1.]
        #from dataloader import SecenFlowLoader as DA

    elif args.model_types ==  "Hybrid_Net":
        model = Hybrid_Net(args)
        args.loss_weights = [0.5, 0.7, 1., 1., 1.]
        #from dataloader import SecenFlowLoader as DA

    elif args.model_types == "Hybrid_Net_DSM" :
        model = Hybrid_Net(args)
        args.loss_weights = [0.5, 0.7, 1., 1., 1.]
        #from dataloader import SecenFlowLoader as DA



    else:

        AssertionError("model error")






    model = nn.DataParallel(model).cuda()



    for i in range (30):

        #print("test_left_img", test_left_img[i])
        log.info("=> test_left_img '{}'".format(test_left_img[i]))


    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)



    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    #if args.testing:
    test(TestImgLoader, model, log)







def test(dataloader, model, log):

    stages = 1#len(args.loss_weights)
    EPEs = [AverageMeter() for _ in range(stages)]
    thres1 = [AverageMeter() for _ in range(stages)]

    length_loader = len(dataloader)

    model.eval()

    padding_len = 16

    inference_time = 0

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp


        if imgL.shape[2] % padding_len != 0:
            times = imgL.shape[2]//padding_len
            #print("times:", times)
            if times % 2 == 0:
                top_pad = (times + 2) * padding_len - imgL.shape[2]

            else:
                top_pad = (times+1)*padding_len -imgL.shape[2]


        else:
            top_pad = 0


        right_pad = 0




        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))
        #print("Ａfter padding imgL size:", imgL.shape)


        with torch.no_grad():

            time_start = time.perf_counter()

            outputs = model(imgL, imgR)

            single_inference_time = time.perf_counter() - time_start

            #print("single_inference_time:", single_inference_time )

            inference_time += single_inference_time


            #print("output size:", outputs[0].shape)


            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    thres1[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)

                if top_pad != 0:
                    output = output[:, top_pad:, :]
                else:
                    output = output

                if args.visualization and batch_idx <= 50:

                    #vis
                    # print("outputs", output.shape )
                    # print("disp_L size:", disp_L.shape)

                    # GT = disp_L[:, :, :]/255.0
                    # torchvision.utils.save_image(GT, join(args.save_path, "iter-%d.jpg" % batch_idx))


                    #output = output.cpu()
                    _, H, W = output.shape
                    # all_results_color = torch.zeros((H, 2*W))
                    GT_color = torch.zeros((H, W ))
                    GT_color[:, :W] = disp_L[0,:, :]
                    GT_color = cv.applyColorMap(np.array(GT_color * 2, dtype=np.uint8), cv.COLORMAP_JET)
                    cv.imwrite(join(args.save_path, "iter-%d_GT_color.jpg" % batch_idx), GT_color)


                    pred_color = torch.zeros((H, W ))
                    pred_color[:, :W] = output[0, :, :]


                    # all_results_color = torch.zeros((H, 2 * W + 20))
                    #
                    #
                    # all_results_color[:,:W]= output[0, :, :]
                    #
                    # #all_results_color[:, W:30] = output[0, :, :]
                    #
                    # all_results_color[:,W+20:2*W+20]= disp_L[0,:, :]

                    pred_color = cv.applyColorMap(np.array(pred_color*2, dtype=np.uint8), cv.COLORMAP_JET)
                    error = (output[mask] - disp_L[mask]).abs().mean()
                    cv.imwrite(join(args.save_path, "iter-%d_pred_color-%.3f.jpg" %( batch_idx , error)),pred_color)






                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())
                thres1[x].update(thres_metric(output, disp_L, mask, 1.0))

        if not batch_idx % args.print_freq:


            print("single_inference_time:", single_inference_time)
            info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])

            log.info('EPEs　　[{}/{}] {}'.format(batch_idx, length_loader, info_str))




            info_str = '\t'.join(['Stage {} = {:.3f}({:.3f})'.format(x, thres1[x].val, thres1[x].avg) for x in range(stages)])

            log.info('thres1　　[{}/{}] {}'.format(batch_idx, length_loader, info_str))






    log.info(('=> Mean inference time for %d images: %.3fs' % (
        length_loader, inference_time / length_loader)))

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)


    info_str = ', '.join(['Stage {}={:.3f}'.format(x, thres1[x].avg) for x in range(stages)])
    log.info('Average test thres1 = ' + info_str)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()
