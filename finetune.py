import argparse
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from torch.autograd import Variable
from dataloader import KITTIdatalist as ls
from dataloader import KITTILoader as DA
import utils.logger as logger
import numpy as np
import torch.distributed as dist
from utils.metric import d1_metric
import pdb

from HybridNet.Hybrid_Net import Hybrid_Net, PSMNet, PSMNet_DSM

# python3 -m pip install -i https://pypi.douban.com/simple torchvision==0.2.1

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



parser = argparse.ArgumentParser(description='HybridNet with KITTI Stereo datasets')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1., 1.,1.])
parser.add_argument('--residual_disparity_range', type=int, default=3)
parser.add_argument('--datapath2015', default='/data6/wsgan/KITTI/KITTI2015/training/', help='datapath')
parser.add_argument('--datapath2012', default='/data6/wsgan/KITTI/KITTI2012/training/', help='datapath')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--split_for_val', type =bool, default=True,  help='finetune for submission or for validation')
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=2, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')

parser.add_argument('--datatype', default='2015', help='finetune dataset: 2012, 2015, mix')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--pretrained', type=str, default='./result/sceneflow/checkpoint_34.tar', help='pretrained model path')
parser.add_argument('--print_freq', type=int, default=10, help='print frequence')
parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--with_residual_cost', type =bool, default=True,  help='with residual cost network or not')
parser.add_argument('--with_cspn', type =bool, default=True,  help='with cspn network or not')
parser.add_argument('--model_types', type=str, default='Hybrid_Net_DSM', help='model_types: PSMNet, PSMNet_DSM, Hybrid_Net, Hybrid_Net_DSM')
parser.add_argument('--activation_types1', type=str, default='ELU', help='activation_function_types (for feature extraction) : ELU, Relu, Mish ')
parser.add_argument('--activation_types2', type=str, default='Relu', help='activation_function_types (for feature aggregation): ELU, Relu, Mish ')
parser.add_argument('--conv_3d_types1', type=str, default='DSM', help='model_types: 3D, P3D, DSM, 2D')
parser.add_argument('--conv_3d_types2', type=str, default='2D', help='model_types: 3D, P3D, DSM, 2D')

parser.add_argument('--save_path', type=str, default='./result/test/',help='the path of saving checkpoints and log')
parser.add_argument('--supervise_types', type=str, default='semi_supervised', help='supervise_types: supervised, semi_supervised, self_supervised')

parser.add_argument('--train', type =bool, default=True,  help='train or test ')

parser.add_argument('--disparity_mask', type =bool, default=False,  help='Replace with gt disparity in self_supervised loss ')
parser.add_argument('--denormalization', type =bool, default=True,  help='denormalization the img in self_supervised loss ')


parser.add_argument('--loss_weight', type=float, default=0.2, help='for balancing the loss')

parser.add_argument('--CSPN_step', type=int, default=4, help='CSPN iteration times')

# distribute related argument
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', type =bool, default=True, help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--distributed', type =bool, default=True,  help='distributed or not ')
parser.add_argument('--deterministic', action='store_true')


args = parser.parse_args()

# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 finetune.py


log = logger.setup_logger(args.save_path + '/training.log')
print("opt_level = {}".format(args.opt_level))
print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))


def main(log):
    global args

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## init dist ##
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.model_types == "PSMNet":
        model = PSMNet(args)
        args.loss_weights = [0.5, 0.7, 1.]

    elif args.model_types == "PSMNet_DSM":
        model = PSMNet_DSM(args)
        args.loss_weights = [0.5, 0.7, 1.]

    elif args.model_types == "Hybrid_Net_DSM" or "Hybrid_Net":
        model = Hybrid_Net(args)
        args.loss_weights = [0.5, 0.7, 1., 1., 1.]

    else:
        AssertionError("model error")


    if args.datatype == '2015':
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader2015(
            args.datapath2015, split = args.split_for_val)

    elif args.datatype == '2012':
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader2012(
            args.datapath2012, split = False)

    elif args.datatype == 'mix':
        all_left_img_2015, all_right_img_2015, all_left_disp_2015, test_left_img_2015, test_right_img_2015, test_left_disp_2015 = ls.dataloader2015(
            args.datapath2015, split = False)
        all_left_img_2012, all_right_img_2012, all_left_disp_2012, test_left_img_2012, test_right_img_2012, test_left_disp_2012 = ls.dataloader2012(
            args.datapath2012, split = False)
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = \
            all_left_img_2015 + all_left_img_2012, all_right_img_2015 + all_right_img_2012, \
            all_left_disp_2015 + all_left_disp_2012, test_left_img_2015 + test_left_img_2012, \
            test_right_img_2015 + test_right_img_2012, test_left_disp_2015 + test_left_disp_2012
    else:

        AssertionError("please define the finetune dataset")



    train_set = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True)
    val_set = DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    else:
        train_sampler = None
        val_sampler = None


    TrainImgLoader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_bsize, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler,
        drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.test_bsize, shuffle=False, num_workers=4, pin_memory=True, sampler=None, drop_last=False)


    num_train = len(TrainImgLoader)
    num_test = len(TestImgLoader)

    if args.local_rank == 0:

        for key, value in sorted(vars(args).items()):

            log.info(str(key) + ': ' + str(value))


    stages = len(args.loss_weights)


    # note
    if args.sync_bn:
        import apex
        model = apex.parallel.convert_syncbn_model(model)
        if args.local_rank == 0:
            log.info("using apex synced BN-----------------------------------------------------")

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=args.opt_level,
                                    keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                    loss_scale=args.loss_scale
                                    )

    if args.distributed:
        model = DDP(model, delay_allreduce=True)
        if args.local_rank == 0:
            log.info("using distributed-----------------------------------------------------")


    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained, map_location='cpu')

            model.load_state_dict(checkpoint['state_dict'], strict=True)

            if args.local_rank == 0:
                log.info("=> loaded pretrained model '{}'".format(args.pretrained))

        else:
            if args.local_rank == 0:
                log.info("=> no pretrained model found at '{}'".format(args.pretrained))
                log.info("=> Will start from scratch.")
    args.start_epoch = 0


    if args.resume:
        if os.path.isfile(args.resume):
            if args.local_rank == 0:
                log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                log.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                log.info("=> no checkpoint found at '{}'".format(args.resume))
                log.info("=> Will start from scratch.")
    else:
        if args.local_rank == 0:
            log.info('Not Resume')


    if args.local_rank == 0:
        log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    start_full_time = time.time()

    for epoch in range(args.start_epoch , args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        total_train_loss = 0
        total_d1 = 0
        total_epe = 0
        adjust_learning_rate(optimizer, epoch)

        losses = [AverageMeter() for _ in range(stages)]

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):


            loss = train(imgL_crop, imgR_crop, disp_crop_L, model, optimizer)

            for idx in range(stages):
                losses[idx].update(loss[idx].item() / args.loss_weights[idx])


            # # record loss
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)
            if args.local_rank == 0:
                log.info('losses　　Epoch{} [{}/{}] {}'.format(epoch, batch_idx, len(TrainImgLoader), info_str))


            total_train_loss += loss[-1]

        if args.local_rank == 0:
            log.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / num_train))



        if epoch % 50 == 0:
            ## Test ##
            inference_time = 0

            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):


                epe, d1, single_inference_time = test(imgL, imgR, disp_L, model)

                inference_time += single_inference_time

                total_d1 += d1
                total_epe += epe

                if args.distributed:
                    total_epe = reduce_tensor(total_epe.data)

                    total_d1 = reduce_tensor(total_d1.data)


                else:

                    total_epe = total_epe
                    total_d1 = total_d1


            if args.local_rank == 0:
                log.info('epoch %d avg_3-px error in val = %.3f' % (epoch, total_d1 / num_test * 100))
                log.info('epoch %d avg_epe  in val = %.3f' % (epoch, total_epe / num_test))
                log.info(('=> Mean inference time for %d images: %.3fs' % (num_test, inference_time / num_test)))


            if args.local_rank == 0:
                if epoch % 100 == 0:
                    savefilename = args.save_path + '/finetune_' + str(epoch) + '.tar'
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'train_loss': total_train_loss / len(TrainImgLoader),
                        'test_loss': total_d1 / len(TestImgLoader) * 100,
                    }, savefilename)


    if args.local_rank == 0:
        log.info('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))



def train(imgL, imgR, disp_L, model, optimizer):

    stages = len(args.loss_weights)

    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))


    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < args.maxdisp)
    mask.detach_()
    # ----

    optimizer.zero_grad()


    outputs, self_supervised_loss = model(imgL, imgR, disp_L)


    outputs = [torch.squeeze(output, 1) for output in outputs]

    loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_true[mask], size_average=True)
            for x in range(stages)]

    if args.supervise_types == 'self_supervised':

        self_supervised_loss = sum(self_supervised_loss)
        if args.local_rank == 0:
            print('self_supervised_loss:', self_supervised_loss)
        sum_loss = self_supervised_loss


    elif args.supervise_types == 'semi_supervised':

        GT_loss = sum(loss)
        if args.local_rank == 0:
            print("GT loss:", GT_loss)
        self_supervised_loss = sum(self_supervised_loss)

        if args.local_rank == 0:
            print('self_supervised_loss:', self_supervised_loss)
        sum_loss = args.loss_weight * self_supervised_loss + GT_loss


    elif args.supervise_types == 'supervised':
        sum_loss = sum(loss)


    else:
        AssertionError



    with amp.scale_loss(sum_loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    optimizer.step()

    if args.distributed:
        reduced_loss = [reduce_tensor(loss.data) for loss in loss]

    else:
        reduced_loss = loss

    return reduced_loss




def test(imgL, imgR, disp_true, model):

    model.eval()

    imgL = imgL.float().cuda()
    imgR = imgR.float().cuda()
    disp_L = disp_true.float().cuda()

    with torch.no_grad():

        time_start = time.perf_counter()
        output3, self_supervised_loss = model(imgL, imgR, disp_L)
        output3 = [output3[-1]]
        single_inference_time = time.perf_counter() - time_start


    pred_disp = output3[-1].squeeze(1)

    # computing 3-px error#
    mask = (disp_L > 0) & (disp_L < args.maxdisp)



    epe = F.l1_loss(disp_L[mask], pred_disp[mask], reduction='mean')
    d1 = d1_metric(pred_disp, disp_L, mask)


    return epe, d1, single_inference_time



def adjust_learning_rate(optimizer, epoch):
    if epoch <= 300:
        lr = args.lr
    elif 300< epoch <= 600:
        lr = args.lr*0.1
    else:
        lr = args.lr*0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main(log)
