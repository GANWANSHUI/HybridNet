import argparse
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import torch.distributed as dist
import utils.logger as logger
import numpy as np
import torch.nn as nn
import pdb
import apex
from utils.metric import thres_metric
from utils.count_flops  import profile
from torch.autograd import Variable
from HybridNet.Hybrid_Net import Hybrid_Net, PSMNet, PSMNet_DSM

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


parser = argparse.ArgumentParser(description='HybridNet with Scene flow')
# Hyperparameter
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1., 1., 1.])
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--residual_disparity_range', type=int, default=3)
parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')
parser.add_argument('--CSPN_step', type=int, default=4, help='CSPN iteration times')

# data path
parser.add_argument('--datapath', default='/data6/wsgan/SenceFlow/train/', help='datapath')
parser.add_argument('--save_path', type=str, default='./result/sceneflow/TEST/', help='the path of saving result')
parser.add_argument('--resume', type=str, default= './result/sceneflow/checkpoint_34.tar', help='resume path')

# train setting
parser.add_argument('--train', type =bool, default=False,  help='train or test ')
parser.add_argument('--supervise_types', type=str, default='supervised', help='supervise_types :  supervised, semi_supervised,  self_supervised')
parser.add_argument('--disparity_mask', type =bool, default=False,  help='train or test ')
parser.add_argument('--print_freq', type=int, default=400, help='print frequence')
parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=2, help='batch size for training (default: 12), 2 for DDP')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--with_residual_cost', type =bool, default=True,  help='with residual cost network or not')
parser.add_argument('--with_cspn', type =bool, default=True,  help='with cspn network or not')
parser.add_argument('--count_flops', type =bool, default=False,  help='with count flops or not')
parser.add_argument('--model_types', type=str, default='Hybrid_Net_DSM', help='model_types: PSMNet, PSMNet_DSM, Hybrid_Net, Hybrid_Net_DSM')
parser.add_argument('--activation_types1', type=str, default='ELU', help='(for feature extraction) : ELU, Relu, Mish ')
parser.add_argument('--activation_types2', type=str, default='Relu', help='(for feature aggregation): ELU, Relu, Mish ')
parser.add_argument('--conv_3d_types1', type=str, default='DSM', help='model_types: 3D, P3D, DSM, 2D')
parser.add_argument('--conv_3d_types2', type=str, default='2D', help='model_types: 3D, P3D, DSM, 2D')

# distribute related parameters
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', type =bool, default=True, help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--distributed', type =bool, default=True,  help='distributed or not ')

args = parser.parse_args()

# clear GPU
# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

# distributed
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py

# no distributed
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py


def main():

    global args

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)



    if args.distributed:

        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1

        args.world_size = 1

        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."



    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)



    train_set = DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True)
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


    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')

    if args.local_rank == 0:
        log.info('len train_left_img: {}'.format(len(train_left_img)))
        log.info('len test_left_img: {}'.format(len(test_left_img)))


    if args.local_rank == 0:
        for key, value in sorted(vars(args).items()):
            log.info(str(key) + ': ' + str(value))


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


    if args.count_flops:

        FLOPs, param = count_flops(model.cuda())
        if args.local_rank == 0:
            log.info("macs:{}".format(FLOPs))
            log.info("parameters:{} ".format(param))


    if args.sync_bn:
        if args.local_rank == 0:
            log.info("using apex synced BN-----------------------------------------------------")
        model = apex.parallel.convert_syncbn_model(model)


    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    if args.distributed:
        if args.local_rank == 0:
            log.info("using distributed-----------------------------------------------------")
        model = DDP(model, delay_allreduce=True)


    if args.local_rank == 0:
        log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



    args.start_epoch = 0
    if args.resume:

        if os.path.isfile(args.resume):

            checkpoint = torch.load(args.resume,  map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank == 0:
                log.info("=> loading checkpoint '{}'".format(args.resume))
                log.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                log.info("=> no checkpoint found at '{}'".format(args.resume))
                log.info("=> Will start from scratch.")
    else:
        if args.local_rank == 0:
            log.info('Not Resume')



    start_full_time = time.time()



    if args.train :

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            if args.local_rank == 0:
                log.info('This is {}-th epoch'.format(epoch))
            adjust_learning_rate(optimizer, epoch)


            train(TrainImgLoader, model, optimizer, log, epoch)

            # SAVE
            if args.local_rank == 0:
                savefilename = args.save_path + '/checkpoint_' + str(epoch) + '.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, savefilename)



            if not epoch % 10:
                test(TestImgLoader, model, log)




    test(TestImgLoader, model, log)

    if args.local_rank == 0:
        log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))




def train(dataloader, model, optimizer, log, epoch=0):

    stages = len(args.loss_weights)
    losses = [AverageMeter() for _ in range(stages)]
    losses_pix = [AverageMeter() for _ in range(stages)]
    self_loss = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        #print("imgL:", imgL.shape)

        optimizer.zero_grad()
        mask = (disp_L < args.maxdisp) &  (disp_L > 0)
        if mask.float().sum() == 0:
            continue

        mask.detach_()
        outputs, self_superised_loss = model(imgL, imgR, disp_L)


        outputs = [torch.squeeze(output, 1) for output in outputs]


        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]

        thres1 = [thres_metric(outputs[x], disp_L, mask, 1.0) for x in range (stages)]


        if args.supervise_types == 'self_supervised':

            GT_loss = sum(loss)
            sum_loss = sum(self_superised_loss) + GT_loss


        else:
            sum_loss =sum(loss)


        with amp.scale_loss(sum_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()


        if args.distributed:
            reduced_loss = [reduce_tensor(loss.data) for loss in loss ]
            reduced_thres1 = [reduce_tensor(thres1.data) for thres1 in thres1]
            if args.supervise_types == 'self_supervised':
                reduced_self_loss = [reduce_tensor(loss.data) for loss in self_superised_loss]

        else:
            reduced_loss = loss
            reduced_thres1 = thres1
            if args.supervise_types == 'self_supervised':
                reduced_self_loss = self_superised_loss




        for idx in range(stages):
            losses[idx].update(reduced_loss[idx].item()/args.loss_weights[idx])
            losses_pix[idx].update(reduced_thres1[idx].item())

            if args.supervise_types == 'self_supervised':
                self_loss[idx].update(reduced_self_loss[idx].item() / args.loss_weights[idx])


        if args.local_rank == 0:
            # record loss and thres1
            if not batch_idx % args.print_freq:


                info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
                info_str = '\t'.join(info_str)
                log.info('losses　　Epoch{} [{}/{}] {}'.format(epoch, batch_idx, length_loader, info_str))


                info_str = ['Stage {} = {:.3f}({:.3f})'.format(x, losses_pix[x].val, losses_pix[x].avg) for x in range(stages)]
                info_str = '\t'.join(info_str)
                log.info('losses_pix　　Epoch{} [{}/{}] {}'.format(epoch, batch_idx, length_loader, info_str))

                if args.supervise_types == 'self_supervised':
                    info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, self_loss[x].val, self_loss[x].avg) for x in range(stages)]
                    info_str = '\t'.join(info_str)
                    log.info('self_supervised_losses　　Epoch{} [{}/{}] {}'.format(epoch, batch_idx, length_loader, info_str))

        else:
            pass

    if args.local_rank == 0:

        info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
        log.info('Average train loss = ' + info_str)
        info_str = '\t'.join(['Stage {} = {:.3f}'.format(x, losses_pix[x].avg) for x in range(stages)])
        log.info('Average train thres1 = ' + info_str)
        if args.supervise_types == 'self_supervised':
            info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, self_loss[x].avg) for x in range(stages)])
            log.info('Average train self_supervised_losses = ' + info_str)


    else:
        pass


def test(dataloader, model, log):

    stages = 1
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

        mask = (disp_L < args.maxdisp) &  (disp_L > 0)

        if mask.float().sum() == 0:
            continue



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

            outputs, self_supervised_loss = model(imgL, imgR, disp_L)
            outputs = [outputs[-1]]

            single_inference_time = time.perf_counter() - time_start

            #print("single_inference_time:", single_inference_time )

            inference_time += single_inference_time


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


                EPEs_record = (output[mask] - disp_L[mask]).abs().mean()
                thres1_record = thres_metric(output, disp_L, mask, 1.0)


                if args.distributed:
                    EPEs_record = reduce_tensor(EPEs_record.data)
                    thres1_record = reduce_tensor(thres1_record.data)

                else:
                    EPEs_record = EPEs_record
                    thres1_record = thres1_record

                EPEs[x].update(EPEs_record.item())
                thres1[x].update(thres1_record.item())


            if args.local_rank == 0:

                if not batch_idx % args.print_freq:

                    info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
                    log.info('EPEs　　[{}/{}] {}'.format(batch_idx, length_loader, info_str))

                    info_str = '\t'.join(['Stage {} = {:.3f}({:.3f})'.format(x, thres1[x].val, thres1[x].avg) for x in range(stages)])
                    log.info('thres1　　[{}/{}] {}'.format(batch_idx, length_loader, info_str))


                else:
                    pass



    if args.local_rank == 0:

        log.info(('=> Mean inference time for %d images: %.3fs' % (
            length_loader, inference_time / length_loader)))

        info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
        log.info('Average test EPE = ' + info_str)



        info_str = ', '.join(['Stage {}={:.3f}'.format(x, thres1[x].avg) for x in range(stages)])
        log.info('Average test thres1 = ' + info_str)

    else:
        pass





def count_flops(model):

    input = Variable(torch.randn(1, 3, 544, 960)).cuda()

    macs, params = profile(model, inputs = (input,input))

    return macs, params




def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = args.lr

    elif 10 <epoch <= 20:
        lr = args.lr * 0.5

    elif 20 <epoch <= 25:
        lr = args.lr * 0.25

    elif 25< epoch <= 30:
        lr = args.lr * 0.1

    else:
        lr = args.lr * 0.01


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


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



# 快捷指令

# https://www.cnblogs.com/kaiye/p/6275207.html

# tmux new -s name  创建session
# tmux a -t  name 进入session
# tmux ls 查看session
# tmux kill-session -t name 杀掉session 进程
# ctrl + B 松开 D 退出当前session的全部窗口
# ctrl + B 松开+ C 基于当前session新建窗口


# exit 退出当前bash的窗口
# ctrl +B 松开 + x 删除当前bash


# ctrl + B  松开 -   水平分屏
# ctrl + B 松开 shift + - 垂直分屏

# tmux kill-server # 删除所有的会话

