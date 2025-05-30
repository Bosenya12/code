import sys
import time
import torch
from tools.utils import AverageMeter, accuracy
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

def train_one_epoch(epoch, train_loader, module_list, criterion_list, optimizer, opt, device):
    """ One epoch distillation """
    # Set modules as train()
    for module in module_list:
        module.train()

    # Set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    # for signal, label in tqdm(train_loader, desc="Training", ncols=100, colour='green'):
    for idx, data in enumerate(train_loader):
        signal, label = data
        # measure data loading time
        data_time.update(time.time() - end)

        signal, label = signal.to(device), label.to(device)
        # ===================forward=====================
        feat_s, logit_s = model_s(signal)
        with torch.no_grad():
            feat_t, logit_t = model_t(signal)

        # Classification (CE) + KL div
        loss_cls = criterion_cls(logit_s, label.long())
        loss_div = criterion_div(logit_s, logit_t)
        # Other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, label, topk=(1, 5))
        losses.update(loss.item(), signal.size(0))
        top1.update(acc1[0], signal.size(0))
        top5.update(acc5[0], signal.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================print======================
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    """ Initialization """
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate

    optimizer = optim.SGD(init_modules.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd', 'rrd'] and opt.memory_type == 'momentum':
                    contrast_idx = contrast_idx.cuda()

            # ==============forward===============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplementedError('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # ===================print======================
        logger.add_scalar('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
            epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
