import os
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import *

from tools.config import parse_option
from tools.lr_scheduler import adjust_learning_rate
# 导入模型
from models.CLDNN import CLDNN
from models.util import ConvReg, LinearEmbed, Connector, Translator, Paraphraser
# 导入数据集
import dataset2016a, dataset2016b
# 导入蒸馏损失类
from distillers import DistillKL, HintLoss, ABLoss, Attention, NSTLoss, Similarity, RKDLoss, ITLoss, PKT, KDSVD, EGA, Correlation, VIDLoss, RRDLoss, FSP, FactorTransfer, CRDLoss, RRDLoss
# 导入KD训练函数
from tools.train import train_one_epoch, init
from tools.validation import validate


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():

    args = parse_option()
    #=====================tensorboard=====================
    logger = SummaryWriter(log_dir=args.tb_folder)

    # =====================dataset======================
    if args.dataset == 'a':
        (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = dataset2016a.load_data(
            train_idx_size=100)
        n_classes = 11
    else:
        (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = dataset2016b.load_data()
        n_classes = 10

    X_train = np.expand_dims(X_train, axis=1)  # add a channel dimension
    X_test = np.expand_dims(X_test, axis=1)
    # Load dataset
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # =======================model======================
    model_t = CLDNN(n_classes=n_classes)
    model_t.load_state_dict(torch.load(args.checkpoint))  # 加载教师模型
    model_s = CLDNN(n_classes=n_classes)

    #=====================mock data=====================
    data = torch.randn(2, 1, 2, 128)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data)
    feat_s, _ = model_s(data)

    #=====================modules=====================
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([]) # 可训练模块列表
    trainable_list.append(model_s)

    # =====================criteria====================
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失函数
    criterion_div = DistillKL(args.kd_T)   # 蒸馏损失函数

    if args.distill == 'kd':
        # 原始知识蒸馏
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif args.distill == 'itrd':
         args.s_dim = feat_s[-1].shape[1]
         args.t_dim = feat_t[-1].shape[1]
         # args.n_data = n_data
         args.n_data = len(train_data)
         criterion_kd = ITLoss(args)
         module_list.append(criterion_kd.embed)
         trainable_list.append(criterion_kd.embed)
    elif args.distill == 'attention':
        criterion_kd = Attention()
    elif args.distill == 'nst':
        criterion_kd = NSTLoss()
    elif args.distill == 'similarity':
        criterion_kd = Similarity()
    elif args.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif args.distill == 'pkt':
        criterion_kd = PKT()
    elif args.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif args.distill == 'ega':
        criterion_kd = EGA()
    elif args.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], args.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], args.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif args.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList([VIDLoss(s, t, t) for s, t in zip(s_n, t_n)])
        trainable_list.append(criterion_kd)
    elif args.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, args)
        module_list.append(connector)
    elif args.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, args)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif args.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, args)
        pass
    elif args.distill == 'crd':
        args.s_dim = feat_s[-1].shape[1]
        args.t_dim = feat_t[-1].shape[1]
        # args.n_data = n_data
        args.n_data = len(train_data)
        criterion_kd = CRDLoss(args)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif args.distill == 'rrd':
        args.s_dim = feat_s[-1].shape[1]
        args.t_dim = feat_t[-1].shape[1]
        # args.n_data = n_data
        args.n_data = len(train_data)
        criterion_kd = RRDLoss(args)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    else:
        raise NotImplementedError(args.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # Classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # Other knowledge distillation loss

    # =====================optimizer=====================
    optimizer = torch.optim.Adam(trainable_list.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4)

    # Append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    #=====================cuda=====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    module_list.to(device)
    criterion_list.to(device)
    # 设置cuDNN优化
    cudnn.benchmark = True

    # =====================eval teacher=====================
    teacher_acc, _, _ = validate(test_loader, model_t, criterion_cls, args, device)
    print(f'Teacher accuracy: {teacher_acc.item()}%')

    # =====================routine=====================
    best_acc = 0

    time_tmp = datetime.now().strftime("%y%m%d_%H-%M-%S")

    for epoch in range(1, args.epochs+1):
        # adjust_learning_rate(epoch, args, optimizer)
        print("==> Training...")

        time1 = time.time()
        train_acc, train_loss = train_one_epoch(epoch, train_loader, module_list, criterion_list, optimizer, args, device)
        time2 = time.time()
        print('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, tect_acc_top5, test_loss = validate(test_loader, module_list[0], criterion_cls, args, device)

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(args.save_folder, '{}_best.pth'.format(time_tmp))
            print('Saving the best model!')
            torch.save(state, save_file)

        # # Regular saving
        # # 定期保存模型，检查点
        # if epoch % args.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'accuracy': test_acc,
        #     }
        #     save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

        # =====================loggging=====================
        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('test_loss', test_loss, epoch)
        logger.add_scalar('test_acc_top5', tect_acc_top5, epoch)
        logger.add_scalar('best_acc', best_acc, epoch)

    # This best accuracy is only for printing purpose.
    print('==> Best student accuracy:', best_acc)

    # # Save last model
    # state = {
    #     'args': args,
    #     'model': model_s.state_dict(),
    # }
    # save_file = os.path.join(args.save_folder, '{}_last.pth'.format(time_tmp))
    # torch.save(state, save_file)

    logger.close()

if __name__ == '__main__':
    main()