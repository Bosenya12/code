import argparse
import os
import torch

# 参数列表
def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch Knowledge Distillation - Student training for AMC')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')  # 打印频率
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency') # tensorboard 记录频率
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=500, help='batch_size') # 批大小
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs') # 训练轮次
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # Dataset
    parser.add_argument('--dataset', type=str, default='a', help='Dataset to load: a for RML2016.10A and b for RML2016.10B (default: a)')

    # Model
    parser.add_argument('--model_s', type=str, default='CLDNN',
                        choices=['CLDNN','VGG16',
                                 'resnet8', 'resnet14', 'resnet20', 'resnet32',
                                 'resnet44', 'resnet56', 'resnet110', 'resnet8x4',
                                 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1',
                                 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16',
                                 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/CLDNN_2016a_all.pth', help='Path of the pre-trained model to be loaded')

    # Distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'kdsvd',
                                                                      'fsp','rkd', 'pkt', 'abound', 'factor',
                                                                      'nst', 'itrd', 'ega',
                                                                      'crd', 'rrd',])
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('-r', '--gamma', type=float, default=0.3, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.7, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=2.0, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t_s', default=0.1, type=float, help='student temperature parameter for softmax')
    parser.add_argument('--nce_t_t', default=0.02, type=float, help='teacher temperature parameter for softmax')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--memory_type', default='fifo', type=str, choices=['fifo', 'momentum'])

    # Other
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    args = parser.parse_args()

    # Set different learning rate from these 4 models
    if args.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        args.learning_rate = 0.01

    # Set the path according to the environment
    args.model_path = './save/student/student_model/'
    args.tb_path = './save/student/student_tensorboards/'

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.dataset == 'a':
        args.dataset_name = 'RML2016.10A'
    elif args.dataset == 'b':
        args.dataset_name = 'RML2016.10B'


    # args.model_t = get_teacher_name(args.path_t)
    args.model_t = 'CLDNN'
    args.model_name = ('{}_{}_S_{}_T_{}_r_{}_a_{}_b_{}_trial_{}'
                       .format(args.distill.upper(), args.dataset_name, args.model_s, args.model_t, args.gamma, args.alpha, args.beta, args.trial))

    # 创建目录
    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # 环境测试
    print("CUDA available:", torch.cuda.is_available())
    print("GPUs: ", torch.cuda.device_count())
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU Name: {gpu_info.name}")
        print(f"Total VRAM: {gpu_info.total_memory / 1e9} GB")

    # 打印参数
    options_dict = vars(args)
    for key, value in options_dict.items():
        print(f"{key}: {value}")

    return args

# args = parse_option()