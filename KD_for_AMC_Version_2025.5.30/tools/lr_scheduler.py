import numpy as np

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """ Learning rate schedule according to RotNet """
    """
    函数参数：
    epoch：当前训练轮次
    optimizer：PyTorch 优化器对象
    LUT：查找表，格式为[(max_epoch1, lr1), (max_epoch2, lr2), ...]
    """
    # 使用生成器表达式查找第一个max_epoch > epoch的条目，返回对应的学习率
    # 如果没有找到更大的max_epoch，则使用LUT中最后一个条目的学习率
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    # 参数更新：遍历优化器的所有参数组，将学习率设置为确定的值
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(epoch, opt, optimizer):
    """ Sets the learning rate to the initial LR decayed by decay rate every steep step """
    # 阶梯式衰减（Step Decay）
    # 计算当前轮次跨越了多少个衰减里程碑
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        # 新学习率 = 初始学习率 × (衰减率 ^ 跨越的里程碑数)
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr