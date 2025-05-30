import torch

def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk) # ：Top-K准确率就是用来计算预测结果中概率最大的前K个结果包含正确标签的占比。
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """ Computes and stores the average and current value """
    """
    AverageMeter 类是深度学习中常用的辅助工具，用于动态跟踪和计算一组数据的当前值、累加和、平均值。
    这个类在训练和验证过程中非常实用，可以实时监控损失函数值、准确率等指标的变化
    """
    def __init__(self):
        # 初始化计数器，调用 reset() 方法。
        self.reset()

    def reset(self):
        # 重置所有计数器为初始值。
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
