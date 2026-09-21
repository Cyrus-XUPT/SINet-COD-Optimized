import torch
from torch.autograd import Variable
from datetime import datetime
import os
import torch.nn.functional as F
def iou_loss(pred, mask):
    """
    计算交并比(IoU)损失，专门针对图像边缘和整体结构进行惩罚
    """
    pred = torch.sigmoid(pred)
    # 计算交集 (Intersection)
    inter = (pred * mask).sum(dim=(2, 3))
    # 计算并集 (Union)
    union = (pred + mask).sum(dim=(2, 3))
    # IoU = 交集 / (并集 - 交集)
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy)


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images)
        gts = Variable(gts)

        #cam_sm, cam_im = model(images)
        #loss_sm = loss_func(cam_sm, gts)
        #loss_im = loss_func(cam_im, gts)
        #loss_total = loss_sm + loss_im
        # ===== 替换为新的加权混合 Loss =====
        cam_sm, cam_im = model(images)
        
        # 1. 基础像素级惩罚 (BCE)
        loss_bce_sm = loss_func(cam_sm, gts)
        loss_bce_im = loss_func(cam_im, gts)
        
        # 2. 结构边缘惩罚 (IoU)
        loss_iou_sm = iou_loss(cam_sm, gts)
        loss_iou_im = iou_loss(cam_im, gts)
        
        # 3. 组合第一阶段 (粗搜) 和第二阶段 (精抠) 的误差
        loss_sm = loss_bce_sm + loss_iou_sm
        loss_im = loss_bce_im + loss_iou_im
        
        # 4. 终极指挥棒：赋予精细抠图阶段 2.0 倍的权重！
        loss_total = loss_sm + 2.0 * loss_im
        # =================================

        loss_total.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data))

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch+1))
