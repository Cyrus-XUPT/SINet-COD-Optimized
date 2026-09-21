import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
opt = parser.parse_args()

model = SINet_ResNet50()
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
model.eval()

for dataset in ['COD10K']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image
        # inference
        cam_sm, cam = model(image) # 💥 修改点 1：把 '_' 换成 'cam_sm'，接住第一阶段的输出
        # reshape and squeeze
        # --- 💥 修改点 2：新增对第一阶段粗搜图 (SM) 的处理和保存 ---
        cam_sm = F.interpolate(cam_sm, size=gt.shape, mode='bilinear', align_corners=True) # 用 interpolate 顺手消除上采样警告
        cam_sm = cam_sm.sigmoid().data.cpu().numpy().squeeze()
        cam_sm = (cam_sm - cam_sm.min()) / (cam_sm.max() - cam_sm.min() + 1e-8)
        
        name_sm = name.replace('.png', '_sm.png').replace('.jpg', '_sm.jpg') # 给粗搜图加个 _sm 后缀
        imageio.imwrite(save_path + name_sm, (cam_sm * 255).astype(np.uint8)) 
        
        # --- 处理第二阶段精抠图 (IM，保持原逻辑，修复保存报错) ---
        cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        imageio.imwrite(save_path + name, (cam * 255).astype(np.uint8))
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
