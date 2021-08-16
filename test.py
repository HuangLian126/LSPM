import torch, os, argparse
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import misc
from ourModels import model_VGG
from torch.autograd import Variable
from data import get_loader_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()
    model = model_VGG()
    model.load_state_dict(torch.load('/model path/'))

    model.cuda()
    model.eval()

    CE = torch.nn.BCEWithLogitsLoss()

    def calculateF1Measure(output_image,gt_image,thre):
        output_image = np.squeeze(output_image)
        gt_image     = np.squeeze(gt_image)

        out_bin      = output_image>thre
        gt_bin       = gt_image>thre
        recall       = np.sum(gt_bin*out_bin)/np.maximum(1, np.sum(gt_bin))
        prec         = np.sum(gt_bin*out_bin)/np.maximum(1, np.sum(out_bin))
        F1           = 2*recall*prec/np.maximum(0.001, recall+prec)
        return F1


    with torch.no_grad():

        sum_val_false_ratio = 0
        sum_val_detect_ratio = 0
        sum_val_F1 = 0

        save_path = '/path/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = '/path/'
        gt_root    = '/path/'

        test_loader = get_loader_test(image_root, gt_root, opt.testsize)

        for i, data in enumerate(test_loader):
            image, gt, name = data
            gts = Variable(gt)
            gts = gts.cuda()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res2 = res.sigmoid()
            res2 = F.interpolate(res2, size=(gts.shape[2], gts.shape[3]), mode='bilinear', align_corners=False)
            res1 = res.sigmoid().data.cpu().numpy().squeeze()
            imo = cv2.resize(res1, (gts.shape[3], gts.shape[2]))
            misc.imsave(save_path + name[0], imo)
            minVar = torch.zeros_like(gts)
            maxVar = torch.ones_like(gts)

            train_F1 = calculateF1Measure(res2.cpu().numpy(), gts.cpu().numpy(), 0.7)
            sum_val_F1 += train_F1

        avg_val_F1_g1 = sum_val_F1 / 100
        print("==========F1 measure is %0.4f" % (avg_val_F1_g1))
