import torch, os, argparse
import torch.nn.functional as F
import numpy as np
import cv2

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

        sum_test_FM = 0

        img_save_path = '/path/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = '/path/'
        gt_root    = '/path/'

        test_loader = get_loader_test(image_root, gt_root, opt.testsize)

        for i, data in enumerate(test_loader):
            image, gt, name = data
         
            gt[gt > 0.5] = 1
            gt[gt != 1] = 0
     
            image = image.cuda()
            res = model(image)
            
            img = img.sigmoid().data.cpu().numpy().squeeze()
            cv2.imwrite(img_save_path + name[0], 255*img)

            test_F1 = calculateF1Measure(res.sigmoid().cpu().numpy(), gt.numpy(), 0.7)
            sum_test_FM += test_F1

        FM = sum_test_FM / 100
        print("==========Fmeasure is %0.4f" % (FM))
