from tools.dataloader import MyDataset
from tools.fmeasure import calculateF1Measure

import torch

from torch.utils.data import DataLoader
from torchvision import transforms

import glob

composed = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]
)
test_dataset = MyDataset('/home/hl/hl/CPD-master-origional-DYM/models/CPD_VGG-352-4/ourACM+FCMCA/results/*', '/home/hl/hl/MDvsFA-master/data/test_gt/*', transform=composed)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)



filenames = sorted(glob.glob('/home/hl/hl/MDvsFA-master/data/test_org/*.png'))
filenames = [x.split('/')[-1] for x in filenames]



with torch.no_grad():
    sum_val_loss_g1 = 0
    sum_val_false_ratio_g1 = 0
    sum_val_detect_ratio_g1 = 0

    sum_val_F1_g1 = 0
    g1_time = 0

    sum_val_loss_g2 = 0
    sum_val_false_ratio_g2 = 0
    sum_val_detect_ratio_g2 = 0

    sum_val_F1_g2 = 0
    g2_time = 0

    sum_val_loss_g3 = 0
    sum_val_false_ratio_g3 = 0
    sum_val_detect_ratio_g3 = 0

    sum_val_F1_g3 = 0

    for ix, (x, y) in (enumerate(test_dataloader)):
        result1 = x.to(device='cuda: 0', dtype=torch.float)
        mask    = y.to(device='cuda: 0', dtype=torch.float)
        #print(result1.shape)

        minVar = torch.zeros_like(mask)
        maxVar = torch.ones_like(mask)

        #print('mask: ', mask.shape)

        train_false_ratio = torch.mean(torch.max(minVar, result1 - mask))
        sum_val_false_ratio_g1 += train_false_ratio.item()

        train_detect_ratio = torch.sum(result1 * mask) / torch.max(mask.sum(), maxVar)
        sum_val_detect_ratio_g1 += torch.mean(train_detect_ratio)

        train_F1 = calculateF1Measure(result1.cpu().numpy(), mask.cpu().numpy(), 0.7)
        sum_val_F1_g1 += train_F1

    avg_val_false_ratio_g1 = sum_val_false_ratio_g1 / 100
    avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1 / 100
    avg_val_F1_g1 = sum_val_F1_g1 / 100

    print("==========falseAlarm_rate is %f" % (avg_val_false_ratio_g1))
    print("==========detection_rate is %f" % (avg_val_detect_ratio_g1))
    print("==========F1 measure is %f" % (avg_val_F1_g1))