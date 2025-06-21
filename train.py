import torch
import argparse
import time

from ourModels import model_VGG
from data import get_loader
from utils import clip_gradient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=50,    help='epoch number')
    parser.add_argument('--lr',          type=float, default=1e-4,  help='learning rate')
    parser.add_argument('--batchsize',   type=int,   default=4,     help='training batch size')
    parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
    parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
    parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,   default=50,    help='every n epochs decay learning rate')
    parser.add_argument('--testsize',    type=int,   default=352,   help='testing size')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = parse_args()
    model = model_VGG()
    model = model.to(device)
    params = model.parameters()

    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '/path/to/images/'
    gt_root    = '/path/to/groundtruths/'

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    CE = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, opt.epoch+1):
        start_time = time.time()

        model.train()
        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts = pack
            
            images = images.to(device)
            gts = gts.to(device)

            dets = model(images)
            loss = CE(dets, gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if i % 200 == 0 or i == total_step:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(
                    epoch, opt.epoch, i, total_step, loss.item()))

        end_time = time.time()
        epoch_duration = end_time - start_time
        print('Epoch [{:03d}] completed in {:.2f} seconds.'.format(epoch, epoch_duration))

        save_path_pth = '/path/to/save/models/'
        if epoch % 1 == 0:
            torch.save(model.state_dict(), save_path_pth + 'LSPM-{:d}.pth'.format(epoch))
