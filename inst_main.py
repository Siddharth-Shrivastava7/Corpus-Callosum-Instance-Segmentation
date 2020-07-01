from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter

from networks.cenet import CE_Net_,UNet
from framework import MyFrame
from loss import weighted_cross_entropy,metrics
from data import ImageFolder
#from Visualizer import Visualizer
import torchvision

#from pytorchtools import EarlyStopping

import Constants
import image_utils
import numpy as np
import matplotlib.pyplot as plt

writer = SummaryWriter()

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from custom_functions import *



def CE_Net_Train():
    NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    # viz = Visualizer(env=NAME)

#     solver = MyFrame(CE_Net_, weighted_cross_entropy, 2e-4)
    solver = MyFrame(UNet, weighted_cross_entropy, 2e-4)
    # batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    batchsize = Constants.BATCHSIZE_PER_CARD
    batchsize_v = Constants.BATCH_VALID

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='Brain',mode ='train')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)


    valid = ImageFolder(root_path=Constants.ROOT, datasets='Brain', mode = 'valid')
    data_loader_v = torch.utils.data.DataLoader(
        valid,
        batch_size=batchsize_v,
        shuffle=True,
        num_workers=4)

    # start the logging files
    # mylog = open('logs/' + NAME + '.txt', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    valid_epoch_best_loss = 0    
                        

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        data_loader_iter_v = iter(data_loader_v)
        valid_epoch_loss = 0
        train_epoch_dice_loss = 0
        valid_epoch_dice_loss = 0

        for img, mask in data_loader_iter:

            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()

            train_dice_loss = solver.eval()
            train_epoch_loss += train_loss
            train_epoch_dice_loss += train_dice_loss        
        
        b = 0
        
        for img, mask in data_loader_iter_v:
            
            solver.set_input(img, mask)
            # valid_loss, pred, valid_dice_loss = solver.optimize_test()
            valid_loss, pred = solver.optimize_test()
            valid_dice_loss = solver.eval()
            valid_epoch_loss += valid_loss
            valid_epoch_dice_loss += valid_dice_loss


            if( b % 20 == 0):

                label = pred_to_label(pred[0])

                pred = label_to_image(label)

                parcel = label_to_image(mask[0])

                img = img[0].permute(1,2,0)

                img = img.cpu().numpy() 

                print('saving_pred_per_epoch_batch')

                plt.imsave('valid/pred/' + str(epoch) + ' ' + str(b) +'.jpg', pred )

                plt.imsave('valid/parcel/' + str(epoch) + ' ' + str(b) + '.jpg', parcel )

                plt.imsave('valid/img/' + str(epoch) + ' ' + str(b) + '.jpg', img )

            b = b + 1

        
        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        train_epoch_dice_loss = train_epoch_dice_loss/len(data_loader_iter)
        valid_epoch_dice_loss = valid_epoch_dice_loss/len(data_loader_iter_v)
        valid_epoch_loss = valid_epoch_loss/len(data_loader_iter_v)


        writer.add_scalars('Epoch_loss',{'train_epoch_loss': train_epoch_loss, 'train_epoch_dice_loss': train_epoch_dice_loss,
        'valid_epoch_loss':valid_epoch_loss,'valid_epoch_dice_loss':valid_epoch_dice_loss},epoch)
    


        print('in Training')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('train_dice_loss', train_epoch_dice_loss)

        # print('SHAPE:', Constants.Image_size)
        print('in VALIDATION')
        print('valid_loss:', valid_epoch_loss)
        print('valid_dice_loss', valid_epoch_dice_loss)
        print('--------------------------------------------------')

        if valid_epoch_best_loss == 0:
            valid_epoch_best_loss = valid_epoch_loss

        elif valid_epoch_loss >= valid_epoch_best_loss:
            no_optim += 1

        elif valid_epoch_loss < valid_epoch_best_loss:
            no_optim = 0
            valid_epoch_best_loss = valid_epoch_loss
            solver.save('./weights/' + 'best' + '.pth')

        elif no_optim > 20:
            #  print(mylog, 'early stop at %d epoch' % epoch)
             print('early stop at %d epoch' % epoch)
             break

        elif no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + 'best' + '.pth')
            solver.update_lr(2.0, factor=True)

        #model = CE_Net_()
        
        # early_stopping(valid_epoch_loss, model)
        
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # solver.save('./weights/' + 'best2' + '.th')

        # mylog.flush()

    # print(mylog, 'Finish!')
    writer.close()
    print('Finish!')
    # mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()



    
