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

from networks.cenet import CE_Net_
from framework import MyFrame
from loss import weighted_cross_entropy,metrics
from data import ImageFolder
#from Visualizer import Visualizer
import torchvision

#from pytorchtools import EarlyStopping

import Constants
import image_utils

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def CE_Net_Train():
    NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    # viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, weighted_cross_entropy, 2e-4)
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

    #early_stopping = EarlyStopping(patience=20, verbose=True)

    color_map = {
        0: (0,0,0),
        1: (255,0,0), 
        2: (0,240,0),
        3: (100,0,0),
        4: (0,120,0),
        5: (0,0,250)              
        }

    def viz_image(pred):    
        
        color = torch.zeros(3, pred.shape[1], pred.shape[2]) 

        for k in color_map:

            if(pred[k] == 1):

                color[0] = color_map[k][0]
                color[1] = color_map[k][1]
                color[2] = color_map[k][2]

        return color
                        

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        data_loader_iter_v = iter(data_loader_v)
        valid_epoch_loss = 0
        # train_epoch_dice_loss = 0
        # valid_epoch_dice_loss = 0

        for img, mask in data_loader_iter:

            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            # train_loss, pred, train_dice_loss = solver.optimize()
            # tsens,tspec,tacc,tprec = solver.eval()
            train_epoch_loss += train_loss
            # train_epoch_dice_loss += train_dice_loss        
        
        a = 0
        
        for img, mask in data_loader_iter_v:

            solver.set_input(img, mask)
            # valid_loss, pred, valid_dice_loss = solver.optimize_test()
            valid_loss, pred = solver.optimize_test()
            # sens,spec,acc,prec = solver.eval()
            valid_epoch_loss += valid_loss
            # valid_epoch_dice_loss += valid_dice_loss

            #print(torch.unique(pred[a][4])) #print it
            print(pred[a])

            if( a % 10 == 0):
                print('saving_pred_images after 10 batches')

                img  = viz_image(pred[a,:,:,:])

                torchvision.utils.save_image(img, "valid/pred"+str(epoch)+str(' ') + str(a) + ".jpg", nrow=1, padding=0)            

            a = a + 1  
        
        # print(sens)
        # torchvision.utils.save_image(img[0, :, :, :], "valid/image_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        #torchvision.utils.save_image(mask[0, :, :, :], "valid/mask"+str(epoch) + ".jpg", nrow=1, padding=0)
        #torchvision.utils.save_image(pred[0, :, :, :], "valid/pred"+str(epoch) + ".jpg", nrow=1, padding=0)
 
        
        # show the original images, predication and ground truth on the visdom.
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        # train_epoch_dice_loss = train_epoch_dice_loss/len(data_loader_iter)
        # valid_epoch_dice_loss = valid_epoch_dice_loss/len(data_loader_iter_v)
        valid_epoch_loss = valid_epoch_loss/len(data_loader_iter_v)

        # print("saving images")
        # print("length of (data_loader_iter) ", len(data_loader_iter))
        # print(mylog, '-----------------------------------------')
        # print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        # print(mylog, 'train_loss:', train_epoch_loss)
        # print(mylog, 'valid_loss:', valid_epoch_loss)
        # print(mylog, 'train_dice_loss:', train_epoch_dice_loss)
        # print(mylog, 'valid_dice_loss:', valid_epoch_dice_loss)
        # print(mylog, 'sens:', sens)
        # print(mylog, 'SHAPE:', Constants.Image_size)
        print('in Training')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        # print('train_dice_loss', train_epoch_dice_loss)

        # print('SHAPE:', Constants.Image_size)
        print('in VALIDATION')
        print('valid_loss:', valid_epoch_loss)
        # print('valid_dice_loss', valid_epoch_dice_loss)
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
    print('Finish!')
    # mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()



