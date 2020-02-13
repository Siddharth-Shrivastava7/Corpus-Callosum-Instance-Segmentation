from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
import shutil
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter

from networks.cenet import CE_Net_
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

import argparse 

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from custom_functions import *


parser = argparse.ArgumentParser()


parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)') 

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay') 

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)') 

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model') 

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set') 

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

writer = SummaryWriter()

best_acc = 0

args = parser.parse_args() 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar') 


def CE_Net_Train(args):

    global best_acc

    # optionally resume from a checkpoint

    if args.resume:

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


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

    # dic = {
    #     0: (0,0,0),
    #     1: (255,0,0), 
    #     2: (0,240,0),
    #     3: (100,0,0),
    #     4: (0,120,0),
    #     5: (0,0,250)              
    #     }

    # def viz_image(pred):
            
    #     (h,w) = (pred.shape[1],pred.shape[2])

    #     #color = np.zeros((h, w,3))

    #     color = torch.zeros(3,h,w)

    #     ind = torch.argmax(abs(pred),0)
             
    #     for i in range(h):   
    #         for j in range(w):
    #                 k = ind[i][j]
    #                 k = k.item()
    #                 #print(k) 
    #                 #print(dic[k][0])
    #                 color[0][i][j] = dic[k][0]
    #                 color[1][i][j] = dic[k][1]
    #                 color[2][i][j] = dic[k][2]

    #     return color
                        

    for epoch in range(args.start_epoch, args.epochs): 
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        data_loader_iter_v = iter(data_loader_v)
        valid_epoch_loss = 0
        train_epoch_dice_loss = 0
        valid_epoch_dice_loss = 0

        for img, mask in data_loader_iter:

            #print(img.shape)
            #print(mask.shape)

            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()

            # train_loss, pred, train_dice_loss = solver.optimize()
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

            #print(torch.unique(pred[a][4])) #print it
            # print(pred[a].shape)

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

            #     print('valid loss after 2 batch:', valid_loss)

                # print('saving_pred_images after 10 batches')

                # img  = viz_image(pred[a,:,:,:])

                # torchvision.utils.save_image(img, "valid/pred"+str(epoch)+str(' ') + str(a) + ".jpg", nrow=1, padding=0)            

            # a = a + 1  
        
        # print(sens)

        # print('saving_pred_per_epoch')

        # img  = viz_image(pred[3]) 

        

        #plt.imsave('valid/pred_plt'+str(epoch)+".jpg",img)

        # #imgt = torch.from_numpy(img)
        # torchvision.utils.save_image(img, "valid/pred"+str(epoch) + ".jpg", nrow=1, padding=0)

        # print('saving_mask_ep')

        # #print(mask.shape)
        # #plt.imsave('valid/mask_plt'+str(epoch)+".jpg",mask[0])
        # torchvision.utils.save_image(mask[3], "valid/mask"+str(epoch) + ".jpg", nrow=1, padding=0)
        
        # ## remove later

        # print('saving_pred_ep_npy')
        
        # np.save('valid/pred'+str(epoch)+".npy",pred[3].cpu().numpy())

        ## till here

        # torchvision.utils.save_image(img, "valid/pred"+str(epoch)+ ".jpg", nrow=1, padding=0)  

        # torchvision.utils.save_image(img[0, :, :, :], "valid/image_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        #torchvision.utils.save_image(mask[0, :, :, :], "valid/mask"+str(epoch) + ".jpg", nrow=1, padding=0)
        #torchvision.utils.save_image(pred[0, :, :, :], "valid/pred"+str(epoch) + ".jpg", nrow=1, padding=0)
 
        
        # show the original images, predication and ground truth on the visdom.
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        train_epoch_dice_loss = train_epoch_dice_loss/len(data_loader_iter)
        valid_epoch_dice_loss = valid_epoch_dice_loss/len(data_loader_iter_v)
        valid_epoch_loss = valid_epoch_loss/len(data_loader_iter_v)

        # writer.add_scalar('trin_epoch_loss', train_epoch_loss,epoch)
        # writer.add_scalar('train_epoch_dice_loss',train_epoch_dice_loss,epoch)
        # writer.add_scalar('valid_epoch_loss',valid_epoch_loss,epoch)
        # writer.add_scalar('valid_epoch_dice_loss',valid_epoch_dice_loss,epoch)

        writer.add_scalars('Epoch_loss',{'train_epoch_loss': train_epoch_loss, 'train_epoch_dice_loss': train_epoch_dice_loss,
        'valid_epoch_loss':valid_epoch_loss,'valid_epoch_dice_loss':valid_epoch_dice_loss},epoch) 
    

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

        acc = valid_epoch_loss

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc1, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        print('in Training')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('train_dice_loss', train_epoch_dice_loss)

        # print('SHAPE:', Constants.Image_size)
        print('in VALIDATION')
        print('valid_loss:', valid_epoch_loss)
        print('valid_dice_loss', valid_epoch_dice_loss)
        print('--------------------------------------------------')




        # if valid_epoch_best_loss == 0:
        #     valid_epoch_best_loss = valid_epoch_loss

        # elif valid_epoch_loss >= valid_epoch_best_loss:
        #     no_optim += 1

        # elif valid_epoch_loss < valid_epoch_best_loss:
        #     no_optim = 0
        #     valid_epoch_best_loss = valid_epoch_loss
        #     solver.save('./weights/' + 'best' + '.pth')

        # elif no_optim > 20:
        #     #  print(mylog, 'early stop at %d epoch' % epoch)
        #      print('early stop at %d epoch' % epoch)
        #      break

        # elif no_optim > Constants.NUM_UPDATE_LR:
        #     if solver.old_lr < 5e-7:
        #         break 
        #     solver.load('./weights/' + 'best' + '.pth')
        #     solver.update_lr(2.0, factor=True)

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
    CE_Net_Train(args)



    
