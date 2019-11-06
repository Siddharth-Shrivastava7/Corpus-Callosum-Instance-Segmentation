import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch 
from torch.nn import functional as F
import glob 



dic_il = { 

    0: 0,
    260:1,
    450:2,
    380:3,
    510:4,
    255:5
}

dic_li ={ 

    0: (0,0,0),
    1: (250,0,10),
    2:(0,250,200),
    3: (200,10,170),
    4:(200,200,110),
    5:(0,0,255)
}
 
def image_to_label(mask):

    sm = np.sum(mask,2)

    lb = sm.copy()

    for i in dic_il:

        lb[sm==i] = dic_il[i]

    return lb


def label_to_image(label):

    label = label.cpu().numpy()

    out = np.zeros((label.shape[0],label.shape[1],3))

    for j in dic_li:

        out[label==j] = dic_li[j]

    out = out.astype('uint8')

    return out

def parcellation(mask):

        img_n = mask.copy()

        # print(b_img.shape)

        ret, b_img = cv2.threshold(img_n,127,255,cv2.THRESH_BINARY)

        # b_img = b_img/255

        # b_img[b_img>=0.5] = 255
        # b_img[b_img<0.5] = 0

        #b_img = cv2.flip(b_img,0)  ## for images inverted initially (rotated it with 180 degrees)

        # for one j value we r seeing different values of i, for which pixel intensity is 255. 

        L = [] #list of pixels corresponding for corpus callosum.
        
        for j in range(b_img.shape[1]): 
            for i in range(b_img.shape[0]):
                if(b_img[i][j][0]==255):
                    L.append((j,i))   
        
    
        psx = L[0][0] #start point (antenior)
        pex = L[-1][0] #end point (posterior)
        distance = L[-1][0] - L[0][0]
        regions = []
        regions.append(psx)

        #Weitelson
        # regions.append(psx+int((1/3)*distance))
        # regions.append(psx+int((1/2)*distance))
        # regions.append(psx+int((2/3)*distance))
        # regions.append(psx+int((4/5)*distance))

        #Hofer
        regions.append(psx+int((1/6)*distance))
        regions.append(psx+int((1/2)*distance))
        regions.append(psx+int((2/3)*distance))
        regions.append(psx+int((3/4)*distance))
        regions.append

        regions.append(pex)

        #filling colors (parcellating)... according to mentioned scheme i.e. here hofer

        for i in range(b_img.shape[0]):
            for j in range(regions[0], regions[1]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 250)
                    b_img.itemset((i, j, 1), 0)
                    b_img.itemset((i, j, 2), 10)
            for j in range(regions[1], regions[2]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0),  0)
                    b_img.itemset((i, j, 1), 250)
                    b_img.itemset((i, j, 2), 200)
            for j in range(regions[2], regions[3]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 200)
                    b_img.itemset((i, j, 1), 10)
                    b_img.itemset((i, j, 2), 170)
            for j in range(regions[3], regions[4]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 200)
                    b_img.itemset((i, j, 1), 200)
                    b_img.itemset((i, j, 2), 110)
            for j in range(regions[4], regions[5]+1):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 0)
                    b_img.itemset((i, j, 1), 0)
                    b_img.itemset((i, j, 2), 255)

        #anterior 
        a = b_img[L[0][1]][L[0][0]][0]
        b = b_img[L[0][1]][L[0][0]][1]
        c = b_img[L[0][1]][L[0][0]][2]

        #posterior
        d = b_img[L[-1][1]][L[-1][0]][0]
        e = b_img[L[-1][1]][L[-1][0]][1]
        f = b_img[L[-1][1]][L[-1][0]][2]

        #correcting errors i.e. maintaining bottom curve portion, present on both lateral sides with single region respectively. 
        #parcellate using small assumption that the geometric baseline being the horizontal line for each
        #half i.e. on anterior and  as well as posterior side of corpus callosum.

        #for upright image (original)

        for j in range(regions[0],regions[2]):
            for i in range(L[0][1], b_img.shape[0]):

                if(b_img[i][j][1] != 0):

                    b_img[i][j][0] = a 
                    b_img[i][j][1] = b
                    b_img[i][j][2] = c
                    
        for j in range(regions[2],regions[5]):
            for i in range(L[-1][1], b_img.shape[0]):

                if(b_img[i][j][1] != 0):

                    b_img[i][j][0] = d 
                    b_img[i][j][1] = e
                    b_img[i][j][2] = f

        return b_img


def pred_to_label(pred):

    pb = F.softmax(pred,dim=0)
    label = torch.argmax(pb,dim=0)

    return label
    


def main():

    direc = glob.glob('./dataset/valid/mask/*.jpg')

    direc = sorted(direc)

    for k in direc:

        mask = plt.imread(k)

        parcel = parcellation(mask)

        save_path1 = k.replace('mask','Parcels')

        plt.imsave(save_path1,parcel)

        label = image_to_label(parcel)

        out = label_to_image(label)

        save_path = k.replace('mask','parcel_out')

        plt.imsave(save_path,out)

    print('Finished it')

# if __name__ == "__main__":

#     main()




        

        

        

    

