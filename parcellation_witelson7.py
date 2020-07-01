import cv2
import glob

import numpy as np

import matplotlib.pyplot as plt

direc = glob.glob('./dataset/train/mask/*.jpg') 

direc = sorted(direc)

# print('direc', direc)

for k in direc:

    img = cv2.imread(k)

    # print(k)

    _, b_imgi = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # print(b_imgi.shape)

    # for one j value we r seeing different values of i, for which pixel intensity is 255. 

    L = []
    # = []

    # print(np.unique(b_imgi))

    for j in range(b_imgi.shape[1]): 
        for i in range(b_imgi.shape[0]):
            if(b_imgi[i][j][0]==255):
                L.append((j,i))   


    def get_line(x1, y1, x2, y2):
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        # Reverse the list if the coordinates were reversed
        if rev:
            points.reverse()
        return points
    
    # print(L)
    
    pts = get_line(L[0][0], L[0][1], L[-1][0], L[-1][1]) 

    in_conx = 0
    in_cony = 0
    # in_con = []
    for i in pts: 
        if (b_imgi[i[1]][i[0]][0] == 0 ):
    #         in_con.append((i[0],i[1]))
            in_conx = i[0]
            in_cony = i[1]
            break
  
    psx = L[0][0] #start point (antenior)
    pex = L[-1][0] #end point (posterior)
    distance = L[-1][0] - L[0][0]
    regions = []
    regions.append(psx)

    #Weitelson
    regions.append(in_conx)  # aditional
    regions.append(psx+int((1/3)*distance))
    regions.append(psx+int((1/2)*distance))
    regions.append(psx+int((2/3)*distance))
    regions.append(psx+int((4/5)*distance))

    #Hofer
    # regions.append(psx+int((1/6)*distance))
    # regions.append(psx+int((1/2)*distance))
    # regions.append(psx+int((2/3)*distance))
    # regions.append(psx+int((3/4)*distance))
    # regions.append

    regions.append(pex)

    #filling colors (parcellating)... according to mentioned scheme i.e. here hofer

    for i in range(b_imgi.shape[0]):
        for j in range(regions[0], regions[1]):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0), 255)
                b_imgi.itemset((i, j, 1), 128)
                b_imgi.itemset((i, j, 2), 0)
        for j in range(regions[1], regions[2]):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0),  255)
                b_imgi.itemset((i, j, 1),  255)
                b_imgi.itemset((i, j, 2),  0)
        for j in range(regions[2], regions[3]):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0),  0)
                b_imgi.itemset((i, j, 1),  200)
                b_imgi.itemset((i, j, 2),  0)
        for j in range(regions[3], regions[4]):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0),   255)
                b_imgi.itemset((i, j, 1),   0)
                b_imgi.itemset((i, j, 2),    250)
        for j in range(regions[4], regions[5]):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0), 0)
                b_imgi.itemset((i, j, 1), 0)
                b_imgi.itemset((i, j, 2),  225)
        for j in range(regions[5], regions[6]+1):
            if(b_imgi[i][j][0]!=0):
                b_imgi.itemset((i, j, 0),  53)
                b_imgi.itemset((i, j, 1), 153)
                b_imgi.itemset((i, j, 2), 200)

    # #anterior 
    # a = b_img[L[0][1]][L[0][0]][0]
    # b = b_img[L[0][1]][L[0][0]][1]
    # c = b_img[L[0][1]][L[0][0]][2]

    # #posterior
    # d = b_img[L[-1][1]][L[-1][0]][0]
    # e = b_img[L[-1][1]][L[-1][0]][1]
    # f = b_img[L[-1][1]][L[-1][0]][2]

    #correcting errors i.e. maintaining bottom curve portion, present on both lateral sides with single region respectively. 
    #parcellate using small assumption that the geometric baseline being the horizontal line for each
    #half i.e. on anterior and  as well as posterior side of corpus callosum.

    #for upright image (original)

    for i in range(in_cony, b_imgi.shape[0]):
        for j in range(regions[1] ,regions[2]):
            if(b_imgi[i][j][1] == 255):
                b_imgi[i][j][0] = 255
                b_imgi[i][j][1] = 0
                b_imgi[i][j][2] = 0

    
    save_path = k.replace('mask','parcel7')

    plt.imsave(save_path,b_imgi)

print('Done !')