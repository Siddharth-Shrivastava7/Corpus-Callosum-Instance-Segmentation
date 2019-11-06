
import torch 
import numpy as np
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths):   # initial logic happens like transform
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()
        self.mapping = {
            0: 0,
            255:1, 
            240:2,
            100:3,
            120:4,
            250:5              
        }
    
    def mask_to_label(self, mask):

        img = plt.imread(mask)

        img_n = img.copy()

        ret, b_img = cv2.threshold(img_n,127,255,cv2.THRESH_BINARY)

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
                    b_img.itemset((i, j, 0), 255)
                    b_img.itemset((i, j, 1), 0)
                    b_img.itemset((i, j, 2), 0)
            for j in range(regions[1], regions[2]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0),  0)
                    b_img.itemset((i, j, 1), 240)
                    b_img.itemset((i, j, 2), 0)
            for j in range(regions[2], regions[3]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 100)
                    b_img.itemset((i, j, 1), 0)
                    b_img.itemset((i, j, 2), 0)
            for j in range(regions[3], regions[4]):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 0)
                    b_img.itemset((i, j, 1), 120)
                    b_img.itemset((i, j, 2), 0)
            for j in range(regions[4], regions[5]+1):
                if(b_img[i][j][0]!=0):
                    b_img.itemset((i, j, 0), 0)
                    b_img.itemset((i, j, 1), 0)
                    b_img.itemset((i, j, 2), 250)

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

        maskn = b_img

        for k in self.mapping:
            maskn[maskn==k] = self.mapping[k]
            label = torch.zeros(maskn.shape[0],maskn.shape[1])
            label = maskn[:,:,0] + maskn[:,:,1] + maskn[:,:,2] 

        return label

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        t_image = self.transforms(image)
        mask = torch.from_numpy(np.array(mask))
        label = self.mask_to_label(mask)
        return t_image, label

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)


train_dataset = CustomDataset(train_image_paths, train_mask_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

for data, target in train_loader:
    print(torch.unique(target))