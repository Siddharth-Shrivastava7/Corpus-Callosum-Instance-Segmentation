            # 0: (255,0,0),
            # 1: (0,255,0),
            # 2: (0,0,255),
            # 3: (255,255,0),
            # 4: (255,0,255),
            # 5: (0,0,0)


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
    
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        t_image = self.transforms(image)
        mask = torch.from_numpy(np.array(mask))
        mask = self.mask_to_class(mask)
        return t_image, mask

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)


train_dataset = CustomDataset(train_image_paths, train_mask_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

for data, target in train_loader:
    print(torch.unique(target))