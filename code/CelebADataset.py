import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import PIL
import os

# reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CelebADataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: csv file containing the image names & labels
        root_dir: root directory containing the image files
        transform: transformation function, which can be found in torchvision.transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # read the image
        # since the index passed in is an integer, we use iloc instead of loc
        # as loc uses label instead on numerical index
        image_name = os.path.join(self.root_dir, self.label_df.iloc[index, 0])
        image = PIL.Image.open(image_name)
        # read the label
        label = self.label_df.iloc[index, 1]

        # preprocess the image if applicable
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    TRAIN_CSV = os.path.dirname(os.path.realpath(__file__)) + '/../data/celeba-train.csv'
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data/img_align_celeba'
    dataset = CelebADataset(TRAIN_CSV, ROOT_DIR, transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    npimg = torchvision.utils.make_grid(images).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
