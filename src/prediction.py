import torch
import torch.nn as nn
import sklearn.metrics as skm
import torchvision as tv 
import torch.utils.data 
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import ast

def image_dataset_normalization(dataset_tensor):
    '''
    Takes a pytorch dataset, calculates the mean and std 
    of each channel for the sample as a whole
    '''
    #getting image
    image = dataset_tensor[0][0]
    #getting channels of images 
    channels = image.shape[0]
    #init running channel sum 
    channel_sum = torch.zeros(channels)
    #getting elements of each channel
    elements = torch.numel(dataset_tensor[0][0][0, :, :])

    # Calculate mean
    for image, label in dataset_tensor:
        channel_sum += torch.sum(image, dim=(1, 2))
    mean = channel_sum / (len(dataset_tensor) * elements)

    # Calculate std
    std_sum = torch.zeros(channels)
    for image, label in dataset_tensor:
        std_sum += torch.sum((image - mean)**2, dim=(1, 2))
    std = torch.sqrt(torch.div(std_sum, len(dataset_tensor) * elements))

    return mean, std

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self._forward = nn.Sequential(
        #conv layer 1 
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        #pool 1 
        nn.MaxPool2d(kernel_size=2, stride=2),

        #conv layer 2
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        #pool 2 
        nn.MaxPool2d(kernel_size=2, stride=2),

        #conv layer 3
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        #pool 3
        nn.MaxPool2d(kernel_size=2, stride=2),

        #conv layer 4 
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        #pool 4 
        nn.MaxPool2d(kernel_size=2, stride=2),

        #conv layer 5
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        #pool 5
        nn.MaxPool2d(kernel_size=2, stride=2),

        #flatten 
        nn.Flatten(),

        #fc layer 1 
        nn.Linear(8192, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        #fc layer 2
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        #fclayer 3 
        nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self._forward(x)

#load image that will be classified 
image = Image.open(r'C:\Users\1234z\Desktop\Jakes Stuff\dataset\Mild_Demented\mild_1.jpg')

#transforming to tensor, getting mean and std to use to normalize the image 
norm_transform = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
norm_tensor =  norm_transform(image)
mean = torch.mean(norm_tensor)
std = torch.std(norm_tensor)

#def the preprocessing transfomration for the next step 
transform = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.Normalize(mean, std), transforms.Resize((128, 128))]) 
#preprocessing image adding 4th demention for the input to the CNN layers                     
tensor = transform(image).unsqueeze(0)

#load classes from file 
with open(r'C:\Users\1234z\Desktop\Jakes Stuff\model_results\classes.txt', 'r') as file:
    content = file.read()
    classes = ast.literal_eval(content)

# #init the model 
model = CNN(4)
#getting the state dict and applying to the model
model.load_state_dict(torch.load(r'C:\Users\1234z\Desktop\Jakes Stuff\model_results\model.pth'))

#setting to eval 
model.eval()
with torch.no_grad():
    output = model(tensor)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()

print(classes[predicted_class])