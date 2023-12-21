import torch
import torch.nn as nn
import torchvision as tv 
import torch.utils.data 
import torchvision.transforms as transforms
from PIL import Image
import ast

def dementia_stage_prediction(path):
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
    image = Image.open(path)

    #transforming to tensor, getting mean and std to use to normalize the image 
    norm_transform = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    norm_tensor =  norm_transform(image)
    mean = torch.mean(norm_tensor)
    std = torch.std(norm_tensor)

    #def the preprocessing transfomration for the next step 
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.Normalize(mean, std), transforms.Resize((128, 128), antialias=True)]) 
    #preprocessing image adding 4th demention for the input to the CNN layers                     
    tensor = transform(image).unsqueeze(0)

    #load classes from file 
    with open(r'model_results\classes.txt', 'r') as file:
        content = file.read()
        classes = ast.literal_eval(content)

    # #init the model 
    model = CNN(4)
    #getting the state dict and applying to the model
    model.load_state_dict(torch.load(r'model_results\model.pth'))

    #setting to eval 
    model.eval()
    with torch.no_grad():
        output = model(tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    print(classes[predicted_class])

dementia_stage_prediction(r'75yo_male.jpg')