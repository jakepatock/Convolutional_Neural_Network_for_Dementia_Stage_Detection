import torch
import torch.nn as nn
import sklearn.metrics as skm
import torchvision as tv 
import torch.utils.data 
import torchvision.transforms as transforms
import numpy as np
import random

#setting variables to make repoducabiliy 
#these variables are used to accelerate gpu learning, especialy when the input size varies into the network
#false will result in loss of performance but deterministic results, need both of these varibles to be set (benchmark = False, deterministic = True) to be deterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#this varible configures the pytorch's use of deterministic algorithms instead of non-deterministic ones 
#(however throws error is there is no alternative to a deterministic algorithm) in this case it thrwos an error 
#torch.use_deterministic_algorithms(True)
#set seeds of libs
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

#getting device, setting to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data path to directory with sub directories that contain images of subdirectory class
data = r'C:\Users\1234z\Desktop\Jakes Stuff\Dataset'

#fuction that takes list of different transfomration to apply to the list of images, converting to tensor and grayscale
tensor_gray_transformation = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])

#feed directory of data with each different class in a different directroy in the parent directory 
#this will be used to calculate the mean and std used for normalization of the data 
norm_dataset = tv.datasets.ImageFolder(data, transform=tensor_gray_transformation)

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

mean, std = image_dataset_normalization(norm_dataset)

#composing final preprocessing transfomrations and making the proprocessed dataset
tensor_gray_norm_transformations = tv.transforms.Compose([tv.transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), tv.transforms.Normalize(mean, std)])
dataset = tv.datasets.ImageFolder(data, transform=tensor_gray_norm_transformations)

#division this data to validation and trianing 
# 80% for training
train_size = int(0.8 * len(dataset)) 
#20% for validation 
val_size = len(dataset) - train_size 

#split the data into train and val 
training_data, validation_data = torch.utils.data.random_split(dataset, [train_size, val_size])

#batch size of the data used due to the fact that loading all the trianing data into ram at one point will not be possible
#therefore we split it up into batches of training data 
batch_size = 16

#feeding the datasets to the loader 
train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(validation_data, batch_size, pin_memory=True)

#init the early stopping class that will be used to stop training after a specified number of epcohs
#without inprovment to the f1 score
class Early_Stopping_F1():
    def __init__(self, patience):
        self._patience = patience
        self.current_patience = 0
        self.max_f1 = float('-inf')
        self.min_state_dict = None
        self.epoch = 0

    def stopper(self, current_model, f1):
        self.epoch += 1
        if f1 > self.max_f1:
            self.max_f1 = f1
            self.current_patience = 0
            self.min_state_dict = current_model.state_dict()
        else:
            self.current_patience += 1
            
        if self.current_patience >= self._patience:
            return True
        else:
            return False
        
    def get_state_dict(self):
        return self.min_state_dict  
    
    def get_current_pacients(self):
        return self.current_patience
    
    def get_max_f1(self):
        return self.max_f1
    
    def get_kept_epoch(self):
        return self.epoch - self._patience

#setting up the model 
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

#init model 
model = CNN(len(dataset.classes))
model.to(device)

#loss function
loss_func = nn.CrossEntropyLoss()

#optimizer 
optimizer = torch.optim.Adam(model.parameters())

#init early stopper 
early_stopper = Early_Stopping_F1(50)

#list to track accuracy and loss 
training_loss_lst = []
training_accuracy_lst = []
test_loss_lst = []
test_acc_lst = []
f1_score_lst = []

#training loop
epoch = 0
while True:
    #set model into training mode 
    model.train()
    #setting running loss, correct, and total varibles for this epoch 
    running_loss = 0 
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        #setting the device 
        features, labels = features.to(device), labels.to(device)
        #clear optimizer
        optimizer.zero_grad()
        #get y predicted
        predicted_labels = model(features)

        #get loss 
        batch_loss = loss_func(predicted_labels, labels)
        #calculated direction of gradient
        batch_loss.backward()
        #get the new theta
        optimizer.step()
        #multipe the loss by the size of the batch to scale it to make the loss 
        #representative of the loss across the entire batch (accounts for batch of different sizes)
        running_loss += batch_loss.item() * features.size(0)

        #getting accuracy 
        total += labels.size(0)
        predicted_arg = torch.argmax(predicted_labels, dim=1)
        correct += (predicted_arg == labels).sum().item()

    #calculating the loss for this epoch
    train_loss = running_loss / len(train_loader.dataset)
    #getting the accuracy for this epoch 
    training_accuracy = correct / total

    #printing status of loss and accuracy to consle
    epoch += 1
    print(f"Epoch {epoch}:")
    print(f'Training: Loss = {train_loss}, Accuracy = {training_accuracy}')

    #evaluating the model 
    model.eval()
    #init all varibles to track accuracy and loss 
    test_loss_cul = 0
    correct = 0
    total = 0 

    #these will be used to calculate the f1 score for this current epoch on the training data
    cul_predictions = torch.tensor([]).to(device)
    cul_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        for features, labels in test_loader:
            #setting to run on cuda 
            features, labels = features.to(device), labels.to(device)
            #calculating predicted 
            predicted_labels = model(features)

            #getting loss from labels 
            test_batch_loss = loss_func(predicted_labels, labels)
            test_loss_cul += test_batch_loss.item() * features.size(0) 

            #getting accuracy
            #getting size of batch 
            total += labels.size(0)
            #getting the predicted label 
            predicted_arg = torch.argmax(predicted_labels, dim=1)
            #counting how many correct labels there are 
            correct += (predicted_arg == labels).sum().item()

            #getting cul prediction and labels to use for f1
            cul_predictions = torch.cat((cul_predictions, predicted_arg))
            cul_labels = torch.cat((cul_labels, labels))
            
        #getting loss 
        test_loss = test_loss_cul / len(test_loader.dataset)
        #getting accuracy 
        test_accuracy = correct / total

        #getting f1 score 
        #average = 'macro' calculates the average f1 scored for each class and averages them 
        #beta > 1 shifts the f1 score to be more in favor or recall while beta < 1 makes it favored towards precision 
        f1 = skm.fbeta_score(cul_predictions.cpu().numpy(), cul_labels.cpu().numpy(), average="weighted", beta=2)
    
    #printing status to consel
    print(f"Test: Loss = {test_loss}, Accuracy = {test_accuracy}, F-1 Score = {f1}")
    

    #adding loss, accuracy, and f1 score to the lists that will be used to produce a plot of the progress of the model 
    training_loss_lst.append(train_loss)
    training_accuracy_lst.append(training_accuracy)
    test_loss_lst.append(test_loss)
    test_acc_lst.append(test_accuracy)
    f1_score_lst.append(float(f1))

    #earlier stopper condition check to see if the pacients of the model has run out 
    #if it has revert model to highest f1 
    if early_stopper.stopper(model, f1):
        state_dict = early_stopper.get_state_dict()
        model.load_state_dict(state_dict)
        max_f1 = early_stopper.get_max_f1()
        kept_epoch = early_stopper.get_kept_epoch()
        print()
        print(f'Kept Epoch: {kept_epoch}, Max F1: {max_f1}')
        break
    else:
        print(f'Current Patience: {early_stopper.get_current_pacients()}')
        print()

torch.save(model, r'C:\Users\1234z\Desktop\Jakes Stuff\Data\model.pth')

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\training_loss.txt', 'w+') as file:
    file.write(str(training_loss_lst))
with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\training_accuracy.txt', 'w+') as file:
    file.write(str(training_accuracy_lst))
with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\test_loss.txt', 'w+') as file:
    file.write(str(test_loss_lst))
with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\test_accuracy.txt', 'w+') as file:
    file.write(str(test_acc_lst))
with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\f1_score.txt', 'w+') as file:
    file.write(str(f1_score_lst))