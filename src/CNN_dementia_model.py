import torch
import torch.nn as nn
import sklearn.metrics as skm
import sklearn.model_selection as skms
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
data = r'dataset'

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
#writting classes to file to use later 
with open(r'model_results\classes.txt', 'w+') as file:
    file.write(str(dataset.classes))


#division this data to validation and traiing 80, 20 in a stratified manner geting index from train_val_split
lables = dataset.targets
#first arg passes the list of indexes of all lables and elements, stratified are the lables of the indexes passesd to the function to split hte data in a stratified manner 
train_idx, val_idx = skms.train_test_split(np.arange(len(lables)), test_size=.2, train_size=.8, random_state=random_seed, shuffle=True, stratify=lables)

#now transform these list of indexs into pytorch dataset sampler which will be passed to the dataloader 
train_sample = torch.utils.data.Subset(dataset, train_idx)
val_sample = torch.utils.data.Subset(dataset, val_idx)

#batch size of the data used due to the fact that loading all the trianing data into ram at one point will not be possible
#therefore we split it up into batches of training data 
batch_size = 16

#feeding the datasets to the loader, the sampler list a pytorch object created from a list of indexes that specific what samples will be loaded into that loader
train_loader = torch.utils.data.DataLoader(train_sample, batch_size, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_sample, batch_size, pin_memory=True)

#init the early stopping class that will be used to stop training after a specified number of epcohs
#without inprovment to the f1 score
class Early_stopping_loss():
    def __init__(self, patience):
        self._patience = patience
        self.current_patience = 0
        self.min_loss = float('inf')
        self.val_accuracy = None
        self.min_state_dict = None
        self.train_loss = None
        self.train_accuracy = None
        self.epoch = 0

    def stopper(self, current_model,  val_loss, train_loss, val_accuracy, train_accuracy):
        self.epoch += 1
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.current_patience = 0
            self.train_loss = train_loss
            self.min_state_dict = current_model.state_dict()
            self.val_accuracy = val_accuracy
            self.train_accuracy = train_accuracy

        else:
            self.current_patience += 1
            
        if self.current_patience >= self._patience:
            return True
        else:
            return False
        
    def get_state_dict(self):
        return self.min_state_dict  
    
    def get_current_patience(self):
        return self.current_patience
    
    def get_min_loss(self):
        return self.min_loss
    
    def get_kept_epoch(self):
        return self.epoch - self._patience
    
    def get_final_stats(self):
        return self.epoch - self._patience, self.train_loss, self.min_loss, self.train_accuracy, self.val_accuracy
    

#setting up the model 
class CNN(nn.Module):
    def __init__(self, num_classes, l2_reg=0):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.l2_reg = l2_reg

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
model = CNN(len(dataset.classes), l2_reg=.00001)
model.to(device)

#loss function, contains softmax already 
loss_func = nn.CrossEntropyLoss()

#optimizer 
#adding l2 regulatization in weigh decay parameter 
optimizer = torch.optim.Adam(model.parameters(), weight_decay=model.l2_reg)

#init early stopper 
early_stopper = Early_stopping_loss(20)

#list to track accuracy and loss 
training_loss_lst = []
training_accuracy_lst = []
val_loss_lst = []
val_acc_lst = []
f1_score_lst = []

#training loop
epoch = 0
while True:
    #set model into training mode 
    model.train()
    #setting running loss, correct, and total varibles for this epoch 
    train_running_loss = 0 
    train_correct_pred = 0
    
    for features, labels in train_loader:
        #setting the device 
        features, labels = features.to(device), labels.to(device)
        #clear optimizer
        optimizer.zero_grad()
        #get y predicted
        train_predicted_labels = model(features)

        #get loss 
        train_batch_loss = loss_func(train_predicted_labels, labels)
        #multipe the loss by the size of the batch to scale it to make the loss 
        #representative of the loss across the entire batch (accounts for batch of different sizes)
        train_running_loss += train_batch_loss.item() * len(labels)

        #calculated direction of gradient
        train_batch_loss.backward()
        #get the new theta
        optimizer.step()

        #getting accuracy 
        train_predicted_arg = torch.argmax(train_predicted_labels, dim=1)
        train_correct_pred += (train_predicted_arg == labels).sum().item()

    #calculating the loss for this epoch
    train_loss = train_running_loss / len(train_loader.dataset)
    #getting the accuracy for this epoch 
    training_accuracy = train_correct_pred / len(train_loader.dataset)

    #printing status of loss and accuracy to consle
    epoch += 1
    print(f"Epoch {epoch}:")
    print(f'Training: Loss = {train_loss}, Accuracy = {training_accuracy}')

    #evaluating the model 
    model.eval()
    #init all varibles to track accuracy and loss 
    val_running_loss = 0 
    val_correct_pred = 0

    #these will be used to calculate the f1 score for this current epoch on the training data
    cul_predictions = torch.tensor([]).to(device)
    cul_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        for features, labels in val_loader:
            #setting to run on cuda 
            features, labels = features.to(device), labels.to(device)
            #calculating predicted 
            val_predicted_labels = model(features)

            #getting loss from labels 
            val_batch_loss = loss_func(val_predicted_labels, labels)
            val_running_loss += val_batch_loss.item() * len(labels)

            #getting accuracy 
            val_predicted_arg = torch.argmax(val_predicted_labels, dim=1)
            val_correct_pred += (val_predicted_arg == labels).sum().item()

            #getting cul prediction and labels to use for f1
            cul_predictions = torch.cat((cul_predictions, val_predicted_arg))
            cul_labels = torch.cat((cul_labels, labels))
            
        #getting loss 
        val_loss = val_running_loss / len(val_loader.dataset)
        #getting accuracy 
        val_accuracy = val_correct_pred / len(val_loader.dataset)

        #getting f1 score 
        #average = 'macro' calculates the average f1 scored for each class and averages them 
        #beta > 1 shifts the f1 score to be more in favor or recall while beta < 1 makes it favored towards precision 
        f1 = skm.fbeta_score(cul_predictions.cpu().numpy(), cul_labels.cpu().numpy(), average="weighted", beta=2)
    
    #printing status to consel
    print(f"Val: Loss = {val_loss}, Accuracy = {val_accuracy}, F-1 Score = {f1}")
    

    #adding loss, accuracy, and f1 score to the lists that will be used to produce a plot of the progress of the model 
    training_loss_lst.append(train_loss)
    training_accuracy_lst.append(training_accuracy)
    val_loss_lst.append(val_loss)
    val_acc_lst.append(val_accuracy)
    f1_score_lst.append(float(f1))

    #earlier stopper condition check to see if the pacients of the model has run out 
    #if it has revert model to highest f1 
    if early_stopper.stopper(model, val_loss, train_loss, val_accuracy, training_accuracy):
        state_dict = early_stopper.get_state_dict()
        model.load_state_dict(state_dict)
        kept_epoch, final_train_loss, final_val_loss, final_train_accuracy, final_val_accuracy = early_stopper.get_final_stats()
        print()
        print(f'Kept Epoch: {kept_epoch}, Train Loss: {final_train_loss}, Val Loss: {final_val_loss}, Train Accuracy: {final_train_accuracy}, Val Accuracy: {final_val_accuracy}')
        break
    else:
        print(f'Current Patience: {early_stopper.get_current_patience()}')
        print()

torch.save(model.state_dict(), r'model_results\model.pth')

with open(r'model_results\16_batch_data\training_loss.txt', 'w+') as file:
    file.write(str(training_loss_lst))
with open(r'model_results\16_batch_data\training_accuracy.txt', 'w+') as file:
    file.write(str(training_accuracy_lst))
with open(r'model_results\16_batch_data\val_loss.txt', 'w+') as file:
    file.write(str(val_loss_lst))
with open(r'model_results\16_batch_data\val_accuracy.txt', 'w+') as file:
    file.write(str(val_acc_lst))
with open(r'model_results\16_batch_data\f1_score.txt', 'w+') as file:
    file.write(str(f1_score_lst))
with open(rf'model_results\kept_model_stats.txt', 'w+') as file:
    file.write(str(f'Kept Model, Epoch: {kept_epoch}, Train Loss: {final_train_loss}, Val Loss: {final_val_loss}, Train Accuracy: {final_train_accuracy}, Val Accuracy: {final_val_accuracy}'))