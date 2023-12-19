import torch 
import torchvision as tv

def rgbimage_dataset_normalization(dataset_tensor):
    channel_sum = torch.zeros(3)
    for image, label in dataset_tensor:
        channel_sum += torch.sum(image, dim=(1,2))
    mean = channel_sum / len(dataset_tensor)

    std_sum = torch.zeros(3)
    for image, label in dataset_tensor:
        std_sum += (torch.sum(image, dim=(1,2)) - mean)**2
    std = torch.sqrt(torch.div(std_sum, len(dataset_tensor)))
    
    return mean, std

class Early_stopping_loss():
    def __init__(self, patience):
        self._patience = patience
        self.current_patience = 0
        self.min_loss = float('inf')
        self.min_state_dict = None

    def stopper(self, current_model,  test_loss):
        if test_loss < self.min_loss:
            self.min_loss = test_loss
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
    
class Early_stopping_F1():
    def __init__(self, patience):
        self._patience = patience
        self.current_patience = 0
        self.max_f1 = float('-inf')
        self.min_state_dict = None

    def stopper(self, current_model, f1):
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
    
def standardize_numpy(X):
    """
    X is a 2D Numpy array that each row is a feature vector of a data point (columns are features)
    Returns a standardized 2D Numpy array 
    """
    #cacluting mean along the first axis 
    feature_mean = X.mean(0)
    #calculating the mean along the first axis 
    feature_std = X.std(0)
    #standardize the features
    standard_X = (X - feature_mean) / feature_std
    return standard_X

def standardize_tensor(X):
    feature_mean = torch.mean(X, dim=0)
    feature_std = torch.std(X, dim=0)
    return (X - feature_mean) / feature_std

class IndividualNormalization(object):
    def __call__(self, image):
        image_mean = image.mean()
        image_stdev = image.std()
        return tv.transforms.Normalize(mean=[image_mean], std=[image_stdev])(image)