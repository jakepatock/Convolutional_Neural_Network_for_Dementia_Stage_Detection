import ast
import numpy as np
import torch
import matplotlib
import sklearn as sk
import numpy as np
import torchmetrics
import PIL 
import ast
with open(r'model_results\16_batch_data\test_accuracy.txt', 'r') as file:
    content = file.read()
    test_accruacy = ast.literal_eval(content)

with open(r'model_results\16_batch_data\test_loss.txt', 'r') as file:
    content = file.read()
    test_loss = ast.literal_eval(content)

with open(r'model_results\16_batch_data\training_accuracy.txt', 'r') as file:
    content = file.read()
    train_accruacy = ast.literal_eval(content)

with open(r'model_results\16_batch_data\training_loss.txt', 'r') as file:
    content = file.read()
    train_loss = ast.literal_eval(content)

with open(r'model_results\16_batch_data\f1_score.txt', 'r') as file:
    content = file.read()
    f1_score = ast.literal_eval(content)

arr = np.array(test_accruacy)
idx = np.max(arr)
print(idx)

#print(train_loss[idx])


