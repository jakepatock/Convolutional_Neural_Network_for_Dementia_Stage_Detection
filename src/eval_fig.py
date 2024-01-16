import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics.functional as tm
import sklearn.metrics as skm
import torchvision as tv 
import torch.utils.data 
import torchvision.transforms as transforms
from PIL import Image
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

x = np.linspace(1, len(train_accruacy), num=len(train_accruacy))


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))

arr = np.array(f1_score)
saved_epoch_idx = np.argmax(arr)

ax1.plot(x, train_accruacy, label="Train Accuracy")
ax1.scatter(saved_epoch_idx + 1, train_accruacy[saved_epoch_idx], color='red', marker='.', zorder=2)
ax1.plot(x, test_accruacy, label="Test Accuracy")
ax1.scatter(saved_epoch_idx + 1, test_accruacy[saved_epoch_idx], color='red', marker='.', zorder=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.1)
ax1.legend()

ax2.plot(x, train_loss, label='Train Loss')
ax2.scatter(saved_epoch_idx + 1, train_loss[saved_epoch_idx], color='red', marker='.', zorder=2)
ax2.plot(x, test_loss, label='Test Loss')
ax2.scatter(saved_epoch_idx + 1, test_loss[saved_epoch_idx], color='red', marker='.', zorder=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel("Loss")
ax2.set_ylim(0, 1)
ax2.legend()

ax1.set_title('Accuracy')
ax2.set_title('Loss')

ax3.plot(x, f1_score)
ax3.scatter(saved_epoch_idx + 1, f1_score[saved_epoch_idx], color='red', marker='.', zorder=2)
ax3.set_ylabel('F1 Score')
ax3.set_xlabel("Epoch")
ax3.set_title('F1 Score')
ax3.set_ylim(0, 1.1)

fig.suptitle('Evaluation Metrics')

plt.subplots_adjust(wspace=0.2)
fig.savefig(r'model_results\plots\CNN_Eval_Plot.png')
plt.show()

