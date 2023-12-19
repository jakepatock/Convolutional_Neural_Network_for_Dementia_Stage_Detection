import ast
import numpy as np

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\test_accuracy.txt', 'r') as file:
    content = file.read()
    test_accruacy = ast.literal_eval(content)

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\test_loss.txt', 'r') as file:
    content = file.read()
    test_loss = ast.literal_eval(content)

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\training_accuracy.txt', 'r') as file:
    content = file.read()
    train_accruacy = ast.literal_eval(content)

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\training_loss.txt', 'r') as file:
    content = file.read()
    train_loss = ast.literal_eval(content)

with open(r'C:\Users\1234z\Desktop\Jakes Stuff\Data\f1_score.txt', 'r') as file:
    content = file.read()
    f1_score = ast.literal_eval(content)

arr = np.array(f1_score)
print(np.argmax(arr))