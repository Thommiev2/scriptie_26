import torch

print(torch.cuda.is_available())

with open('dataset/interviews/ground truth/verzorgings-staat.txt') as f:
    with open('dataset/interviews/ground truth/verzorgings-staat.txt') as f_w:
        for line in f:
            print(line)
