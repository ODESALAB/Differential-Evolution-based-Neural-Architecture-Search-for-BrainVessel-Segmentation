import glob
import pickle
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.losses import *
from utils.metrics import *
from torch.utils.data import DataLoader
from utils.drive_dataset import CustomImageDataset
from utils.save_best_model import BestModelCheckPoint

def getBestModelNumbers():
    result = []
    for file in glob.glob("results/*.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)
            result.append((data.fitness, data.cost, data.solNo))

    return sorted(result, key=lambda x: x[0])[-5:]

def readPickleFile(file):
    with open(f"results/model_{file}.pkl", "rb") as f:
        data = pickle.load(f)
    
    return data

from torchinfo import summary

# test_images.txt dosyalarını değiştirdin UNUTMA!!!
seed = 0
modelNo = 404

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

checkpoint = BestModelCheckPoint(modelNo)
device = torch.device('cuda')

model = readPickleFile(modelNo)
#model.reset()
print("Model No:", model.solNo, "Seed:", seed)
summary(model, input_size=(2, 3, 512, 512))

model.to(device)

dataset = CustomImageDataset(mode='train', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"))
val_dataset = CustomImageDataset(mode='val', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"))
test_dataset = CustomImageDataset(mode='test', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"))

print("Dataset:", dataset.__len__())

train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss = DiceLoss()
metric = DiceCoef()
iou_metric = IoU()

log = ""

for epoch in range(200): 
  train_loss = []
  train_dice = []
  
  # Train Phase
  model.train()
  for inputs, labels in train_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.set_grad_enabled(True):
      output = model(inputs)
      error = loss(output, labels)
      train_loss.append(error.item())
      train_dice.append(metric(output, labels).item())
      optimizer.zero_grad()
      error.backward()
      optimizer.step()

  # Validation Phase
  val_loss = []
  val_dice = []
  model.eval()
  for inputs, labels in val_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    error = loss(output, labels)
    val_loss.append(error.item())
    val_dice.append(metric(output, labels).item())
    
  avg_tr_loss = sum(train_loss) / len(train_loss)
  avg_tr_score = sum(train_dice) / len(train_dice)
  avg_val_loss = sum(val_loss) / len(val_loss)
  avg_val_score = sum(val_dice) / len(val_dice)
  txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_dice_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_dice: {avg_val_score}"
  log += txt
  print(txt)
  checkpoint.check(avg_val_score, model, seed)

# Get Best Model
print("Load Model...")
model.load_state_dict(torch.load(f"model_{modelNo}_seed_{seed}.pt"))
model.to(device)

# Testing
test_loss = []
test_dice = []
test_iou = []

for m in model.modules():
    for child in m.children():
        if type(child) == nn.BatchNorm2d:
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None
model.eval()

for inputs, labels in test_dataloader:
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        error = loss(output, labels)
        test_dice.append(metric(output, labels).item())
        test_iou.append(iou_metric(output, labels).item())
        test_loss.append(error.item())

log += f"\ntest_loss: {sum(test_loss) / len(test_loss)}, test_dice: {sum(test_dice) / len(test_dice)}, test_iou: {sum(test_iou) / len(test_iou)}"
print(f"test_loss: {sum(test_loss) / len(test_loss)}, test_dice: {sum(test_dice) / len(test_dice)}, test_iou: {sum(test_iou) / len(test_iou)}")

# Write Log
with open(f"log_{modelNo}_seed_{seed}.txt", "w") as f:
    f.write(log)

torch.cuda.empty_cache()