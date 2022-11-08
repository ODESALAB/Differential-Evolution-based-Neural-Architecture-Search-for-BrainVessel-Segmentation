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

# 273 > 568 > 495 > 599 > 434 - rand1 -v1
# 403 > 587 > 239 > 423 > 419 - currenttobest1 -v2
# 446 > 489 > 269 > 640 > 561 - v3
# 518 > 120 > 160 > 378 > 460 - v4
# 42 > 406 > 153 > 459 > 197  - v5
# 598 > 274 > 626 > 664 > 140 - v6
# 224 > 437 > 61 > 261 > 553 - v7
# 630 > 626 > 289 > 723 > 553 - v8
# 332 > 477 > 427 > 351 > 456 - v9
# 442 > 598 > 577 > 341 > 501 - v10
# 302 > 234 > 542 > 370 > 693 > 121 > 605 > 204 > 435 - v12
# 467 > 650 > 456 > 591 > 301 - v13 - DEOP_Cell_Nuclei
# 656 > 493 > 519 > 459 > 215 - v14 - DEOP_Vessel
# 166 > 175 > 406 > 625 > 437 - v15 - DE_Vessel
# 490 > 201 > 588 > 605 > 197 - v16 - DEOP_Vessel_v2
# 330 > 510 > 123 > 347 > 524 - v17 - DEOP w/ Asymmetric Conv - vessel
# 615 > 452 > 334 > 496 > 393 
# 240 > 196 > 477 > 671 > 377 

# 654 > 329 > 167 > 534 > 326 - v20 - DEOP DRIVE
# 194 > 421 > 596 > 564 > 628 - v21 - DEOP DRIVE Operations_v2
# 581 > 479 > 191 > 430 > 369 - v22 - v21 + EMONAS Fitness Function
# 719 > 739 > 767 > 630 > 666 - v23 - v20 + EMONAS Fitness Function
# 725 > 167 > 229 > 411 > 596 - v24 - v20 + CLAHE DRIVE
# 112 > 442 > 91 > 274 > 579 - v25 - v24 + 120 train epoch, 30 patience for earlyStopping
# 414 > 600 > 255 > 225 > 404

#bestSol = getBestModelNumbers() 
#print(bestSol)

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