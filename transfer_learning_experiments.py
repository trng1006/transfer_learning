import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'CIFAR10_Fake_Real'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=10, unfreeze_at=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Gradual unfreezing logic
        if unfreeze_at is not None and epoch == unfreeze_at:
            print("Unfreezing all layers...")
            for param in model.parameters():
                param.requires_grad = True
            # Re-initialize optimizer to include all parameters
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, time_elapsed, best_acc.item(), history

def run_experiment(strategy_name):
    print(f"\n{'='*20}")
    print(f"Running Strategy: {strategy_name}")
    print(f"{'='*20}")
    
    model_ft = models.resnet18(weights='DEFAULT')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    unfreeze_at = None
    if strategy_name == "Freeze Backbone":
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
    elif strategy_name == "Fine-tune All":
        for param in model_ft.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    elif strategy_name == "Gradual Unfreeze":
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.fc.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
        unfreeze_at = 5  # Unfreeze at epoch 5
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model, duration, best_acc, history = train_model(model_ft, criterion, optimizer, 
                                                     exp_lr_scheduler, num_epochs=10, 
                                                     unfreeze_at=unfreeze_at)
    return duration, best_acc, history

if __name__ == "__main__":
    results = {}
    strategies = ["Freeze Backbone", "Fine-tune All", "Gradual Unfreeze"]
    
    for strategy in strategies:
        duration, best_acc, history = run_experiment(strategy)
        results[strategy] = {
            'duration': duration,
            'best_acc': best_acc,
            'history': history
        }
    
    # Save results summary
    with open('results_summary.txt', 'w') as f:
        f.write("Strategy | Duration (s) | Best Val Acc\n")
        f.write("-" * 40 + "\n")
        for s, r in results.items():
            f.write(f"{s} | {r['duration']:.2f} | {r['best_acc']:.4f}\n")
    
    print("\nAll experiments completed. Summary saved to results_summary.txt")
