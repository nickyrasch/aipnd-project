import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)

    if chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized")
        return

    model.class_to_idx = chpt['class_to_idx']

    # Create the classifier
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(4096, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )

    # Put the classifier on the pretrained network
    model.classifier = classifier

    model.load_state_dict(chpt['state_dict'])

    return model