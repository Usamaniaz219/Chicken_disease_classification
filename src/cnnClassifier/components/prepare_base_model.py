import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models 
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def get_base_model(self):
        self.model = models.vgg16(pretrained=True if self.config.params_weights == "imagenet" else False)
        
        # If include_top is False, remove the classifier
        if not self.config.params_include_top:
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # Removes classifier
        
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, learning_rate):
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # If the model has no classifier (params_include_top=False), add one
        if isinstance(model, nn.Sequential):
            model = nn.Sequential(
                model,  # Keep the base model
                nn.Flatten(),  # Flatten output for fully connected layers
                nn.Linear(512 * 7 * 7, 4096),  # VGG feature map size
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, classes),  # Final classification layer
                nn.Softmax(dim=1)  # Softmax for classification
            )
        else:
            model.classifier[6] = nn.Linear(in_features=4096, out_features=classes)

        # Define optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion
    
    def update_base_model(self):
        self.Full_model,optimizer,criterion = self._prepare_full_model(model=self.model,
                                                                       classes=self.config.params_classes,learning_rate=self.config.params_learning_rate)
        self.save_model(path=self.config.updated_base_model_path,model=self.Full_model)
    
    @staticmethod
    def save_model(path:Path,model):
        torch.save(model.state_dict(),path)

