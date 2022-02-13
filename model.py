import torch
import torch.nn as nn
import torchvision.models as models


def ResNet18(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18


def ResNet50(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet50 = models.resnet50(pretrained=True).cuda()
    resnet50.fc = nn.Linear(2048, num_classes).cuda()
    return resnet50
