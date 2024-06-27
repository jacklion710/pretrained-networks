from torchvision import models

# print(dir(models))

# alexnet = models.AlexNet()

resnet = models.resnet101(pretrained=True)
print(resnet)
