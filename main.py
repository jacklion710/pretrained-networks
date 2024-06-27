from torchvision import models
from torchvision.models import ResNet101_Weights # Import the weights directly
from torchvision import transforms
from PIL import Image
import torch

img = Image.open("./data/bobby.jpg")
# img.show()
with open("./data/imagenet_classes.txt") as f: # Load the labels for ImageNet dataset
    labels = [line.strip() for line in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.255]
    )])

img_t = preprocess(img) # Process image through preprocessing pipeline
batch_t = torch.unsqueeze(img_t, 0) # Adds a dimension of size 1 to the tensor to fulfil (batch_size, channels, height, width)

resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1) # `pretrained=True` is deprecated therefor explicitly importing the weights is required

resnet.eval() # Eval mode enabled for inferencing

out = resnet(batch_t)

_, index = torch.max(out, 1) # Get the max value from the out tensor and its index

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
