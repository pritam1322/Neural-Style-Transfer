import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

img_size = 512
img = "San-diego-california-beach-sunset-wallpaper.jpg"
style = "style/d15c7fbb8d6960762356a94aa6a27ca4--modern-art-paintings-original-paintings.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose(
    [
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

unloader = transforms.Compose([transforms.ToPILImage()])

def pre_processing(img_name):
    img = Image.open(img_name)
    img = loader(img).unsqueeze(0)
    return img.to(device,torch.float)

def imshow(img):
    image = img.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)

content_img = pre_processing(img)
style_img = pre_processing(style)

"""
plt.figure()
imshow(content_img)

plt.figure()
imshow(style_img)
"""

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = models.vgg16(pretrained=True).features[:29]
        self.feature = ['0', '5', '10', '19', '28']

    def forward(self,x):
        features = []
        for layer_num,layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.feature:
                features.append(x)

        return features


model = VGG().to(device).eval()
generated = content_img.clone().requires_grad_(True)

#Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)




for step in range(total_steps):
    generated_features = model(generated)
    content_img_features = model(content_img)
    style_img_features = model(style_img)

    style_loss = content_img_loss  = 0

    for generated_feature, content_img_feature, style_img_feature in zip(generated_features, content_img_features, style_img_features):

        b,c,h,w = generated_feature.shape
        content_img_loss += torch.mean((generated_feature - content_img_feature)**2)

        G = generated_feature.view(c,h*w).mm(generated_feature.view(c,h*w).t())
        A = style_img_feature.view(c,h*w).mm(style_img_feature.view(c,h*w).t())




        style_loss += torch.mean((G-A)**2)

    total_loss = alpha * content_img_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")















