
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
from DN import models

def vgg16_netvlad(pretrained=False):
    base_model = models.create('vgg16', pretrained=False)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    if pretrained:
        model.load_state_dict('pretrained_model/vgg16_netvlad.pth', map_location=torch.device('cpu'))
    return model

print("Loading model")
model =vgg16_netvlad(pretrained=False)
print("Loaded successfully")
# read image
img = Image.open('sample/image.PNG').convert('RGB')
transformer = transforms.Compose([transforms.Resize((480, 640)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
img = transformer(img)
model = model
img = img
# use GPU (optional)
# model = model.cuda()
# img = img.cuda()

# extract descriptor (4096-dim)
with torch.no_grad():
    print("Procssing images")
    descriptor = model(img.unsqueeze(0))[0]
descriptor = descriptor.numpy()
pd.DataFrame(descriptor).to_csv("result/descriptor.csv")
print("Saved results on CSV")