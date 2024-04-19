import pickle
import torch
import numpy as np
from torchvision import transforms
import random
from train import QuadraticKappaLoss
clip = QuadraticKappaLoss.clip

# results = pickle.load(open("results.obj", "rb"))
# labels = pickle.load(open("labels.obj", "rb"))
# print(len(labels)), print(len(results))

# An example with a single image
def func_run(logits):
    # results has all the images tiled into patches
    sz = 256
    N = 16
    img_res = [results[0][i]["img"] for i in range(N)]
    img_res_tensor = torch.tensor(np.array(img_res)).view(
        (N, 3, sz, sz)
    )  # torch.Size([16, 3, 128, 128])


    # Define transformation to apply to the image
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Resize to match the input size expected by the model
        ]
    )

    # Apply transformation to the image
    image_tensor = transform(img_res)


    model = torch.load("ensemble.pth")

    with model.eval():
        out = model(image_tensor.float())

## Do for all images
# create a tensor of all the patches for all the images
# pass through the model
# get the output
# create a tensor of all the patches for all the images using results
def func():
    images = []
    for i in range(len(results)):
        img_res = [results[i][j]["img"] for j in range(N)]
        img_res_tensor = torch.tensor(np.array(img_res)).view(
            (N, 3, sz, sz)
        )  # torch.Size([16, 3, 128, 128])
        images.append(img_res_tensor)

    images = torch.stack(images)
    output = model(images.float())
    print(output)

        
def softmax(logits):
    return torch.softmax(clip(logits), dim=1)
