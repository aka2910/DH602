import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import EfficientNet_B6_Weights

N = 16
FEATURE_SIZE = 2304


def build_model_feature(
    pretrained=True,
    fine_tune=True,
):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
    else:
        print("[INFO]: Not loading pre-trained weights")
    model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.classifier = nn.Identity()
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model


class efficienet_pool(nn.Module):
    def __init__(self, num_classes=3):
        # once this is 3 and then it is 6
        super(efficienet_pool, self).__init__()
        self.model = build_model_feature()
        self.num_classes = num_classes
        self.fc = nn.Linear(FEATURE_SIZE, num_classes)
        self.feature_layer = None

    def forward(self, x):
        x = self.model(x)
        # group by 16 and take the mean
        x = x.view(-1, N, x.size(1))
        x = torch.mean(x, dim=1)
        # add classifier
        self.feature_layer = x
        x = self.fc(x)
        return x
    
    def forward_infer(self, x):
        x = self.model(x)
        # group by 16 and take the mean
        x = x.view(-1, N, x.size(1))
        x = torch.mean(x, dim=1)
        return x


class ensemble(nn.Module):
    def __init__(self, num_classes, model1, model2):
        super(ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(FEATURE_SIZE * 2, num_classes)

    def forward(self, x):
        x1 = self.model1.forward_infer(x)
        x2 = self.model2.forward_infer(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model1 = efficienet_pool(3)
    model2 = efficienet_pool(6)
    model = ensemble(6,model1,model2)