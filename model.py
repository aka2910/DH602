import torch
import torchvision.models as models
import torch.nn as nn

N = 16


def build_model_feature(
    pretrained=True,
    fine_tune=True,
):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
    else:
        print("[INFO]: Not loading pre-trained weights")
    model = models.efficientnet_b6()
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
        self.fc = nn.Linear(1280, num_classes)
        self.feature_layer = None

    def forward(self, x):
        x = self.model(x)
        # group by 16 and take the mean
        x = x.view(-1, N, x.size(1))
        x = torch.mean(x, dim=1)
        # add classifier
        self.feature_layer = x
        x = self.fc(x)  # TODO
        return x


class ensemble(nn.Module):
    def __init__(self, num_classes, model1, model2):
        super(ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(3 * num_classes, num_classes)

    def forward(self, x):
        self.model1(x)
        self.model2(x)
        x1 = self.model1.feature_layer
        x2 = self.model2.feature_layer
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
