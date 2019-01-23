import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    """
    Model modified.
    """
    def __init__(self, n_class):
        super(DenseNet121, self).__init__()

        # 4-channel to 3-channel transformation:
        self.conv_block = [
            nn.Conv2d(4, 3, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        ]
        self.conv_block = nn.Sequential(*self.conv_block)

        self.model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
                    nn.Linear(num_ftrs*4, n_class),
                    nn.Sigmoid()
                )

    def forward(self, x):
        return self.model(self.conv_block(x))
        