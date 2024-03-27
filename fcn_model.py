import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        #encoder
        pool1 = self.features_block1(x)
        pool2 = self.features_block2(pool1)
        pool3 = self.features_block3(pool2)
        pool4 = self.features_block4(pool3)
        pool5 = self.features_block5(pool4)

        #classifier
        class_score = self.classifier(pool5)

        #decoder(upsample using skip connections)
        #first upsampling using the conv transposse 2d layer
        upscore2 = self.upscore2(class_score)
        #second upscoring by adding the skip connectiond from pool4 to the upscore2 layer.
        upscore_pool4 = self.upscore_pool4(upscore2 + self.score_pool4(pool4))
        #final upscoring by adding skip connections from pool3 to upscore_pool4 layer.
        upscore_final = self.upscore_final(upscore_pool4 + self.score_pool3(pool3))

        #trimming the output to get an output image as the same size as input image
        h = x.size()[2]
        w = x.size()[3]

        #calculating the trimming offsets
        trim_h = (upscore_final.size()[2] - h) // 2
        trim_w = (upscore_final.size()[3] - w) // 2
        output = upscore_final[:, :, trim_h:trim_h+h, trim_w+w+trim_w]
        return output
        raise NotImplementedError("Implement the forward method")
