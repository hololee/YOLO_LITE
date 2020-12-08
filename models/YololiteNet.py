import torch
import torch.nn as nn


class yoloLite(nn.Module):

    def __init__(self, classes, bbox=2):
        super(yoloLite, self).__init__()

        self.C1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.C11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.CB1 = nn.BatchNorm2d(num_features=32)
        self.CB11 = nn.BatchNorm2d(num_features=32)
        self.MP1 = nn.MaxPool2d(kernel_size=(2, 2)) # 256
        self.C2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.C21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.CB2 = nn.BatchNorm2d(num_features=64)
        self.CB21 = nn.BatchNorm2d(num_features=64)
        self.MP2 = nn.MaxPool2d(kernel_size=(2, 2)) # 128
        self.C3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.C31 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.CB3 = nn.BatchNorm2d(num_features=128)
        self.CB31 = nn.BatchNorm2d(num_features=128)
        self.MP3 = nn.MaxPool2d(kernel_size=(2, 2)) #  64
        self.C4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.C41 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.CB4 = nn.BatchNorm2d(num_features=256)
        self.CB41 = nn.BatchNorm2d(num_features=256)
        self.MP4 = nn.MaxPool2d(kernel_size=(2, 2)) # 32
        self.C5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.C51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.CB5 = nn.BatchNorm2d(num_features=512)
        self.CB51 = nn.BatchNorm2d(num_features=512)
        self.MP5 = nn.MaxPool2d(kernel_size=(2, 2)) # 16
        self.C6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.C61 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.CB6 = nn.BatchNorm2d(num_features=256)
        self.CB61 = nn.BatchNorm2d(num_features=256)
        self.MP6 = nn.MaxPool2d(kernel_size=(2, 2))  # 8
        self.C7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding = 1)
        self.C71 = nn.Conv2d(in_channels=256, out_channels=bbox * 5 + classes, kernel_size=(1, 1))
        self.CB7 = nn.BatchNorm2d(num_features=256)

    def __call__(self, x):
        out = self.MP1(nn.functional.relu(self.CB11(self.C11(nn.functional.relu(self.CB1(self.C1(x)))))))
        out = self.MP2(nn.functional.relu(self.CB21(self.C21(nn.functional.relu(self.CB2(self.C2(out)))))))
        out = self.MP3(nn.functional.relu(self.CB31(self.C31(nn.functional.relu(self.CB3(self.C3(out)))))))
        out = self.MP4(nn.functional.relu(self.CB41(self.C41(nn.functional.relu(self.CB4(self.C4(out)))))))
        out = self.MP5(nn.functional.relu(self.CB51(self.C51(nn.functional.relu(self.CB5(self.C5(out)))))))
        out = self.MP6(nn.functional.relu(self.CB61(self.C61(nn.functional.relu(self.CB6(self.C6(out)))))))
        out = torch.sigmoid(self.C71(nn.functional.relu(self.CB7(self.C7(out)))))

        return out
