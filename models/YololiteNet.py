import torch.nn as nn


class yoloLite(nn.Module):

    def __init__(self, classes, bbox=2):
        super(yoloLite, self).__init__()

        self.C1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.MP1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.C2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.MP2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.C3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.MP3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.C4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.MP4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.C5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.MP5 = nn.MaxPool2d(kernel_size=(2, 2))
        self.C6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.C7 = nn.Conv2d(in_channels=256, out_channels=bbox * 5 + classes, kernel_size=(1, 1))


    def __call__(self, x):
        out = self.MP1(nn.functional.relu(self.C1(x)))
        out = self.MP2(nn.functional.relu(self.C2(out)))
        out = self.MP3(nn.functional.relu(self.C3(out)))
        out = self.MP4(nn.functional.relu(self.C4(out)))
        out = self.MP5(nn.functional.relu(self.C5(out)))
        out = nn.functional.relu(self.C6(out))
        out = self.C7(out)

        return out
