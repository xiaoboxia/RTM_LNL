import torch.nn as nn

class NewsNet(nn.Module):
    def __init__(self, num_classes=20):
        super(NewsNet, self).__init__()
        #self.avgpool = nn.AdaptiveAvgPool1d(16 * num_classes)
        self.fc1 = nn.Linear(300, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, num_classes)


        # self.fc1 = nn.Linear(50, 40)
        # self.bn1 = nn.BatchNorm1d(40)
        # self.ac = nn.Softsign()
        # self.fc2 = nn.Linear(40, 30)
        # self.bn2 = nn.BatchNorm1d(30)
        # self.fc3 = nn.Linear(30, num_classes)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out