from torch import nn


class FCNetowrk(nn.Module):
    def __init__(self):
        super(FCNetowrk, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(1*28*28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 10),
        )

    def forward(self, x, output_feature=False):
        if output_feature:
            print('*************')
            feature = self.features(x.view(-1, 28*28))
            return self.classifier(feature), feature.detach().cpu().numpy()
        else:
            return self.classifier(self.features(x.view(-1, 28*28)))


