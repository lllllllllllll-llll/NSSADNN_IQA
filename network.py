import torch
import torch.nn as nn
import torch.nn.functional as F

class NSSADNN(nn.Module):
    def __init__(self):
        super(NSSADNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, stride=1, padding=1)

        self.fc1_1 = nn.Linear(2025, 1024)
        self.fc2_1 = nn.Linear(1024, 36)

        self.fc1_2 = nn.Linear(2025, 1024)
        self.fc2_2 = nn.Linear(2048, 1024)
        self.fc3_2 = nn.Linear(1024, 1)


    def forward(self, x):
        # print('x', x.size())#x torch.Size([128, 1, 32, 32])
        x_distort = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # print('input size:' + str(x_distort.size()))#input size:torch.Size([1, 1, 32, 32])

        x1 = F.relu(self.conv1(x_distort))
        # print('x1:' + str(x1.size()))#x1:torch.Size([1, 50, 28, 28])
        x2 = F.relu(self.conv2(x1))
        # print('x2:' + str(x2.size()))#x2:torch.Size([1, 50, 26, 26])
        x3 = F.relu(self.conv3(x2))
        # print('x3:' + str(x3.size()))#x3:torch.Size([1, 50, 26, 26])

        x3 = torch.add(x3, x2)
        # print('x3:' + str(x3.size()))#x3:torch.Size([1, 50, 26, 26])
        x3 = F.max_pool2d(x3, (2, 2), stride=2)
        # print('x3:' + str(x3.size()))#x3:torch.Size([1, 50, 13, 13])

        x4 = F.relu(self.conv4(x3))
        # print('x4:' + str(x4.size()))#x4:torch.Size([1, 25, 11, 11])
        x5 = F.relu(self.conv5(x4))
        # print('x5:' + str(x5.size()))#x5:torch.Size([1, 25, 9, 9])
        x6 = F.relu(self.conv6(x5))
        # print('x6:' + str(x6.size()))#x6:torch.Size([1, 25, 9, 9])

        x6 = torch.add(x6, x5)

        fc = x6.view(-1, self.num_flat_features(x6))
        # print('fc', fc.size())#fc torch.Size([1, 2025])

        fc1_1 = self.fc1_1(fc)
        # print('fc1_1', fc1_1.size())#fc1_1 torch.Size([1, 1024])
        fc2_1 = self.fc2_1(fc1_1)
        # print('fc2_1', fc2_1.size())#fc2_1 torch.Size([1, 36])

        fc1_2 = self.fc1_2(fc)
        # print('fc1_2', fc1_2.size())#fc1_2 torch.Size([1, 1024])
        cat = F.relu(torch.cat((fc1_2, fc1_1), 1))
        # print('cat', cat.size())#cat torch.Size([1, 2048])
        fc2_2 = self.fc2_2(cat)
        quality = self.fc3_2(fc2_2)

        # print('quality', quality.size())#quality torch.Size([128, 1])
        # print('fc2_1', fc2_1.size())#torch.Size([128, 36])

        return quality, fc2_1


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features















