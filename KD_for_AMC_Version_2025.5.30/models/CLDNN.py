import torch.nn as nn

class CLDNN(nn.Module):
    def __init__(self, n_classes):
        super(CLDNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 256, (1,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (2,3), padding=(0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 80, (1,3), padding='same'),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 80, (1,3), padding='same'),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=80, hidden_size=50, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(6400, 128),   
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1, 80)
        x, _ = self.lstm(x)  # [b, 128, 50]

        x = x.reshape(x.size(0), -1)
        features = x
        out = self.fc(x)

        return features, out
        # return x




# model = CLDNN(n_classes=11)
# # 计算模型参数总数
# num_params = sum(p.numel() for p in model.parameters())
# # 计算参数占用的内存空间（MB）
# total_size = num_params * 4 / (1024 ** 2)
# print(f"total param: {num_params}")
# print(f"Total Model Parameters Size: {total_size:.2f} MB")