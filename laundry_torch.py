import torch
import torch.nn as nn
import torch.optim as optim

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)  # CrossEntropyLoss 계산
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# CNN Branch
class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjust size as per input image
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return x

# FNN Branch
class FNNBranch(nn.Module):
    def __init__(self):
        super(FNNBranch, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# Hybrid Model (CNN + FNN)
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn_branch = CNNBranch()
        self.fnn_branch = FNNBranch()
        self.fc = nn.Linear(128 + 32, 4)  # Combine CNN (128) and FNN (32)

    def forward(self, image, weight):
        cnn_out = self.cnn_branch(image)
        fnn_out = self.fnn_branch(weight)
        combined = torch.cat((cnn_out, fnn_out), dim=1)  # Concatenate features
        output = self.fc(combined)
        return output

# 모델, 손실 함수, 최적화 정의
model = HybridModel()
focal_loss = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
l2_lambda = 0.01  # L2 Regularization 강도

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(image_data, weight_data)

    # Focal Loss 계산
    loss = focal_loss(outputs, labels)

    # L2 정규화 추가
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    loss += l2_lambda * l2_reg

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
