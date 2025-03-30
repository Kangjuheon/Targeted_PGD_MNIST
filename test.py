import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 훈련 함수
def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# PGD Targeted 공격 함수
def pgd_targeted(model, x, target, k, eps, eps_step):
    x_orig = x.clone().detach().to(device)
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    for _ in range(k):
        output = model(x_adv)
        loss = F.cross_entropy(output, target.to(device))
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv - eps_step * grad_sign
        eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + eta
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
        x_adv.requires_grad = True

    return x_adv.detach()

# 공격 성공률 측정 함수
def test_pgd_targeted_success_rate(model, test_loader, k=10, eps=0.3, eps_step=0.03, fixed_target=0):
    model.eval()
    total = 0
    success = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        mask = label != fixed_target
        if mask.sum() == 0:
            continue

        data = data[mask]
        target = torch.full_like(label[mask], fixed_target).to(device)

        data_adv = pgd_targeted(model, data, target, k, eps, eps_step)
        output_adv = model(data_adv)
        pred_adv = output_adv.argmax(dim=1)

        success += pred_adv.eq(target).sum().item()
        total += len(data)

    rate = 100 * success / total if total > 0 else 0
    print(f"Target: {fixed_target}, Success Rate: {rate:.2f}%")
    return rate

# 실행부
if __name__ == "__main__":
    # 데이터 로딩
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 모델 훈련
    model = SimpleCNN().to(device)
    train(model, train_loader, epochs=3)

    # PGD 공격 파라미터
    k = 10
    eps = 0.3
    eps_step = 0.03

    # 0~9 각 타겟에 대해 성공률 출력
    print("\n[PGD Targeted Attack Success Rate by Target Class]")
    for target_class in range(10):
        test_pgd_targeted_success_rate(model, test_loader, k, eps, eps_step, fixed_target=target_class)
