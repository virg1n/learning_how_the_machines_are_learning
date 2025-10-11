import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset



class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            nn.Conv2d(3, 1, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d((2,2)),
            torch.nn.Flatten(),
            torch.nn.Linear(121, 10)
        )
    def forward(self, x):
        return self.main(x)
    
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


indices_train = list(range(4000))
indices_test = list(range(4000)) 


train_dataset = Subset(train_dataset, indices_train)
test_dataset = Subset(test_dataset, indices_test)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = CNN(1)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

for epoch in range(5):
    loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outs = model(inputs)
        loss = F.cross_entropy(outs, labels)
        loss.backward()
        optimizer.step()
    
    print(123)
    
correct = 0
total = 0
model.eval() 
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print("END:")
print(correct/total)

