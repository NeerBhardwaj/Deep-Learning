import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
scheduler_step_size = 10
scheduler_gamma = 0.1

transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           transform=transforms,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./',
                                          train =False,
                                          transform=transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=scheduler_step_size,
                                            gamma=scheduler_gamma)

total_step = len(train_loader)
start_time = time.time()

for epoch in range(num_epochs):
    scheduler.step()
    correct = 0
    total = 0
    for images, labels in train_loader:
        
        images = images.reshape(-1,28*28)
        labels = labels
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_accuracy = correct/total
    with torch.no_grad():
        correct =0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1,28*28)
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct/total
    print ('Epoch {}, Time{:.4f}, Loss: {:.4f}, Train Accuarcy: {:.4f},Test Accuarcy :{:.4f}'
           .format(epoch, time.time()-start_time, loss.item(), train_accuracy, test_accuracy))
    torch.save(model.state_dict(), 'epoch-{}.ckpt'.format(epoch))
