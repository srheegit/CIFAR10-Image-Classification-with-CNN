# CIFAR-10 Image Classification with Convolutional Neural Networks

This repository contains code for training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Dataset

We use the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset includes the following classes:

- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is automatically downloaded and transformed using the following steps:
- Images are converted from PIL format to tensors.
- Tensors are normalized to have values in the range [-1, 1].

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
)
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

## Data Loaders

Data Loaders are used to load the training and test datasets in batches.

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)
```

## Model Architecture

The CNN model consists of two convolyutional layers followed by three fully connected layers.

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Training

The model is trained using the CrossEntropyLoss function and the SGD optimizer. The training loop includes forward and backward passes, and the model’s parameters are updated accordingly.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0 and epoch % 4 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{n_total_steps}], loss: {loss.item():.4f}")
```

## Evaluation

The model’s performance is evaluated on the test dataset, and the accuracy is calculated for each class.

```python
with torch.no_grad():
    n_correct, n_samples = 0, 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100. * n_correct / n_samples
    print(f"Accuracy of the network: {acc} %")

    for i in range(10):
        acc = 100. * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc} %")
```

## Evluation Results

After running 20 epochs for training, we run the model against the test data, getting the following accuracies below.

The accuracy of a class means the recall. For example, "accuracy of a deer" is the percentage of deer images that were correctly identified by the model.

- Accuracy of the network: 62.76 %
- Accuracy of plane: 70.8 %
- Accuracy of car: 67.2 %
- Accuracy of bird: 43.5 %
- Accuracy of cat: 44.0 %
- Accuracy of deer: 55.7 %
- Accuracy of dog: 50.1 %
- Accuracy of frog: 72.1 %
- Accuracy of horse: 67.5 %
- Accuracy of ship: 81.3 %
- Accuracy of truck: 75.4 %

## Room for Improvement

Admittedly, the model's accuracy of 62.76% is not satisfactory. As the training took tens of minutes, the model's performance can be improved by using a GPU rather than a CPU we used for the model training above.
We also can add more complexity to the model structure, for example by adding more iterations of conv-layer-followed-by-max-pooling-layer, as well as increasing the number of fully connected layers and/or increasing the number of nuerons in the fully connected layers.
We also could have tried out varying the hyperparameters, such as the learning rate and the sizes of kernels, and we also could have tried out different optimizers. As our computational resources were limited, there are various rooms for improvement.
