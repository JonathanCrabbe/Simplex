import torch
import torchvision
import torch.optim as optim
import os
import torch.nn.functional as F
from models.image_recognition import MnistClassifier
from visualization.images import plot_mnist



# Set Hyperparameters:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
load_model = True

# Prepare data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# Create save directories
save_path = './results/mnist_toy/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create the model
classifier = MnistClassifier()
classifier.to(device)
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)

# Train the model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(classifier.state_dict(), os.path.join(save_path, 'model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pth'))


def test():
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


if not load_model:
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

classifier = MnistClassifier()
classifier.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
classifier.to(device)
with torch.no_grad():
    test_id = 69
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.to(device)
    example_targets = example_targets.to(device)
    output = classifier(example_data)
    title = f'Prediction: {output.data.max(1, keepdim=True)[1][test_id].item()}'
    plot_mnist(example_data[test_id][0].cpu().numpy(), title)
    print(classifier.latent_representation(example_data)[test_id])

    # Here: try the corpus explainer for a test example
