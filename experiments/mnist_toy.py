import torch
import torchvision
import torch.optim as optim
import os
import numpy as np
import pickle as pkl
import torch.nn.functional as F
from visualization.images import plot_mnist
from models.image_recognition import MnistClassifier
from explainers.corpus import Corpus
from utils.schedulers import ExponentialScheduler


def train_model(n_epoch: int = 10, batch_size_train: int = 64, batch_size_test: int = 1000,
                random_seed: int = 42, learning_rate=0.01, momentum=0.5, log_interval=100,
                save_path='./experiments/results/mnist_toy/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

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
    test_counter = [i * len(train_loader.dataset) for i in range(n_epoch + 1)]

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

    test()
    for epoch in range(1, n_epoch + 1):
        train(epoch)
        test()


def fit_corpus(n_epoch=10000, corpus_size=1000, test_size=100, learning_rate=100.0, momentum=0.5,
               save_path='./experiments/results/mnist_toy/', random_seed: int = 40,
               fraction_keep=0.005, reg_factor_init=0.1, reg_factor_final=1000,
               ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
    classifier.to(device)
    classifier.eval()
    corpus_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=corpus_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=test_size, shuffle=True)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test, (example_data, example_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device)
    corpus_target = corpus_target.to(device)
    example_data = example_data.to(device)
    example_targets = example_targets.to(device)

    reg_factor_scheduler = ExponentialScheduler(reg_factor_init, reg_factor_final, n_epoch)
    corpus = Corpus(examples=corpus_data.detach().cpu().numpy(),
                    latent_reps=(classifier.latent_representation(corpus_data)).detach().cpu().numpy())
    weights = corpus.fit(classifier.latent_representation(example_data).detach(), n_epoch=n_epoch,
                         learning_rate=learning_rate, momentum=momentum, reg_factor=reg_factor_init,
                         fraction_keep=fraction_keep, reg_factor_scheduler=reg_factor_scheduler)
    corpus.plot_hist()
    corpus_path = os.path.join(save_path, 'corpus.pkl')
    with open(corpus_path, 'wb') as f:
        print(f'Saving the corpus decomposition in {corpus_path}.')
        pkl.dump(corpus, f)


def plot_decomposition(test_id=42, load_path='./experiments/results/mnist_toy/', random_seed=40, test_size=100):

    # There is a problem here in matching the random seeds:
    # find a way to recover the same test example as in the above function

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=test_size, shuffle=True)
    test_examples = enumerate(test_loader)
    batch_id_test, (example_data, example_targets) = next(test_examples)
    example_data = example_data.to(device)
    example_targets = example_targets.to(device)
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(load_path, 'model.pth')))
    classifier.to(device)
    classifier.eval()
    corpus_path = os.path.join(load_path, 'corpus.pkl')
    with open(corpus_path, 'rb') as f:
        corpus = pkl.load(f)
    decomposition = corpus.decompose(test_id)
    output = classifier(example_data)
    print(torch.exp(output[test_id]))
    title = f'Prediction: {output.data.max(1, keepdim=True)[1][test_id].item()}'
    plot_mnist(example_data[test_id][0].cpu().numpy(), title)
    plot_mnist(decomposition[0][1][0], title=str(decomposition[0][0]))
    plot_mnist(decomposition[1][1][0], title=str(decomposition[1][0]))
    plot_mnist(decomposition[2][1][0], title=str(decomposition[2][0]))
    plot_mnist(decomposition[3][1][0], title=str(decomposition[3][0]))
    plot_mnist(decomposition[4][1][0], title=str(decomposition[4][0]))


#fit_corpus()
plot_decomposition()