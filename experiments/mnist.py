import torch
import torchvision
import torch.optim as optim
import os
import pickle as pkl
import torch.nn.functional as F
from visualization.images import plot_mnist
from models.image_recognition import MnistClassifier
from explainers.corpus import Corpus
from explainers.nearest_neighbours import NearNeighLatent
from utils.schedulers import ExponentialScheduler


# Load data
def load_mnist(batch_size: int, train: bool):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=train, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)


# Train model
def train_model(device: torch.device, n_epoch: int = 10, batch_size_train: int = 64, batch_size_test: int = 1000,
                random_seed: int = 42, learning_rate=0.01, momentum=0.5, log_interval=100,
                save_path='./results/mnist/', cv: int = 0):
    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.enabled = False

    # Prepare data
    train_loader = load_mnist(batch_size_train, train=True)
    test_loader = load_mnist(batch_size_test, train=False)

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
                torch.save(classifier.state_dict(), os.path.join(save_path, f'model_cv{cv}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_cv{cv}.pth'))

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
    torch.save(classifier.state_dict(), os.path.join(save_path, f'model_cv{cv}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_cv{cv}.pth'))
    return classifier


# Train explainers
def fit_explainers(device: torch.device, explainers_name: list, corpus_size=1000, test_size=100,
                   n_epoch=10000, learning_rate=100.0, momentum=0.5, save_path='./results/mnist/',
                   random_seed: int = 42, n_keep=5, reg_factor_init=0.1, reg_factor_final=1000, cv: int = 0
                   ):
    torch.random.manual_seed(random_seed + cv)
    explainers = []

    # Load model:
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    # Load data:
    corpus_loader = load_mnist(corpus_size, train=True)
    test_loader = load_mnist(test_size, train=False)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test, (example_data, example_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device)
    example_data = example_data.to(device)

    # Fit corpus:
    reg_factor_scheduler = ExponentialScheduler(reg_factor_init, reg_factor_final, n_epoch)
    corpus = Corpus(corpus_examples=corpus_data.detach(),
                    corpus_latent_reps=(classifier.latent_representation(corpus_data)).detach())
    weights = corpus.fit(test_examples=example_data.detach(),
                         test_latent_reps=classifier.latent_representation(example_data).detach(),
                         n_epoch=n_epoch, learning_rate=learning_rate, momentum=momentum,
                         reg_factor=reg_factor_init, n_keep=n_keep,
                         reg_factor_scheduler=reg_factor_scheduler)
    explainers.append(corpus)

    # Fit nearest neighbors:
    nn_uniform = NearNeighLatent(corpus_examples=corpus_data,
                                 corpus_latent_reps=(classifier.latent_representation(corpus_data)).detach())
    nn_uniform.fit(test_examples=example_data.detach(),
                   test_latent_reps=classifier.latent_representation(example_data).detach(),
                   n_keep=n_keep)
    explainers.append(nn_uniform)
    nn_dist = NearNeighLatent(corpus_examples=corpus_data,
                              corpus_latent_reps=(classifier.latent_representation(corpus_data)).detach(),
                              weights_type='distance')
    nn_dist.fit(test_examples=example_data.detach(),
                test_latent_reps=classifier.latent_representation(example_data).detach(),
                n_keep=n_keep)
    explainers.append(nn_dist)

    # Save explainers:
    for explainer, explainer_name in zip(explainers, explainers_name):
        explainer_path = os.path.join(save_path, f'{explainer_name}_cv{cv}_n{n_keep}.pkl')
        with open(explainer_path, 'wb') as f:
            print(f'Saving {explainer_name} decomposition in {explainer_path}.')
            pkl.dump(explainer, f)
    return explainers


# Approximation Quality experiment
def approximation_quality(cv: int = 0, random_seed: int = 42, n_keep: int = 15, load_model: bool = False,
                          save_path: str = './results/mnist/'):
    print(100*'-' + '\n'+'Welcome in the approximation quality experiment for MNIST. \n'
          f'Settings: random_seed = {random_seed} ; n_keep = {n_keep} ; load_model = {load_model}.\n'
          + 100 * '-')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainers_name = ['corpus', 'nn_uniform', 'nn_dist']

    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)
    if not load_model:
        print(100 * '-' + '\n' + 'Now fitting the model. \n' + 100 * '-')
        train_model(device, random_seed=random_seed, cv=cv, save_path=save_path)
    print(100 * '-' + '\n' + 'Now fitting the explainers. \n' + 100 * '-')
    explainers = fit_explainers(device=device, random_seed=random_seed, cv=cv, test_size=100, corpus_size=100,
                                n_keep=n_keep, save_path=save_path, explainers_name=explainers_name)
    print(100 * '-' + '\n' + 'Results. \n' + 100 * '-')
    for explainer, explainer_name in zip(explainers, explainers_name):
        avg_residual, std_residual = explainer.residual_statistics(normalize=True)
        r2_score = explainer.r2_score()
        print(f'{explainer_name} residual: {avg_residual:.2g} +/- {std_residual:.2g} ; r2_score = {r2_score}.')


approximation_quality(load_model=True)

# Make the repetitions automatic!
