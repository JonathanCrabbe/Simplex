import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr._utils.visualization import visualize_image_attr
from mnist import load_mnist
from models.image_recognition import MnistClassifier
from explainers.simplex import Simplex
from visualization.images import plot_mnist
from utils.schedulers import ExponentialScheduler


def mnist_use_case(random_seed=42, save_path='./results/use_case/mnist/', train_model: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    batch_size_train = 64
    batch_size_test = 1000
    corpus_size = 1000
    n_epoch = 10
    log_interval = 100
    n_keep = 3
    test_id = 42

    print(20 * '-' + f'Welcome in the use case for MNIST' + 20 * '-')

    # Create save directory
    if not os.path.exists(save_path):
        print(f'Creating a saving path at {save_path}')
        os.makedirs(save_path)

    # Prepare data
    train_loader = load_mnist(batch_size_train, train=True)
    test_loader = load_mnist(batch_size_test, train=False)

    # Create the model
    classifier = MnistClassifier()
    classifier.to(device)

    # Train the model
    if train_model:
        # Train the model
        optimizer = optim.Adam(classifier.parameters())
        train_losses = []
        train_counter = []
        test_losses = []

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
                    torch.save(classifier.state_dict(), os.path.join(save_path, f'model.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer.pth'))

        def test():
            classifier.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = classifier(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
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

    # Load the model
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model.pth')))
    classifier.to(device)
    classifier.eval()

    # Prepare corpus and test data
    corpus_loader = load_mnist(corpus_size, train=True)
    corpus_examples = enumerate(corpus_loader)
    _, (corpus_inputs, corpus_target) = next(corpus_examples)
    corpus_inputs = corpus_inputs.to(device).detach()
    test_examples = enumerate(test_loader)
    _, (test_inputs, test_target) = next(test_examples)
    test_inputs = test_inputs[test_id:test_id+1].to(device).detach()

    simplex = Simplex(corpus_inputs, classifier.latent_representation(corpus_inputs).detach())
    scheduler = ExponentialScheduler(x_init=0.1, x_final=1000, n_epoch=20000)
    weights = simplex.fit(test_inputs, classifier.latent_representation(test_inputs).detach(), n_keep=n_keep,
                          n_epoch=20000, reg_factor_scheduler=scheduler, reg_factor=0.1)

    input_baseline = torch.zeros(corpus_inputs.shape, device=device)
    jacobian_projections = simplex.jacobian_projection(test_id=0, model=classifier, input_baseline=input_baseline,
                                                       n_bins=100)
    decomposition = simplex.decompose(0)

    output = classifier(test_inputs)
    title = f'Prediction: {output.data.max(1, keepdim=True)[1][0].item()}'
    plot_mnist(simplex.test_examples[0][0].cpu().numpy(), title)
    for i in range(n_keep):
        image = decomposition[i][1].cpu().numpy().transpose((1, 2, 0))
        saliency = decomposition[i][2].cpu().numpy().transpose((1, 2, 0))
        title = f'Weight: {decomposition[i][0]:.2g}'
        #plot_mnist(data, title)
        #plot_mnist(saliency, title)
        visualize_image_attr(saliency, image, method='blended_heat_map', sign='all', title=title)
        plt.show()



mnist_use_case(train_model=False)