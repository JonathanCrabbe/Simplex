import numpy as np
import torch
import os
import torch.optim as optim
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr._utils.visualization import visualize_image_attr
from mnist import load_mnist
from matplotlib import cm
from prostate_cancer import load_seer, ProstateCancerDataset, load_cutract
from models.image_recognition import MnistClassifier
from models.tabular_data import MortalityPredictor
from explainers.simplex import Simplex
from visualization.images import plot_mnist
from utils.schedulers import ExponentialScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from visualization.tables import plot_prostate_patient


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
    test_id = 22

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
    test_inputs = test_inputs[test_id:test_id + 1].to(device).detach()

    simplex = Simplex(corpus_inputs, classifier.latent_representation(corpus_inputs).detach())
    scheduler = ExponentialScheduler(x_init=0.1, x_final=1000, n_epoch=20000)
    weights = simplex.fit(test_inputs, classifier.latent_representation(test_inputs).detach(), n_keep=n_keep,
                          n_epoch=20000, reg_factor_scheduler=scheduler, reg_factor=0.1)

    input_baseline = -0.4242 * torch.ones(corpus_inputs.shape, device=device)
    jacobian_projections = simplex.jacobian_projection(test_id=0, model=classifier, input_baseline=input_baseline,
                                                       n_bins=500)
    decomposition = simplex.decompose(0)

    output = classifier(test_inputs)
    title = f'Prediction: {output.data.max(1, keepdim=True)[1][0].item()}'
    fig = plot_mnist(simplex.test_examples[0][0].cpu().numpy(), title)
    # plt.savefig(os.path.join(save_path, f'test_image_id{test_id}'))
    for i in range(n_keep):
        image = decomposition[i][1].cpu().numpy().transpose((1, 2, 0))
        saliency = decomposition[i][2].cpu().numpy().transpose((1, 2, 0))
        title = f'Weight: {decomposition[i][0]:.2g}'
        figure, axis = visualize_image_attr(saliency, image, method='blended_heat_map',
                                            sign='all', title=title, use_pyplot=True)
        plt.show()
        # plt.savefig(os.path.join(save_path, f'corpus_image{i + 1}_id{test_id}'))


def prostate_use_case(random_seed=42, save_path='./results/use_case/prostate/', train_model: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    batch_size_train = 64
    batch_size_test = 1000
    corpus_size = 1000
    n_epoch = 10
    log_interval = 100
    n_keep = 3
    test_id = 123

    print(20 * '-' + f'Welcome in the use case for Prostate Cancer' + 20 * '-')

    # Create save directory
    if not os.path.exists(save_path):
        print(f'Creating a saving path at {save_path}')
        os.makedirs(save_path)

    # Prepare data
    X, y = load_seer(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_seed,
                                                        stratify=y)

    train_data = ProstateCancerDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_data = ProstateCancerDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    # Create the model
    classifier = MortalityPredictor()
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
                target = target.type(torch.LongTensor).to(device)
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
                    target = target.type(torch.LongTensor).to(device)
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
    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=True)
    corpus_examples = enumerate(corpus_loader)
    _, (corpus_inputs, corpus_target) = next(corpus_examples)
    corpus_inputs = corpus_inputs.to(device).detach()
    test_examples = enumerate(test_loader)
    _, (test_inputs, test_target) = next(test_examples)
    test_inputs = test_inputs[test_id:test_id + 1].to(device).detach()
    simplex = Simplex(corpus_inputs, classifier.latent_representation(corpus_inputs).detach())
    scheduler = ExponentialScheduler(x_init=0.1, x_final=100, n_epoch=20000)
    weights = simplex.fit(test_inputs, classifier.latent_representation(test_inputs).detach(), n_keep=n_keep,
                          n_epoch=20000, reg_factor_scheduler=scheduler, reg_factor=0.1)

    # input_baseline = torch.zeros(corpus_inputs.shape, device=device)
    # input_baseline[:, :3] = torch.mean(corpus_inputs, dim=0, keepdim=True)[:, :3]
    input_baseline = torch.mean(corpus_inputs, dim=0, keepdim=True).repeat(corpus_size, 1)
    jacobian_projections = simplex.jacobian_projection(test_id=0, model=classifier, input_baseline=input_baseline,
                                                       n_bins=500)
    decomposition = simplex.decompose(0)

    output = classifier(test_inputs)
    title = f'Predicted Mortality: {output.data.max(1, keepdim=True)[1][0].item()}'
    plot_prostate_patient(test_inputs[0].cpu().numpy(), title)
    plt.show()
    plt.savefig(os.path.join(save_path, f'test_patient_id{test_id}'))
    for i in range(n_keep):
        input = decomposition[i][1].cpu().numpy()
        saliency = decomposition[i][2].cpu().numpy()
        title = f'Weight: {decomposition[i][0]:.2g}'
        plot_prostate_patient(input, title, saliency)
        plt.savefig(os.path.join(save_path, f'corpus_patient{i + 1}_id{test_id}'))
        plt.show()


def prostate_two_corpus(random_seed=42, save_path='./results/use_case/prostate/', train_model: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    batch_size_train = 64
    batch_size_test = 1000
    corpus_size = 1000
    n_epoch = 10
    log_interval = 100
    n_keep = 3
    test_id = 123

    print(20 * '-' + f'Welcome in the two corpus use case for Prostate Cancer' + 20 * '-')

    # Create save directory
    if not os.path.exists(save_path):
        print(f'Creating a saving path at {save_path}')
        os.makedirs(save_path)

    # Prepare data
    X_usa, y_usa = load_seer(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X_usa, y_usa, test_size=0.15, random_state=random_seed,
                                                        stratify=y_usa)

    train_data = ProstateCancerDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_data = ProstateCancerDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    # Create the model
    classifier = MortalityPredictor()
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
                target = target.type(torch.LongTensor).to(device)
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
                    target = target.type(torch.LongTensor).to(device)
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

    # Prepare USA corpus
    corpus_loader_usa = DataLoader(train_data, batch_size=corpus_size, shuffle=True)
    corpus_examples = enumerate(corpus_loader_usa)
    _, (corpus_usa_inputs, corpus_usa_target) = next(corpus_examples)
    corpus_usa_inputs = corpus_usa_inputs.to(device).detach()
    corpus_usa_predictions = torch.argmax(classifier(corpus_usa_inputs), dim=-1).cpu().numpy()

    # Prepare UK corpus and UK test data
    X_uk, y_uk = load_cutract(random_seed)
    uk_data = ProstateCancerDataset(X_uk, y_uk)
    data_loader_uk = DataLoader(uk_data, batch_size=corpus_size, shuffle=True)
    uk_examples = enumerate(corpus_loader_usa)
    _, (corpus_uk_inputs, corpus_uk_target) = next(uk_examples)
    corpus_uk_inputs = corpus_uk_inputs.to(device).detach()
    corpus_uk_predictions = torch.argmax(classifier(corpus_uk_inputs), dim=-1).cpu().numpy()
    _, (test_uk_inputs, test_uk_target) = next(uk_examples)
    test_uk_inputs = test_uk_inputs.to(device).detach()
    test_uk_target = test_uk_target.cpu().numpy()
    test_uk_predictions = torch.argmax(classifier(test_uk_inputs), dim=-1).cpu().numpy()

    print(f'Accuracy score of the model on the UK cohort: {accuracy_score(test_uk_target, test_uk_predictions):.2g}.')

    # Extract a mislabeled example that we will interpret with the two corpus
    mislabeled_examples = np.nonzero(test_uk_predictions != test_uk_target)[0]
    selected_id = mislabeled_examples[66]  # Type 1: 66 12     ; Type 2: 31 42 14
    selected_input = test_uk_inputs[selected_id:selected_id+1]
    selected_latent_rep = classifier.latent_representation(selected_input).detach()
    scheduler = ExponentialScheduler(x_init=0.1, x_final=100, n_epoch=20000)

    print(20*'-'+'Now fitting the USA SimplEx'+20*'-')
    simplex_usa = Simplex(corpus_usa_inputs, classifier.latent_representation(corpus_usa_inputs).detach())
    _ = simplex_usa.fit(selected_input, selected_latent_rep, n_keep=n_keep,
                        n_epoch=20000, reg_factor=0.1, reg_factor_scheduler=scheduler)
    input_baseline_usa = torch.mean(corpus_usa_inputs, dim=0, keepdim=True).repeat(corpus_size, 1)
    jacobian_projections_usa = simplex_usa.jacobian_projection(test_id=0, model=classifier,
                                                               input_baseline=input_baseline_usa, n_bins=500)
    decomposition_usa, corpus_ids_usa = simplex_usa.decompose(0, return_id=True)

    print(20 * '-' + 'Now fitting the UK SimplEx' + 20 * '-')
    simplex_uk = Simplex(corpus_uk_inputs, classifier.latent_representation(corpus_uk_inputs).detach())
    _ = simplex_uk.fit(selected_input, selected_latent_rep, n_keep=n_keep,
                       n_epoch=20000, reg_factor=0.1, reg_factor_scheduler=scheduler)
    input_baseline_uk = torch.mean(corpus_uk_inputs, dim=0, keepdim=True).repeat(corpus_size, 1)
    jacobian_projections_uk = simplex_uk.jacobian_projection(test_id=0, model=classifier,
                                                             input_baseline=input_baseline_uk, n_bins=500)
    decomposition_uk, corpus_ids_uk = simplex_uk.decompose(0, return_id=True)

    title = f'Predicted Mortality: {test_uk_predictions[selected_id]} ; True Mortality: {test_uk_target[selected_id]}'
    plot_prostate_patient(selected_input[0].cpu().numpy(), title)
    plt.show()
    #plt.savefig(os.path.join(save_path, f'test_patient_id{test_id}'))
    for i in range(n_keep):
        input_usa = decomposition_usa[i][1].cpu().numpy()
        saliency_usa = decomposition_usa[i][2].cpu().numpy()
        title = f'USA Patient ; Weight: {decomposition_usa[i][0]:.2g} ; ' \
                f'Predicted Mortality: {corpus_usa_predictions[corpus_ids_usa[i]]} ;' \
                f'True Mortality: {corpus_usa_target[corpus_ids_usa[i]]}'
        plot_prostate_patient(input_usa, title, saliency_usa)
        #plt.savefig(os.path.join(save_path, f'corpus_patient{i + 1}_id{test_id}'))
        plt.show()
        input_uk = decomposition_uk[i][1].cpu().numpy()
        saliency_uk = decomposition_uk[i][2].cpu().numpy()
        title = f'UK Patient ; Weight: {decomposition_uk[i][0]:.2g} ; ' \
                f'Predicted Mortality: {corpus_uk_predictions[corpus_ids_uk[i]]}  ;' \
                f'True Mortality: {corpus_uk_target[corpus_ids_uk[i]]}'
        plot_prostate_patient(input_uk, title, saliency_uk)
        # plt.savefig(os.path.join(save_path, f'corpus_patient{i + 1}_id{test_id}'))
        plt.show()


if __name__ == '__main__':
    # mnist_use_case(train_model=False)
    # prostate_use_case(train_model=False)
    prostate_two_corpus(train_model=False)
