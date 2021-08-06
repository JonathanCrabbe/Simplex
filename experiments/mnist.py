import captum.attr
import numpy as np
import torch
import torchvision
import torch.optim as optim
import os
import math
import seaborn as sns
import math
import sklearn
import argparse
import pickle as pkl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_influence_functions as ptif
from models.image_recognition import MnistClassifier
from explainers.simplex import Simplex
from explainers.nearest_neighbours import NearNeighLatent
from explainers.representer import Representer
from utils.schedulers import ExponentialScheduler
from torch.utils.data import Dataset
from visualization.images import plot_mnist


# Load data

class MNISTSubset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        return self.X[i], self.y[i]


def load_mnist(batch_size: int, train: bool, subset_size=None, shuffle=True):
    dataset = torchvision.datasets.MNIST('./data/', train=train, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    if subset_size:
        dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:subset_size])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_emnist(batch_size: int, train: bool):
    return torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST('./data/', train=train, download=True, split='letters',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size, shuffle=True)


# Train model
def train_model(device: torch.device, n_epoch: int = 10, batch_size_train: int = 64, batch_size_test: int = 1000,
                random_seed: int = 42, learning_rate=0.01, momentum=0.5, log_interval=100, model_reg_factor=0.01,
                save_path='./results/mnist/', cv: int = 0):
    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.enabled = False

    # Prepare data
    train_loader = load_mnist(batch_size_train, train=True)
    test_loader = load_mnist(batch_size_test, train=False)

    # Create the model
    classifier = MnistClassifier()
    classifier.to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum, weight_decay=model_reg_factor)

    # Train the model
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
    torch.save(classifier.state_dict(), os.path.join(save_path, f'model_cv{cv}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_cv{cv}.pth'))
    return classifier


# Train explainers
def fit_explainers(device: torch.device, explainers_name: list, corpus_size=1000, test_size=100,
                   n_epoch=10000, learning_rate=100.0, momentum=0.5, save_path='./results/mnist/',
                   random_seed: int = 42, n_keep=5, reg_factor_init=0.1, reg_factor_final=1000, cv: int = 0):
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
    batch_id_test, (test_data, test_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device).detach()
    test_data = test_data.to(device).detach()
    corpus_latent_reps = classifier.latent_representation(corpus_data).detach()
    corpus_probas = classifier.probabilities(corpus_data).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[torch.arange(corpus_size), corpus_target] = 1
    test_latent_reps = classifier.latent_representation(test_data).detach()

    # Fit corpus:
    reg_factor_scheduler = ExponentialScheduler(reg_factor_init, reg_factor_final, n_epoch)
    corpus = Simplex(corpus_examples=corpus_data,
                     corpus_latent_reps=corpus_latent_reps)
    weights = corpus.fit(test_examples=test_data,
                         test_latent_reps=test_latent_reps,
                         n_epoch=n_epoch, learning_rate=learning_rate, momentum=momentum,
                         reg_factor=reg_factor_init, n_keep=n_keep,
                         reg_factor_scheduler=reg_factor_scheduler)
    explainers.append(corpus)

    # Fit nearest neighbors:
    nn_uniform = NearNeighLatent(corpus_examples=corpus_data,
                                 corpus_latent_reps=corpus_latent_reps)
    nn_uniform.fit(test_examples=test_data,
                   test_latent_reps=test_latent_reps,
                   n_keep=n_keep)
    explainers.append(nn_uniform)
    nn_dist = NearNeighLatent(corpus_examples=corpus_data,
                              corpus_latent_reps=corpus_latent_reps,
                              weights_type='distance')
    nn_dist.fit(test_examples=test_data,
                test_latent_reps=test_latent_reps,
                n_keep=n_keep)
    explainers.append(nn_dist)

    # Save explainers and data:
    for explainer, explainer_name in zip(explainers, explainers_name):
        explainer_path = os.path.join(save_path, f'{explainer_name}_cv{cv}_n{n_keep}.pkl')
        with open(explainer_path, 'wb') as f:
            print(f'Saving {explainer_name} decomposition in {explainer_path}.')
            pkl.dump(explainer, f)
    corpus_data_path = os.path.join(save_path, f'corpus_data_cv{cv}.pkl')
    with open(corpus_data_path, 'wb') as f:
        print(f'Saving corpus data in {corpus_data_path}.')
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = os.path.join(save_path, f'test_data_cv{cv}.pkl')
    with open(test_data_path, 'wb') as f:
        print(f'Saving test data in {test_data_path}.')
        pkl.dump([test_latent_reps, test_targets], f)
    return explainers


# Fit a representer
def fit_representer(model_reg_factor, load_path: str, cv: int = 0):
    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    corpus_data_path = os.path.join(load_path, f'corpus_data_cv{cv}.pkl')
    with open(corpus_data_path, 'rb') as f:
        corpus_latent_reps, corpus_probas, corpus_true_classes = pkl.load(f)
    test_data_path = os.path.join(load_path, f'test_data_cv{cv}.pkl')
    with open(test_data_path, 'rb') as f:
        test_latent_reps, test_targets = pkl.load(f)
    representer = Representer(corpus_latent_reps=corpus_latent_reps,
                              corpus_probas=corpus_probas,
                              corpus_true_classes=corpus_true_classes,
                              reg_factor=model_reg_factor)
    representer.fit(test_latent_reps=test_latent_reps)
    explainer_path = os.path.join(load_path, f'representer_cv{cv}.pkl')
    with open(explainer_path, 'wb') as f:
        print(f'Saving representer decomposition in {explainer_path}.')
        pkl.dump(representer, f)
    return representer


'''
        ----------------------------
        Precision experiment
        ----------------------------
'''

def approximation_quality(n_keep_list: list, cv: int = 0, random_seed: int = 42,
                          model_reg_factor=0.1, save_path: str = './results/mnist/quality/'):
    print(100 * '-' + '\n' + 'Welcome in the approximation quality experiment for MNIST. \n'
                             f'Settings: random_seed = {random_seed} ; cv = {cv}.\n'
          + 100 * '-')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainers_name = ['simplex', 'nn_uniform', 'nn_dist', 'representer']

    # Create saving directory if inexistent
    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)

    # Training a model, save it
    print(100 * '-' + '\n' + 'Now fitting the model. \n' + 100 * '-')
    train_model(device, random_seed=random_seed, cv=cv, save_path=save_path, model_reg_factor=model_reg_factor)

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    # Fit the explainers
    print(100 * '-' + '\n' + 'Now fitting the explainers. \n' + 100 * '-')
    for i, n_keep in enumerate(n_keep_list):
        print(100 * '-' + '\n' + f'Run number {i + 1}/{len(n_keep_list)} . \n' + 100 * '-')
        explainers = fit_explainers(device=device, random_seed=random_seed, cv=cv, test_size=100, corpus_size=1000,
                                    n_keep=n_keep, save_path=save_path, explainers_name=explainers_name)
        # Print the partial results
        print(100 * '-' + '\n' + 'Results. \n' + 100 * '-')
        for explainer, explainer_name in zip(explainers, explainers_name[:-1]):
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy())
            output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
            print(f'{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}.')

    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    representer = fit_representer(model_reg_factor, save_path, cv)
    latent_rep_true = representer.test_latent_reps
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_approx = representer.output_approx()
    output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
    print(f'representer output r2 = {output_r2_score:.2g}.')


'''
        ----------------------------
        Influence Function experiment
        ----------------------------
'''


def influence_function(n_keep_list: list, cv: int = 0, random_seed: int = 42,
                       save_path: str = './results/mnist/influence/', batch_size: int = 20,
                       corpus_size: int = 1000, test_size: int = 100):
    print(100 * '-' + '\n' + 'Welcome in the influence function computation for MNIST. \n'
                             f'Settings: random_seed = {random_seed} ; cv = {cv}.\n'
          + 100 * '-')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed + cv)

    # Create saving directory if inexistent
    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    corpus_loader = load_mnist(subset_size=corpus_size, train=True, batch_size=corpus_size)
    test_loader = load_mnist(subset_size=100, train=False, batch_size=20, shuffle=False)

    corpus_features = next(iter(corpus_loader))[0].to(device)
    corpus_latent_reps = classifier.latent_representation(corpus_features).detach()
    scheduler = ExponentialScheduler(0.1, 100, 20000)

    with open(os.path.join(save_path, f'corpus_latent_reps_cv{cv}.pkl'), 'wb') as f:
        pkl.dump(corpus_latent_reps.cpu().numpy(), f)

    test_latent_reps = np.zeros((test_size, corpus_latent_reps.shape[-1]))
    for batch_id, batch in enumerate(test_loader):
        test_latent_reps[batch_id * len(batch[0]):(batch_id + 1) * len(batch[0]), :] = \
            classifier.latent_representation(batch[0].to(device)).detach().cpu().numpy()
    with open(os.path.join(save_path, f'test_latent_reps_cv{cv}.pkl'), 'wb') as f:
        pkl.dump(test_latent_reps, f)

    print(30*'-' + f'Fitting SimplEx' + 30*'-')
    for n_keep in n_keep_list:
        print(20*'-' + f'Training Simplex by allowing to keep {n_keep} corpus examples' + 20*'-')
        weights = np.zeros((test_size, corpus_size))
        for batch_id, batch in enumerate(test_loader):
            print(20 * '-' + f'Working with batch {batch_id+1} / {math.ceil(test_size/batch_size)}' + 20 * '-')
            simplex = Simplex(corpus_features, corpus_latent_reps)
            test_features = batch[0].to(device)
            test_latent_reps = classifier.latent_representation(test_features).detach()
            simplex.fit(test_features, test_latent_reps, n_keep=n_keep, reg_factor=0.1,
                        reg_factor_scheduler=scheduler, n_epoch=20000)
            weights_batch = simplex.weights.cpu().numpy()
            weights[batch_id*len(weights_batch):(batch_id+1)*len(weights_batch), :] = weights_batch

        with open(os.path.join(save_path, f'simplex_weights_cv{cv}_n{n_keep}.pkl'), 'wb') as f:
            pkl.dump(weights, f)

    print(30 * '-' + f'Computing Influence Functions' + 30 * '-')
    ptif.init_logging()
    config = ptif.get_default_config()
    config['outdir'] = save_path
    config['test_sample_num'] = False
    ptif.calc_img_wise(config, classifier, corpus_loader, test_loader)


'''
        ----------------------------
        Outlier Detection experiment
        ----------------------------
'''


def outlier_detection(cv: int = 0, random_seed: int = 42, save_path: str = './results/mnist/outlier/',
                      train: bool = True):
    torch.random.manual_seed(random_seed + cv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epoch_simplex = 10000
    K = 5

    print(100 * '-' + '\n' + 'Welcome in the outlier detection experiment for MNIST. \n'
                             f'Settings: random_seed = {random_seed} ; cv = {cv}.\n'
          + 100 * '-')

    # Create saving directory if inexistent
    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)

    # Training a model, save it
    if train:
        print(100 * '-' + '\n' + 'Now fitting the model. \n' + 100 * '-')
        train_model(device, random_seed=random_seed, cv=cv, save_path=save_path, model_reg_factor=0)

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    # Load data:
    corpus_loader = load_mnist(batch_size=1000, train=True)
    mnist_test_loader = load_mnist(batch_size=100, train=False)
    emnist_test_loader = load_emnist(batch_size=100, train=True)
    corpus_examples = enumerate(corpus_loader)
    batch_id_corpus, (corpus_features, corpus_target) = next(corpus_examples)
    corpus_features = corpus_features.to(device).detach()
    mnist_test_examples = enumerate(mnist_test_loader)
    batch_id_test_mnist, (mnist_test_features, mnist_test_target) = next(mnist_test_examples)
    mnist_test_features = mnist_test_features.to(device).detach()
    emnist_test_examples = enumerate(emnist_test_loader)
    batch_id_test_emnist, (emnist_test_features, emnist_test_target) = next(emnist_test_examples)
    emnist_test_features = emnist_test_features.to(device).detach()
    test_features = torch.cat([mnist_test_features, emnist_test_features], dim=0)
    corpus_latent_reps = classifier.latent_representation(corpus_features).detach()
    test_latent_reps = classifier.latent_representation(test_features).detach()

    # Fit corpus:
    simplex = Simplex(corpus_examples=corpus_features,
                      corpus_latent_reps=corpus_latent_reps)
    weights = simplex.fit(test_examples=test_features,
                          test_latent_reps=test_latent_reps,
                          n_epoch=n_epoch_simplex, learning_rate=100.0, momentum=0.5,
                          reg_factor=0, n_keep=corpus_features.shape[0])
    explainer_path = os.path.join(save_path, f'simplex_cv{cv}.pkl')
    with open(explainer_path, 'wb') as f:
        print(f'Saving simplex decomposition in {explainer_path}.')
        pkl.dump(simplex, f)
    nn_uniform = NearNeighLatent(corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps)
    nn_uniform.fit(test_features, test_latent_reps, n_keep=K)
    nn_dist = NearNeighLatent(corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps,
                              weights_type='distance')
    nn_dist.fit(test_features, test_latent_reps, n_keep=K)
    explainer_path = os.path.join(save_path, f'nn_dist_cv{cv}.pkl')
    with open(explainer_path, 'wb') as f:
        print(f'Saving nn_dist decomposition in {explainer_path}.')
        pkl.dump(nn_dist, f)
    explainer_path = os.path.join(save_path, f'nn_uniform_cv{cv}.pkl')
    with open(explainer_path, 'wb') as f:
        print(f'Saving nn_uniform decomposition in {explainer_path}.')
        pkl.dump(nn_uniform, f)

    simplex_latent_approx = simplex.latent_approx()
    simplex_residuals = torch.sqrt(((test_latent_reps - simplex_latent_approx) ** 2).mean(dim=-1))
    n_inspected = [n for n in range(simplex_residuals.shape[0])]
    simplex_n_detected = [torch.count_nonzero(torch.topk(simplex_residuals, k=n)[1] > 99) for n in n_inspected]
    nn_dist_latent_approx = nn_dist.latent_approx()
    nn_dist_residuals = torch.sqrt(((test_latent_reps - nn_dist_latent_approx) ** 2).mean(dim=-1))
    nn_dist_n_detected = [torch.count_nonzero(torch.topk(nn_dist_residuals, k=n)[1] > 99) for n in n_inspected]
    nn_uniform_latent_approx = nn_uniform.latent_approx()
    nn_uniform_residuals = torch.sqrt(((test_latent_reps - nn_uniform_latent_approx) ** 2).mean(dim=-1))
    nn_uniform_n_detected = [torch.count_nonzero(torch.topk(nn_uniform_residuals, k=n)[1] > 99) for n in n_inspected]
    sns.set()
    plt.plot(n_inspected, simplex_n_detected, label='Simplex')
    plt.plot(n_inspected, nn_dist_n_detected, label=f'{K}NN Distance')
    plt.plot(n_inspected, nn_uniform_n_detected, label=f'{K}NN Uniform')
    plt.xlabel('Number of inspected examples')
    plt.ylabel('Number of outliers detected')
    plt.legend()
    plt.show()


'''
        ----------------------------
        Jacobian Projection experiment
        ----------------------------
'''


def jacobian_projection_check(random_seed=42, save_path='./results/mnist/jacobian_projections/',
                              corpus_size=500, test_size=50, n_bins=100, batch_size=20):
    print(100 * '-' + '\n' + 'Welcome in the Jacobian Projection check for MNIST. \n'
                             f'Settings: random_seed = {random_seed} .\n'
          + 100 * '-')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed)
    n_pert_list = [1, 5, 10, 50]

    # Create saving directory if inexistent
    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model.pth')))
    classifier.to(device)
    classifier.eval()

    # Prepare the corpus and the test set
    corpus_loader = load_mnist(subset_size=corpus_size, train=True, batch_size=batch_size)
    test_loader = load_mnist(subset_size=test_size, train=False, batch_size=1, shuffle=False)
    Corpus_inputs = torch.zeros((corpus_size, 1, 28, 28), device=device)

    # Prepare the IG baseline
    ig_explainer = captum.attr.IntegratedGradients(classifier)
    Corpus_inputs_pert_jp = torch.zeros((len(n_pert_list), corpus_size, 1, 28, 28), device=device)
    Corpus_inputs_pert_ig = torch.zeros((len(n_pert_list), corpus_size, 1, 28, 28), device=device)


    for test_id, (test_input, _) in enumerate(test_loader):
        print(25*'=' + f'Now working with test sample {test_id+1}/{test_size}' + 25*'=')
        for batch_id, (corpus_inputs, corpus_targets) in enumerate(corpus_loader):
            print(f'Now working with corpus batch {batch_id+1}/{math.ceil(corpus_size/batch_size)}.')
            test_input = test_input.to(device)
            corpus_inputs = corpus_inputs.to(device).requires_grad_()
            baseline_inputs = -0.4242 * torch.ones(corpus_inputs.shape, device=device)
            input_shift = corpus_inputs - baseline_inputs
            test_latent = classifier.latent_representation(test_input).detach()
            baseline_latents = classifier.latent_representation(baseline_inputs)
            latent_shift = test_latent - baseline_latents
            latent_shift_sqrdnorm = torch.sum(latent_shift ** 2, dim=-1, keepdim=True)
            input_grad = torch.zeros(corpus_inputs.shape, device=corpus_inputs.device)
            for n in range(1, n_bins + 1):
                t = n / n_bins
                inputs = baseline_inputs + t * (corpus_inputs - baseline_inputs)
                latent_reps = classifier.latent_representation(inputs)
                latent_reps.backward(gradient=latent_shift / latent_shift_sqrdnorm)
                input_grad += corpus_inputs.grad
                corpus_inputs.grad.data.zero_()
            jacobian_projections = input_shift * input_grad/n_bins
            integrated_gradients = ig_explainer.attribute(corpus_inputs, baseline_inputs,
                                                          target=corpus_targets.to(device),
                                                          n_steps=n_bins)
            saliency_jp = torch.abs(jacobian_projections).detach()
            saliency_ig = torch.abs(integrated_gradients).detach()

            lower_id =  batch_id * batch_size
            higher_id = lower_id + batch_size
            Corpus_inputs[lower_id:higher_id] = corpus_inputs.detach()

            for pert_id, n_pert in enumerate(n_pert_list):
                top_pixels_jp = torch.topk(saliency_jp.view(batch_size, -1), k=n_pert)[1]
                top_pixels_ig = torch.topk(saliency_ig.view(batch_size, -1), k=n_pert)[1]
                mask_jp = torch.zeros(corpus_inputs.shape, device=device)
                mask_ig = torch.zeros(corpus_inputs.shape, device=device)
                for k in range(n_pert):
                    mask_jp[:, 0, top_pixels_jp[:, k] // 28, top_pixels_jp[:, k] % 28] = 1
                    mask_ig[:, 0, top_pixels_ig[:, k] // 28, top_pixels_ig[:, k] % 28] = 1
                corpus_inputs_pert_jp = mask_jp*baseline_inputs + (1-mask_jp)*corpus_inputs
                corpus_inputs_pert_ig = mask_ig * baseline_inputs + (1 - mask_ig) * corpus_inputs
                Corpus_inputs_pert_jp[pert_id, lower_id:higher_id] = corpus_inputs_pert_jp
                Corpus_inputs_pert_ig[pert_id, lower_id:higher_id] = corpus_inputs_pert_ig

        print('Now fitting the uncorrupted SimplEx')
        test_latent = classifier.latent_representation(test_input).detach().to(device)
        simplex = Simplex(Corpus_inputs, classifier.latent_representation(Corpus_inputs).detach())
        simplex.fit(test_input, test_latent, reg_factor=0)
        residual = torch.sum((test_latent - simplex.latent_approx()) ** 2)

        for pert_id, n_pert in enumerate(n_pert_list):

            print(f'Now fitting the JP-corrupted SimplEx with {n_pert} perturbation per image')
            simplex_jp = Simplex(Corpus_inputs_pert_jp[pert_id],
                                 classifier.latent_representation(Corpus_inputs_pert_jp[pert_id]).detach())
            simplex_jp.fit(test_input, test_latent, reg_factor=0)
            residual_jp = torch.sum((test_latent - simplex_jp.latent_approx())**2)

            print(f'Now fitting the IG-corrupted SimplEx with {n_pert} perturbation per image')
            simplex_ig = Simplex(Corpus_inputs_pert_ig[pert_id],
                                 classifier.latent_representation(Corpus_inputs_pert_ig[pert_id]).detach())
            simplex_ig.fit(test_input, test_latent, reg_factor=0)
            residual_ig = torch.sum((test_latent - simplex_ig.latent_approx()) ** 2)
            print(residual_jp-residual)
            print(residual_ig-residual)



'''
corpus_latents = classifier.latent_representation(corpus_inputs.detach()).detach()
corpus_latents_pert_jp = classifier.latent_representation(corpus_inputs_pert_jp).detach()
corpus_latents_pert_ig = classifier.latent_representation(corpus_inputs_pert_ig).detach()
proj_initial = torch.einsum('bd,bd->b', corpus_latents, test_latent)
cos_initial = proj_initial/torch.sqrt(torch.sum(test_latent ** 2)*torch.sum(corpus_latents**2, dim=-1))
proj_jp = torch.einsum('bd,bd->b', corpus_latents_pert_jp, test_latent)
cos_jp = proj_jp/torch.sqrt(torch.sum(test_latent ** 2)*torch.sum(corpus_latents_pert_jp**2, dim=-1))
proj_ig = torch.einsum('bd,bd->b', corpus_latents_pert_ig, test_latent)
cos_ig = proj_ig/torch.sqrt(torch.sum(test_latent ** 2)*torch.sum(corpus_latents_pert_ig**2, dim=-1))
cos_shift_jp = torch.abs(cos_initial - cos_jp).cpu()
cos_shift_ig = torch.abs(cos_initial - cos_ig).cpu()

   print(metrics_tensor[0].mean(dim=-1))
   print(metrics_tensor[0].std(dim=-1))
   print(metrics_tensor[1].mean(dim=-1))
   print(metrics_tensor[1].std(dim=-1))
'''




def main(experiment: str, cv: int):

    if experiment == 'approximation_quality':
        approximation_quality(cv=cv, n_keep_list=[n for n in range(2, 51)])
    elif experiment == 'outlier':
        outlier_detection(cv)



parser = argparse.ArgumentParser()
parser.add_argument('-experiment', type=str, default='approximation_quality', help='Experiment to perform')
parser.add_argument('-cv', type=int, default=0, help='Cross validation parameter')
args = parser.parse_args()

if __name__ == '__main__':
    jacobian_projection_check()
    #main(args.experiment, args.cv)

'''

def approximation_quality_single(cv: int = 0, random_seed: int = 42, n_keep: int = 10, load_model: bool = False,
                                 model_reg_factor=0.1, save_path: str = './results/mnist/'):
    print(100 * '-' + '\n' + 'Welcome in the approximation quality experiment for MNIST. \n'
                             f'Settings: random_seed = {random_seed} ; n_keep = {n_keep} ; load_model = {load_model}.\n'
          + 100 * '-')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainers_name = ['corpus', 'nn_uniform', 'nn_dist', 'representer']

    # Create saving directory if inexistent
    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.makedirs(save_path)

    # Training a model from scratch
    if not load_model:
        print(100 * '-' + '\n' + 'Now fitting the model. \n' + 100 * '-')
        train_model(device, random_seed=random_seed, cv=cv, save_path=save_path, model_reg_factor=model_reg_factor)

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    # Fit the explainers
    print(100 * '-' + '\n' + 'Now fitting the explainers. \n' + 100 * '-')
    explainers = fit_explainers(device=device, random_seed=random_seed, cv=cv, test_size=100, corpus_size=1000,
                                n_keep=n_keep, save_path=save_path, explainers_name=explainers_name,
                                model_reg_factor=model_reg_factor)

    # Print the partial results
    print(100 * '-' + '\n' + 'Results. \n' + 100 * '-')
    for explainer, explainer_name in zip(explainers, explainers_name):
        if not explainer_name == 'representer':
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy())
        else:
            latent_rep_true = explainer.test_latent_reps
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            output_approx = explainer.output_approx()
            latent_r2_score = 0
        output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
        print(f'{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}.')

'''
