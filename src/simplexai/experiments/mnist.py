import argparse
import math
import os
import pickle as pkl
import time
from pathlib import Path

import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_influence_functions as ptif
import seaborn as sns
import sklearn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from simplexai.explainers.nearest_neighbours import NearNeighLatent
from simplexai.explainers.representer import Representer
from simplexai.explainers.simplex import Simplex
from simplexai.models.image_recognition import MnistClassifier
from simplexai.utils.schedulers import ExponentialScheduler


# Load data
class MNISTSubset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple:
        if torch.is_tensor(i):
            i = i.tolist()
        return self.X[i], self.y[i]


def load_mnist(
    batch_size: int, train: bool, subset_size=None, shuffle: bool = True
) -> DataLoader:
    dataset = torchvision.datasets.MNIST(
        "./data/",
        train=train,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_emnist(batch_size: int, train: bool) -> DataLoader:
    return DataLoader(
        torchvision.datasets.EMNIST(
            "./data/",
            train=train,
            download=True,
            split="letters",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def train_model(
    save_path: Path,
    device: torch.device,
    n_epoch: int = 10,
    batch_size_train: int = 64,
    batch_size_test: int = 1000,
    random_seed: int = 42,
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    log_interval: int = 100,
    model_reg_factor: float = 0.01,
    cv: int = 0,
) -> MnistClassifier:
    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.enabled = False

    # Prepare data
    train_loader = load_mnist(batch_size_train, train=True)
    test_loader = load_mnist(batch_size_test, train=False)

    # Create the model
    classifier = MnistClassifier()
    classifier.to(device)
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=model_reg_factor,
    )

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
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(
                    classifier.state_dict(),
                    os.path.join(save_path, f"model_cv{cv}.pth"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(save_path, f"optimizer_cv{cv}.pth"),
                )

    def test():
        classifier.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = classifier(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
            f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )

    test()
    for epoch in range(1, n_epoch + 1):
        train(epoch)
        test()
    torch.save(classifier.state_dict(), os.path.join(save_path, f"model_cv{cv}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, f"optimizer_cv{cv}.pth"))
    return classifier


def fit_explainers(
    device: torch.device,
    explainers_name: list,
    save_path: Path,
    corpus_size: int = 1000,
    test_size: int = 100,
    n_epoch: int = 10000,
    random_seed: int = 42,
    n_keep: int = 5,
    reg_factor_init: float = 0.1,
    reg_factor_final: float = 100,
    cv: int = 0,
    train_only: bool = False,
) -> list:
    torch.random.manual_seed(random_seed + cv)
    explainers = []

    # Load model:
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    # Load data:
    corpus_loader = load_mnist(corpus_size, train=True)
    test_loader = load_mnist(test_size, train=train_only)
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

    # Fit SimplEx:
    reg_factor_scheduler = ExponentialScheduler(
        reg_factor_init, reg_factor_final, n_epoch
    )
    simplex = Simplex(
        corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
    )
    simplex.fit(
        test_examples=test_data,
        test_latent_reps=test_latent_reps,
        n_epoch=n_epoch,
        reg_factor=reg_factor_init,
        n_keep=n_keep,
        reg_factor_scheduler=reg_factor_scheduler,
    )
    explainers.append(simplex)

    # Fit nearest neighbors:
    nn_uniform = NearNeighLatent(
        corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
    )
    nn_uniform.fit(
        test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
    )
    explainers.append(nn_uniform)
    nn_dist = NearNeighLatent(
        corpus_examples=corpus_data,
        corpus_latent_reps=corpus_latent_reps,
        weights_type="distance",
    )
    nn_dist.fit(
        test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
    )
    explainers.append(nn_dist)

    # Save explainers and data:
    for explainer, explainer_name in zip(explainers, explainers_name):
        explainer_path = save_path / f"{explainer_name}_cv{cv}_n{n_keep}.pkl"
        with open(explainer_path, "wb") as f:
            print(f"Saving {explainer_name} decomposition in {explainer_path}.")
            pkl.dump(explainer, f)
    corpus_data_path = save_path / f"corpus_data_cv{cv}.pkl"
    with open(corpus_data_path, "wb") as f:
        print(f"Saving corpus data in {corpus_data_path}.")
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = save_path / f"test_data_cv{cv}.pkl"
    with open(test_data_path, "wb") as f:
        print(f"Saving test data in {test_data_path}.")
        pkl.dump([test_latent_reps, test_targets], f)
    return explainers


def fit_representer(
    model_reg_factor: float, load_path: Path, cv: int = 0
) -> Representer:
    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    corpus_data_path = os.path.join(load_path, f"corpus_data_cv{cv}.pkl")
    with open(corpus_data_path, "rb") as f:
        corpus_latent_reps, corpus_probas, corpus_true_classes = pkl.load(f)
    test_data_path = os.path.join(load_path, f"test_data_cv{cv}.pkl")
    with open(test_data_path, "rb") as f:
        test_latent_reps, test_targets = pkl.load(f)
    representer = Representer(
        corpus_latent_reps=corpus_latent_reps,
        corpus_probas=corpus_probas,
        corpus_true_classes=corpus_true_classes,
        reg_factor=model_reg_factor,
    )
    representer.fit(test_latent_reps=test_latent_reps)
    explainer_path = os.path.join(load_path, f"representer_cv{cv}.pkl")
    with open(explainer_path, "wb") as f:
        print(f"Saving representer decomposition in {explainer_path}.")
        pkl.dump(representer, f)
    return representer


def approximation_quality(
    n_keep_list: list,
    cv: int = 0,
    random_seed: int = 42,
    model_reg_factor=0.1,
    save_path: str = "experiments/results/mnist/quality/",
    train_only=False,
) -> None:
    print(
        100 * "-"
        + "\n"
        + "Welcome in the approximation quality experiment for MNIST. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv}.\n" + 100 * "-"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainers_name = ["simplex", "nn_uniform", "nn_dist", "representer"]

    current_path = Path.cwd()
    save_path = current_path / save_path
    # Create saving directory if inexistent
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Training a model, save it
    print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
    train_model(
        device=device,
        random_seed=random_seed,
        cv=cv,
        save_path=save_path,
        model_reg_factor=model_reg_factor,
    )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    # Fit the explainers
    print(100 * "-" + "\n" + "Now fitting the explainers. \n" + 100 * "-")
    for i, n_keep in enumerate(n_keep_list):
        print(30 * "-" + f"n_keep = {n_keep}" + 30 * "-")
        explainers = fit_explainers(
            device=device,
            random_seed=random_seed,
            cv=cv,
            test_size=100,
            corpus_size=1000,
            n_keep=n_keep,
            save_path=save_path,
            explainers_name=explainers_name,
            train_only=train_only,
        )
        # Print the partial results
        print(100 * "-" + "\n" + "Results. \n" + 100 * "-")
        for explainer, explainer_name in zip(explainers, explainers_name[:-1]):
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy()
            )
            output_r2_score = sklearn.metrics.r2_score(
                output_true.cpu().numpy(), output_approx.cpu().numpy()
            )
            print(
                f"{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}."
            )

    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    representer = fit_representer(model_reg_factor, save_path, cv)
    latent_rep_true = representer.test_latent_reps
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_approx = representer.output_approx()
    output_r2_score = sklearn.metrics.r2_score(
        output_true.cpu().numpy(), output_approx.cpu().numpy()
    )
    print(f"representer output r2 = {output_r2_score:.2g}.")


def influence_function(
    n_keep_list: list,
    cv: int = 0,
    random_seed: int = 42,
    save_path: str = "experiments/results/mnist/influence/",
    corpus_size: int = 1000,
    test_size: int = 100,
) -> None:
    print(
        100 * "-" + "\n" + "Welcome in the influence function computation for MNIST. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv}.\n" + 100 * "-"
    )
    device = torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.random.manual_seed(random_seed + cv)
    current_path = Path.cwd()
    save_path = current_path / save_path

    # Create saving directory if inexistent
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Training a model if necessary, save it
    if not (save_path / f"model_cv{cv}.pth").exists():
        print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
        train_model(
            device=device,
            random_seed=random_seed,
            cv=cv,
            save_path=save_path,
            model_reg_factor=0,
        )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    corpus_loader = load_mnist(
        subset_size=corpus_size, train=True, batch_size=corpus_size
    )
    test_loader = load_mnist(
        subset_size=test_size, train=False, batch_size=50, shuffle=False
    )

    corpus_features = next(iter(corpus_loader))[0].to(device)
    corpus_latent_reps = classifier.latent_representation(corpus_features).detach()
    scheduler = ExponentialScheduler(0.1, 100, 20000)

    with open(save_path / f"corpus_latent_reps_cv{cv}.pkl", "wb") as f:
        pkl.dump(corpus_latent_reps.cpu().numpy(), f)

    test_latent_reps = np.zeros((test_size, corpus_latent_reps.shape[-1]))
    for batch_id, batch in enumerate(test_loader):
        test_latent_reps[
            batch_id * len(batch[0]) : (batch_id + 1) * len(batch[0]), :
        ] = (
            classifier.latent_representation(batch[0].to(device)).detach().cpu().numpy()
        )
    with open(save_path / f"test_latent_reps_cv{cv}.pkl", "wb") as f:
        pkl.dump(test_latent_reps, f)

    print(30 * "-" + "Fitting SimplEx" + 30 * "-")
    for n_keep in n_keep_list:
        print(
            20 * "-"
            + f"Training Simplex by allowing to keep {n_keep} corpus examples"
            + 20 * "-"
        )
        weights = np.zeros((test_size, corpus_size))
        for batch_id, batch in enumerate(test_loader):
            print(
                20 * "-"
                + f"Working with batch {batch_id + 1} / {len(test_loader)}"
                + 20 * "-"
            )
            simplex = Simplex(corpus_features, corpus_latent_reps)
            test_features = batch[0].to(device)
            test_latent_reps = classifier.latent_representation(test_features).detach()
            simplex.fit(
                test_features,
                test_latent_reps,
                n_keep=n_keep,
                reg_factor=0.1,
                reg_factor_scheduler=scheduler,
                n_epoch=20000,
            )
            weights_batch = simplex.weights.cpu().numpy()
            weights[
                batch_id * len(weights_batch) : (batch_id + 1) * len(weights_batch), :
            ] = weights_batch

        with open(save_path / f"simplex_weights_cv{cv}_n{n_keep}.pkl", "wb") as f:
            pkl.dump(weights, f)

    print(30 * "-" + "Computing Influence Functions" + 30 * "-")
    ptif.init_logging()
    config = ptif.get_default_config()
    config["outdir"] = str(save_path)
    config["test_sample_num"] = False
    ptif.calc_img_wise(config, classifier, corpus_loader, test_loader)
    # Delete temporary files, rename the result file
    for root, dirs, files in os.walk(save_path):
        for name in files:
            if "influence_results_tmp" in name or "influences_results_meta_0" in name:
                os.remove(save_path / name)
            elif name == "influence_results_0_False.json":
                os.rename(
                    save_path / name, save_path / f"influence_functions_cv{cv}.json"
                )


def outlier_detection(
    cv: int = 0,
    random_seed: int = 42,
    save_path: str = "experiments/results/mnist/outlier/",
    train: bool = True,
) -> None:
    torch.random.manual_seed(random_seed + cv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epoch_simplex = 10000
    K = 5

    print(
        100 * "-" + "\n" + "Welcome in the outlier detection experiment for MNIST. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv}.\n" + 100 * "-"
    )
    current_path = Path.cwd()
    save_path = current_path / save_path
    # Create saving directory if inexistent
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Training a model, save it
    if train:
        print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
        train_model(
            device=device,
            random_seed=random_seed,
            cv=cv,
            save_path=save_path,
            model_reg_factor=0,
        )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(os.path.join(save_path, f"model_cv{cv}.pth")))
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
    batch_id_test_mnist, (mnist_test_features, mnist_test_target) = next(
        mnist_test_examples
    )
    mnist_test_features = mnist_test_features.to(device).detach()
    emnist_test_examples = enumerate(emnist_test_loader)
    batch_id_test_emnist, (emnist_test_features, emnist_test_target) = next(
        emnist_test_examples
    )
    emnist_test_features = emnist_test_features.to(device).detach()
    test_features = torch.cat([mnist_test_features, emnist_test_features], dim=0)
    corpus_latent_reps = classifier.latent_representation(corpus_features).detach()
    test_latent_reps = classifier.latent_representation(test_features).detach()

    # Fit corpus:
    simplex = Simplex(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps
    )
    simplex.fit(
        test_examples=test_features,
        test_latent_reps=test_latent_reps,
        n_epoch=n_epoch_simplex,
        reg_factor=0,
        n_keep=corpus_features.shape[0],
    )
    explainer_path = save_path / f"simplex_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving simplex decomposition in {explainer_path}.")
        pkl.dump(simplex, f)
    nn_uniform = NearNeighLatent(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps
    )
    nn_uniform.fit(test_features, test_latent_reps, n_keep=K)
    nn_dist = NearNeighLatent(
        corpus_examples=corpus_features,
        corpus_latent_reps=corpus_latent_reps,
        weights_type="distance",
    )
    nn_dist.fit(test_features, test_latent_reps, n_keep=K)
    explainer_path = save_path / f"nn_dist_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving nn_dist decomposition in {explainer_path}.")
        pkl.dump(nn_dist, f)
    explainer_path = save_path / f"nn_uniform_cv{cv}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving nn_uniform decomposition in {explainer_path}.")
        pkl.dump(nn_uniform, f)


def jacobian_corruption(
    random_seed=42,
    save_path="experiments/results/mnist/jacobian_corruption/",
    corpus_size=500,
    test_size=500,
    n_bins=100,
    batch_size=50,
    train: bool = True,
) -> None:
    print(
        100 * "-" + "\n" + "Welcome in the Jacobian Projection check for MNIST. \n"
        f"Settings: random_seed = {random_seed} .\n" + 100 * "-"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(random_seed)
    n_pert_list = [1, 5, 10, 50]
    metric_data = []
    df_columns = ["Method", "N_pert", "Residual Shift"]
    current_folder = Path.cwd()
    save_path = current_folder / save_path
    # Create saving directory if inexistent
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Training a model, save it
    if train:
        print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
        train_model(
            device=device,
            random_seed=random_seed,
            cv=0,
            save_path=save_path,
            model_reg_factor=0,
        )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / "model_cv0.pth"))
    classifier.to(device)
    classifier.eval()

    # Prepare the corpus and the test set
    corpus_loader = load_mnist(
        subset_size=corpus_size, train=True, batch_size=batch_size
    )
    test_loader = load_mnist(
        subset_size=test_size, train=False, batch_size=1, shuffle=False
    )
    Corpus_inputs = torch.zeros((corpus_size, 1, 28, 28), device=device)

    # Prepare the IG baseline
    ig_explainer = captum.attr.IntegratedGradients(classifier)
    Corpus_inputs_pert_jp = torch.zeros(
        (len(n_pert_list), corpus_size, 1, 28, 28), device=device
    )
    Corpus_inputs_pert_ig = torch.zeros(
        (len(n_pert_list), corpus_size, 1, 28, 28), device=device
    )

    for test_id, (test_input, _) in enumerate(test_loader):
        print(
            25 * "="
            + f"Now working with test sample {test_id + 1}/{test_size}"
            + 25 * "="
        )
        for batch_id, (corpus_inputs, corpus_targets) in enumerate(corpus_loader):
            print(
                f"Now working with corpus batch {batch_id + 1}/{math.ceil(corpus_size / batch_size)}."
            )
            test_input = test_input.to(device)
            corpus_inputs = corpus_inputs.to(device).requires_grad_()
            baseline_inputs = -0.4242 * torch.ones(corpus_inputs.shape, device=device)
            input_shift = corpus_inputs - baseline_inputs
            test_latent = classifier.latent_representation(test_input).detach()
            baseline_latents = classifier.latent_representation(baseline_inputs)
            latent_shift = test_latent - baseline_latents
            latent_shift_sqrdnorm = torch.sum(latent_shift**2, dim=-1, keepdim=True)
            input_grad = torch.zeros(corpus_inputs.shape, device=corpus_inputs.device)
            for n in range(1, n_bins + 1):
                t = n / n_bins
                inputs = baseline_inputs + t * (corpus_inputs - baseline_inputs)
                latent_reps = classifier.latent_representation(inputs)
                latent_reps.backward(gradient=latent_shift / latent_shift_sqrdnorm)
                input_grad += corpus_inputs.grad
                corpus_inputs.grad.data.zero_()
            jacobian_projections = input_shift * input_grad / n_bins
            integrated_gradients = ig_explainer.attribute(
                corpus_inputs,
                baseline_inputs,
                target=corpus_targets.to(device),
                n_steps=n_bins,
            )
            saliency_jp = torch.abs(jacobian_projections).detach()
            saliency_ig = torch.abs(integrated_gradients).detach()
            lower_id = batch_id * batch_size
            higher_id = lower_id + batch_size
            Corpus_inputs[lower_id:higher_id] = corpus_inputs.detach()

            for pert_id, n_pert in enumerate(n_pert_list):
                top_pixels_jp = torch.topk(saliency_jp.view(batch_size, -1), k=n_pert)[
                    1
                ]
                top_pixels_ig = torch.topk(saliency_ig.view(batch_size, -1), k=n_pert)[
                    1
                ]
                mask_jp = torch.zeros(corpus_inputs.shape, device=device)
                mask_ig = torch.zeros(corpus_inputs.shape, device=device)
                for k in range(n_pert):
                    mask_jp[
                        :, 0, top_pixels_jp[:, k] // 28, top_pixels_jp[:, k] % 28
                    ] = 1
                    mask_ig[
                        :, 0, top_pixels_ig[:, k] // 28, top_pixels_ig[:, k] % 28
                    ] = 1
                corpus_inputs_pert_jp = (
                    mask_jp * baseline_inputs + (1 - mask_jp) * corpus_inputs
                )
                corpus_inputs_pert_ig = (
                    mask_ig * baseline_inputs + (1 - mask_ig) * corpus_inputs
                )
                Corpus_inputs_pert_jp[
                    pert_id, lower_id:higher_id
                ] = corpus_inputs_pert_jp
                Corpus_inputs_pert_ig[
                    pert_id, lower_id:higher_id
                ] = corpus_inputs_pert_ig

        print("Now fitting the uncorrupted SimplEx")
        test_latent = classifier.latent_representation(test_input).detach().to(device)
        simplex = Simplex(
            Corpus_inputs, classifier.latent_representation(Corpus_inputs).detach()
        )
        simplex.fit(test_input, test_latent, reg_factor=0)
        residual = torch.sqrt(torch.sum((test_latent - simplex.latent_approx()) ** 2))

        for pert_id, n_pert in enumerate(n_pert_list):
            print(
                f"Now fitting the JP-corrupted SimplEx with {n_pert} perturbation(s) per image"
            )
            simplex_jp = Simplex(
                Corpus_inputs_pert_jp[pert_id],
                classifier.latent_representation(
                    Corpus_inputs_pert_jp[pert_id]
                ).detach(),
            )
            simplex_jp.fit(test_input, test_latent, reg_factor=0)
            residual_jp = torch.sqrt(
                torch.sum((test_latent - simplex_jp.latent_approx()) ** 2)
            )

            print(
                f"Now fitting the IG-corrupted SimplEx with {n_pert} perturbation(s) per image"
            )
            simplex_ig = Simplex(
                Corpus_inputs_pert_ig[pert_id],
                classifier.latent_representation(
                    Corpus_inputs_pert_ig[pert_id]
                ).detach(),
            )
            simplex_ig.fit(test_input, test_latent, reg_factor=0)
            residual_ig = torch.sqrt(
                torch.sum((test_latent - simplex_ig.latent_approx()) ** 2)
            )
            metric_data.append(
                ["SimplEx", n_pert, (residual_jp - residual).cpu().numpy().item()]
            )
            metric_data.append(
                ["IG", n_pert, (residual_ig - residual).cpu().numpy().item()]
            )

    metric_df = pd.DataFrame(metric_data, columns=df_columns)
    sns.set_palette("colorblind")
    sns.boxplot(data=metric_df, x="N_pert", y="Residual Shift", hue="Method")
    plt.xlabel("Number of pixels perturbed")
    plt.savefig(save_path / "box_plot.pdf")


def timing_experiment() -> None:
    print(100 * "-" + "\n" + "Welcome in timing experiment for MNIST. \n" + 100 * "-")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(42)
    n_bins = 30
    batch_size = 50
    test_size = 100
    CV = 10
    times = np.zeros((4, CV))
    load_path = Path.cwd() / "experiments/results/mnist/quality"
    if not (load_path / "model_cv0.pth").exists():
        raise RuntimeError(
            "The timing experiment should be run after the approximation quality experiment."
        )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(load_path / "model_cv0.pth"))
    classifier.to(device)
    classifier.eval()

    for cv in range(CV):
        # Prepare the corpus and the test set
        corpus_loader = load_mnist(subset_size=1000, train=True, batch_size=batch_size)
        test_loader_single = load_mnist(
            subset_size=1, train=False, batch_size=1, shuffle=False
        )
        test_loader_multiple = load_mnist(
            subset_size=test_size, train=False, batch_size=test_size, shuffle=False
        )
        Corpus_inputs = torch.zeros((1000, 1, 28, 28), device=device)

        for test_id, (test_input, _) in enumerate(test_loader_single):
            print(
                25 * "="
                + f"Now working with test sample {test_id + 1}/{len(test_loader_single)}"
                + 25 * "="
            )
            test_latent = classifier.latent_representation(
                test_input.to(device)
            ).detach()
            t1 = time.time()
            for batch_id, (corpus_inputs, corpus_targets) in enumerate(corpus_loader):
                print(
                    f"Now working with corpus batch {batch_id + 1}/{len(corpus_loader)}."
                )
                test_input = test_input.to(device)
                corpus_inputs = corpus_inputs.to(device).requires_grad_()
                baseline_inputs = -0.4242 * torch.ones(
                    corpus_inputs.shape, device=device
                )
                baseline_latents = classifier.latent_representation(baseline_inputs)
                latent_shift = test_latent - baseline_latents
                latent_shift_sqrdnorm = torch.sum(
                    latent_shift**2, dim=-1, keepdim=True
                )
                input_grad = torch.zeros(
                    corpus_inputs.shape, device=corpus_inputs.device
                )
                for n in range(1, n_bins + 1):
                    t = n / n_bins
                    inputs = baseline_inputs + t * (corpus_inputs - baseline_inputs)
                    latent_reps = classifier.latent_representation(inputs)
                    latent_reps.backward(gradient=latent_shift / latent_shift_sqrdnorm)
                    input_grad += corpus_inputs.grad
                    corpus_inputs.grad.data.zero_()
                lower_id = batch_id * batch_size
                higher_id = lower_id + batch_size
                Corpus_inputs[lower_id:higher_id] = corpus_inputs.detach()
            t2 = time.time()
            times[2, cv] = t2 - t1

        for test_id, (test_inputs, _) in enumerate(test_loader_multiple):
            test_latents = classifier.latent_representation(
                test_inputs.to(device)
            ).detach()
            t1 = time.time()
            Corpus_latents = classifier.latent_representation(
                Corpus_inputs.to(device)
            ).detach()
            simplex = Simplex(Corpus_inputs, Corpus_latents)
            simplex.fit(test_inputs, test_latents, reg_factor=0, n_epoch=1000)
            simplex.latent_approx()
            t2 = time.time()
            times[0, cv] = (t2 - t1) / test_size

            t1 = time.time()
            Corpus_latents = classifier.latent_representation(
                Corpus_inputs.to(device)
            ).detach()
            knn = NearNeighLatent(Corpus_inputs, Corpus_latents)
            knn.fit(test_inputs, test_latents)
            knn.latent_approx()
            t2 = time.time()
            times[1, cv] = (t2 - t1) / test_size

        # Influence Functions
        t1 = time.time()
        ptif.init_logging()
        config = ptif.get_default_config()
        config["outdir"] = str(load_path)
        config["test_sample_num"] = False
        ptif.calc_img_wise(config, classifier, corpus_loader, test_loader_single)
        t2 = time.time()
        times[3, cv] = t2 - t1

    print(np.mean(times, axis=-1))
    print(np.std(times, axis=-1))


def main(experiment: str, cv: int) -> None:
    if experiment == "approximation_quality":
        approximation_quality(cv=cv, n_keep_list=[3, 5, 10, 20, 50])
    elif experiment == "outlier_detection":
        outlier_detection(cv)
    elif experiment == "jacobian_corruption":
        jacobian_corruption(test_size=100, train=True)
    elif experiment == "influence":
        influence_function(n_keep_list=[2, 5, 10, 20, 50], cv=cv)
    elif experiment == "timing":
        timing_experiment()
    else:
        raise ValueError(
            "The name of the experiment is not valid. "
            "Valid names are: "
            "approximation_quality , outlier_detection , jacobian_corruption, influence, timing."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    args = parser.parse_args()
    main(args.experiment, args.cv)
