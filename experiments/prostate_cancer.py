import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch
import sklearn
import os
import torch.nn.functional as F
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.tabular_data import MortalityPredictor
from sklearn.model_selection import train_test_split
from explainers.corpus import Corpus
from explainers.nearest_neighbours import NearNeighLatent
from explainers.representer import Representer
from utils.schedulers import ExponentialScheduler


class ProstateCancerDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y.astype(int)

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = torch.tensor(self.X.iloc[i, :], dtype=torch.float32)
        target = self.y.iloc[i]
        return data, target


def load_seer(random_seed: int=42):
    features = ['age', 'psa', 'comorbidities', 'treatment_CM', 'treatment_Primary hormone therapy',
                'treatment_Radical Therapy-RDx', 'treatment_Radical therapy-Sx', 'grade_1.0', 'grade_2.0', 'grade_3.0',
                'grade_4.0', 'grade_5.0', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'gleason1_1', 'gleason1_2',
                'gleason1_3', 'gleason1_4', 'gleason1_5', 'gleason2_1', 'gleason2_2', 'gleason2_3', 'gleason2_4',
                'gleason2_5']
    label = 'mortCancer'
    df = pd.read_csv('./data/Prostate Cancer/seer_external_imputed_new.csv')
    mask = df[label] == True
    df_dead = df[mask]
    df_survive = df[~mask]
    df = pd.concat([df_dead.sample(12000, random_state=random_seed),
                    df_survive.sample(12000, random_state=random_seed)])
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]


def approximation_quality(cv: int = 0, random_seed: int = 42, save_path: str = './results/prostate/quality/',
                          train_model: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.random.manual_seed(random_seed + cv)

    print(100 * '-' + '\n' + 'Welcome in the approximation quality experiment for Prostate Cancer. \n'
                             f'Settings: random_seed = {random_seed} ; cv = {cv} ; device = {device}.\n'
          + 100 * '-')

    # Define parameters
    n_epoch_model = 5
    log_interval = 100
    weight_decay = 1e-5
    corpus_size = 100
    test_size = 100
    n_keep_list = [n for n in range(2, 10)] + [n for n in range(10, 55, 5)]
    reg_factor_init = 0.01
    reg_factor_final = 10.0
    n_epoch_simplex = 10000
    learning_rate_simplex = 100
    momentum_simplex = 0.5


    if not os.path.exists(save_path):
        print(f'Creating the saving directory {save_path}')
        os.mkdir(save_path)

    # Load the data
    X, y = load_seer(random_seed=random_seed+cv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_seed + cv,
                                                        stratify=y)

    train_data = ProstateCancerDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_data = ProstateCancerDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=True)


    if train_model:
        # Create the model
        classifier = MortalityPredictor(n_cont=3)
        classifier.to(device)
        optimizer = optim.Adam(classifier.parameters(), weight_decay=weight_decay)

        # Train the model
        print(100 * '-' + '\n' + 'Now fitting the model. \n' + 100 * '-')
        train_losses = []
        train_counter = []
        test_losses = []

        def train(epoch):
            classifier.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.type(torch.LongTensor)
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
                        (batch_idx * 128) + ((epoch - 1) * len(train_loader.dataset)))
                    torch.save(classifier.state_dict(), os.path.join(save_path, f'model_cv{cv}.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_cv{cv}.pth'))

        def test():
            classifier.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.type(torch.LongTensor)
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
        for epoch in range(1, n_epoch_model + 1):
            train(epoch)
            test()
        torch.save(classifier.state_dict(), os.path.join(save_path, f'model_cv{cv}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_cv{cv}.pth'))

    # Load model:
    classifier = MortalityPredictor(n_cont=3)
    classifier.load_state_dict(torch.load(os.path.join(save_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()

    # Load data for the explainers
    print(100 * '-' + '\n' + 'Now fitting the explainers. \n' + 100 * '-')

    explainer_names = ['simplex', 'nn_uniform', 'nn_dist']
    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test, (test_data, test_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device).detach()
    test_data = test_data.to(device).detach()
    corpus_latent_reps = classifier.latent_representation(corpus_data).detach()
    corpus_probas = classifier.probabilities(corpus_data).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[torch.arange(corpus_size), corpus_target.type(torch.LongTensor)] = 1
    test_latent_reps = classifier.latent_representation(test_data).detach()


    # Save data:
    corpus_data_path = os.path.join(save_path, f'corpus_data_cv{cv}.pkl')
    with open(corpus_data_path, 'wb') as f:
        print(f'Saving corpus data in {corpus_data_path}.')
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = os.path.join(save_path, f'test_data_cv{cv}.pkl')
    with open(test_data_path, 'wb') as f:
        print(f'Saving test data in {test_data_path}.')
        pkl.dump([test_latent_reps, test_targets], f)

    # Fit the explainers
    for n_keep in n_keep_list:
        print(30*'-' + f'n_keep = {n_keep}' + 30*'-')
        explainers = []
        # Fit corpus:
        reg_factor_scheduler = ExponentialScheduler(reg_factor_init, reg_factor_final, n_epoch_simplex)
        corpus = Corpus(corpus_examples=corpus_data,
                        corpus_latent_reps=corpus_latent_reps)
        weights = corpus.fit(test_examples=test_data,
                             test_latent_reps=test_latent_reps,
                             n_epoch=n_epoch_simplex, learning_rate=learning_rate_simplex, momentum=momentum_simplex,
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

        # Save explainers:
        for explainer, explainer_name in zip(explainers, explainer_names):
            explainer_path = os.path.join(save_path, f'{explainer_name}_cv{cv}_n{n_keep}.pkl')
            with open(explainer_path, 'wb') as f:
                print(f'Saving {explainer_name} decomposition in {explainer_path}.')
                pkl.dump(explainer, f)
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy())
            output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
            print(f'{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}.')

    # Fit the representer:
    representer = Representer(corpus_latent_reps=corpus_latent_reps,
                              corpus_probas=corpus_probas,
                              corpus_true_classes=corpus_true_classes,
                              reg_factor=weight_decay)
    representer.fit(test_latent_reps=test_latent_reps)
    explainer_path = os.path.join(save_path, f'representer_cv{cv}.pkl')
    with open(explainer_path, 'wb') as f:
        print(f'Saving representer decomposition in {explainer_path}.')
        pkl.dump(representer, f)
    latent_rep_true = representer.test_latent_reps
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_approx = representer.output_approx()
    output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
    print(f'representer output r2 = {output_r2_score:.2g}.')



def main(experiment: str = 'approximation_quality', cv: int = 0):
    if experiment == 'approximation_quality':
        approximation_quality(cv=cv)


parser = argparse.ArgumentParser()
parser.add_argument('-experiment', type=str, default='approximation_quality', help='Experiment to perform')
parser.add_argument('-cv', type=int, default=0, help='Cross validation parameter')
args = parser.parse_args()

if __name__ == '__main__':
    main(args.experiment, args.cv)
