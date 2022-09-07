import os

import pandas as pd
import sklearn
import torch
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BreastCancerDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X.iloc[i, :], dtype=torch.float32)
        target = self.y.iloc[i]
        return data, target


def load_breast_cancer_seer(random_seed: int = 42) -> tuple:
    features = [
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Reginal Node Positive",
        "Race_Black",
        "Race_Other",
        "Race_White",
        "Marital Status_Divorced",
        "Marital Status_Married",
        "Marital Status_Separated",
        "Marital Status_Single",
        "Marital Status_Widowed",
        "T Stage_T1",
        "T Stage_T2",
        "T Stage_T3",
        "T Stage_T4",
        "N Stage_N1",
        "N Stage_N2",
        "N Stage_N3",
        "6th Stage_IIA",
        "6th Stage_IIB",
        "6th Stage_IIIA",
        "6th Stage_IIIB",
        "6th Stage_IIIC",
        "Grade I",
        "Grade II",
        "Grade III",
        "Grade IV",
        "A Stage_Distant",
        "A Stage_Regional",
        "Estrogen Status_Positive",
        "Progesterone Status_Positive",
    ]
    label = "Status_Dead"
    df = pd.read_csv(
        os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "./data/Breast Cancer/SEER Breast Cancer Dataset_preprocessed.csv",
            )
        )
    )
    mask = df[label] is True
    df_dead = df[mask]
    df_survive = df[~mask]
    df = pd.concat(
        [
            df_dead.sample(
                600, random_state=random_seed
            ),  # Sample number low due to small amounts of data for patients who died (sample normally 12000)
            df_survive.sample(600, random_state=random_seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)
    return df[features], df[label]
