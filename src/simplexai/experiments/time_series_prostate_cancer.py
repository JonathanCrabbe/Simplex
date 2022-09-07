import os

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

pd.set_option("display.max_columns", None)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TimeSeriesProstateCancerDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X[i], dtype=torch.float32)
        target = torch.tensor(self.y[i], dtype=torch.float32)
        return data, target


def load_time_series_prostate_cancer(random_seed: int = 42) -> tuple:
    temporal_features = [
        "Days Since Diagnosis",
        "Repeat PSA",
        "Repeat Biopsy Core Total",
        "Repeat Biopsy Core Positive",
        "Repeat Biopsy Primary Gleason",
        "Repeat Biopsy Secondary Gleason",
        "Repeat Biopsy Grade Group",
        "Repeat MRI PRECISE Scoring",
        "Repeat MRI Stage",
        "Repeat MRI Volume",
        "Repeat MRI PSAd",
    ]
    constant_features = [
        "Exact age at diagnosis",
        "Number of negative biopsies before diagnosis",
        "Number of MRI-visible lesions",
        "Ethnicity",
        "Family History of Prostate Cancer",
        "CPG",
        "PI-RADS score",
        "STRATCANS (simplified)",
        "Days since diagnosis.3",
    ]
    categorical_features = [
        "Repeat Biopsy Primary Gleason",
        "Repeat Biopsy Secondary Gleason",
        "Repeat Biopsy Grade Group",
        "Repeat MRI PRECISE Scoring",
        "Repeat MRI Stage",
        "Ethnicity",
        "Family History of Prostate Cancer",
        "CPG",
        "PI-RADS score",
        "STRATCANS (simplified)",
    ]
    label = "Coding.3"
    temporal_df = pd.read_csv(
        os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "./data/Time series Prostate Cancer/temporal.csv",
            )
        )
    )
    max_time_points = temporal_df["New ID"].value_counts().max()
    const_df = pd.read_csv(
        os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "./data/Time series Prostate Cancer/baseline.csv",
            )
        )
    )
    y = const_df[label].to_numpy()

    df = temporal_df.merge(const_df, on="New ID", how="left")
    # categorical_df = df[categorical_features]

    # Get dummies
    all_features = constant_features + temporal_features + ["New ID"]
    df = pd.get_dummies(df[all_features], columns=categorical_features)
    all_features = [col for col in df.columns if col != "New ID"]

    # Scaling
    rescale_dict = df[all_features].max().to_dict()
    scaler = MinMaxScaler()
    df[all_features] = scaler.fit_transform(df[all_features])

    # Balance the dataset (not in use - use weights instead)
    # grouped_df = df.groupby(by="New ID")
    # df = pd.concat([df for idx, df in grouped_df][:200])

    # Limit columns to features
    df = df[all_features + ["New ID"]]

    group_df = df.groupby(by="New ID").cumcount()
    mux = pd.MultiIndex.from_product([df["New ID"].unique(), group_df.unique()])
    X = (
        df.set_index(["New ID", group_df])
        .reindex(mux, fill_value=0)
        .groupby(level=0)
        .apply(lambda x: x.values)
        .to_numpy()
    )

    # mask = const_df[label] == 1
    # const_df = sklearn.utils.shuffle(const_df, random_state=random_seed)
    # const_df = const_df.reset_index(drop=True)

    return X, y, all_features, max_time_points, rescale_dict
