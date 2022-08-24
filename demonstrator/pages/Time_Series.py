import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from explainers.simplex import Simplex
from models.tabular_data import MortalityPredictor
from models.two_linear_layers import TwoLayerMortalityPredictor
from models.linear_regression import LinearRegression
from models.recurrent_neural_net import MortalityGRU
from experiments.prostate_cancer import (
    load_seer,
    load_cutract,
    ProstateCancerDataset,
)
from experiments.time_series_prostate_cancer import (
    load_time_series_prostate_cancer,
    TimeSeriesProstateCancerDataset,
)
from experiments.breast_cancer import load_breast_cancer_seer

# page config
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 550px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 550px;
        margin-left: -550px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Page
st.write(
    """
# SimplEx Demo

On this page SimplEx is used to explain the predictions of recurrent Neural
Network on a time series prostate cancer dataset. Then in the side bar,
please select a patient to review from the test set with the Patient ID slider.
You can also change the number of example patients from the corpus that are
displayed by adjusting the "Minimum Example Importance" slider.
"""
)


def apply_sort_order(in_list, sort_order):
    if isinstance(in_list, list):
        return [in_list[idx] for idx in sort_order]
    if torch.is_tensor(in_list):
        return [in_list.numpy()[idx] for idx in sort_order]


def load_data(random_seed=42, corpus_size=100, batch_size=50):

    # Load corpus and test inputs
    (
        X,
        y,
        feature_names,
        max_time_points,
        rescale_dict,
    ) = load_time_series_prostate_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_seed, stratify=y
    )
    class_imbalance_weighting = sum(y_train == 0) / len(y_train)

    train_data = TimeSeriesProstateCancerDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TimeSeriesProstateCancerDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    test_examples = enumerate(test_loader)
    batch_id_test, (test_inputs, test_targets) = next(test_examples)

    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=False)
    corpus_examples = enumerate(corpus_loader)
    batch_id_corpus, (corpus_inputs, corpus_targets) = next(corpus_examples)

    input_baseline = torch.mean(torch.mean(corpus_inputs, 1), 0).expand(
        100, max_time_points, -1
    )  # Baseline tensor of the same shape as corpus_inputs
    return (
        train_loader,
        test_loader,
        corpus_inputs,
        corpus_targets,
        test_inputs,
        test_targets,
        max_time_points,
        feature_names,
        class_imbalance_weighting,
        input_baseline,
        rescale_dict,
    )


def load_trained_model(model, trained_model_state_path):
    model.load_state_dict(torch.load(trained_model_state_path))
    # model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def get_simplex_decomposition(i, model, input_baseline):
    simplex.jacobian_projection(test_id=i, model=model, input_baseline=input_baseline)
    result, sort_order = simplex.decompose(i, return_id=True)
    return result, sort_order


st.write("## Patient Breakdown")

model_name2mortalitymodel = {
    "Recurrent Neural Net": MortalityGRU,
}

model_name2trained_model_repo_name = {
    "Recurrent Neural Net": "RNN",
}

# Load the data
(
    train_loader,
    test_loader,
    corpus_inputs,
    corpus_targets,
    test_inputs,
    test_targets,
    max_time_points,
    feature_names,
    class_imbalance_weighting,
    input_baseline,
    rescale_dict,
) = load_data(
    random_seed=5,
    corpus_size=100,
    batch_size=50,
)

# Get a trained model
model = MortalityGRU(
    input_dim=len(feature_names),
    hidden_dim=5,
    output_dim=1,
    n_layers=1,
)  # Model should have the BlackBox interface

TRAINED_MODEL_STATE_PATH = os.path.abspath(
    f"./resources/trained_models/{model_name2trained_model_repo_name['Recurrent Neural Net']}/Time series Prostate Cancer/model_cv1.pth"
)
model = load_trained_model(model, TRAINED_MODEL_STATE_PATH)

# Compute the corpus and test latent representations
corpus_latents = model.latent_representation(corpus_inputs).detach()
test_latents = model.latent_representation(test_inputs).detach()


# Load fitted SimplEx
with open(
    f"./resources/trained_models/{model_name2trained_model_repo_name['Recurrent Neural Net']}/Time series Prostate Cancer/simplex.pkl",
    "rb",
) as f:
    simplex = pkl.load(f)


# Compute corpus and test model predictions
corpus_predictions = model.forward(corpus_inputs).detach().round()
test_predictions = model.forward(test_inputs).detach().round()

##################################################################################
def user_input_features():
    test_patient_id = st.sidebar.slider("Patient ID", 1, 50, 1)
    example_importance_threshold = st.sidebar.slider(
        "Minimum Example Importance", 0.0, 1.0, 0.1
    )
    corpus_patient_sort = st.sidebar.radio(
        "Corpus patient feature sort:",
        ["None", "Highest importance first", "Lowest importance first"],
    )
    return test_patient_id - 1, example_importance_threshold, corpus_patient_sort


# Sidebar
st.sidebar.header("Test Patient")
(
    test_patient_id,
    example_importance_threshold,
    corpus_patient_sort,
) = user_input_features()


def show_thumb(label):
    if label == 0:
        return st.image("./resources/thumb_up.png", width=40)
    elif label == 1:
        return st.image("./resources/thumb_down.png", width=40)


# Test patient image and data

st.sidebar.header(f"Patient number: {test_patient_id+1}")

show_patient_outcome = st.sidebar.checkbox(
    "Show test patient Real-world Outcome", value=False, key="show_patient_outcome"
)
col1, col2, col3 = st.sidebar.columns(3, gap="small")
with col1:
    st.image("./resources/patient_male.png", width=100)
with col2:
    st.write("Model Prediction:")
    if show_patient_outcome:
        st.write("Real-world Outcome:")
with col3:
    prediction = test_predictions[test_patient_id]
    show_thumb(prediction)
    if show_patient_outcome:
        label = test_targets[test_patient_id].item()
        st.write("")
        show_thumb(label)


# Test patient sidebar
st.sidebar.write("Patient features")
test_patient_last_time_step_idx = (
    simplex.test_examples[test_patient_id][
        ~np.all(simplex.test_examples[test_patient_id].numpy() == 0, axis=1)
    ].shape[0]
    - 1
)

test_patient_df = pd.DataFrame(
    simplex.test_examples[test_patient_id][
        test_patient_last_time_step_idx - 2 : test_patient_last_time_step_idx + 1, :
    ].numpy(),
    columns=feature_names,
    index=["(t_max) - 2", "(t_max) - 1", "(t_max)"],
)

st.sidebar.write(test_patient_df.transpose().astype(float).round(4))

# Main corpus decomposition area
result, sort_order = get_simplex_decomposition(test_patient_id, model, input_baseline)

last_time_step_idx = [
    result[j][1][~np.all(result[j][1].numpy() == 0, axis=1)].shape[0] - 1
    for j in range(len(result))
]

corpus_dfs = [
    pd.DataFrame(
        result[j][1][idx - 2 : idx + 1].numpy(),
        columns=feature_names,
    )
    for j, idx in zip(range(len(result)), last_time_step_idx)
]
for corpus_df in corpus_dfs:
    for col_name, rescale_value in rescale_dict.items():
        corpus_df[col_name] = corpus_df[col_name].apply(lambda x: x * rescale_value)
corpus_data = [
    {
        "feature_vals": corpus_dfs[i].transpose(),
        "Label": apply_sort_order(corpus_targets, sort_order)[i],
        "Prediction": apply_sort_order(corpus_predictions, sort_order)[i],
        "Example Importance": result[i][0],
    }
    for i in range(len(corpus_dfs))
]


importance_dfs = [
    pd.DataFrame(
        result[j][2][idx - 2 : idx + 1].numpy(),
        columns=[f"{col}_fi" for col in feature_names],
    )
    for j, idx in zip(range(len(result)), last_time_step_idx)
]
importance_data = [
    {
        "importance_vals": importance_dfs[i].transpose(),
        "Label": apply_sort_order(corpus_targets, sort_order)[i],
        "Prediction": apply_sort_order(corpus_predictions, sort_order)[i],
        "Example Importance": result[i][0],
    }
    for i in range(len(corpus_dfs))
]


corpus_data = [
    example
    for example in corpus_data
    if example["Example Importance"] >= example_importance_threshold
]
importance_data = [
    example
    for example in importance_data
    if example["Example Importance"] >= example_importance_threshold
]

if len(corpus_data) == 0:
    st.write(
        "All corpus examples have a lower importance than the Minimum Example Importance threshold. Use the slider in the sidebar to reduce the Minimum Example Importance."
    )


def df_values_to_colors(df):
    """Gets color values based in values relative to all other values in df."""

    min_val = np.nanmin(df.values)
    max_val = np.nanmax(df.values)
    for col in df:
        # map values to colors in hex via
        # creating a hex Look up table table and apply the normalized data to it
        norm = mcolors.Normalize(
            vmin=min_val,
            vmax=max_val,
            # vmin=np.nanmin(df[col].values),
            # vmax=np.nanmax(df[col].values),
            clip=True,
        )
        lut = plt.cm.bwr(np.linspace(0.2, 0.75, 256))
        lut = np.apply_along_axis(mcolors.to_hex, 1, lut)
        a = (norm(df[col].values) * 255).astype(np.int16)
        df[col] = lut[a]
    return df


def highlight(x):
    return pd.DataFrame(importance_df_colors.values, index=x.index, columns=x.columns)


# Display icons
display_icon_cols = 3
max_display_corpus_patients = 6
if len(corpus_data) > max_display_corpus_patients:
    corpus_data = corpus_data[:max_display_corpus_patients]
    importance_data = importance_data[:max_display_corpus_patients]

corpus_patient_columns = st.columns(display_icon_cols)

for example_i in range(len(corpus_data)):
    with corpus_patient_columns[example_i % display_icon_cols]:
        st.image("./resources/patient_male.png", width=100)
        if corpus_patient_sort == "None":
            importance_df_colors = df_values_to_colors(
                importance_data[example_i]["importance_vals"].copy()
            )
        else:
            ascending = (
                True if corpus_patient_sort == "Lowest importance first" else False
            )
            importance_df_colors = df_values_to_colors(
                importance_data[example_i]["importance_vals"]
                .copy()
                .sort_values(
                    by=importance_data[example_i]["importance_vals"].columns[-1],
                    ascending=ascending,
                )
            )
        importance_df_colors = importance_df_colors.applymap(
            lambda x: f"background-color: {x}"
        )
        st.write(
            f"SimplEx Example importance: {corpus_data[example_i]['Example Importance']*100:0.1f}%"
        )
        # # TODO: get thumb on same line as text see: https://discuss.streamlit.io/t/image-and-text-next-to-each-other/7627/17
        st.write("Model Prediction:")
        show_thumb(corpus_data[example_i]["Prediction"])
        st.write("Real-world Outcome:")
        show_thumb(corpus_data[example_i]["Label"])
        with pd.option_context("display.max_colwidth", 5):
            if corpus_patient_sort == "None":
                styled_display_df = (
                    corpus_data[example_i]["feature_vals"].rename(
                        columns={0: "(t_max) - 2", 1: "(t_max) - 1", 2: "(t_max)"}
                    )
                ).style.apply(highlight, axis=None)
            else:
                ascending = (
                    True if corpus_patient_sort == "Lowest importance first" else False
                )
                styled_display_df = (
                    corpus_data[example_i]["feature_vals"]
                    .assign(
                        s=importance_data[example_i]["importance_vals"]
                        .iloc[:, -1]
                        .values
                    )
                    .sort_values(by="s", ascending=ascending)
                    .drop("s", axis=1)
                    .rename(columns={0: "(t_max) - 2", 1: "(t_max) - 1", 2: "(t_max)"})
                ).style.apply(highlight, axis=None)

            st.write(styled_display_df)
