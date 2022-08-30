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

# Main Page
st.write(
    """
# SimplEx Demo

First, please select the dataset you wish to view and the model you wish
to debug. Then in the side bar, please select a patient to review from the
test set with the Patient ID slider. You can also change the number of
example patients from the corpus that are displayed by adjusting the
"Minimum Example Importance" slider.
"""
)


def apply_sort_order(in_list, sort_order):
    if isinstance(in_list, list):
        return [in_list[idx] for idx in sort_order]
    if torch.is_tensor(in_list):
        return [in_list.numpy()[idx] for idx in sort_order]


def load_data(
    data_source="breast_seer", random_seed=42, corpus_size=100, batch_size=50
):

    # Load corpus and test inputs
    # LOAD DATA from file
    data_load_funcs = {
        "prostate_seer": load_seer(random_seed=random_seed),
        "breast_seer": load_breast_cancer_seer(random_seed=random_seed),
        "prostate_cutract": load_cutract(random_seed=random_seed),
    }
    X, y = data_load_funcs[data_source.lower()]

    feature_names = X.columns

    # Get data into shape and produce corpus
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_seed, stratify=y
    )

    train_data = ProstateCancerDataset(X_train, y_train)

    test_data = ProstateCancerDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=False)
    test_examples = enumerate(test_loader)
    batch_id_test, (test_inputs, test_targets) = next(test_examples)

    corpus_loader = DataLoader(train_data, batch_size=corpus_size, shuffle=False)
    corpus_examples = enumerate(corpus_loader)
    batch_id_corpus, (corpus_inputs, corpus_targets) = next(corpus_examples)
    input_baseline = torch.median(corpus_inputs, dim=0, keepdim=True).values.expand(
        100, -1
    )  # Baseline tensor of the same shape as corpus_inputs

    return (
        corpus_inputs,
        corpus_targets,
        test_inputs,
        test_targets,
        input_baseline,
        feature_names,
    )


def load_trained_model(model, trained_model_state_path):
    model.load_state_dict(torch.load(trained_model_state_path))
    model.to("cpu")
    model.eval()
    return model


def get_simplex_decomposition(i, model, input_baseline):
    simplex.jacobian_projection(test_id=i, model=model, input_baseline=input_baseline)
    result, sort_order = simplex.decompose(i, return_id=True)
    return result, sort_order


dropdown_col1, dropdown_col2, _, col4 = st.columns(4)
with dropdown_col1:
    data_source = st.selectbox(
        "Dataset:",
        (
            "breast_seer",
            "prostate_seer",
            "prostate_cutract",
        ),
    )

with dropdown_col2:
    model_name = st.selectbox(
        "Predictive Model:",
        (
            "Multilayer Perceptron",
            "Linear Regression",
        ),
    )
st.write("## Patient Breakdown")

data2n_cont = {
    "breast_seer": 4,
    "prostate_seer": 3,
    "prostate_cutract": 3,
}

model_name2mortalitymodel = {
    "Multilayer Perceptron": MortalityPredictor,
    "Linear Regression": LinearRegression,
    "Two Layer Linear Regression": TwoLayerMortalityPredictor,
}

model_name2trained_model_repo_name = {
    "Multilayer Perceptron": "MLP",
    "Linear Regression": "LinearRegression",
    "Two Layer Linear Regression": "TwoLayerLinearRegression",
}

# Load the data
(
    corpus_inputs,
    corpus_targets,
    test_inputs,
    test_targets,
    input_baseline,
    feature_names,
) = load_data(data_source=data_source, random_seed=42, corpus_size=100, batch_size=50)

# Get a trained model
model = model_name2mortalitymodel[model_name](
    n_cont=data2n_cont[data_source], input_feature_num=len(feature_names)
)  # Model should have the BlackBox interface

TRAINED_MODEL_STATE_PATH = os.path.abspath(
    f"./resources/trained_models/{model_name2trained_model_repo_name[model_name]}/{data_source}/model_cv1.pth"
)
model = load_trained_model(model, TRAINED_MODEL_STATE_PATH)


# Compute the corpus and test latent representations
corpus_latents = model.latent_representation(corpus_inputs).detach()
test_latents = model.latent_representation(test_inputs).detach()


# Load fitted SimplEx
with open(
    f"./resources/trained_models/{model_name2trained_model_repo_name[model_name]}/{data_source}/simplex.pkl",
    "rb",
) as f:
    simplex = pkl.load(f)


# Compute corpus and test model predictions
corpus_predictions = [
    1 if prediction[1] > 0.5 else 0
    for prediction in model.probabilities(corpus_inputs).detach()
]

test_predictions = [
    1 if prediction[1] > 0.5 else 0
    for prediction in model.probabilities(test_inputs).detach()
]

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
    else:
        pass


# Test patient sidebar
st.sidebar.write("Patient features")

test_patient_df = pd.DataFrame(
    [simplex.test_examples[test_patient_id].numpy()],
    columns=feature_names,
    index=["Value"],
)


st.sidebar.write(test_patient_df.transpose())

# Main corpus decomposition area
result, sort_order = get_simplex_decomposition(test_patient_id, model, input_baseline)

corpus_df = pd.DataFrame(
    [result[j][1].numpy() for j in range(len(result))],
    columns=feature_names,
)
corpus_df.insert(
    loc=0, column="Label", value=apply_sort_order(corpus_targets, sort_order)
)
corpus_df.insert(
    loc=0, column="Prediction", value=apply_sort_order(corpus_predictions, sort_order)
)
corpus_df.insert(
    loc=0,
    column="Example Importance",
    value=[result[j][0] for j in range(len(result))],
)

feature_df = pd.DataFrame(
    [result[j][2].numpy() for j in range(len(result))],
    columns=[f"{col}_fi" for col in feature_names],
)
feature_df = feature_df.loc[
    corpus_df["Example Importance"] >= example_importance_threshold
]
display_df = corpus_df.loc[
    corpus_df["Example Importance"] >= example_importance_threshold
].copy()

if len(display_df) == 0:
    st.write(
        "All corpus examples have a lower importance than the Minimum Example Importance threshold. Use the slider in the sidebar to reduce the Minimum Example Importance."
    )


def df_values_to_colors(df):

    for col in df:
        # map values to colors in hex via
        # creating a hex Look up table table and apply the normalized data to it
        norm = mcolors.Normalize(
            vmin=np.nanmin(df[col].values),
            vmax=np.nanmax(df[col].values),
            clip=True,
        )
        lut = plt.cm.bwr(np.linspace(0.2, 0.75, 256))
        lut = np.apply_along_axis(mcolors.to_hex, 1, lut)
        a = (norm(df[col].values) * 255).astype(np.int16)
        df[col] = lut[a]
    return df


def highlight(x):
    return pd.DataFrame(feature_df_colors.values, index=x.index, columns=x.columns)


# Display icons
display_icon_cols = 3
display_df = display_df.transpose()
if len(display_df.columns) > 6:
    display_df = display_df.iloc[:, :6].copy()

corpus_patient_columns = st.columns(display_icon_cols)
corpus_patient_row_number = len(display_df.columns) // display_icon_cols
feature_df = feature_df.transpose()

for example_i in display_df.columns:
    with corpus_patient_columns[example_i % display_icon_cols]:
        st.image("./resources/patient_male.png", width=100)
        if corpus_patient_sort == "None":
            feature_df_colors = df_values_to_colors(feature_df[[example_i]].copy())
        else:
            ascending = (
                True if corpus_patient_sort == "Lowest importance first" else False
            )
            feature_df_colors = df_values_to_colors(
                feature_df[[example_i]]
                .copy()
                .sort_values(by=example_i, ascending=ascending)
            )
        feature_df_colors = feature_df_colors.applymap(
            lambda x: f"background-color: {x}"
        )
        st.write(
            f"SimplEx Example importance: {display_df.loc['Example Importance', example_i]*100:0.1f}%"
        )
        # # TODO: get thumb on same line as text see: https://discuss.streamlit.io/t/image-and-text-next-to-each-other/7627/17
        st.write("Model Prediction:")
        show_thumb(display_df.loc["Prediction", example_i])
        st.write("Real-world Outcome:")
        show_thumb(display_df.loc["Label", example_i])
        with pd.option_context("display.max_colwidth", 5):
            if corpus_patient_sort == "None":
                styled_display_df = (
                    display_df.drop(
                        index=["Example Importance", "Label", "Prediction"]
                    )[[example_i]]
                ).style.apply(highlight, axis=None)

            else:
                ascending = (
                    True if corpus_patient_sort == "Lowest importance first" else False
                )
                styled_display_df = (
                    display_df.drop(
                        index=["Example Importance", "Label", "Prediction"]
                    )[[example_i]]
                    .assign(s=feature_df[[example_i]].values)
                    # .assign(s2=feature_df_colors[[example_i]].values)
                    .sort_values(by="s", ascending=ascending)
                    .drop("s", axis=1)
                    # .drop("s2", axis=1)
                ).style.apply(highlight, axis=None)
            st.write(styled_display_df)
