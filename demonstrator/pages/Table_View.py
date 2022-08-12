import sys
import os
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
from experiments.prostate_cancer import (
    load_seer,
    load_cutract,
    ProstateCancerDataset,
)


def apply_sort_order(in_list, sort_order):
    if isinstance(in_list, list):
        return [in_list[idx] for idx in sort_order]
    if torch.is_tensor(in_list):
        return [in_list.numpy()[idx] for idx in sort_order]


def load_data(data_source="seer", random_seed=42, corpus_size=100):

    # Load corpus and test inputs
    if data_source.lower() in ["cutract", "seer"]:
        # LOAD DATA from file
        prostate_load_funcs = {
            "cutract": load_cutract(random_seed=random_seed),
            "seer": load_seer(random_seed=random_seed),
        }
        X, y = prostate_load_funcs[data_source.lower()]

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
    # model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def get_simplex_decomposition(i, model, input_baseline):
    simplex.jacobian_projection(test_id=i, model=model, input_baseline=input_baseline)
    result, sort_order = simplex.decompose(i, return_id=True)
    return result, sort_order


data_source = "prostate_seer"
# Load the data
(
    corpus_inputs,
    corpus_targets,
    test_inputs,
    test_targets,
    input_baseline,
    feature_names,
) = load_data(data_source=data_source, random_seed=42, corpus_size=100)

# Get a trained model
model = MortalityPredictor(n_cont=3)  # Model should have the BlackBox interface
TRAINED_MODEL_STATE_PATH = os.path.abspath(
    "../experiments/results/prostate/outlier/model_cv0.pth"
)
model = load_trained_model(model, TRAINED_MODEL_STATE_PATH)

# Compute the corpus and test latent representations
corpus_latents = model.latent_representation(corpus_inputs).detach()
test_latents = model.latent_representation(test_inputs).detach()


# Load fitted SimplEx
with open(f"./resources/trained_models/{data_source}/simplex.pkl", "rb") as f:
    simplex = pkl.load(f)


# Compute corpus and test model predictions
corpus_predictions = [
    1 if prediction[0] < 0.5 else 0
    for prediction in model.probabilities(corpus_inputs).detach()
]

test_predictions = [
    1 if prediction[0] < 0.5 else 0
    for prediction in model.probabilities(test_inputs).detach()
]

##################################################################################
def user_input_features():
    test_patient_id = st.sidebar.slider("Patient ID", 1, 50, 1)
    example_importance_threshold = st.sidebar.slider(
        "Minimum Example Importance", 0.0, 1.0, 0.1
    )
    return test_patient_id - 1, example_importance_threshold


# Sidebar
st.sidebar.header("Test Patient")
st.sidebar.write(
    "Please select a patient to review from the test set with the Patient ID slider. You can also change the number of example patients from the corpus by adjusting the minimum example importance slider."
)
(
    test_patient_id,
    example_importance_threshold,
) = user_input_features()  # user input sliders
# Test patient image and data

st.sidebar.header(f"Patient number: {test_patient_id+1}")

col1, col2, col3 = st.sidebar.columns(3, gap="small")
with col1:
    st.image("./resources/patient_male.png", width=100)
with col2:
    st.write("Patient Outcome:")
    st.write("Patient Prediction:")
with col3:
    label = test_targets[test_patient_id].item()
    if label == 0:
        st.image("./resources/thumb_up.png", width=40)
    elif label == 1:
        st.image("./resources/thumb_down.png", width=40)
    prediction = test_predictions[test_patient_id]
    st.write("")
    if prediction == 0:
        st.image("./resources/thumb_up.png", width=40)
    elif prediction == 1:
        st.image("./resources/thumb_down.png", width=40)


st.sidebar.write("Patient features")

# Main Page
st.write(
    """
# SimplEx demonstrator

This app explains your models predictions using **SimplEx**! 
"""
)

test_patient_df = pd.DataFrame(
    [simplex.test_examples[test_patient_id].numpy()],
    columns=feature_names,
    index=["Value"],
)

st.sidebar.write(test_patient_df.transpose())

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
feature_df.insert(loc=0, column="Label", value=0)
feature_df.insert(loc=0, column="Prediction", value=0)
feature_df.insert(loc=0, column="Example Importance", value=0)


def df_values_to_colors(df, feature_names):

    df = df.transpose()
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
        # st.write(type((norm(df[col].values[3:]) * 255).astype(np.int16)))
        a = np.concatenate(
            ([140, 140, 140], (norm(df[col].values[3:]) * 255).astype(np.int16)),
            axis=None,
        )
        if col not in ["Example Importance", "Label", "Prediction"]:
            df[col] = lut[a]
        else:
            df[col] = lut[140]
    return df.transpose()


def highlight(x):
    return pd.DataFrame(feature_df_colors.values, columns=x.columns)


feature_df = feature_df.loc[
    corpus_df["Example Importance"] >= example_importance_threshold
]

feature_df_colors = df_values_to_colors(feature_df.copy(), feature_names)
feature_df_colors = feature_df_colors.applymap(lambda x: f"background-color: {x}")

display_df = corpus_df.loc[
    corpus_df["Example Importance"] >= example_importance_threshold
].copy()


st.write(display_df.style.apply(highlight, axis=None))
# st.write(feature_df)  # display importances behind colours
