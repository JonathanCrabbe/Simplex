import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_prostate_patient(
    input: np.ndarray, title: str, saliency=None
) -> matplotlib.axes.Axes:
    sns.set()
    treatment_list = np.array(["CM", "PHT", "RT-RDx", "RT-Sx"])
    fig, ax = plt.subplots()
    ax.set_axis_off()
    age, psa, comorbidities = input[:3]
    treatments = treatment_list[np.nonzero(input[3:7])]
    grade = np.argmax(input[7:12]) + 1
    stage = np.argmax(input[12:16]) + 1
    gleason1 = np.argmax(input[16:21]) + 1
    gleason2 = np.argmax(input[21:]) + 1
    rowLabels = [
        "Age",
        "PSA",
        "Comorbidities",
        "Treatments",
        "Grade",
        "Stage",
        "Gleason1",
        "Gleason2",
    ]
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
    colLabels = [""]
    table = ax.table(
        cellText=[
            [age],
            [psa],
            [comorbidities],
            [treatments],
            [grade],
            [stage],
            [gleason1],
            [gleason2],
        ],
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="upper left",
    )
    table.set_fontsize(20)
    table.auto_set_column_width(0)
    table.scale(1.6, 1.6)
    ax.set_title(title)
    fig.tight_layout()
    if saliency is not None:
        saliency_reduced = np.concatenate(
            (
                saliency[:3],
                saliency[3:7].sum(keepdims=True),
                saliency[7:12].sum(keepdims=True),
                saliency[12:16].sum(keepdims=True),
                saliency[16:21].sum(keepdims=True),
                saliency[21:].sum(keepdims=True),
            )
        )
        saliency_reduced /= 0.25
        saliency_reduced = 0.5 + 0.5 * saliency_reduced
        for i in range(len(rowLabels)):
            color = cmap(saliency_reduced[i])
            table.get_celld()[(i + 1, 0)].set_facecolor(color)
    return ax
