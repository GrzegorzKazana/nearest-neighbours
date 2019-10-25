import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from constants import OBJECT_LABELS, ROOM_LABELS, LABELS_DATA_PATH

LABELS = ROOM_LABELS + OBJECT_LABELS


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true.values.argmax(axis=1), y_pred.argmax(axis=1))
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()

    filename = f"cm_{'with_normalization' if normalize else 'without_normalization'}"
    fig.savefig(filename)
    plt.close()


def divide(df: pd.DataFrame, labels: list):
    X = df.drop(labels, axis=1, inplace=False).copy()
    y = df[labels].copy()

    return X, y


def main():

    df = pd.read_csv(LABELS_DATA_PATH, usecols=LABELS)
    X, y = divide(df, ROOM_LABELS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    df_results = pd.DataFrame()

    # model

    clf = RandomForestClassifier(class_weight="balanced", max_depth=15, n_estimators=1000)
    # clf = GradientBoostingClassifier(max_depth=10, n_estimators=500)
    clf.fit(X_train, y_train)

    # evaluation

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=ROOM_LABELS, title="Confusion matrix, without normalization",normalize=True)

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    # print(f"AUC: {round(auc(fpr, tpr),3)}")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred),3)}")
    # print("F1-score:",f1_score(y_test, y_pred))


if __name__ == "__main__":
    main()
