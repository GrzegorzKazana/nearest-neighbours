import pandas as pd
import numpy as np
import click
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from constants import LABELS_DATA_PATH, ALL_COLS, ROOM_LABELS, TASK2_CLASS_COL, TASK2_CLASS_LABELS


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
    cm = confusion_matrix(y_true, y_pred)
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


def read_data(one):
    X, y = 0.0, 0.0

    # use task2_class
    if one:
        df = pd.read_csv(LABELS_DATA_PATH, usecols=ALL_COLS + [TASK2_CLASS_COL])
        df = df.drop(ROOM_LABELS, axis=1)

        one_hot = pd.get_dummies(df[TASK2_CLASS_COL])
        df = df.drop(TASK2_CLASS_COL, axis=1)
        df = df.join(one_hot)
        df = df.drop("validation", axis=1)
        X, y = divide(df, TASK2_CLASS_LABELS)

    # respective cols
    else:
        df = pd.read_csv(LABELS_DATA_PATH, usecols=ALL_COLS)
        X, y = divide(df, ROOM_LABELS)
        df = df.drop("House", axis=1)

    return X, y


@click.command()
@click.option("--label", default="Bathroom", help="")
@click.option("--normalize", is_flag=True, help="")
@click.option("--search", is_flag=True, help="")
@click.option(
    "--one", is_flag=True, help="Use labels from task2_class; otherwise use data from cols respective to rom names"
)
@click.option("--test_size", default=0.3, help="test set size")
@click.option("--n_estimators", default=100, help="")
@click.option("--max_depth", default=15, help="")
@click.option("--max_features", default=3, help="")
@click.option("--min_samples_split", default=3, help="")
@click.option("--max_leaf_nodes", default=3, help="")
@click.option("--min_samples_leaf", default=3, help="")
@click.option("--min_weight_fraction_leaf", default=0.5, help="")
def main(
    test_size,
    label,
    normalize,
    one,
    min_samples_leaf,
    min_weight_fraction_leaf,
    n_estimators,
    max_depth,
    search,
    max_features,
    min_samples_split,
    max_leaf_nodes,
):

    # data

    # X, y = read_data(one)
    df = pd.read_csv(LABELS_DATA_PATH, usecols=ALL_COLS + [TASK2_CLASS_COL])
    df = df[df[TASK2_CLASS_COL] != "validation"]
    X, y = divide(df, TASK2_CLASS_COL)
    # y = y[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # model

    # clf = RandomForestClassifier(
    #     class_weight="balanced",
    #     max_depth=max_depth,
    #     n_estimators=n_estimators,
    #     max_features=max_features,
    #     bootstrap=False,
    # )
    clf = GradientBoostingClassifier(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
    )
    clf.fit(X_train, y_train)

    # grid search
    if search:
        param_grid = {
            "max_depth": [3, 5, 10],
            "max_leaf_nodes": [3, 5, 10],
            "max_features": [3, 5, 10],
            "n_estimators": [64, 128],
            "min_samples_split": [3, 5],
            "min_samples_leaf": [3, 5],
            "min_weight_fraction_leaf": [0.01, 0.05, 0.1, 0.2, 0.5],
        }
        grid = GridSearchCV(estimator=clf, param_grid=param_grid, verbose=1, n_jobs=-1)
        grid.fit(X, y)
        print(grid)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_params_)

    # cross val

    scores = cross_val_score(clf, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # evaluation

    y_pred = clf.predict(X_test)

    # print(classification_report(y_test['Bathroom'], y_pred['Bathroom']))

    classes = TASK2_CLASS_LABELS if one else ROOM_LABELS
    title = f"Confusion matrix, {'with normalization' if normalize else 'without normalization'}"
    plot_confusion_matrix(y_test, y_pred, classes=classes, title=title, normalize=normalize)

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    # print(f"AUC: {round(auc(fpr, tpr),3)}")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred),3)}")
    print("F1-score:",f1_score(y_test, y_pred, average='weighted'))

    dump(clf, f"clf_task2_xgboost_{'single' if one else 'multiple'}.joblib")


if __name__ == "__main__":
    main()
