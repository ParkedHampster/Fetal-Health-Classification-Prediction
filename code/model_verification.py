import numpy as np
import pandas as pd

from IPython.display import Markdown
from sklearn.metrics import confusion_matrix, \
                            recall_score, accuracy_score, \
                            roc_curve, roc_auc_score, \
                            auc, \
                            RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# from yellowbrick.classifier import ROCAUC
# from yellowbrick.classifier.rocauc import roc_auc
import matplotlib.pyplot as plt

def model_scoring(model,X,y,average=None,plot_curve=False,ax=None,class_names=None,cv=5,
        scoring='recall_macro',figsize=(14,8),multi_class='ovr',is_binary=False,
        **kwargs):
    """_summary_
    model: model with a .predict() method
        the model being used for predictions and scoring
    X: array-like or DataFrame
        array-like representation of all X values for
        the model to assess
    y: array-like or Series
        array-like representation of all y values for
        the model to assess
    average: (default=None) {‘micro’, ‘macro’,
        ‘samples’, ‘weighted’, ‘binary'}:
        the average method for the recall score to use
        see:
            recall_score documentation - 
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
            default=None, defaults to 'binary'
    plot_curve: boolean (default=False)
        determines whether or not to plot the roc_curve
        for the given model and y values
        !!! ONLY WORKING WITH MACRO AND OVR SETTINGS !!!
    ax: pyplot axis
        the axis for a curve plot to be drawn on
    class_names: array of str
        an array of class names for a plotted ROC curve
        in case the y values are not the desired names
    cv: (default=5) int
        number of folds for cross_val_score
    scoring: (default='recall_macro')
        scoring method for cross_val_score
    figsize: (default=(14,8))
        the figsize to use if ax is not defined

    returns: tuple
        recall, rocauc, cv_score
            - scores of the same names in order
            - cv_score is an array of scores

    """
    predictions = model.predict(X)
    proba_predictions = model.predict_proba(X)
    if is_binary:
        proba_predictions = proba_predictions[:,1]

    print(f"""
Model recall:         {(recall := recall_score(y,predictions,average=average))}
Median ROC AUC score: {(rocauc := roc_auc_score(y,proba_predictions, multi_class=multi_class,average=average))}
Cross Val Score:      {(cv_score := cross_val_score(model,X,y,cv=cv,scoring=scoring)).mean()}
    """)

    # multi-class roc_plot modified from sample in sklearn
    # documentation, available here:
    #  https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-using-the-ovr-macro-average
    if plot_curve:
        if ax==None:
            fig, ax = plt.subplots(figsize=figsize)
        if class_names==None:
            class_names = str(np.unique(y))
        if is_binary:
            RocCurveDisplay.from_predictions(
                y,
                proba_predictions,
                ax=ax,
                name=f"ROC curve for {class_names[1].title()}"
                )
        elif not is_binary:
            n_classes = len(np.unique(y))
            ohe = pd.get_dummies(y).values
            for class_id in range(n_classes):
                RocCurveDisplay.from_predictions(
                    ohe[:,class_id],
                    proba_predictions[:,class_id],
                    ax=ax,
                    name=f"ROC curve for {class_names[class_id].title()}"
                    )
    ax.set(
        title="ROC One-v-Rest Multiclass",
        ylabel="True Positive Rate",
        xlabel="False Positive Rate"
    )
    return {'recall':recall, 'rocauc':rocauc, 'cv_score':cv_score, 'ax':ax}



def model_scoring_table(model_results,model_names):

    assert len(model_results) == len(model_names)

    results_df = pd.DataFrame(model_results)
    results_df['cv_mean'] = results_df['cv_score'].map(lambda x: x.mean())
    tbody = ""
    max_recall = round(results_df['recall'].max(),3)
    max_rocauc = round(results_df['rocauc'].max(),3)
    max_cv_score = round((results_df['cv_mean']).max(),3)

    for i, name in enumerate(model_names):
        recstar = '**' if round(model_results[i]['recall'],3) == max_recall else ' '
        rocstar = '**' if round(model_results[i]['rocauc'],3) == max_rocauc else ' '
        cv_star = '***' if round(model_results[i]['cv_score'].mean(),3) == max_cv_score else ' '
        tbody += f"""| {name} | {
    recstar }{ model_results[i]['recall'] :.3f}{ recstar
    } | {
    rocstar }{ model_results[i]['rocauc'] :.3f}{ rocstar
    } | {
    cv_star }{ model_results[i]['cv_score'].mean() :.3f}{ cv_star
    } |
"""
    table_string = f"""
| Model | Recall | ROC AUC | CV Score |
|---:|:---:|:---:|:---:|
{tbody}"""
    return {'md':Markdown(table_string), 'df':results_df}
