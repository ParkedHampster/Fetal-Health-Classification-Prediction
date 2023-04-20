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
        print_scores=True,**kwargs):
    """_summary_
    Generate a score dict automatically using recall,
    rocauc, and cross validation. 
    If plot_curve=True, also plot a ROC curve with a
    given pyplot axis if available.
    Defaults to multi-class.

    Args:
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
        multi_class: (default='ovr' - One vs. Rest)
            use 'raise' for binary data, is_binary will
            also set this. Otherwise, look at
            roc_auc_score documentation for other
            values.
        is_binary: (default=False)
            Determine if the target is binary (2
            values.) Defaults to False for multi_class.
        print_scores: (default=True)
            Whether or not to run print() with
            resulting metrics.

        returns: dict
            recall, rocauc, cv_score, ax
                - scores of the same names in order
                - cv_score is an array of scores
                - ax is the resulting pyplot ax for
                    limited use later.
    """
    predictions = model.predict(X)
    proba_predictions = model.predict_proba(X)
    if is_binary:
        proba_predictions = proba_predictions[:,1]
        multi_class = 'raise'
        rec_avg = 'binary'
        scoring = 'recall'
    else:
        rec_avg = average
    print(rec_avg)
    scores_info= f"""
Model recall:         {(recall := recall_score(y,predictions,average=rec_avg))}
Median ROC AUC score: {(rocauc := roc_auc_score(y,proba_predictions, multi_class=multi_class,average=average))}
Cross Val Score:      {(cv_score := cross_val_score(model,X,y,cv=cv,scoring=scoring)).mean()}
    """
    if print_scores:
        print(scores_info)

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
    """_summary_
    Generate a markdown table and data frame from given
    model_results and model_names. model_results is
    obtained from model_scoring.

    Args:
        model_results (list-like):
            a list-like of result dicts from
            model_scoring
        model_names (list-like):
            a list of model names to be used as the
            model names on the left-hand column of the
            markdown table or the 'names' column of the
            resulting data frame
    """    


    assert len(model_results) == len(model_names)

    results_df = pd.DataFrame(model_results)
    results_df['names'] = model_names
    results_df['cv_mean'] = results_df['cv_score'].map(lambda x: x.mean())
    tbody = ""
    max_recall = round(results_df['recall'].max(),3)
    max_rocauc = round(results_df['rocauc'].max(),3)
    max_cv_score = round((results_df['cv_mean']).max(),3)
    
    for i, name in enumerate(model_names):
        recstar = '**' if round(results_df.iloc[i]['recall'],3) == max_recall else ' '
        rocstar = '**' if round(results_df.iloc[i]['rocauc'],3) == max_rocauc else ' '
        cv_star = '***' if round(results_df.iloc[i]['cv_score'].mean(),3) == max_cv_score else ' '
        tbody += f"""| {name} | {
    recstar }{ results_df.iloc[i]['recall'] :.3f}{ recstar
    } | {
    rocstar }{ results_df.iloc[i]['rocauc'] :.3f}{ rocstar
    } | {
    cv_star }{ results_df.iloc[i]['cv_mean'] :.3f}{ cv_star
    } |
"""
    table_string = f"""
| Model | Recall | ROC AUC | CV Score |
|---:|:---:|:---:|:---:|
{tbody}"""
    return {'md':Markdown(table_string), 'df':results_df}
