import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pingouin as pg
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from matplotlib import cm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score
import scikit_posthocs as sp
from scipy.stats import t as student_t

import math

from analysis.plot_style import apply_style

def calc_regression_metrics(df, cycle_col, val_col, pred_col, thresh):
    """
    Calculate regression metrics (MAE, MSE, R2, prec, recall) for each method and split

    :param df: input dataframe must contain columns [method, split] as well the columns specified in the arguments
    :param cycle_col: column indicating the cross-validation fold
    :param val_col: column with the ground truth value
    :param pred_col: column with predictions
    :param thresh: threshold for binary classification
    :return: a dataframe with [cv_cycle, method, split, mae, mse, r2, prec, recall]
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall"]
    metric_list = []
    df_in['true_class'] = df_in[val_col] > thresh
    # Make sure the thresh variable creates 2 classes
    assert len(df_in.true_class.unique()) == 2, "Binary classification requires two classes"
    df_in['pred_class'] = df_in[pred_col] > thresh

    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        mae = mean_absolute_error(v[val_col], v[pred_col])
        mse = mean_squared_error(v[val_col], v[pred_col])
        r2 = r2_score(v[val_col], v[pred_col])
        recall = recall_score(v.true_class, v.pred_class)
        prec = precision_score(v.true_class, v.pred_class)
        rho, _ = spearmanr(v[val_col], v[pred_col])
        metric_list.append([cycle, method, split, mae, mse, r2, rho, prec, recall])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split"] + metric_ls)
    return metric_df


def rm_tukey_hsd(df, metric, group_col, alpha=0.05, sort = False, direction_dict=None):
    """
    Perform repeated measures Tukey HSD test on the given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric (str): The metric column name to perform the test on.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the test. Default is 0.05.
    sort (bool): Whether to sort the output tables. Default is False.

    Returns:
    tuple: A tuple containing:
        - result_tab (pd.DataFrame): DataFrame with pairwise comparisons and adjusted p-values.
        - df_means (pd.DataFrame): DataFrame with mean values for each group.
        - df_means_diff (pd.DataFrame): DataFrame with mean differences between groups.
        - pc (pd.DataFrame): DataFrame with adjusted p-values for pairwise comparisons.
    """
    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == 'maximize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=False)
        elif direction_dict[metric] == 'minimize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=True)
        else:
            raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
    else:
        df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero encountered in scalar divide')
        aov = pg.rm_anova(dv=metric, within=group_col, subject='cv_cycle', data=df, detailed=True)
    mse = aov.loc[1, 'MS']
    df_resid = aov.loc[1, 'DF']

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons),
                              columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)

    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

    return result_tab, df_means, df_means_diff, pc


# -------------- Plotting routines -------------------#


def make_boxplots_parametric(df, metric_ls):
    """
    Create boxplots for each metric using repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metric column names to create boxplots for.

    Returns:
    None
    """
    apply_style()
    figure, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8))
    # figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    for i, stat in enumerate(metric_ls):
        model = AnovaRM(data=df, depvar=stat, subject='cv_cycle', within=['method']).fit()
        p_value = model.anova_table['Pr > F'].iloc[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        title = stat.upper()
        ax.set_title(f"p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels, rotation=45, ha='right', va='top')

    plt.tight_layout()

def make_boxplots_nonparametric(df, metric_ls):
    apply_style()
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=False, figsize=(28, 8))

    for i, stat in enumerate(metric_ls):
        friedman = pg.friedman(df, dv=stat, within="method", subject="cv_cycle")['p-unc'].values[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        title = stat.replace("_", " ").upper()
        ax.set_title(f"p={friedman:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()
        
def make_sign_plots_nonparametric(df, metric_ls):
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': True, 'square': True}
    apply_style()
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=True, figsize=(26, 8))

    for i, stat in enumerate(metric_ls):
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", p_adjust="holm",
                                         melted=True)
        sub_ax, sub_c = sp.sign_plot(pc, **heatmap_args, ax=axes[i], xticklabels=True)  # Update xticklabels parameter
        sub_ax.set_title(stat.upper())

def make_critical_difference_diagrams(df, metric_ls):
    figure, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(16, 10))
    for i, stat in enumerate(metric_ls):
        avg_rank = df.groupby("cv_cycle")[stat].rank(pct=True).groupby(df.method).mean()
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", p_adjust="holm",
                                         melted=True)
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(stat.upper())
    plt.tight_layout()

def make_normality_diagnostic(df, metric_ls):
    """
    Create a normality diagnostic plot grid with histograms and QQ plots for the given metrics.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metrics to create plots for.

    Returns:
    None
    """
    df_norm = df.copy()
    
    for metric in metric_ls:
        df_norm[metric] = df_norm[metric] - df_norm.groupby("method")[metric].transform("mean")

    df_norm = df_norm.melt(id_vars=["cv_cycle", "method", "split"],
                                   value_vars=metric_ls,
                                   var_name="metric",
                                   value_name="value")

    apply_style()
    
    metrics = df_norm['metric'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, n_metrics, figsize=(20, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        sns.histplot(df_norm[df_norm['metric'] == metric]['value'], kde=True, ax=ax)
        ax.set_title(f'{metric}')
    
    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        metric_data = df_norm[df_norm['metric'] == metric]['value']
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title("")
    
    plt.tight_layout()


def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
             ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
             show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
    """
    Create a multiple comparison of means plot using a heatmap.

    Parameters:
    pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
    effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
    means (pd.Series): Series containing mean values for each group.
    labels (bool): Whether to show labels on the axes. Default is True.
    cmap (str): Colormap to use for the heatmap. Default is None.
    cbar_ax_bbox (tuple): Bounding box for the colorbar axis. Default is None.
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap. Default is None.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    show_cbar (bool): Whether to show the colorbar. Default is True.
    reverse_cmap (bool): Whether to reverse the colormap. Default is False.
    vlim (float): Limit for the colormap. Default is None.
    **kwargs: Additional keyword arguments for the heatmap.

    Returns:
    matplotlib.axes.Axes: The axes with the heatmap.
    """
    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = '***'
    significance[(pc < 0.01) & (pc >= 0.001)] = '**'
    significance[(pc < 0.05) & (pc >= 0.01)] = '*'
    significance[(pc >= 0.05)] = ''

    np.fill_diagonal(significance.values, '')

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                      annot_kws={"size": cell_text_size},
                      vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)
    hax.tick_params(axis='x', pad=12)   # space between x-axis and tick labels
    hax.tick_params(axis='y', pad=6)    # small padding on y as well
  

    if labels:
        label_list = list(means.index)
        x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
        y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=45,
                            rotation_mode='anchor')
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                            rotation_mode='anchor')

    hax.set_xlabel('')
    hax.set_ylabel('')
    return hax


def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                       figsize=(20, 10), direction_dict={}, effect_dict={}, show_diff=True,
                       cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False):
    """
    Create a grid of multiple comparison of means plots using Tukey HSD test results.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    stats (list of str): List of statistical metrics to create plots for.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
    figsize (tuple): Size of the figure. Default is (20, 10).
    direction_dict (dict): Dictionary indicating whether to minimize or maximize each metric.
    effect_dict (dict): Dictionary with effect size limits for each metric.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    title_text_size (int): Font size for the title text. Default is 16.
    sort (bool): Whether to sort the axes. Default is False.

    Returns:
    None
    """
    nrow = math.ceil(len(stats) / 3)
    fig, ax = plt.subplots(nrow, 3, figsize=figsize)
    #edit add ax dim
    ax = np.atleast_2d(ax)
    # Set defaults
    for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
        direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')

    for key in ['r2', 'rho', 'prec', 'recall']:
        effect_dict.setdefault(key, 0.1)

    direction_dict = {k.lower(): v for k, v in direction_dict.items()}
    effect_dict = {k.lower(): v for k, v in effect_dict.items()}

    for i, stat in enumerate(stats):
        stat = stat.lower()

        row = i // 3
        col = i % 3

        if stat not in direction_dict:
            raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
        if stat not in effect_dict:
            raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")

        reverse_cmap = False
        if direction_dict[stat] == 'minimize':
            reverse_cmap = True

        _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha,
                                                       sort_axes, direction_dict)

        hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                       show_diff=show_diff, ax=ax[row, col], cbar=True,
                       cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                       reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
        hax.set_title(stat.upper(), fontsize=title_text_size)
        for tick in hax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('center')
            tick.set_va('top')

        for tick in hax.get_yticklabels():
            tick.set_rotation(45)
        hax.tick_params(axis='x', pad=25, labelsize=axis_text_size)
        hax.tick_params(axis='y', pad=25,  labelsize=axis_text_size)

    # If there are less plots than cells in the grid, hide the remaining cells
    if (len(stats) % 3) != 0:
        for i in range(len(stats), nrow * 3):
            row = i // 3
            col = i % 3
            ax[row, col].set_visible(False)

    plt.tight_layout()


def make_scatterplot(df, val_col, pred_col, thresh, cycle_col="cv_cycle", group_col="method"):
    """
    Create scatter plots for each method showing the relationship between predicted and measured values.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    val_col (str): The column name for the ground truth values.
    pred_col (str): The column name for the predicted values.
    thresh (float): Threshold for binary classification.
    cycle_col (str): The column name indicating the cross-validation fold. Default is "cv_cycle".
    group_col (str): The column name indicating the groups/methods. Default is "method".

    Returns:
    None
    """
    df_split_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col,
                                               thresh=thresh)
    methods = df[group_col].unique()

    fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(25, 10))

    for ax, method in zip(axs, methods):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot([df_method[val_col].min(), df_method[val_col].max()],
                [df_method[val_col].min(), df_method[val_col].max()], 'k--', lw=1)

        ax.axhline(y=thresh, color='r', linestyle='--')
        ax.axvline(x=thresh, color='r', linestyle='--')
        ax.set_title(method)

        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics_text = f"MAE: {df_metrics['mae'].mean():.2f}\nMSE: {df_metrics['mse'].mean():.2f}\nR2: {df_metrics['r2'].mean():.2f}\nrho: {df_metrics['rho'].mean():.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        ax.text(0.05, .5, metrics_text, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Measured')

    plt.tight_layout()
    plt.show()


def ci_plot(result_tab, ax_in, name):
    """
    Create a confidence interval plot for the given result table.

    Parameters:
    result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
    ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
    name (str): The title of the plot.

    Returns:
    None
    """
    result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                           result_tab['upper'] - result_tab['meandiff']])
    apply_style()
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.set_xlim(-0.2, 0.2) 


def make_ci_plot_grid( df_in,
    metric_list,
    group_col: str = "method",
    figsize: tuple | None = None,
    x_tick_size: int = 9,
    y_tick_size: int = 9,
    title_size: int = 14,
    ci_xlim: tuple | None = None,   # e.g. (-0.05, 0.05)
):
    """
     Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

     Parameters:
     df_in (pd.DataFrame): Input dataframe containing the data.
     metric_list (list of str): List of metric column names to create confidence interval plots for.
     group_col (str): The column name indicating the groups. Default is "method".

     Returns:
     None
     """
    if figsize is None:
        figsize = (12, max(2.5 * len(metric_list), 6))

    fig, axes = plt.subplots(len(metric_list), 1, figsize=figsize, sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        ax = axes[i]

        # draw the CI strip plot on this axis
        ci_plot(df_tukey, ax_in=ax, name=metric.upper())

        # styling: small ticks, optional x limits, cleaner labels
        ax.tick_params(axis='x', labelsize=x_tick_size)
        ax.tick_params(axis='y', labelsize=y_tick_size)
        if ci_xlim is not None:
            ax.set_xlim(ci_xlim)
        # make axis labels/titles consistent
        ax.set_xlabel("Mean Difference", fontsize=x_tick_size + 1)
        ax.set_ylabel("", fontsize=y_tick_size)
        ax.set_title(metric.upper(), fontsize=title_size)

    fig.suptitle("Multiple Comparison of Means — Tukey HSD (FWER = 0.05)", fontsize=title_size + 2)
    fig.tight_layout()
    # leave some space for the suptitle
    fig.subplots_adjust(top=0.92)


def recall_at_precision(y_true, y_score, precision_threshold=0.5, direction='greater'):
    if direction not in ['greater', 'lesser']:
        raise ValueError("Invalid direction. Expected one of: ['greater', 'lesser']")

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)

    if direction == 'greater':
        thresholds = np.sort(thresholds)
    else:  
        thresholds = np.sort(thresholds)[::-1]

    for threshold in thresholds:
        if direction == 'greater':
            y_pred = y_score >= threshold
        else:  
            y_pred = y_score <= threshold

        precision = precision_score(y_true, y_pred)
        if precision >= precision_threshold:
            recall = recall_score(y_true, y_pred)
            return recall, threshold
    return np.nan, None

def calc_classification_metrics(df_in, cycle_col, val_col, prob_col, pred_col):
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        mcc = matthews_corrcoef(v[val_col], v[pred_col])
        
        recall, _ = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='greater')
        tnr, _ = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='lesser')

        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr])
        
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split",
                                                    "roc_auc", "pr_auc", "mcc", "recall", "tnr"])
    return metric_df


def make_curve_plots(df):
    df_plot = df.query("cv_cycle == 0 and split == 'scaffold'").copy()
    color_map = plt.get_cmap('tab10')
    le = LabelEncoder()
    df_plot['color'] = le.fit_transform(df_plot['method'])
    colors = color_map(df_plot['color'].unique())
    val_col = "Sol"
    prob_col = "Sol_prob"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for (k, v), color in zip(df_plot.groupby("method"), colors):
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        fpr, recall_pos, thresholds_roc = roc_curve(v[val_col], v[prob_col])
        precision, recall, thresholds_pr = precision_recall_curve(v[val_col], v[prob_col])

        _, threshold_recall_pos = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='greater')
        _, threshold_recall_neg = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='lesser')

        fpr_recall_pos = fpr[np.abs(thresholds_roc - threshold_recall_pos).argmin()]
        fpr_recall_neg = fpr[np.abs(thresholds_roc - threshold_recall_neg).argmin()]
        recall_recall_pos = recall[np.abs(thresholds_pr - threshold_recall_pos).argmin()]
        recall_recall_neg = recall[np.abs(thresholds_pr - threshold_recall_neg).argmin()]

        axes[0].plot(fpr, recall_pos, label=f"{k} (ROC AUC={roc_auc:.03f})", color=color, alpha=0.75)
        axes[1].plot(recall, precision, label=f"{k} (PR AUC={pr_auc:.03f})", color=color, alpha=0.75)

        axes[0].axvline(fpr_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[0].axvline(fpr_recall_neg, color=color, linestyle='--', alpha=0.75)
        axes[1].axvline(recall_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[1].axvline(recall_recall_neg, color=color, linestyle='--', alpha=0.75)

    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()







import numpy as np
import pandas as pd

def plot_boxplots_features(resample_df, value_metric="mcc", title=None, figsize=(12,6)):
    df = resample_df.copy()
    # order by mean
    order = (df.groupby("var_tag")[value_metric]
               .mean()
               .sort_values(ascending=False)
               .index.tolist())
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df, x="var_tag", y=value_metric, order=order, whis=1.5, fliersize=2)
    sns.stripplot(data=df, x="var_tag", y=value_metric, order=order, color="k", alpha=0.35, size=2, jitter=0.15)
    ax.set_xlabel("Feature set (coalition)")
    ax.set_ylabel(value_metric.upper())
    ax.set_title(title or f"Per-cycle {value_metric.upper()} by feature set")
    plt.xticks(rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# usage:
# plot_boxplots_features(resample_df, value_metric=VALUE_METRIC, 
#                        title=f"{MODEL_TYPE}: per-cycle {VALUE_METRIC.upper()} by feature set")


def plot_pairwise_q_heatmap(pval_matrix, means, title=None, figsize=(10,8), vmax=0.10):
    """
    pval_matrix: DataFrame (var_tag x var_tag) with BH-adjusted p-values (q-values).
    means: Series of mean metric per var_tag to sort rows/cols (best at top).
    vmax: top of colorbar (e.g., 0.10 shows 0–10% FDR nicely).
    """
    # sort by mean (descending)
    order = means.sort_values(ascending=False).index.tolist()
    q = pval_matrix.loc[order, order].copy()

    # mask upper triangle
    mask = np.triu(np.ones_like(q, dtype=bool), k=1)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(q, mask=mask, annot=False, cmap="viridis_r", vmin=0, vmax=vmax, cbar_kws={"label":"BH q-value"})
    ax.set_title(title or "Pairwise BH-adjusted q-values (lower triangle)")
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# usage:
# wide.columns.name = VALUE_METRIC.upper()  # optional label
# ci_strips_vs_best(wide, best, pval_matrix)


def normality_diag_centered(resample_df, value_metric="mcc", bins=12, figsize=(12,5)):
    g = resample_df[["var_tag", value_metric]].dropna().copy()
    g["centered"] = g[value_metric] - g.groupby("var_tag")[value_metric].transform("mean")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(g["centered"], bins=bins, kde=True, ax=axes[0], alpha=0.85)
    axes[0].set_title(f"{value_metric.upper()} (centered within feature set)")
    axes[0].set_xlabel("Value"); axes[0].set_ylabel("Count")

    stats.probplot(g["centered"].values, dist="norm", plot=axes[1])
    axes[1].set_title("QQ plot (centered values)")
    plt.tight_layout(); plt.show()

# usage:
# normality_diag_centered(resample_df, value_metric=VALUE_METRIC)



def _mean_ci_bootstrap(vals, n_boot=10000, alpha=0.05, rng=None):
    """Bootstrap CI for the mean of 1D array vals."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    m = float(np.mean(vals))
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return m, float(lo), float(hi)

def _mean_ci_t(vals, alpha=0.05):
    """t-interval CI for the mean of 1D array vals."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n <= 1:
        m = float(vals.mean()) if n == 1 else np.nan
        return m, m, m
    m = float(vals.mean())
    s = float(vals.std(ddof=1))
    se = s / np.sqrt(n)
    q = float(student_t.ppf(1 - alpha/2, df=n-1))
    return m, m - q*se, m + q*se




def _mean_ci_bootstrap(vals, n_boot=10000, alpha=0.05, rng=None):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(12345) if rng is None else rng
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    m = float(np.mean(vals))
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return m, float(lo), float(hi)

def _mean_ci_t(vals, alpha=0.05):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n <= 1:
        m = float(vals.mean()) if n == 1 else np.nan
        return m, m, m
    m = float(vals.mean())
    s = float(vals.std(ddof=1))
    se = s / np.sqrt(n)
    q = float(student_t.ppf(1 - alpha/2, df=n-1))
    return m, m - q*se, m + q*se

def plot_sci_vs_best_pat(
    wide: pd.DataFrame,
    best: str,
    pval_matrix: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "bootstrap",   # "bootstrap" or "t"
    title: str | None = None,
    value_label: str | None = None,
    figsize=(9.5, None),
    savepath: str | None = None
):
    """
    SCI-style plot of mean differences (method − best) with CI bars.
    Colors by Conover+BH decision: best=blue, not different=grey, significantly worse=red.
    Legend in bottom-left. Only zero vertical line (no best-band lines).
    """
    # order best → worst by original mean
    means_orig = wide.mean(axis=0).sort_values(ascending=False)
    order = means_orig.index.tolist()
    wide = wide[order]

    # best CI (for its own row)
    best_vals = wide[best].values
    if method == "bootstrap":
        best_mean, best_lo, best_hi = _mean_ci_bootstrap(best_vals, n_boot=10000, alpha=alpha)
    elif method == "t":
        best_mean, best_lo, best_hi = _mean_ci_t(best_vals, alpha=alpha)
    else:
        raise ValueError("method must be 'bootstrap' or 't'")
    # map best CI into difference scale for drawing on best row
    best_ci_lo_rel = best_lo - best_mean
    best_ci_hi_rel = best_hi - best_mean

    # per-method Δ and CI(Δ) vs best
    rows, rng = [], np.random.default_rng(12345)
    for c in wide.columns:
        diffs = (wide[c] - wide[best]).dropna().values
        if method == "bootstrap":
            m, lo, hi = _mean_ci_bootstrap(diffs, n_boot=10000, alpha=alpha, rng=rng)
        else:
            m, lo, hi = _mean_ci_t(diffs, alpha=alpha)
        q = float(pval_matrix.loc[best, c]) if (best in pval_matrix.index and c in pval_matrix.columns) else np.nan
        rows.append({"var_tag": c, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "q_bh": q})

    df = pd.DataFrame(rows)
    # enforce exact 0 for best and display its own CI on its row
    df.loc[df["var_tag"] == best, ["mean_diff","ci_lo","ci_hi","q_bh"]] = [0.0, best_ci_lo_rel, best_ci_hi_rel, np.nan]

    # sort best → worst
    df["mean_orig"] = df["var_tag"].map(means_orig.to_dict())
    df = df.sort_values("mean_orig", ascending=False).reset_index(drop=True)

    # colors
    colors = []
    for _, r in df.iterrows():
        if r["var_tag"] == best:
            colors.append("#1f77b4")  # blue
        else:
            if (not np.isnan(r["q_bh"])) and (r["q_bh"] < alpha) and (r["mean_diff"] < 0):
                colors.append("#d62728")  # red
            else:
                colors.append("#9e9e9e")  # grey
    df["color"] = colors

    # figure height
    h = max(4.0, 0.38 * len(df))
    if figsize[1] is None:
        figsize = (figsize[0], h)

    fig, ax = plt.subplots(figsize=figsize)

    # only zero line
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)

    # draw CI segments + points
    y = np.arange(len(df))
    for i, r in df.iterrows():
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=r["color"], linewidth=3)
        ax.scatter([r["mean_diff"]], [i], color=r["color"], s=28, zorder=3)

    ax.set_yticks(y, df["var_tag"])
    ax.set_xlabel(f"Δ {value_label or ''}  (method − {best})".strip())
    ax.set_title(title or f"SCI vs best — {best}  (dashed=0)")
    ax.grid(True, axis="x", alpha=0.3)

    # legend bottom-left inside plot
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color="#1f77b4", lw=3, label="Best"),
        Line2D([0],[0], color="#9e9e9e", lw=3, label=f"Not different vs best (BH q≥{alpha})"),
        Line2D([0],[0], color="#d62728", lw=3, label=f"Significantly worse (BH q<{alpha})"),
        Line2D([0],[0], color="k", lw=1, ls="--", label="Zero (no difference)"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="lower left")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    return df


def _fmt_p(p):
    return f"{p:.3g}" if p >= 1e-3 else f"{p:.1e}"
def plot_boxplots_with_significance(
    resample_df, pval_matrix, value_metric="mcc", alpha=0.05,
    title=None, figsize=(12,6), savepath=None,
    friedman_table=None, friedman_p_override=None,
    show_points="outliers",  # "outliers" | "all" | "none"
    whis=1.5
):
    import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

    def _fmt_p(p): return f"{p:.3g}" if p >= 1e-3 else f"{p:.1e}"

    df = resample_df[["var_tag","cv_cycle",value_metric]].dropna().copy()
    means = df.groupby("var_tag")[value_metric].mean().sort_values(ascending=False)
    order = means.index.tolist(); best = order[0]; best_mean = float(means.iloc[0])

    # palette from BH decisions
    palette = []
    for tag in order:
        if tag == best:
            palette.append("#1f77b4")
        else:
            q = pval_matrix.loc[best, tag] if (best in pval_matrix.index and tag in pval_matrix.columns) else np.nan
            if (not np.isnan(q)) and (q < alpha) and (means.loc[tag] < best_mean):
                palette.append("#c45c5c")
            else:
                palette.append("#9e9e9e")

    # Friedman p for title (optional)
    if friedman_p_override is not None:
        p_show = friedman_p_override
    else:
        if friedman_table is None:
            try:
                import pingouin as pg
                friedman_table = pg.friedman(
                    data=df.rename(columns={value_metric:"value"}),
                    dv="value", within="var_tag", subject="cv_cycle"
                )
            except Exception:
                friedman_table = None
        p_show = None
        if friedman_table is not None:
            if "p-unc" in friedman_table.columns: p_show = float(friedman_table["p-unc"].iloc[0])
            elif "p" in friedman_table.columns:  p_show = float(friedman_table["p"].iloc[0])

    # --- plot ---
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=df, x="var_tag", y=value_metric, order=order,
        whis=whis, fliersize=2,
        showfliers=(show_points != "outliers"),  # avoid double-drawing if we add jittered outliers
        palette=palette
    )

    # optional points
    if show_points == "all":
        sns.stripplot(data=df, x="var_tag", y=value_metric, order=order,
                      color="k", alpha=0.25, size=2, jitter=0.15)
    elif show_points == "outliers":
        # compute outliers per group using the same 1.5*IQR rule as the box
        outs = []
        for tag, g in df.groupby("var_tag"):
            q1 = g[value_metric].quantile(0.25)
            q3 = g[value_metric].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - whis*iqr
            hi = q3 + whis*iqr
            og = g[(g[value_metric] < lo) | (g[value_metric] > hi)].copy()
            outs.append(og)
        outs = pd.concat(outs, ignore_index=True) if outs else df.iloc[0:0]
        if not outs.empty:
            sns.stripplot(data=outs, x="var_tag", y=value_metric, order=order,
                          color="k", alpha=0.6, size=2.5, jitter=0.15)

    ax.axhline(best_mean, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.9)
    ax.set_xlabel("")              # your preference
    ax.set_ylabel(value_metric.upper())
    base_title = title or f"{value_metric.upper()} by feature set"
    ax.set_title(f"{base_title})" if p_show is not None else base_title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=60, ha="right")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="#1f77b4", lw=6, label="Best"),
        Line2D([0],[0], color="#9e9e9e", lw=6, label=f"Not statistically different vs best (BH q≥{alpha})"),
        Line2D([0],[0], color="#c45c5c", lw=6, label=f"Significantly worse (BH q<{alpha})"),
        Line2D([0],[0], color="#1f77b4", lw=1, ls="--", label="Best mean"),
    ], frameon=False, loc="lower left")

    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

from matplotlib import colors
from matplotlib.patches import Patch

def _fmt_q(v):
    if v is None or np.isnan(v): return ""
    return f"{v:.2f}" if v >= 1e-2 else f"{v:.1e}"

def plot_pairwise_q_heatmap_sig(
    pval_matrix: pd.DataFrame,
    means: pd.Series,
    alpha: float = 0.05,
    title: str | None = None,
    figsize=(10, 8),
    cmap_non_sig: str = "OrRd",     # gradient for q ≥ alpha
    sig_color: str = "#1f77b4",     # solid color for q < alpha
    annotate: bool = True,
    annot_size: int = 7,
    cbar_decimals: int = 2,
    savepath: Path | None = None
):
    # Order best → worst by mean metric
    order = means.sort_values(ascending=False).index.tolist()
    q = pval_matrix.loc[order, order].copy()
    np.fill_diagonal(q.values, np.nan)

    # Lower triangle only
    mask_upper = np.triu(np.ones_like(q, dtype=bool), k=1)

    # --- Build a single colormap: [0, alpha) = solid blue; [alpha, 1] = gradient ---
    base = plt.get_cmap(cmap_non_sig, 256)
    n_total = 256
    n_blue  = max(1, int(np.floor(alpha * n_total)))    # size of the blue band
    n_grad  = n_total - n_blue
    grad = base(np.linspace(0, 1, max(n_grad, 1)))
    blue = np.tile(colors.to_rgba(sig_color), (n_blue, 1))
    merged = np.vstack([blue, grad])
    cmap = colors.ListedColormap(merged)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)         # full [0,1] range

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        q, mask=mask_upper, cmap=cmap, norm=norm, vmin=0.0, vmax=1.0,
        cbar=True, linewidths=0.5, linecolor="white", square=True, ax=ax
    )

    # Colorbar: full 0→1, with blue band at bottom
    cbar = hm.collections[0].colorbar
    ticks = np.array([0.0, alpha, 0.25, 0.5, 0.75, 1.0])
    ticks = np.unique(np.clip(ticks, 0, 1))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.{cbar_decimals}f}" for t in ticks])
    cbar.set_label("Adjusted p-value")

    # Annotations (smaller font)
    if annotate:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if i <= j:  # lower triangle only
                    continue
                val = q.iat[i, j]
                if np.isnan(val): 
                    continue
                ax.text(j + 0.5, i + 0.5, _fmt_q(val), ha="center", va="center",
                        fontsize=annot_size, color="black")

    ax.set_title(title or "Pairwise adjusted p-values")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
