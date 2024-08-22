
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_roc_val(data, section, var):

    if var == 'Tumor&Normal-Tissue-Ratio':
        patch_cal = r'$\frac{Tissue - (Tumor+Normal)}{Tissue}$'
    if var == 'Tumor&Normal-Stroma-Ratio':
        patch_cal = r'$\frac{Stroma}{Stroma+(Tumor+Normal)}$'
    if var == 'Tumor-Tissue-Ratio':
        patch_cal = r'$\frac{Tissue - Tumor}{Tissue}$'
    if var == 'Tumor-Stroma-Ratio':
        patch_cal = r'$\frac{Stroma}{Stroma+Tumor}$'

    data = data[data.section.values == section]
    binary_label = [int(i == 'high') for i in data.label.values]
    tsp = data[var].values.astype('float32')
    label = np.array(binary_label).astype('float32')

    nan_mask = ~np.isnan(tsp)
    tsp = tsp[nan_mask]
    label = label[nan_mask]
    fpr, tpr, threshold = roc_curve(label, tsp)
    roc_auc = auc(fpr, tpr)

    print('{} : tsp median {}'.format(section, np.median(tsp)))
    # print('{} : kappa {}'.format(section, kappa))

    _label = "AUC : " + patch_cal + " = " + "{:0.3}".format(roc_auc)
    plt.plot(fpr, tpr,
             label=_label)
    plt.legend(loc='lower right', fontsize=10)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('False Positive Rate', size=12)
    plt.ylabel('True Positive Rate', size=12)


def plot_roc(data, section):
    binary_label = [int(i == 'high') for i in data.label.values]
    tsp = data['Tumor&Normal-Tissue-Ratio'].values.astype('float32')
    label = np.array(binary_label).astype('float32')

    nan_mask = ~np.isnan(tsp)
    tsp = tsp[nan_mask]
    label = label[nan_mask]

    fpr, tpr, threshold = roc_curve(label, tsp)
    roc_auc = auc(fpr, tpr)
    # kappa = cohen_kappa_score((tsp > 0.85).astype(int), label)

    print('{} : tsp median {}'.format(section, np.median(tsp)))
    # print('{} : kappa {}'.format(section, kappa))

    _label = "$AUC_{" + section + "} = " + "{:0.3}$".format(
        roc_auc)
    plt.plot(fpr, tpr,
             label=_label)
    plt.legend(loc='lower right', fontsize=8)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('False Positive Rate', size=10)
    plt.ylabel('True Positive Rate', size=10)


tsr_result = pd.read_csv(f'./data/result_stomach_part.csv')
tsr_result0 = pd.read_csv(f'./data/result_colon_part.csv')
tsr_result['section'] = 'stomach'
tsr_result0['section'] = 'colon'
tsr_result = pd.concat((tsr_result, tsr_result0))

data = tsr_result[tsr_result.label.values != 'int']
data.label.values

data.slidename.values

data = data[['slidename', 'label', 'section', 'Tumor-Stroma-Ratio',
       'Tumor-Tissue-Ratio', 'Tumor&Normal-Stroma-Ratio',
       'Tumor&Normal-Tissue-Ratio']].groupby(['slidename', 'label', 'section']).mean()
data = data.reset_index(drop=False)
data.label

aa=data[data['section'] == 'colon']
np.sum((aa.label == 'high').values)



plt.figure(figsize=(4, 4))
plt.title(r'Multiclass-classification method', size=10, pad=8)
section = 'stomach'
data_stomach = data[data.section.values == section]
plot_roc(data_stomach,section)
section = 'colon'
data_colon = data[data.section.values == section]
plot_roc(data_colon,section)
section = 'stomach+colon'
plot_roc(data, section)
plt.savefig('./data/TSR_Ratio_Stomach.png')


section = 'colon'
plt.figure(figsize=(5, 5))
plt.title(rf'Colon Multiclass-classification method', size=12, pad=8)
var = 'Tumor&Normal-Tissue-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor-Tissue-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor&Normal-Stroma-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor-Stroma-Ratio'
plot_roc_val(data, section, var)
plt.savefig('./data/TSR_Ratio_Colon_condition.png')

section = 'stomach'
plt.figure(figsize=(5, 5))
plt.title(rf'Stomach Multiclass-classification method', size=12, pad=8)
var = 'Tumor&Normal-Tissue-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor-Tissue-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor&Normal-Stroma-Ratio'
plot_roc_val(data, section, var)
var = 'Tumor-Stroma-Ratio'
plot_roc_val(data, section, var)
plt.savefig('./data/TSR_Ratio_Stomach_condition.png')