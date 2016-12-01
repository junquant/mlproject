import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utilities import Timer, MetaData, ResultsWriter

# file properties
# -----------------------------------------------------
filePath = '../data/results.txt'

metadata = MetaData()
dataType = metadata.getResultColsDataType()

timer = Timer()
startTime = timer.getTime()
print('Start Time : ', timer.getTime())  # Get the start time for tracking purposes

print('------------------------------------------------------------')
print('Reading files ... ')
print('------------------------------------------------------------')
data = np.loadtxt(filePath, delimiter = ',', skiprows = 1, dtype=dataType)
df = pd.DataFrame(data)

# Separating the subject and activity
activity = df.ix[:,-1]%100
activity.name = 'predicted_activity'
subject = (df.ix[:,-1] - activity) / 100
subject.name = 'predicted_subj'

df = pd.concat([df,subject,activity], axis=1)

readings = df.ix[:,:-6]
true_subj = df.ix[:,-6]
true_activity = df.ix[:,-5]
true_subj_activity = df.ix[:,-4]

pred_subj = df.ix[:,-2]
pred_activity = df.ix[:,-1]
pred_subj_activity = df.ix[:,-3]

accurate = np.array(true_subj_activity == pred_subj_activity)

plt.style.use('ggplot')
# --------------------------------------------------------
# Correct vs Incorrect Plots (Subj)
ct_data = pd.crosstab(true_subj, pred_subj, normalize='index')

lang_names = ct_data.columns.tolist()
tick_indices = np.arange(0.5, len(lang_names) + 0.5)

plt.figure(1,figsize=(10, 10))
plt.pcolor(ct_data.values, cmap='RdBu', vmin=-1, vmax=1)
colorbar = plt.colorbar()
colorbar.set_label('Percentage')
plt.title('Actual Subj vs Predicted Subj')
plt.xticks(tick_indices, lang_names, rotation='vertical')
plt.yticks(tick_indices, lang_names)
plt.xlim(0,len(lang_names))
plt.ylim(0,len(lang_names))
plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

plt.savefig('../plots/class_act_vs_pred_subj.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# --------------------------------------------------------
# Correct vs Incorrect Plots (Activity)
ct_data = pd.crosstab(true_activity, pred_activity, normalize='index')

lang_names = ct_data.columns.tolist()
tick_indices = np.arange(0.5, len(lang_names) + 0.5)

plt.figure(2,figsize=(10, 10))
plt.pcolor(ct_data.values, cmap='RdBu', vmin=-1, vmax=1)
colorbar = plt.colorbar()
colorbar.set_label('Percentage')
plt.title('Actual Activity vs Predicted Activity')
plt.xticks(tick_indices, lang_names, rotation='vertical')
plt.yticks(tick_indices, lang_names)
plt.xlim(0,len(lang_names))
plt.ylim(0,len(lang_names))
plt.gcf().subplots_adjust(bottom=0.25, left=0.25)

plt.savefig('../plots/class_act_vs_pred_acti.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)


# --------------------------------------------------------
# Correct vs Incorrect Plots (Subject-Activity)
ct_data = pd.crosstab(true_subj_activity, pred_subj_activity, normalize='index')

lang_names = ct_data.columns.tolist()
tick_indices = np.arange(0.5, len(lang_names) + 0.5)

plt.figure(3,figsize=(10, 10))
plt.pcolor(ct_data.values, cmap='RdBu', vmin=-1, vmax=1)
colorbar = plt.colorbar()
colorbar.set_label('Percentage')
plt.title('Actual Subj Activity vs Predicted Subj Activity')
plt.xticks(tick_indices, lang_names, rotation='vertical')
plt.yticks(tick_indices, lang_names)
plt.xlim(0,len(lang_names))
plt.ylim(0,len(lang_names))
plt.gcf().subplots_adjust(bottom=0.25, left=0.25)


plt.savefig('../plots/class_act_vs_pred_subj_acti.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

plt.show()