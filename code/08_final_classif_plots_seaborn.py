import numpy as np
import pandas as pd
import seaborn as sns
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

cols = np.array(ct_data.columns)
cols = cols.astype(int)
ct_data.columns = cols

rows = np.array(ct_data.index)
rows = rows.astype(int)
ct_data.index = rows

fig = plt.figure(1,figsize=(10, 10))
fig.suptitle('Actual Subj vs Predicted Subj')
ax = sns.heatmap(ct_data, annot=True,
                 fmt='.3f', annot_kws={"size": 8})

ax.tick_params(axis='both', which='major', labelsize=8)
ax.set(xlabel='Actual', ylabel='Predicted')

plt.savefig('../plots/class_act_vs_pred_subj_seaborn.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# --------------------------------------------------------
# Correct vs Incorrect Plots (Activity)
ct_data = pd.crosstab(true_activity, pred_activity, normalize='index')

cols = np.array(ct_data.columns)
cols = cols.astype(int)
ct_data.columns = cols

rows = np.array(ct_data.index)
rows = rows.astype(int)
ct_data.index = rows

fig = plt.figure(2,figsize=(10, 10))
fig.suptitle('Actual Activity vs Predicted Activity')
ax = sns.heatmap(ct_data, annot=True,
                 fmt='.3f', annot_kws={"size": 8})

ax.tick_params(axis='both', which='major', labelsize=8)
ax.set(xlabel='Actual', ylabel='Predicted')
plt.savefig('../plots/class_act_vs_pred_acti_seaborn.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)

# --------------------------------------------------------
# Correct vs Incorrect Plots (Subject-Activity)
ct_data = pd.crosstab(true_subj_activity, pred_subj_activity, normalize='index')

cols = np.array(ct_data.columns)
cols = cols.astype(int)
ct_data.columns = cols

rows = np.array(ct_data.index)
rows = rows.astype(int)
ct_data.index = rows

fig = plt.figure(3,figsize=(14, 14))
fig.suptitle('Actual Subj Activity vs Predicted Subj Activity')
ax = sns.heatmap(ct_data, annot=False, linewidth=1,
                 fmt='.3f', annot_kws={"size": 8})

ax.tick_params(axis='both', which='major', labelsize=8)
ax.set(xlabel='Actual', ylabel='Predicted')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.savefig('../plots/class_act_vs_pred_subj_acti_seaborn.png', format='png', bbox_inches='tight', pad_inches=0.1,dpi=150)
plt.show()