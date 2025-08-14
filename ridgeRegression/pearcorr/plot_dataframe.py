import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white')

df = pd.read_csv("/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/model_segment_type_value.csv", header=0, sep=',')
df = df[df.pc_value.values != 0]
df['model_type'] = df['model_type'].replace('brainbert', 'brainlm2.0')
df['segment_type'] = df['segment_type'].replace('ave_len', 'fixed length')
df['segment_type'] = df['segment_type'].replace('anno', 'variable length')
sns.set(rc={'legend.loc': 'upper center'})

# model_type	segment_type	pc_value
sns.stripplot(x='segment_type', y='pc_value'
              , data=df
              , hue='model_type'  # 以性别分类
              , dodge=True  # 分开显示
              , palette='Blues'  # 设置颜色调色盘
# , showmeans=True
              )
# sns.boxplot(x='category', y='object', data=df, showmeans=True)

# add overall title
plt.title('', fontsize=16)

# add axis titles
plt.xlabel('Approaches')
plt.ylabel('Pearson Correlation')
sns.despine(right=True, top=True)

# rotate x-axis labels
plt.xticks(rotation=0)
plt.show()
