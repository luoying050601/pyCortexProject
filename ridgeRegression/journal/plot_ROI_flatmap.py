import os
from random import randint
display_num = randint(100, 200)  # 随机生成一个虚拟显示端口
os.system(f"Xvfb :{display_num} -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = f":{display_num}"
import numpy as np
import json
from com.pycortex.src.utils.common import plot_flatmap


# 准备存储每个ROI的平均值
# roi_averages = {}
def get_brain_cortex_data(subj):
    # # 加载你的预测结果数组，假设它是一个 NumPy 数组，大小为220000
    # # EN '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json'  exp3_layer_acl_EN_24_brainlm_pearson_corr_whole_cortex.json
    # # FR /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_FR_brainlm_anno_pearson_corr_whole_cortex_after.json
    # # CN '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp3_acl_ZH_brainlm_anno_pearson_corr_whole_cortex_after.json'
    if subj == 'sub_CN003':
        with open('/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp3_acl_ZH_brainlm_anno_pearson_corr_whole_cortex_after.json', 'r') as f:
            brain_cortex = np.array(json.load(f)['data'])
    elif subj == 'sub_FR025':
        with open('/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_FR_24_brainlm_pearson_corr_whole_cortex.json', 'r') as f:
            brain_cortex = np.array(json.load(f)['data'])
    elif subj == 'sub_EN057':
        with open('/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json', 'r') as f:
            brain_cortex = np.array(json.load(f)['data'])
    return brain_cortex

def get_ROI_list(brain_cortex,subj):
    roi_averages_list = []
    # 加载之前生成的150个ROI索引的JSON文件
    with open('/home/ying/project/pyCortexProj/resource/littlePrince/' + subj + '/' + subj + '_roi_indices.json',
              'r') as f:
        roi_indices = json.load(f)
        roi_names = list(roi_indices.keys())
        # 对于每个ROI，计算其平均值
    for roi_name, indices in roi_indices.items():
        # 从预测结果中提取该ROI的值
        roi_values = brain_cortex[indices]

        # 计算该ROI的平均值
        roi_avg = np.mean(roi_values)

        # 存储到字典中
        # roi_averages[roi_name] = roi_avg
        roi_averages_list.append(roi_avg)

        # Find the top 3 ROI averages and their corresponding names
    top_3_indices = np.argsort(roi_averages_list)[-3:][::-1]  # Get indices of the top 3
    top_3_names = [roi_names[i] for i in top_3_indices]
    top_3_values = [roi_averages_list[i] for i in top_3_indices]

    # Output the top 3 ROIs
    print("Top 3 ROIs by average values:")
    for name, value in zip(top_3_names, top_3_values):
        print(f"{name}: {value:.4f}")
    return roi_averages_list


subj_en = 'sub_EN057' #sub_CN003 sub_FR025 sub_EN057
subj_fr = 'sub_FR025' #sub_CN003 sub_FR025 sub_EN057
subj_zh = 'sub_CN003' #sub_CN003 sub_FR025 sub_EN057
subjects_dir = "/Storage2/brain_group/freesurfer/subjects"  # Replace with the path to your subjects directory
save_dir = "/home/ying/project/pyCortexProj/ridgeRegression/journal/flatmap/"

brain_cortex_en = get_brain_cortex_data(subj_en)
roi_list_en = get_ROI_list(brain_cortex_en,subj_en)
brain_cortex_fr = get_brain_cortex_data(subj_fr)
roi_list_fr = get_ROI_list(brain_cortex_fr, subj_fr)
brain_cortex_zh = get_brain_cortex_data(subj_zh)
roi_list_zh = get_ROI_list(brain_cortex_zh, subj_zh)

# 加载存储6个功能区域的数据
def load_roi_data(roi_filepath):
    with open(roi_filepath, 'r') as f:
        roi_data = json.load(f)
    return roi_data

# 获取功能区域的平均值
def compute_functional_region_avg(brain_cortex, roi_data):
    region_averages = {}
    for region, indices in roi_data.items():
        # 提取该功能区域的所有索引对应的值
        roi_values = brain_cortex[indices]
        # 计算平均值
        roi_avg = np.mean(roi_values)
        # 存储区域的平均值
        region_averages[region] = roi_avg
    return region_averages

# 比较不同功能区域的平均值
def compare_functional_regions(subject):
    # 根据subject获取预测数据
    brain_cortex = get_brain_cortex_data(subject)

    # 加载6个功能区域的ROI数据
    roi_data = load_roi_data('/home/ying/project/pyCortexProj/resource/littlePrince/'+subject+'/roi_index_list.json')

    # 计算每个功能区域的平均值
    region_averages = compute_functional_region_avg(brain_cortex, roi_data)

    # 输出比较结果 Functional Region Average Comparison:
    print(f"{subject}: ")
    for region, avg in region_averages.items():
        print(f"{region}: {avg:.3f}")

    # 返回结果以便进一步处理
    return region_averages

compare_functional_regions(subj_en)
compare_functional_regions(subj_fr)
compare_functional_regions(subj_zh)


import matplotlib.pyplot as plt
import numpy as np

# Data to plot
subjects = ['sub_EN057', 'sub_FR025', 'sub_CN003']
functional_regions = ['Visual', 'Language', 'Auditory']

# Values for left and right hemisphere contributions
left_data = {
    'sub_EN057': [0.127, 0.102, 0.111],
    'sub_FR025': [0.188, 0.175, 0.177],
    'sub_CN003': [0.253, 0.174, 0.210]
}

right_data = {
    'sub_EN057': [0.113, 0.103, 0.104],
    'sub_FR025': [0.206, 0.187, 0.201],
    'sub_CN003': [0.221, 0.166, 0.190]
}

# Number of functional regions
n_regions = len(functional_regions)

# Set positions for each subject's bars
bar_width = 0.2
index = np.arange(n_regions) * (len(subjects) * (bar_width + 0.05) + 0.3)  # Adjust spacing between groups
space_between_subjects = 0.05  # Space between subjects within a functional region

# Colors for left and right hemisphere
colors_left = ['#99CCFF', '#99FF99', '#FF9999']  # Light colors for left hemisphere
colors_right = ['#6699CC', '#66CC66', '#FF6666']  # Darker shades for right hemisphere

# Create figure and axis with less width for reduced whitespace
fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the figure size

# Plot bars for each subject for each functional region
for i, subject in enumerate(subjects):
    bar_position = index + i * (bar_width + space_between_subjects)
    ax.bar(bar_position, left_data[subject], bar_width, label=f'{subject} Left', color=colors_left[i])
    ax.bar(bar_position, right_data[subject], bar_width, bottom=left_data[subject], label=f'{subject} Right', color=colors_right[i])

# Set the labels and ticks
ax.set_xlabel('Functional Regions')
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Correlation Coefficients by Functional Region and Hemisphere')

# Set x-ticks at the center of each functional region group
xtick_positions = index + (len(subjects) * (bar_width + space_between_subjects)) / 2 - bar_width / 2
ax.set_xticks(xtick_positions)
ax.set_xticklabels(functional_regions)

# Move the legend outside the plot
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Avoid duplicate labels
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to ensure everything fits, especially with the legend outside
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reduce the rect area to minimize whitespace

# Display the plot
plt.show()
# 输出每个ROI的平均值
# for roi, avg in roi_averages.items():
#     print(f"ROI: {roi}, Average: {avg}")

# plot_flatmap(subjects_dir, subj_en, _type='differential', score=(np.array(roi_list_fr)-np.array(roi_list_zh)), _cmap="RdBu", save_dir=save_dir)
# plot_flatmap(subjects_dir, subj_en, _type='zh', score=(np.array(roi_list_zh)), _cmap="RdBu", save_dir=save_dir)
# plot_flatmap(subjects_dir, subj_en, _type='fr', score=(np.array(roi_list_fr)), _cmap="RdBu", save_dir=save_dir)
# plot_flatmap(subjects_dir, subj_en, _type='en', score=(np.array(roi_list_en)), _cmap="RdBu", save_dir=save_dir)

# # 如果需要保存结果为JSON文件
# with open('roi_averages.json', 'w') as f:
#     json.dump(roi_averages, f)