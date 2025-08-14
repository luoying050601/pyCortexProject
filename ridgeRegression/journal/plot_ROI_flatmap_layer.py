import os
from random import randint

display_num = randint(100, 200)  # 随机生成一个虚拟显示端口
os.system(f"Xvfb :{display_num} -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = f":{display_num}"
import numpy as np
import json
from com.pycortex.src.utils.common import plot_flatmap

subj_en = 'sub_EN057'  # sub_CN003 sub_FR025 sub_EN057
subj_fr = 'sub_FR025'  # sub_CN003 sub_FR025 sub_EN057
subj_zh = 'sub_CN003'  # sub_CN003 sub_FR025 sub_EN057
subjects_dir = "/Storage2/brain_group/freesurfer/subjects"  # Replace with the path to your subjects directory
save_dir = "/home/ying/project/pyCortexProj/ridgeRegression/journal/flatmap/"


# 准备存储每个ROI的平均值
# roi_averages = {}
def get_ROI_layer_list(subj, layer=None):
    # # 加载你的预测结果数组，假设它是一个 NumPy 数组，大小为220000
    # # EN '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json'
    # # FR /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_FR_brainlm_anno_pearson_corr_whole_cortex_after.json
    # # CN '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp3_acl_ZH_brainlm_anno_pearson_corr_whole_cortex_after.json'
    if subj == 'sub_CN003':
        if layer is None:
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp3_acl_ZH_brainlm_anno_pearson_corr_whole_cortex_after.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])
        else:
            # /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_CN_1_brainlm_pearson_corr_whole_cortex.json
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_CN_' + str(
                            layer) + '_brainlm_pearson_corr_whole_cortex.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])
    elif subj == 'sub_FR025':
        if layer is None:
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_FR_24_brainlm_pearson_corr_whole_cortex.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])
        else:
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_FR_' + str(
                            layer) + '_brainlm_pearson_corr_whole_cortex.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])

    elif subj == 'sub_EN057':
        if layer is None:
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])
        else:
            # /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_EN_12_brainlm_pearson_corr_whole_cortex.json
            with open(
                    '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_EN_' + str(
                            layer) + '_brainlm_pearson_corr_whole_cortex.json',
                    'r') as f:
                brain_cortex = np.array(json.load(f)['data'])
    roi_averages_list = []
    # 加载之前生成的150个ROI索引的JSON文件
    with open('/home/ying/project/pyCortexProj/resource/littlePrince/' + subj + '/' + subj + '_roi_indices.json',
              'r') as f:
        roi_indices = json.load(f)
    # 对于每个ROI，计算其平均值
    print(subj,layer,np.mean(brain_cortex))
    for roi_name, indices in roi_indices.items():
        # 从预测结果中提取该ROI的值
        roi_values = brain_cortex[indices]

        # 计算该ROI的平均值
        roi_avg = np.mean(roi_values)
        # print()

        # 存储到字典中
        # roi_averages[roi_name] = roi_avg
        roi_averages_list.append(roi_avg)
    return roi_averages_list

def compare_multiple_ROI_lists(*roi_lists):
    # 初始化一个150大小的数组，存放结果
    result = []

    # 使用zip()同时遍历多个数组
    for values in zip(*roi_lists):
        # 找到最大值及其对应的索引
        max_value = max(values)
        max_index = values.index(max_value) + 1  # 索引从1开始计数

        # 将索引添加到结果数组中
        result.append(max_index)

    return result

# 示例用法：
roi_list_fr_1 = get_ROI_layer_list(subj_zh, 1)
roi_list_fr_4 = get_ROI_layer_list(subj_zh, 4)
roi_list_fr_12 = get_ROI_layer_list(subj_zh, 12)
roi_list_fr_16 = get_ROI_layer_list(subj_zh, 16)
roi_list_fr_24 = get_ROI_layer_list(subj_zh, 24)

# # 动态传入任意数量的ROI列表
final_result = compare_multiple_ROI_lists(roi_list_fr_1, roi_list_fr_4, roi_list_fr_12, roi_list_fr_16, roi_list_fr_24)
#
# # 输出结果
print(final_result)
# RdBu
plot_flatmap(subjects_dir, subj_en, _type='layer', score=final_result, save_dir=save_dir)
