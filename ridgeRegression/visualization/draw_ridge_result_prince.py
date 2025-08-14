import os

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
print(Proj_dir)
import sys
import matplotlib.pyplot as plt
from com.pycortex.src.utils.common import get_roi_border_vertex

sys.path.append(Proj_dir)
# 实验3可视化绘图
import cortex
import numpy as np
import json
# from ridgeRegression.utils import non_zero_average
parcellation = 'aparc.a2009s'

save_dir = "/home/ying/project/pyCortexProj/"
# freesurfer_subject_dir = "/home/ying/subjects"
# freesurfer_subject_name = "sub_EN057"
# freesurfer_subject_name = "sub_CN003"
freesurfer_subject_name = "sub_FR025"
surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(freesurfer_subject_name, "fiducial")]  # flat
left_index = surfs[0].pts.shape[0]
xfm = "fullhead2"
# method = "anno"
# method = "ave_len"

def non_zero_average(lst):
    non_zero_elements = [x for x in lst if x != 0]
    if not non_zero_elements:
        return 0  # 避免除以零错误
    return sum(non_zero_elements) / len(non_zero_elements)

def top_percentile_cutoff(arr, ratio):
    """
    Find the cutoff value for the top percentile of the sorted array.

    Parameters:
        arr (numpy.ndarray): The input 1D array.
        ratio (float): The ratio for selecting the top percentile.

    Returns:
        float: The cutoff value for the top percentile.
    """
    sorted_arr = np.sort(arr)[::-1]  # Sort array in descending order
    index = int(len(sorted_arr) * ratio)
    return sorted_arr[index]

def filter_array(arr, ratio):
    """
    Filter an ndarray setting values below a certain percentile to zero.

    Parameters:
        arr (numpy.ndarray): The input 1D array.
        ratio (float): The ratio for filtering. Values below this percentile will be set to zero.

    Returns:
        numpy.ndarray: The filtered array.
    """
    threshold = top_percentile_cutoff(arr, ratio)
    filtered_arr = np.where(arr < threshold, 0, arr)
    return filtered_arr


# Example usage:
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# plt.axis('off')
# fig, ax = plt.subplots()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.gca().set_frame_on(False)
# plt.legend(frameon=False)

roi_dict = json.load(
    open('/home/ying/project/pyCortexProj/resource/littlePrince/' + freesurfer_subject_name + '/roi_index_list.json',
         'r'))  # print(roi_index)
language_index_list = roi_dict['lh.language'] + roi_dict['rh.language']

model_list = [
    # 'BERT',
    # 'gpt2',
    # 'mBERT',
    # "bert-base-multilingual-cased",
    'brainlm',
    # 'albert-xlarge-v1',
    # 'albert-xlarge-v2',
]
layers_list = [1, 24] #4, 12, 16,
degree_list = [0.1] #0.05, 0.1,
# layers_list = [1]
for degree in degree_list:
    for l in layers_list:
        for i in model_list:
            print(l, i)
            prep = 'exp3_review_'
            if freesurfer_subject_name == 'sub_EN057':
                if i == 'albert-xlarge-v1':
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_albert-xlarge-v1_anno_pearson_corr_whole_cortex.json'
                if i == 'albert-xlarge-v2':
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_albert-xlarge-v2_anno_pearson_corr_whole_cortex.json'
                if i == 'brainlm':
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json'
                if i == 'BERT':
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_bert-base-uncased_anno_after_pearson_corr_whole_cortex.json'
            elif freesurfer_subject_name == 'sub_FR025':
                if len(layers_list) > 1:
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/exp3_layer_acl_FR_' + str(
                        l) + '_brainlm_pearson_corr_whole_cortex.json'
                else:
                    if i == 'brainlm':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_acl_FR_brainlm_anno_pearson_corr_whole_cortex_before.json'
                    if i == 'bert-base-multilingual-cased':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_acl_FR_bert-base-multilingual-cased_anno_pearson_corr_whole_cortex_before.json'
                    if i == 'mBERT':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_acl_FR_mBERT_anno_pearson_corr_whole_cortex_before.json'
                    if i == 'BERT':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_acl_FR_bert-base-uncased_anno_pearson_corr_whole_cortex_before.json'

                    if i == 'albert-xlarge-v1':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_review_acl_FR_albert-xlarge-v1_anno_pearson_corr_whole_cortex_after.json'
                    if i == 'albert-xlarge-v2':
                        path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp2_acl_FR_albert-xlarge-v2_anno_pearson_corr_whole_cortex_before.json'

            elif freesurfer_subject_name == 'sub_CN003':
                if i == 'brainlm':
                    path = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/new_exp3_acl_ZH_brainlm_anno_pearson_corr_whole_cortex_after.json'

            # brainlm average
            # en /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_brainlm2.0_anno_pearson_corr_whole_cortex.json
            # fr /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/5145_acl_FR_brainlm_anno_pearson_corr_whole_cortex_after.json
            # cn

            # brainlm layers
            # 1
            # 12
            # 24
            brain_cortex = np.array(json.load(open(path, 'r'))['data'])
            filtered_arr = filter_array(brain_cortex, degree)
            # print(filtered_arr)

            # print(np.average(brain_cortex))
            print(i + ": Pearson correlation with p-value average:",
                  brain_cortex.mean(), non_zero_average(brain_cortex))
            print(i + ": Pearson correlation with p-value average left and right:",
                  non_zero_average(brain_cortex[:left_index]), non_zero_average(brain_cortex[left_index:]))
            # zeros_arr = np.zeros_like(brain_cortex)
            # for i in language_index_list:
            #     zeros_arr[i] = 0.5

            min_val = 0
            max_val = 0.5
            # + get_roi_border_vertex(freesurfer_subject_name, parcellation).data
            ver = cortex.Vertex(filtered_arr , freesurfer_subject_name, cmap="hot",
                                vmin=min_val, vmax=max_val)

            # ver = cortex.Vertex(filtered_arr, freesurfer_subject_name, cmap="hot", vmin=min_val, vmax=max_val)
            fig = cortex.quickflat.make_figure(ver, with_colorbar=False)
            plt.axis("off")

            # Add sulci in light yellow
            # _ = cortex.quickflat.composite.add_sulci(fig, ver,
            #                                          with_labels=False,
            #                                          linewidth=2,
            #                                          linecolor=(0.9, 0.85, 0.5))
            # # Add all rois, with a particular color scheme:
            # _ = cortex.quickflat.composite.add_rois(fig, ver,
            #                                         with_labels=False,
            #                                         linewidth=1,
            #                                         linecolor=(0.8, 0.8, 0.8))
            # # Highlight face- and body-selective ROIs:
            # _ = cortex.quickflat.composite.add_rois(fig, ver,
            #                                         # roi_list=['FFA', 'EBA', 'OFA'],
            #                                         # (This defaults to all rois if not specified)
            #                                         with_labels=True,
            #                                         linewidth=5,
            #                                         linecolor=(0.9, 0.5, 0.5),
            #                                         roifill=(0.9, 0.5, 0.5),
            #                                         fillalpha=0.35,
            #                                         dashes=(5, 3)  # Dash length & gap btw dashes
            #                                         )
            # plt.savefig('chk.svg', format='svg')
            # plt.show()
            # plt.title(cl)
            # plt.savefig(
            #     save_dir + 'ridgeRegression/visualization/filter'+str(degree)+'_' + prep + freesurfer_subject_name + '_' + i + '_' + str(
            #         l) + '.svg', format='svg')
            #
            plt.show()
            # plt.close()



