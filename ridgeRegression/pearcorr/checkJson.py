import json
import numpy as np
from ridgeRegression.utils import non_zero_average,average_values_at_indices
proj_dir = '/Storage2/ying/pyCortexProj/'
subject = "sub_FR025"#sub_EN057
roi_dict = json.load(open('/home/ying/project/pyCortexProj/resource/littlePrince/'+subject+'/roi_index_list.json', 'r'))# print(roi_index)
visual_index_list = roi_dict['lh.visual']+roi_dict['rh.visual']
language_index_list = roi_dict['lh.language']+roi_dict['rh.language']
auditory_index_list = roi_dict['lh.auditory']+roi_dict['rh.auditory']


method_list = ["anno"]
key_list = [
# 'before',
    'after'
]
model_list = [
        # exp1
        # 'albert-xlarge-v1',
        # 'albert-xlarge-v2',
        # 'brainlm',
        # 'brainlm2.0'
# 'bert-base-uncased',
# 'bert-base-multilingual-cased',
# 'mBERT',
# 'XLM-RoBERTa',
'brainlm'

]
for model_name in model_list:
    for method in method_list:
        for key in key_list:
            print(model_name, method, key)
        # /Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_FR_bert-base-multilingual-cased_anno_pearson_corr_whole_cortex_after.json
        # exp1
        # corr_dict = json.load(open(
        #     proj_dir + f'ridgeRegression/pearcorr/acl_' + model_name + "_" + method + "_pearson_corr_whole_cortex.json",
        #     'r'))
        # exp2
            corr_dict = json.load(open(
                proj_dir + f'ridgeRegression/pearcorr/TEST_5130_acl_FR_' + model_name + "_" + method + "_pearson_corr_whole_cortex_"+key+".json",
                'r'))
            for k, v in corr_dict.items():
                print(np.array(v).mean(), non_zero_average(v))
                # v_result = average_values_at_indices(v, visual_index_list)
                # print(f"Average value of visual indices: {v_result}")
                # a_result = average_values_at_indices(v, auditory_index_list)
                # print(f"Average value of auditory indices: {a_result}")
                l_result = average_values_at_indices(v, language_index_list)
                print(f"Average value of language indices: {l_result}")
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# print(corr_dict)
