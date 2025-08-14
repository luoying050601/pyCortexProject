import json
import csv
import numpy as np
proj_dir = '/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/'
file_list = ['acl_albert-xlarge-v2_ave_len_pearson_corr_whole_cortex.json', 'acl_albert-xlarge-v2_anno_pearson_corr_whole_cortex.json',
             'acl_brainlm_ave_len_pearson_corr_whole_cortex.json', 'acl_brainlm_anno_pearson_corr_whole_cortex.json',
             'acl_brainbert_ave_len_pearson_corr_whole_cortex.json', 'acl_brainbert_anno_pearson_corr_whole_cortex.json',
             'acl_albert-xlarge-v1_ave_len_pearson_corr_whole_cortex.json', 'acl_albert-xlarge-v1_anno_pearson_corr_whole_cortex.json'
             ]

data_file = proj_dir + 'model_segment_type_value.csv'
with open(data_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model_type', 'segment_type', 'pc_value'])
with open(data_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for file in file_list:
            print(file)
            file_l = file.split("_")
            if "ave_len" in file:
                method = "ave_len"
            else:
                method = "anno"
            file_path = proj_dir + file
            result = json.load(open(file_path, 'r'))
            _result = list(map(abs, result["data"]))
            print(sum(_result) / len(np.nonzero(_result)[0]))
            for i in result['data']:
                writer.writerow([file_l[1], method, abs(i)])
                # print([file_l[1], method, abs(i)])





print("finished")



# 'ridgeRegression/pearcorr/acl_' + model_type + "_" + method + "_pearson_corr_whole_cortex.json",
# albert-xlarge-v2 + average length
# albert-xlarge-v2 + annotation
# BrainLM（alice）+ average length
# BrainLM（alice）+ annotation
# BrainLM2.0（alice+wiki）+ average length
# BrainLM2.0（alice+wiki）+ annotation
# albert-xlarge-v1 + average length
# albert-xlarge-v1 + annotation
