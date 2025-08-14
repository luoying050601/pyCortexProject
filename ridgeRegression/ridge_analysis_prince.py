import os
# for prince English
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
print(Proj_dir)
import sys

sys.path.append(Proj_dir)
import numpy as np
import joblib
import time
import json
from ridgeRegression.create_word_embedding import createDataSet_prince
from utils import correlation_with_pvalue
from sklearn.preprocessing import StandardScaler

start = time.perf_counter()
method = "anno"
# method = "ave_len"
# print(method)

# anno
# test data loading。。。
# brainlm
# brainlm: Pearson correlation with fdr p-value average: -4.188872158699736e-06
# time-cost(hours): 0.08480715009357988
# anno
# test data loading。。。
# brainbert
# brainbert: Pearson correlation with fdr p-value average: 0.0031190628305750936
# time-cost(hours): 0.08210385317694292
model_list = [
    # 'BERT',

    # 'gpt2',
    # 'GloVe',
    # "brainbert",
    # "brainbert2.0"
    'albert-xlarge-v1',
    # 'albert-xlarge-v2',
    # 'brainlm'
]
# laplace
proj_dir = '/Storage2/ying/pyCortexProj/'
test_dir = '/home/ying/project/pyCortexProj/'

for model_type in model_list:
    if model_type == 'GloVe':
        # method1 /home/ying/project/pyCortexProj/ridgeRegression/models/prince_GloVe_ave_len_10000.0.pkl
        file_name = proj_dir + 'ridgeRegression/models/prince_GloVe_' + method + '_10000.0.pkl'
        # method2
        # file_name = proj_dir + 'ridgeRegression/models/prince_GloVe_' + method + '_1000.0.pkl'
    #     /home/ying/project/pyCortexProj/ridgeRegression/models/prince_GloVe_anno_1000.0.pkl
    # if model_type == 'word2vec':
    #     file_name = 'COCO_word2vec_25000.0.pkl'
    if model_type == 'BERT':
        # method1 /home/ying/project/pyCortexProj/ridgeRegression/models/prince_BERT_ave_len_25000.0.pkl
        file_name = proj_dir + 'ridgeRegression/models/prince_BERT_' + method + '_25000.0.pkl'
        # method2
        # file_name = proj_dir + 'ridgeRegression/models/prince_BERT_' + method + '_1000.0.pkl'
    #     /home/ying/project/pyCortexProj/ridgeRegression/models/prince_BERT_anno_1000.0.pkl
    # if model_type == 'bert-base-multilingual-cased':
    #     file_name = 'COCO_bert-base-multilingual-cased_50000.0.pkl'
    # if model_type == 'bert-large-uncased-whole-word-masking':
    #     file_name = 'COCO_bert-large-uncased-whole-word-masking_25000.0.pkl'
    # if model_type == 'bert-large-uncased':
    #     file_name = 'COCO_bert-large-uncased_50000.0.pkl'
    # if model_type == 'roberta-large':
    #     file_name = 'COCO_roberta-large_25000.0.pkl'
    # if model_type == 'roberta-base':
    #     file_name = 'COCO_roberta-base_50000.0.pkl'
    # if model_type == 'albert-base-v1':
    #     file_name = 'COCO_albert-base-v1_50000.0.pkl'
    # if model_type == 'albert-large-v1':
    #     file_name = 'COCO_albert-large-v1_50000.0.pkl'
    if model_type == 'albert-xlarge-v1':
        #         file_name = proj_dir + 'ridgeRegression/models/prince_BERT_' + method + '_25000.0.pkl'
        # /home/ying/project/pyCortexProj/ridgeRegression/models/prince_albert-xlarge-v1_ave_len_10000000.0.pkl
        # method = ave_len
        # file_name = proj_dir + 'ridgeRegression/models/prince_albert-xlarge-v1_ave_len_10000000.0.pkl'
    #     method = anno 100000.0
    #     /home/ying/project/pyCortexProj/ridgeRegression/models/prince_albert-xlarge-v1_anno_100000.0.pkl
        file_name = test_dir + 'ridgeRegression/models/prince_albert-xlarge-v1_anno_100000.0.pkl'

    # if model_type == 'albert-xxlarge-v1':
    #     file_name = 'COCO_albert-xxlarge-v1_25000.0.pkl'
    # if model_type == 'albert-base-v2':
    #     file_name = 'COCO_albert-base-v2_25000.0.pkl'
    # if model_type == 'albert-large-v2':
    #     file_name = 'COCO_albert-large-v2_10000.0.pkl'
    if model_type == 'albert-xlarge-v2':
        # method = ave_len
        # file_name = proj_dir + 'ridgeRegression/models/prince_albert-xlarge-v2_ave_len_10000000.0.pkl'
        # method = anno 100000.0 /home/ying/project/pyCortexProj/ridgeRegression/models/prince_albert-xlarge-v2_anno_1000000.0.pkl
        file_name = '/home/ying/project/pyCortexProj/ridgeRegression/models/prince_albert-xlarge-v2_anno_1000000.0.pkl'

    # 1000000.0
    # if model_type == 'albert-xxlarge-v2':
    #     file_name = 'COCO_albert-xxlarge-v2_25000.0.pkl'
    if model_type == 'gpt2':
        # method1 /home/ying/project/pyCortexProj/ridgeRegression/models/prince_gpt2_ave_len_100000.0.pkl
        file_name = proj_dir + 'ridgeRegression/models/prince_gpt2_' + method + '_100000.0.pkl'
        #
        # method2
        # file_name = proj_dir+'ridgeRegression/models/prince_gpt2_'+method+'_10.0.pkl'
    #     /home/ying/project/pyCortexProj/ridgeRegression/models/prince_gpt2_anno_10.0.pkl
    # if model_type == 'gpt2-medium':
    #     file_name = 'COCO_gpt2-medium_10000000.0.pkl'
    # if model_type == 'gpt2-large':
    #     file_name = 'COCO_gpt2-large_10000000.0.pkl'
    # if model_type == 'gpt2-xl':
    #     file_name = 'COCO_gpt2-xl_10000000.0.pkl'
    if model_type == 'brainlm':
        # anno /Storage2/ying/pyCortexProj/ridgeRegression/models/prince_brainlm_anno_1000000.0.pkl
        file_name = proj_dir + 'ridgeRegression/models/prince_brainlm_anno_1000000.0.pkl'
    #     average
    #     file_name = proj_dir + 'ridgeRegression/models/prince_brainlm_ave_len_10000000.0.pkl'

        # file_name = proj_dir + 'ridgeRegression/models/prince_brainlm_anno_1000000.0.pkl'

    if model_type == 'brainbert':
        # anno /home/ying/project/pyCortexProj/ridgeRegression/models/prince_brainbert17.67_ave_len_10000000.0.pkl
        # file_name = proj_dir + 'ridgeRegression/models/prince_brainbert_anno_1000000.0.pkl'
        # average
        file_name = proj_dir + 'ridgeRegression/models/prince_brainbert_ave_len_10000000.0.pkl'

        # average
        # file_name = proj_dir + 'ridgeRegression/models/prince_brainbert17.67_' + method + '_1000000.0.pkl'
    #     /home/ying/project/pyCortexProj/ridgeRegression/models/prince_brainbert17.67_anno_1000000.0.pkl

    # if model_type == 'brainbert':
    #     file_name = 'COCO_brainbert_1000000.0.pkl'
    # if model_type == 'sentence-camembert-large':
    #     file_name = 'COCO_sentence-camembert-large_100000.0.pkl'
    # if model_type == 'sup-simcse-bert-base-uncased':
    #     file_name = 'COCO_sup-simcse-bert-base-uncased_100000.0.pkl'
    # if model_type == 'unsup-simcse-roberta-large':
    #     file_name = 'COCO_unsup-simcse-roberta-large_100000.0.pkl'
    # if model_type == 'sup-simcse-roberta-large':
    #     file_name = 'COCO_sup-simcse-roberta-large_100000.0.pkl'

    clf = joblib.load(file_name)

    print("test data loading。。。")
    print(model_type)
    Y_test, X_test = createDataSet_prince(embedding_model_name=model_type, type="pred", method=method)
    sc = StandardScaler()
    # X_scaled = sc.fit_transform(X_train)
    text_after = sc.fit_transform(X_test)
    # score = clf.score(text_after, brain_true)
    # print("score:", score)
    brain_pred = clf.predict(text_after)
    corr_list = np.array(correlation_with_pvalue(Y_test, brain_pred))

    # BERT
    print(model_type + ": Pearson correlation with fdr p-value average:", corr_list.mean())
    # print(model_type, two_vs_two(Y_test, brain_pred, _type='cosine'))
    # roi_true, roi_pred = get_roi_data_COCO(brain_true, brain_pred)
    # roi_corr_dict = correlation_roi_dict(roi_true, roi_pred)
    with open(
            proj_dir + 'ridgeRegression/pearcorr/acl_' + model_type + "_" + method + "_pearson_corr_whole_cortex.json",
            'w') as f:
        json.dump({"data": corr_list.tolist()}, f)
        f.close()

end = time.perf_counter()
time_cost = ((end - start) / 3600)
print("time-cost(hours):", time_cost)
# 加载
