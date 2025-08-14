# 验证预测
import numpy as np
import joblib
import time
import json
from ridge import createDataSet, get_roi_data_COCO
from utils import correlation_roi_dict, correlation_roi_with_pvalue, two_vs_two
from sklearn.preprocessing import StandardScaler

start = time.perf_counter()

model_list = [
    # 'sentence-camembert-large',
    #
    # 'bert-base-uncased',
    # 'bert-large-uncased',
    # 'bert-base-multilingual-cased',
    #
    # 'roberta-large',
    # 'roberta-base',
    # 'albert-base-v1',
    # 'albert-large-v1',
    # 'gpt2',
    # 'GloVe',
    # 'word2vec',
    # 'brainbert',
    # 'brainbert2.0'
    # 'brainlm2.0',
    # 'XLM',
    # 'bert-large-uncased-whole-word-masking',
    # 'sup-simcse-bert-base-uncased',
    # 'unsup-simcse-roberta-large',
    # 'sup-simcse-roberta-large',
    # 'albert-xlarge-v1',
    # 'albert-xxlarge-v1',
    # 'albert-base-v2',
    # 'albert-large-v2',
    # 'albert-xlarge-v2',
    # 'albert-xxlarge-v2',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    # 'llama2',
    'llama2',
    't5-l',
    't5',
    'bart'

]
#

for model_type in model_list:
    if model_type == 'GloVe':
        file_name = 'COCO_GloVe_25000.0.pkl'
    if model_type == 'word2vec':
        file_name = 'COCO_word2vec_25000.0.pkl'
    if model_type == 'bert-base-uncased':
        file_name = 'COCO_bert-base-uncased_25000.0.pkl'
    if model_type == 'bert-base-multilingual-cased':
        file_name = 'COCO_bert-base-multilingual-cased_50000.0.pkl'
    if model_type == 'bert-large-uncased-whole-word-masking':
        file_name = 'COCO_bert-large-uncased-whole-word-masking_25000.0.pkl'
    if model_type == 'bert-large-uncased':
        file_name = 'COCO_bert-large-uncased_50000.0.pkl'
    if model_type == 'roberta-large':
        file_name = 'COCO_roberta-large_25000.0.pkl'
    if model_type == 'roberta-base':
        file_name = 'COCO_roberta-base_50000.0.pkl'
    if model_type == 'albert-base-v1':
        file_name = 'COCO_albert-base-v1_50000.0.pkl'
    if model_type == 'albert-large-v1':
        file_name = 'COCO_albert-large-v1_50000.0.pkl'
    if model_type == 'albert-xlarge-v1':
        file_name = 'COCO_albert-xlarge-v1_10000000.0.pkl'
    if model_type == 'albert-xxlarge-v1':
        file_name = 'COCO_albert-xxlarge-v1_25000.0.pkl'
    if model_type == 'albert-base-v2':
        file_name = 'COCO_albert-base-v2_25000.0.pkl'
    if model_type == 'albert-large-v2':
        file_name = 'COCO_albert-large-v2_10000.0.pkl'
    if model_type == 'albert-xlarge-v2':
        file_name = 'COCO_albert-xlarge-v2_10000000.0.pkl'
    if model_type == 'albert-xxlarge-v2':
        file_name = 'COCO_albert-xxlarge-v2_25000.0.pkl'
    if model_type == 'gpt2':
        file_name = 'COCO_gpt2_10000000.0.pkl'
    if model_type == 'gpt2-medium':
        file_name = 'COCO_gpt2-medium_10000000.0.pkl'
    if model_type == 'gpt2-large':
        file_name = 'COCO_gpt2-large_10000000.0.pkl'
    if model_type == 'gpt2-xl':
        file_name = 'COCO_gpt2-xl_10000000.0.pkl'
    if model_type == 'brainbert2.0':
        file_name = 'COCO_brainbert2.0_1000000.0.pkl'
    if model_type == 'brainbert':
        file_name = 'COCO_brainbert_1000000.0.pkl'
    if model_type == 'sentence-camembert-large':
        file_name = 'COCO_sentence-camembert-large_100000.0.pkl'
    if model_type == 'sup-simcse-bert-base-uncased':
        file_name = 'COCO_sup-simcse-bert-base-uncased_100000.0.pkl'
    if model_type == 'unsup-simcse-roberta-large':
        file_name = 'COCO_unsup-simcse-roberta-large_100000.0.pkl'
    if model_type == 'sup-simcse-roberta-large':
        file_name = 'COCO_sup-simcse-roberta-large_100000.0.pkl'
    # TODO /Storage2/ying/pyCortexProj/ridgeRegression/models/
    if model_type == 'brainlm2.0':
        file_name = 'COCO_brainlm2.0_1000000.0.pkl'
    if model_type == 'llama2':
        file_name = 'COCO_llama2_1000000.0.pkl'
    if model_type == 'XLM':
        file_name = 'COCO_XLM_1000000.0.pkl'
    if model_type == 't5':
        file_name = 'COCO_t5_50000.0.pkl'
    if model_type == 't5-l':
        file_name = 'COCO_t5-l_100000.0.pkl'
    if model_type == 'bart':
        file_name = 'COCO_bart_100000.0.pkl'

    clf = joblib.load('/Storage2/ying/pyCortexProj/ridgeRegression/models/' + file_name)

    print("test data loading。。。")
    print(model_type)
    text_before, brain_true = createDataSet('run_', key='test', embedding_model_name=model_type)
    sc = StandardScaler()
    # X_scaled = sc.fit_transform(X_train)
    text_after = sc.fit_transform(text_before)
    # score = clf.score(text_after, brain_true)
    # print("score:", score)
    brain_pred = clf.predict(text_after)
    corr_list = np.array(correlation_roi_with_pvalue(brain_true, brain_pred))
    print(model_type + ": Pearson correlation with fdr p-value average:", corr_list.mean())
    print(model_type, two_vs_two(brain_true, brain_pred, _type='cosine'))
    roi_true, roi_pred = get_roi_data_COCO(brain_true, brain_pred)
    roi_corr_dict = correlation_roi_dict(roi_true, roi_pred)
    with open('/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/acl_' + model_type + "_corr_list_with_ROI.json",
              'w') as f:
        json.dump(roi_corr_dict, f)

end = time.perf_counter()
time_cost = ((end - start) / 3600)
print("time-cost(hours):", time_cost)
# 加载
