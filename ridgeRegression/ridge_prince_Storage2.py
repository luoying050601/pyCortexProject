import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
print(Proj_dir)
import sys
sys.path.append(Proj_dir)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import joblib
import time
from ridgeRegression.create_word_embedding import createDataSet_prince
from ridgeRegression.utils import make_print_to_file
proj_dir = "/Storage2/ying/pyCortexProj/"

# test_flag = False
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # GPU
import torch
from sklearn.model_selection import cross_val_score
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    model_list = [
        # 'BERT',
        # 'gpt2',
        # 'GloVe',
        'brainlm'
    ]
    import sys

    if len(sys.argv) > 1:
        _alpha = sys.argv[4]
    else:
        _alpha = 10.0

    start = time.perf_counter()
    # _type = 'run_'
    make_print_to_file("run_prince_rr", path='.')
    print("data loading is starting:")
    # データ作成
    for model_type in model_list:
        print(model_type)
        # try:
        Y_train, X_train, Y_test, X_test = createDataSet_prince(embedding_model_name=model_type,proj_dir=proj_dir)
        print("data loaded...")
        a_list = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3, 10.0 ** 4, 2.5 * (10.0 ** 4), 5.0 * (10.0 ** 4), 10.0 ** 5,
                  10.0 ** 6, 10.0 ** 7]
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_train)
        X_scaled_t = sc.transform(X_test)
        model = RidgeCV(alphas=a_list, store_cv_values=True)
        # 将数据转换为 cupy 数组
        print("ridge regression is training")
        # 在 GPU 上拟合模型
        model.fit(X_scaled, Y_train)
        print("best alpha_:", model.alpha_)
        # print(model.alpha_)
        # 使用交叉验证评估模型性能
        scores = cross_val_score(model, X_scaled_t, Y_test, cv=5)
        print("Cross-validation scores:", scores)        # 这样生成的结果已经弃用啦
        # /usr/local/lib/python3.6/dist-packages/sklearn/base.py:434:
        # FutureWarning: The default value of multioutput (not exposed in score method)
        # will change from 'variance_weighted' to 'uniform_average' in 0.23 to keep consistent with
        # 'metrics.r2_score'. To specify the default value manually and avoid the warning,
        # please either call 'metrics.r2_score' directly or make a custom scorer with 'metrics.make_scorer'
        # (the built-in scorer 'r2' uses multioutput='uniform_average').
        #   "multioutput='uniform_average').", FutureWarning)
        joblib.dump(model, proj_dir+"ridgeRegression/models/prince_" + model_type + "_" + str(model.alpha_) + ".pkl")
        # Y_pre = model.predict(X_scaled_t)
        # corr_list = np.array(correlation_roi_with_pvalue(Y_test, Y_pre))
        # roi_true, roi_pred = get_roi_data_COCO(Y_test, Y_pre)
        # print(model_type + ": Pearson correlation with fdr p-value average:", corr_list.mean(),
        #       '; 2VS2 under Pearson Correlation:', two_vs_two(Y_test, Y_pre, _type='pearsonr'))
        # with open("/Storage2/ying/pyCortexProj/ridgeRegression/pearcorr/" + model_type + "_pearcorr_with_pvalue.json", 'w') as f:
        #     json.dump({model_type: corr_list.tolist()}, f)
        # print(model_type, two_vs_two(Y_test, Y_pre))
    end = time.perf_counter()
    time_cost = ((end - start) / 3600)
    print("time-cost(hours):", time_cost)
