import json

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import RidgeCV, Ridge, Lasso

from create_word_embedding import createDataSet
import logging

# 配置 logging 的基本设置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 日期时间格式
)

# 示例：打印一条日志
logging.info("Logging has been set up.")
model_list = [
    "bart",
    "t5",
    "t5-l",
    "llama2"
    # 'GloVe',
    # 'word2vec',
    # 'bert-base-uncased',
    # 'bert-large-uncased',
    # 'bert-base-multilingual-cased',
    # 'bert-large-uncased-whole-word-masking',
    # 'roberta-large',
    # 'roberta-base',
    # 'albert-base-v1',
    # 'albert-large-v1',
    # 'albert-xlarge-v1',
    # 'albert-xxlarge-v1',
    # 'albert-base-v2',
    # 'albert-large-v2',
    # 'albert-xlarge-v2',
    # 'albert-xxlarge-v2',
    # 'gpt2',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl'
    #         'brainbert'
]
#
# # 看一下模型在训练和测试数据的误差表现，以及参数的尺度分布。
# def plot_residuals_and_coeff(resid_train, resid_test, coeff):
#     fig, axes = plt.subplots(1, 3, figsize=(12, 3))
#     axes[0].bar(np.arange(len(resid_train)), resid_train)
#     axes[0].set_xlabel("sample number")
#     axes[0].set_ylabel("residual")
#     axes[0].set_title("training data")
#     axes[1].bar(np.arange(len(resid_test)), resid_test)
#     axes[1].set_xlabel("sample number")
#     axes[1].set_ylabel("residual")
#     axes[1].set_title("testing data")
#     axes[2].bar(np.arange(len(coeff)), coeff)
#     axes[2].set_xlabel("coefficient number")
#     axes[2].set_ylabel("coefficient")
#     fig.tight_layout()
#     return fig, axes
#
# # 创建损失计算函数 SSE
# def sse(resid):
#     return np.sum(resid**2)
# 交差検証
# def cross_validate(train_x_all,train_y_all,a_,split_size=5):
#   results = [0 for _ in range(train_y_all.shape[1])]
#   kf = KFold(n_splits=split_size)
#   for train_idx, val_idx in kf.split(train_x_all, train_y_all):
#     train_x = train_x_all[train_idx]
#     train_y = train_y_all[train_idx]
#     val_x = train_x_all[val_idx]
#     val_y = train_y_all[val_idx]
#
#     reg = Ridge(alpha=a_).fit(train_x,train_y)
#     pre_y = reg.predict(val_x)
#
#     y_val_T = val_y.T
#     y_pre_T = pre_y.T
#     k_fold_r = correlation_c(y_val_T,y_pre_T)
#     results = [x + y for (x, y) in zip(results, k_fold_r)]
#
#   results = map(lambda x : x/5,results)
#   results = list(results)
#   return results
#
# def regmodel_param_plot(
#         validation_score, train_score, alphas_to_try, chosen_alpha,
#         scoring, model_name, test_score=None, filename=None):
#     plt.figure(figsize=(8, 8))
#     sns.lineplot(y=validation_score, x=alphas_to_try,
#                  label='validation_data')
#     sns.lineplot(y=train_score, x=alphas_to_try,
#                  label='training_data')
#     plt.axvline(x=chosen_alpha, linestyle='--')
#     if test_score is not None:
#         sns.lineplot(y=test_score, x=alphas_to_try,
#                      label='test_data')
#     plt.xlabel('alpha_parameter')
#     plt.ylabel(scoring)
#     plt.title(model_name + ' Regularisation')
#     plt.legend()
#     if filename is not None:
#         plt.savefig(str(filename) + ".png")
#     plt.show()
# def regmodel_param_test(
#         alphas_to_try, X, y, cv, scoring='r2',
#         model_name='LASSO', X_test=None, y_test=None,
#         draw_plot=False, filename=None):
#     validation_scores = []
#     train_scores = []
#     results_list = []
#     if X_test is not None:
#         test_scores = []
#         scorer = get_scorer(scoring)
#     else:
#         test_scores = None
#
#     for curr_alpha in alphas_to_try:
#
#         if model_name == 'LASSO':
#             regmodel = Lasso(alpha=curr_alpha)
#         elif model_name == 'Ridge':
#             regmodel = Ridge(alpha=curr_alpha)
#         else:
#             return None
#
#         results = cross_validate(
#             regmodel, X, y, scoring=scoring, cv=cv,
#             return_train_score=True)
#
#         validation_scores.append(np.mean(results['test_score']))
#         train_scores.append(np.mean(results['train_score']))
#         results_list.append(results)
#
#         if X_test is not None:
#             regmodel.fit(X, y)
#             y_pred = regmodel.predict(X_test)
#             test_scores.append(scorer(regmodel, X_test, y_test))
#
#     chosen_alpha_id = np.argmax(validation_scores)
#     chosen_alpha = alphas_to_try[chosen_alpha_id]
#     max_validation_score = np.max(validation_scores)
#     if X_test is not None:
#         test_score_at_chosen_alpha = test_scores[chosen_alpha_id]
#     else:
#         test_score_at_chosen_alpha = None
#
#     if draw_plot:
#         regmodel_param_plot(
#             validation_scores, train_scores, alphas_to_try, chosen_alpha,
#             scoring, model_name, test_scores, filename)
#
#     return chosen_alpha, max_validation_score, test_score_at_chosen_alpha

import numpy as np


def correlation_roi(y_true, y_pred):
    """
    计算每个 ROI 的 Pearson 相关系数。

    参数:
        y_true (numpy.ndarray): 真实值矩阵，形状为 (样本数, ROI 数)。
        y_pred (numpy.ndarray): 预测值矩阵，形状为 (样本数, ROI 数)。

    返回:
        list: 每个 ROI 的相关系数列表。
    """
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须相同"
    n_roi = y_true.shape[1]  # ROI 数量
    correlations = []

    for roi in range(n_roi):
        corr = np.corrcoef(y_true[:, roi], y_pred[:, roi])[0, 1]  # 计算 Pearson 相关系数
        correlations.append(corr)

    return correlations


def train_and_evaluate(model_type, X_train, Y_train, X_test, Y_test, alpha_list):
    try:
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_train)
        X_scaled_t = sc.transform(X_test)

        model = RidgeCV(alphas=alpha_list)
        model.fit(X_scaled, Y_train)

        Y_pred = model.predict(X_scaled_t)
        corr_list = np.array(correlation_roi(Y_test, Y_pred))
        print(model_type + "_corr_list and average:", corr_list.mean())
        with open(model_type + "_corr_list_jr.json", 'w') as f:
            json.dump({model_type: corr_list.tolist()}, f)# jr: journal review のため

        return model.alpha_, model.score(X_scaled_t, Y_test), corr_list

    except Exception as e:
        print(f"Error during training and evaluation for {model_type}: {e}")
        return None, None, None


def load_and_preprocess_data(model_list, _type):
    data_dict = {}
    for model_type in model_list:
        try:
            logging.info(f"Loading data for model: {model_type}")
            X_test, Y_test = createDataSet(_type=_type, key='test', embedding_model_name=model_type)
            X_train, Y_train = createDataSet(_type=_type, key='train', embedding_model_name=model_type)
            data_dict[model_type] = (X_train, Y_train, X_test, Y_test)
        except Exception as e:
            logging.error(f"Error loading data for {model_type}: {e}")
    return data_dict


if __name__ == "__main__":
    alpha_list = [0.5, 1.0, 5.0, 10.0, 10.0 ** 2, 10.0 ** 3]
    data_dict = load_and_preprocess_data(model_list, _type="run_")
    # results = []

    for model_type, (X_train, Y_train, X_test, Y_test) in data_dict.items():
        alpha, score, corr_list = train_and_evaluate(model_type, X_train, Y_train, X_test, Y_test, alpha_list)

        if alpha:
            # results.append((model_type, alpha, score, corr_list.mean()))
            logging.info(f"Model {model_type}: alpha={alpha}, score={score}, mean_corr={corr_list.mean()}")
            # 输出结果
            # with open(model_type+"_corr_list_results_jr.json", "w") as f: # jr: journal review のため
            #     json.dump(results, f)


