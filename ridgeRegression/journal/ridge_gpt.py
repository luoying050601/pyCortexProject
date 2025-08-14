import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
import numpy as np
from ridge import createDataSet_prince, compute_p, fdr, plot_flatmap

def perform_grid_search(X_train, Y_train, alphas):
    # 初始化 Ridge 模型和超参数网格
    ridge = Ridge()
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # 超参数搜索
    grid_search.fit(X_train, Y_train)
    best_alpha = grid_search.best_params_['alpha']
    print('Best alpha:', best_alpha)
    return best_alpha

def log_results(save_dir, model_type, method, best_alpha, mean_acc):
    log_file = os.path.join(save_dir, 'logs/experiment_log.txt')
    with open(log_file, 'a') as f:
        f.write(f"Model: {model_type}, Method: {method}, Alpha: {best_alpha}, Mean Accuracy: {mean_acc}\n")
    print(f"Logged results to {log_file}")

def save_predictions(Y_pred, save_dir, model_type, method):
    save_filename = f"results/{model_type}_{method}_predictions.npy"
    save_filepath = os.path.join(save_dir, save_filename)
    # 使用 numpy.save() 保存为 .npy 文件
    np.save(save_filepath, Y_pred)
    print(f"Predictions saved to {save_filepath}")

def save_model(model, save_dir, model_type, method):
    save_filename = f"models/{model_type}_{method}_ridge_model.pkl"
    save_filepath = os.path.join(save_dir, save_filename)
    joblib.dump(model, save_filepath)
    print(f"Model saved to {save_filepath}")

def evaluate_model(Y_val, Y_pred):
    # 计算每个 voxel 的 Pearson 相关系数
    corr_test = np.array([pearsonr(Y_val[:, i], Y_pred[:, i])[0] for i in range(Y_val.shape[1])])
    p_test = compute_p(corr_test, corr_test.shape[0])

    # FDR 校正
    p_rejected, _ = fdr(p_test)

    # 设置拒绝的相关性值
    corr_rejected = np.maximum(0, corr_test)
    corr_rejected[~p_rejected] = 0

    # 返回结果
    return corr_rejected, np.mean(corr_rejected), corr_test


def run_experiment(model_type, method, alphas):
    print(f"model_type: {model_type}, method: {method}")

    # 创建数据集 , Y_test, X_test
    Y_train, X_train= createDataSet_prince(subj, load_dir, embedding_model_name=model_type, _type="training",
                                                            method=method)

    # 执行第一次超参数搜索（原始特征）
    # best_alpha = perform_grid_search(X_train, Y_train, alphas)
    #
    # # 使用找到的最优 alpha 值训练 Ridge 模型（原始特征）
    # ridge_best = Ridge(alpha=best_alpha)
    # ridge_best.fit(X_train, Y_train)

    # 使用多项式特征
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)

    # 第二次超参数搜索（多项式特征）
    best_alpha_poly = perform_grid_search(X_poly_train, Y_train, alphas)

    # 训练 Ridge 模型（使用多项式特征）
    ridge_best_poly = Ridge(alpha=best_alpha_poly)
    ridge_best_poly.fit(X_poly_train, Y_train)

    # 预测阶段
    Y_val, X_val = createDataSet_prince(subj, load_dir, embedding_model_name=model_type, _type="pred", method=method)
    X_poly_val = poly.transform(X_val)  # 使用相同的多项式特征变换
    Y_pred = ridge_best_poly.predict(X_poly_val)

    # 评估结果
    corr_rejected, mean_acc, corr_test = evaluate_model(Y_val, Y_pred)
    n_voxels = np.array(corr_test).shape[0]

    # 打印结果
    print(f"Mean accuracy for {model_type} - {method}: {mean_acc}")
    # 可視化
    # corr_graph(corr_test)
    pred_acc = []
    for i in range(n_voxels):
        pred_acc.append([corr_rejected[i]])
    # print("The number of target voxels: {}".format(len(pred_acc)))
    print(np.mean(pred_acc))

    # Transform the vectorized data to the EPI volumes
    plot_flatmap(model_type, method, best_alpha_poly, pred_acc, volume_size, subj, xfm, cmap="hot", save_dir=save_dir)
    log_results(save_dir, model_type, method, best_alpha_poly, mean_acc)



# 主程序
if __name__ == "__main__":
    alphas = np.logspace(-6, 6, 100)
    model_list = ['brainlm', 'albert-xlarge-v1', "BERT",] #'XLM', 'GloVe', 'gpt2', 'BERT',
    method_list = ['FLA', 'VLA']
    load_dir = "/home/ying/project/pyCortexProj/"
    subj = 'sub_EN057'
    volume_size = [64, 64, 33]
    xfm = "fullhead2"
    save_dir = "/home/ying/project/pyCortexProj/ridgeRegression/journal/"

    for model_type in model_list:
        for method in method_list:
            run_experiment(model_type, method, alphas)

