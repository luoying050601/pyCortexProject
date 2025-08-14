import os
from random import randint
from scipy.stats import pearsonr
import nibabel as nib
from nilearn import plotting

import sys
# from ridgeRegression.ridge_alexhuth import bootstrap_ridge
# load_dir = "/home/ying/project/pyCortexProj/"

# 加载真代码。
import json
# 工具包 laplace 执行
import pandas as pd
import numpy as np
import joblib
import cortex

import matplotlib.pyplot as plt
from scipy import stats
import h5py
from statsmodels.stats.multitest import fdrcorrection as fdr


def make_pair_data(feature, width, delay):
    '''
    ペアデータを作る

    <<parameters>>
    feature: ペアデータにするデータ
    width: 時間幅
    delay: 遅延
    width = 4
    delay = 3

    <<return>>
    paired_data: ペアデータになったもの
    '''
    for i in range(width):
        a = np.roll(feature, delay + i, axis=0)
        if i == 0:
            paired_data = a
        else:
            paired_data = np.concatenate((paired_data, a), axis=1)
    return paired_data


def alpha_graph(alphas, corrs, b_alpha, n_cv):
    '''
    alphaのグラフを可視化する

    <<parameters>>
    alphas: (a, ) アルファ達
    corrs: (a, ) それぞれのアルファの平均の相関係数
    save: (bool) Trueならばグラフを保存する
    '''
    fig = plt.figure()
    mcc = corrs[np.where(alphas == b_alpha)][0]
    plt.title(
        'average corr coef in training(CV:{})\nbest alpha: {:.2e}, mean corr coef: {:.3f}'.format(n_cv, b_alpha, mcc))
    plt.xlabel('alpha')
    plt.ylabel('average correlation coefficient')
    plt.plot(alphas, corrs, marker='o')
    plt.plot(b_alpha, mcc, color='red', marker='o')
    plt.xscale('log')
    plt.minorticks_on()
    plt.grid(axis='x')
    plt.grid(which='both', axis='y', ls='--')
    plt.show()


def compute_p(corrs, n):
    t = np.dot(corrs, np.sqrt(n - 2)) / np.sqrt(1 - np.power(corrs, 2))
    p = (1 - stats.t.cdf(np.absolute(t), n - 2)) * 2
    return p


def corr_graph(corr, range_min=-1.0, range_max=1.0):
    '''
    相関係数のヒストグラムを作成する

    <<parameters>>
    corr: (b, ) 相関係数
    range_min: (float) グラフのメモリの最小
    range_max: (float) グラフのメモリの最大
    '''
    fig = plt.figure()
    mean = sum(corr) / len(corr)
    plt.title('histogram of corr coef in test\nmean corr:{:.3g}'.format(mean))
    plt.xlabel('correlation coefficient')
    plt.ylabel('number of voxels')
    plt.grid()
    plt.hist(corr, range=(range_min, range_max), bins=40)
    plt.show()


def normalization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    # m = normData.shape[0]
    normData = data - np.tile(minVals, np.shape(data))
    normData = normData / np.tile(ranges, np.shape(data))
    return normData, ranges, minVals


# 按照总数/目标time point数
def average_word_embedding(text_embedding, timePoint):
    # 使用循环和切片取三个元素
    X = np.empty((0, text_embedding.shape[1]), float)
    for a, b, c, d, e in zip(text_embedding[::5][:], text_embedding[1::5][:], text_embedding[2::5][:],
                             text_embedding[3::5][:], text_embedding[4::5][:]):
        # print(a, b, c)
        # 将数组按位相加
        sum_array = np.array(a) + np.array(b) + np.array(c) + np.array(d) + np.array(e)
        # 取平均
        average_array = sum_array / 5

        # print("按位相加取平均的结果：", average_array)
        # X = np.append(X,average_array)
        if np.isnan(average_array).any():
            # 将存在 NaN 的元素赋值为 0
            average_array[np.isnan(average_array)] = 0
        X = np.append(X, [average_array], axis=0)

    # 输出 shape 2852，1024
    return X[:timePoint, :]


def annotation_word_embedding(text_embedding, timepoint):
    # 读取annotation 文件
    X = np.empty((0, text_embedding.shape[1]), float)

    df = pd.read_csv('/Storage2/ying/pyCortexProj/resource/littlePrince/' + subj + '/lppEN_word_information_global.csv',
                     sep=',', index_col=0, header=0)
    # 输入shape 15376，1024
    for i in range(timepoint):
        # 取出某列的值的上整值
        df['ceil'] = np.ceil(df['global_onset'])

        # 筛选上整值在0到1之间的所有数据
        result = df[(df['ceil'] > 2 * i) & (df['ceil'] <= 2 * (i + 1))].index.values
        if len(result) == 0:
            average_embedding = np.zeros(text_embedding.shape[1], float)
        else:
            average_embedding = np.average(text_embedding[result][:], axis=0)
        if np.isnan(average_embedding).any():
            # 将存在 NaN 的元素赋值为 0
            average_embedding[np.isnan(average_embedding)] = 0
        X = np.append(X, [average_embedding], axis=0)

    # 输出 shape 2852，1024
    return X[:timepoint, :]


def createDataSet_prince(subj, load_dir, embedding_model_name, _type, method):
    # datasetname = 'prince'

    #     load_dir = "/Storage2/ying/pyCortexProj/"
    # /Storage2/ying/pyCortexProj/resource

    # 脳活動データの次元設定
    # brain_dim = 272386
    H_DIM = 1024
    if embedding_model_name in ['gpt2', "BERT"]:
        H_DIM = 768
    elif embedding_model_name == "GloVe":
        H_DIM = 300
    elif embedding_model_name in ['albert-xlarge-v1', "albert-xlarge-v2"]:
        H_DIM = 2048
    elif embedding_model_name == "XLM":
        H_DIM = 2048
    X = np.empty((0, H_DIM), float)
    #     /home/ying/project/pyCortexProj/resource/littlePrince/sub_FR025/echo-1-cortex-volume.h5
    brain_echo_1_path = load_dir + 'resource/littlePrince/' + subj + '/echo-1-cortex-volume.h5'
    brain_echo_2_path = load_dir + 'resource/littlePrince/' + subj + '/echo-2-cortex-volume.h5'
    brain_echo_3_path = load_dir + 'resource/littlePrince/' + subj + '/echo-3-cortex-volume.h5'

    if embedding_model_name not in ['brainbert', 'brainlm']:
        word_embedding_path = '/Storage2/ying/pyCortexProj/resource/littlePrince/' + subj + '/' + embedding_model_name + '_word_embedding_whole_words.h5'
    else:
        # /home/ying/project/pyCortexProj/resource/littlePrince/sub_EN057/brainLM20.57_word_embedding_whole_words_brainlm.h5
        word_embedding_path = "/home/ying/project/pyCortexProj/resource/littlePrince/" + subj + "/brainLM286230_word_embedding_whole_words_brainlm.h5"
    if _type != 'pred':
        # 获取脑数据 echo-1
        with h5py.File(brain_echo_1_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[1]
            file.close()
        with h5py.File(brain_echo_2_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            # brain_cortex = np.stack(brain_cortex, ,axis=1)
            brain_cortex = np.hstack((brain_cortex, dataset[:]))
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        if embedding_model_name in ['brainbert', 'brainlm']:
            text_embedding1, _, _ = normalization(np.array(loaded_json['brainlm']))
        else:
            text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))
        if method == "VLA":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        if method == "VLA":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)
        brain_cortex = brain_cortex.T

        train_Y = brain_cortex[:, :]
        train_X = X[:, :]
        # train_Y = brain_cortex[:int(0.8 * brain_cortex.shape[0]), :]
        # train_X = X[:int(0.8 * X.shape[0]), :]
        # test_Y = brain_cortex[int(0.8 * brain_cortex.shape[0]):, :]
        # test_X = X[int(0.8 * X.shape[0]):, :]
        train_Y, _, _ = normalization(train_Y)
        train_X, _, _ = normalization(train_X)
        # test_Y, _, _ = normalization(test_Y)
        # test_X, _, _ = normalization(test_X)

        return train_Y, train_X#, test_Y, test_X

    else:
        # 获取脑数据 echo-3 for test
        with h5py.File(brain_echo_3_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]  # cortex , time point
            timepoint = brain_cortex.shape[1]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        # text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))
        if embedding_model_name in ['brainbert', 'brainlm']:
            text_embedding1, _, _ = normalization(np.array(loaded_json['brainlm']))
        else:
            text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))

        if method == "VLA":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)
        X = np.append(X, text_embedding, axis=0)
        brain_cortex = brain_cortex.T
        # 获取数组的time point
        #         size_first_dim = brain_cortex.shape[0]
        # 计算需要提取的数据量
        # num_samples = int(timepoint * 0.1)
        # 随机选择10%的索引
        # selected_indices = np.random.choice(timepoint, num_samples, replace=False)
        # 根据选定的索引提取数据
        # VAL_Y = brain_cortex[selected_indices, :]
        # VAL_X = X[selected_indices, :]
        VAL_Y = brain_cortex[:, :]
        VAL_X = X[:, :]
        VAL_Y, _, _ = normalization(VAL_Y)
        VAL_X, _, _ = normalization(VAL_X)
        # print("VAL_X shape:", VAL_X.shape)
        # print("VAL_Y shape:", VAL_Y.shape)

        return VAL_Y, VAL_X


def plot_flatmap(model_type,method,best_alpha, pred_acc, volume_size, subj, xfm, cmap="hot", save_dir=None):
    vol_to_ver_map = cortex.get_mapper(subj, xfm, 'line_nearest', recache=True)

    epi_data = np.zeros(volume_size)
    epi_data[vol_to_ver_map.mask.T] = np.array(pred_acc).reshape(-1)
    epi_data = epi_data.reshape((volume_size[1], volume_size[0], -1), order="F")
    #
    # max_val = epi_data.max()
    # min_val = 0  # This case uses the minimum value of 0, but you can use any values (e.g. the minimum prediction accuracy that reaches the significance level)
    #
    # # Transform the raw EPI coordination to the pycortex coordination
    rot_data = np.transpose(epi_data, (2, 1, 0))
    # # Show the cortical map
    plt.figure(figsize=(10, 5))
    # plt.imshow(np.array(Image.open(out_ctxmap_file)))
    plt.axis("off")
    plt.tight_layout()
    # Set volume data
    dv = cortex.Volume(rot_data, subj, xfm, cmap=cmap) #, vmin=min_val, vmax=max_val
    # Generate a flattened cortical map of the volume data and save it as a png file
    _ = cortex.quickflat.make_figure(dv, with_colorbar=False)
    # _ = cortex.quickflat.make_figure(cortex.Volume(epi_data, subj, xfm, cmap=cmap), with_colorbar=False)
    plt.plot()
    plt.show()
    # 保存图像
    if save_dir:
        save_filename = f"{model_type}_{method}_alpha_{best_alpha}_flatmap.png"
        save_filepath = os.path.join(save_dir, save_filename)
        plt.savefig(save_filepath)
        print(f"Flatmap image saved to {save_filepath}")
    plt.close()  # 确保关闭绘图窗口，防止内存占用


    # roi_array_volume = cortex.Volume(rot_data, subj, xfm, cmap=cmap)
    # affine = nib.load(
    #     "/Storage2/ying/pyCortexProj/venv/share/pycortex/db/" + subj + "/anatomicals/brainmask.nii.gz").affine
    # _ = plotting.plot_img(nib.Nifti1Image(roi_array_volume.data, affine), display_mode="x")
    # plt.show()


def save_params(wt,model_type, method, best_alpha, pred_acc):
    path = load_dir + "ridgeRegression/journal/models/" + model_type + "_" + method + "_" + str(
        best_alpha) + "_" + str(
        round(np.mean(pred_acc), 3))
    joblib.dump(wt, path + ".pkl")
    path = load_dir + "ridgeRegression/journal/results/" + model_type + "_" + method + "_" + str(
        best_alpha) + "_" + str(
        round(np.mean(pred_acc), 3))
    np.save(path + '_predictions.npy', pred_acc)


if __name__ == "__main__":
    display_num = randint(100, 200)  # 随机生成一个虚拟显示端口
    os.system(f"Xvfb :{display_num} -screen 0 1024x768x24 &")
    os.environ["DISPLAY"] = f":{display_num}"


    Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    print(Proj_dir)
    sys.path.append(Proj_dir)
    load_dir = "/home/ying/project/pyCortexProj/"
    subj = 'sub_EN057'
    volume_size = [64, 64, 33]
    xfm = "fullhead2"
    # 'sub_FR025' # 55764
    # sub_EN057 19926
    # sub_CN003 51267 'XLM', 'GloVe', 'gpt2', 'BERT', 'albert-xlarge-v1', "albert-xlarge-v2",
    n_cv = 5
    model_list = ['brainlm']
    method_list = ['FLA', 'VLA']
    # model_type = "brainlm"  # XLM, gpt2, BERT,GloVe,'albert-xlarge-v1', "albert-xlarge-v2"
    # method = "FLA"  # FLA
    # anno VLA
    width = 20
    delay = 3
    alphas = np.logspace(-6, 6, 100)
    # alphas = np.logspace(2, 5, 20)  # ridge回帰の正則化項
    for model_type in model_list:
        for method in method_list:
            print("model_type: {}, method: {}".format(model_type, method))
            # Subject parameters ------------------
            # EPI volume size of fMRI (the first two indicate the matrix size and the last one indictes # horizontal slices)

            Y_train, X_train = createDataSet_prince(subj, load_dir, embedding_model_name=model_type,
                                                                    _type="training", method=method)
            # wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(X_train, Y_train, X_test,
            #                                                       Y_test,
            #                                                       alphas=alphas,
            #                                                       nboots=5,
            #                                                       chunklen=10, nchunks=15, return_wt=True)
            #TODO
            feature_trn_paired = make_pair_data(X_train, width, delay)
            # feature_test_paired = make_pair_data(X_test, width, delay)
            # wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(feature_trn_paired, Y_train, feature_test_paired,
            #                                                       Y_test,
            #                                                       alphas=alphas,
            #                                                       nboots=5,
            #                                                       chunklen=10, nchunks=15, return_wt=True)

            cv_corr = array_mean = np.mean(bscorrs, axis=2)
            # print('cv_corr shape:', np.array(cv_corr).shape)
            mean_corr = np.nanmean(cv_corr, axis=1)  # alphaごとに平均
            # print('mean_corr shape:', mean_corr.shape)
            best_ind = np.argmax(mean_corr)  # 平均の相関係数が最も高くなるalphaの位置
            best_alpha = alphas[best_ind]  # 平均の相関係数が最も高くなるalpha。これをモデルの正則化項として採用します
            print('index: {}, best_alpha: {}'.format(best_ind, best_alpha))
            alpha_graph(alphas, mean_corr, best_alpha, n_cv=n_cv)
            Y_val, X_val = createDataSet_prince(subj, load_dir, embedding_model_name=model_type, _type="pred",
                                                method=method)
            # Y_pred = np.dot(X_val, wt)
            # TODO
            feature_val_paired = make_pair_data(X_val, width, delay)

            # 予測

            Y_pred = np.dot(feature_val_paired, wt)
            # print('pred_test shape:', Y_pred.shape)
            # 相関&P値
            corr_test = []  # (n_voxels, )

            # ボクセルごとに相関を求める
            for i in range(Y_val.shape[1]):
                r, _ = pearsonr(Y_val.T[i, :], Y_pred.T[i, :])
                corr_test.append(r)

            corr_test = np.array(corr_test)
            p_test = compute_p(corr_test, corr_test.shape[0])  # p値を求める。これはライブラリとかを使っても求められると思います

            n_voxels = np.array(corr_test).shape[0]
            # print('corr_test shape:', np.array(corr_test).shape)
            # print('p_test shape:', np.array(p_test).shape)
            # print('n_voxels:', n_voxels)
            # FDR補正 0.05
            p_rejected, _ = fdr(p_test)
            # print('p_rejected shape:', p_rejected.shape)
            # print('棄却されたボクセル数:', np.count_nonzero(p_rejected))
            # 棄却されたところのみ。可視化のために<0は0にしてしまっている
            corr_rejected = np.array(corr_test)
            corr_rejected[~p_rejected] = 0
            corr_rejected[corr_rejected < 0] = 0
            # 可視化
            corr_graph(corr_test)
            pred_acc = []
            for i in range(n_voxels):
                pred_acc.append([corr_rejected[i]])
            # print("The number of target voxels: {}".format(len(pred_acc)))
            print(np.mean(pred_acc))

            # Transform the vectorized data to the EPI volumes
            plot_flatmap(model_type, method, best_alpha, pred_acc, volume_size, subj, xfm, cmap="hot")
            # save_params(wt, model_type, method, best_alpha, pred_acc)

    # /home/ying/project/pyCortexProj/ridgeRegression/journal/models
