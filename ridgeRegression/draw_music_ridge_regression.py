import os

# TODO shell
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
Resources_dir = '/Storage/ying/resources/'
import pickle
from ridgeRegression.utils import load_music_pickle_data, make_print_to_file
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import trange
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection as fdr
import h5py
import json
import seaborn as sns
import pandas as pd
import time
from scipy import io
from scipy import stats


def get_brain_cortex_index(resp_ids, resp_labels):
    cortex = {}
    # index = []
    for i in range(0, resp_ids.shape[0]):
        if b'ctx_' in resp_labels[i]:
            cortex[resp_labels[i]] = resp_ids[i]
    return cortex


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


def draw_visualization(timestamps, df, filename):
    sns.set(style="whitegrid", color_codes=True)
    np.random.seed(sum(map(ord, "categorical")))
    sns.boxplot(x="brain_roi", y="correlation", data=df)
    plt.savefig(
        '/Storage/ying/pyCortexProj/ridgeRegression/visualization/music/roi_pc2_'
        + filename + '_' + str(timestamps) + '.png')
    # sns.stripplot(x="day", y="total_bill", data=tips)
    plt.show()

    # V_Vol = cortex2Vol(pred_acc, tvoxels, Volshape)
    #
    # dv = cortex.Volume(V_Vol, file, "fullhead", cmap="hot", vmin=min_val, vmax=max_val)
    # _ = cortex.quickflat.make_figure(dv,
    #                                  # roi_list=['FFA', 'EBA', 'OFA'],
    #                                  with_curvature=True,
    #                                  with_sulci=True,
    #                                  recache=True,
    #                                  with_rois=True
    #                                  )
    #
    # plt.axis("on")

    # plt.title(cl)
    # plt.savefig(
    #     '/Storage/ying/pyCortexProj/ridgeRegression/visualization/music/pearson_'
    #     + file + '_' + str(timestamps) + '.png')

    # plt.show()
    plt.close()


if __name__ == "__main__":
    LOG_DIR = Proj_dir + '/results/log/'

    # multiprocessing.freeze_support()  # 开启分布式支持
    start_time = time.perf_counter()
    make_print_to_file(_type='draw_music_RR_analysis_', path=LOG_DIR)

    # count = 0
    files = [
        'S3002', 'S3006', 'S3011', 'S3015', 'S3019',
        'S3023', 'S3027', 'S3031', 'S3035', 'S3039',
        'S3043', 'S3047', 'S3051', 'S3003', 'S3008',
        'S3012', 'S3016', 'S3020', 'S3024', 'S3028',
        'S3032', 'S3036', 'S3040', 'S3044', 'S3048',
        'S3052', 'S3004', 'S3009', 'S3013', 'S3017',
        'S3021', 'S3025', 'S3029', 'S3033', 'S3037',
        'S3041', 'S3045', 'S3049', 'S3005', 'S3010',
        'S3014', 'S3018', 'S3022', 'S3026', 'S3030',
        'S3034', 'S3038', 'S3042', 'S3046', 'S3050'
    ]
    with open(Proj_dir+'/results/models/user_best_rr_model.pickle', 'rb') as handle:
        pkg_dict = pickle.load(handle)


    for file in files:
        pair_path = Proj_dir + '/resource/pair/'

        brain_path = Resources_dir + '/musicDataset/fmri/'
        h5_file = file + '.h5'

        f = h5py.File(brain_path + h5_file, 'r')
        # trn = np.array(f.get('rmat_val'))
        resp_ids = f['roi/resp_ids/']
        resp_labels = np.array(f['roi/labels/'])
        brain_cortex = get_brain_cortex_index(resp_ids, resp_labels)

        Volshape = (72, 96, 96)
        # tvoxels = io.loadmat(Resources_dir + '/musicDataset/tvoxels/' + file + '/vset_099.mat')[
        #               'tvoxels'] - 1

        X_test, Y_test = load_music_pickle_data(pair_path=pair_path, file=file, _type="test")
        # S3002_bestAlpha_206.913808111479.pkl
        with open(Proj_dir+'/results/models/'+pkg_dict[file],
                  mode='rb') as fp:
            clf = pickle.load(fp)
        pred_test = clf.predict(X_test)
        # print(pred_test.shape)

        corr_test = []  # (n_voxels, )
        for j in range(Y_test.shape[1]):
            r, _ = pearsonr(Y_test.T[j, :], pred_test.T[j, :])  # 相関とp値が求められる関数。np.corrcoefとかでも相関は求められる。
            corr_test.append(r)

        corr_test = np.array(corr_test)
        p_test = compute_p(corr_test, corr_test.shape[0])  # p値を求める。これはライブラリとかを使っても求められると思います
        print('corr_test shape:', np.array(corr_test).shape)
        print('p_test shape:', np.array(p_test).shape)
        # FDR補正 0.05
        p_rejected, _ = fdr(p_test)
        print('p_rejected shape:', p_rejected.shape)
        print('棄却されたボクセル数:', np.count_nonzero(p_rejected))
        # corr_lst.append(corr)
        # 棄却されたところのみ。可視化のために<0は0にしてしまっている
        corr_rejected = np.array(corr_test)
        corr_rejected[~p_rejected] = 0
        corr_rejected[corr_rejected < 0] = 0
        # corr_graph(corr_test)
        # corr_graph(corr_rejected)
        #
        # project_cortex = np.mean(corr, 0)

        min_val = 0
        max_val = max(corr_test)
        pred_acc = []
        n_voxels = Y_test.shape[1]  # ボクセル数

        for i in range(n_voxels):
            pred_acc.append([corr_rejected[i]])
        print("The number of target voxels: {}".format(len(pred_acc)))

        pred_acc = np.array(pred_acc).reshape(-1)
        print(pred_acc.mean())
        pred_roi = {}
        if not os.path.isfile(Proj_dir+'/ridgeRegression/visualization/music/roi_pc_' + file + '.tsv'):
            df = pd.DataFrame(columns=['brain_roi', 'correlation'])
            for k, v in brain_cortex.items():
                roi_value = []
                # print(k)
                for i in range(pred_acc.size):
                    if i in v:
                        roi_value.append(pred_acc[i])
                        if pred_acc[i] != 0:
                            df = df.append({"brain_roi": k, "correlation": pred_acc[i]}, ignore_index=True)

            df.to_csv(Proj_dir+f'/ridgeRegression/visualization/music/roi_pc_' + file + '.tsv',
                      index=False,
                      sep='\t')
        else:
            df = pd.read_csv(Proj_dir+'/ridgeRegression/visualization/music/roi_pc_' + file + '.tsv',
                             sep='\t', header=0)
            df.columns = ['brain_roi', 'correlation']
            df.sort_values(by="brain_roi", ascending=False)

        timestamps = time.perf_counter()
        print(file)
        left_df = df.loc[df['brain_roi'].str.contains('lh', case=False)]
        # right_df = df.loc[b'rh' in df['brain_roi']]
        right_df = df.loc[df['brain_roi'].str.contains('rh', case=False)]

        # left_df = df[df["lh"] == ('trn0'+str(k))]['genre'].values[0]
        print(left_df['correlation'].mean())
        print(right_df['correlation'].mean())

        draw_visualization(timestamps, left_df, file + '_left')
        draw_visualization(timestamps, right_df, file + '_right')
