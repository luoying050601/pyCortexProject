# 工具包 laplace 执行
import json
import os
import numpy as np
from transformers import BertModel, BertTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers import AlbertTokenizer, AlbertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import XLMModel, XLMTokenizer
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
# from simcse import SimCSE
import h5py
# import math

# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import lmdb
import msgpack
from lz4.frame import decompress
import torch
from ridgeRegression.utils import normalization
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# from models.sentence2vec import Sentence2Vec


# H_DIM = 1024
# 创建train test 训练集
# type-test/run
# key:train/test
# embedding_model_name: bert-base-uncased etc.
def createDataSet(_type, key, embedding_model_name):
    datasetname = 'COCO2014_2023'
    # 中間表現の次元

    # 脳活動データの次元設定
    brain_dim = 3566
    # データの作成
    if embedding_model_name in ['sup-simcse-bert-base-uncased', 'unsup-simcse-roberta-large',
                                'sup-simcse-roberta-large']:
        if embedding_model_name == 'sup-simcse-bert-base-uncased':
            H_DIM = 768
        else:
            H_DIM = 1024
    elif embedding_model_name == 'sentence-camembert-large':
        H_DIM = 1024
    elif embedding_model_name in \
            ['bert-large-uncased',
             'bert-large-uncased-whole-word-masking']:
        H_DIM = 1024
    elif embedding_model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
        H_DIM = 768
    elif embedding_model_name == 'roberta-large' or embedding_model_name == 'roberta-base':
        # roberta
        H_DIM = 1024
        if embedding_model_name == 'roberta-base':
            H_DIM = 768
    elif embedding_model_name in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                                  'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2',
                                  'albert-xxlarge-v2']:
        if embedding_model_name in ['albert-base-v1', 'albert-base-v2']:
            H_DIM = 768
        if embedding_model_name in ['albert-large-v1', 'albert-large-v2']:
            H_DIM = 1024
        if embedding_model_name in ['albert-xlarge-v1', 'albert-xlarge-v2']:
            H_DIM = 2048
        if embedding_model_name in ['albert-xxlarge-v2', 'albert-xxlarge-v1']:
            H_DIM = 4096
    elif embedding_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        if embedding_model_name in ['gpt2']:
            H_DIM = 768
        if embedding_model_name in ['gpt2-medium']:
            H_DIM = 1024
        if embedding_model_name in ['gpt2-large']:
            H_DIM = 1280
        if embedding_model_name in ['gpt2-xl']:
            H_DIM = 1600
    elif embedding_model_name in ['brainbert', 'brainbert2.0', 'brainlm2.0']:
        H_DIM = 1024
    elif embedding_model_name.lower() == "glove":
        H_DIM = 300
    elif embedding_model_name == 'word2vec':
        H_DIM = 100
    #     TODO 添加新对象在这里加
    elif embedding_model_name == 'llama2':
        H_DIM = 4096
    elif embedding_model_name == 't5-l':
        H_DIM = 1024
    elif embedding_model_name == 't5':
        H_DIM = 512
    elif embedding_model_name == 'bart':
        H_DIM = 1024
    elif embedding_model_name == 'bart-base':
        H_DIM = 768
    elif embedding_model_name == 'XLM':
        H_DIM = 2048

    X = np.empty((0, H_DIM), float)
    Y = np.empty((0, brain_dim), float)
    if key == 'train':
        brain_dbs = ["/Storage2/ying/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_train.db",
                     "/Storage2/ying/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_test.db"
                     ]
        text_dbs = ["/Storage2/ying/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_train.db",
                    "/Storage2/ying/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_test.db"
                    ]
    else:
        text_dbs = ["/Storage2/ying/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_val.db"]
        brain_dbs = ["/Storage2/ying/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_val.db"]
    # 文本向量化的cache 如果不存在则要生成。
    cache_path = '/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/' + datasetname + '/' \
                 + embedding_model_name + f'_text_embedding_' + str(key) + '.json'
    if os.path.exists(cache_path):
        sen_embed_dict = json.load(open(cache_path, 'r'))
    else:
        sen_embed_dict = {}
        # 加载embedding model
        if embedding_model_name in ['sup-simcse-bert-base-uncased', 'unsup-simcse-roberta-large',
                                    'sup-simcse-roberta-large']:
            model = SimCSE("princeton-nlp/" + embedding_model_name)
        elif embedding_model_name == 'sentence-camembert-large':
            model = SentenceTransformer("dangvantuan/sentence-camembert-large")

        elif embedding_model_name in \
                ['bert-large-uncased',
                 'bert-large-uncased-whole-word-masking']:
            tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
            model = BertModel.from_pretrained(embedding_model_name)
        elif embedding_model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:

            tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
            model = BertModel.from_pretrained(embedding_model_name)
        elif embedding_model_name in ['roberta-large', 'roberta-base']:
            # roberta
            config = AutoConfig.from_pretrained(embedding_model_name)
            config.output_hidden_states = True
            tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            model = AutoModelForMaskedLM.from_pretrained(embedding_model_name, config=config)

        elif embedding_model_name in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                                      'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2',
                                      'albert-xxlarge-v2']:

            tokenizer = AlbertTokenizer.from_pretrained(embedding_model_name)
            model = AlbertModel.from_pretrained(embedding_model_name)
        elif embedding_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            tokenizer = GPT2Tokenizer.from_pretrained(embedding_model_name)
            model = GPT2Model.from_pretrained(embedding_model_name)
            # trainデータとtestデータの作成
        elif embedding_model_name in ['brainbert', 'brainbert2.0', 'brainlm2.0']:
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            from brainbert_pretrain_model import BrainBertModel
            #         # brainBERT
            if embedding_model_name == 'brainbert':  # alice only
                brainBERT_checkpoint = '/home/ying/project/brainLM_SA/output/ckpt/alice_ae_0.17647058823529413_6.419957104851218.pt'
            if embedding_model_name == 'brainlm2.0':  # BrainLM
                # 结果无效
                brainBERT_checkpoint = '/home/ying/project/brainLM_SA/output/ckpt/alice_ae_0,1,2,3_20.79_2530.pt'
            # if embedding_model_name == 'brainbert2.0':  # BrainLM
            #     brainBERT_checkpoint = '/Storage2/ying/pyCortexProj/ridgeRegression/models/alice_ae_100.0_286230.pt'

            checkpoint = {k.replace('module.brainbert.', ''): v for k, v in torch.load(brainBERT_checkpoint).items()}
            model_config = '/Storage2/ying/project/brainBERT/config/brainbert-large.json'
            # checkpoint = {k.replace('module.', ''): v for k, v in torch.load(os.path.join(checkpoint)).items()}
            model = BrainBertModel.from_pretrained(model_config, checkpoint)
        elif embedding_model_name.lower() == "glove":
            word2vec_output_file = '/Storage2/ying/pyCortexProj/ridgeRegression/models/glove.42B.300d' + '.word2vec'
            glove2word2vec("/Storage2/ying/pyCortexProj/ridgeRegression/models/glove.42B.300d.txt",
                           word2vec_output_file)
            # load the Stanford GloVe model
            model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        # k = get_glove_embedding(text, model)
        # TODO
        elif embedding_model_name == 'word2vec':
            model = Sentence2Vec('/Storage2/ying/pyCortexProj/ridgeRegression/models/word2vec.model')
        elif embedding_model_name == 'llama2':
            model_id = "/home/ying/llama/models_hf/7B"
            #
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, torch_dtype="auto",
                                                         device_map="auto")

        elif embedding_model_name == 't5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            model_name = "t5-base"  # 或 "t5-base", "t5-large", "google/t5-large-ssm"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)


        elif embedding_model_name == 't5-l':
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            model_name = "t5-large"  # 或 "t5-base", "t5-large", "google/t5-large-ssm"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)



        elif embedding_model_name == 'bart':
            # 加载模型和分词器
            from transformers import BartTokenizer, BartForConditionalGeneration

            model_name = "facebook/bart-large"
            # model_id = 'bart'
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)


        elif embedding_model_name == 'XLM':

            # 选择所需的预训练模型，例如'xlm-mlm-en-2048'
            model_name = 'xlm-mlm-en-2048'
            model = XLMModel.from_pretrained(model_name)
            tokenizer = XLMTokenizer.from_pretrained(model_name)
            # model.eval()

        model.eval()
    # sentence_list = []
    for text_db, brain_db in zip(text_dbs, brain_dbs):
        brain_env = lmdb.open(brain_db, readonly=True, create=False, lock=False)
        brain_txn = brain_env.begin(buffers=True)
        text_env = lmdb.open(text_db, readonly=True, create=False, lock=False)
        text_txn = text_env.begin(buffers=True)
        text_cursor = text_txn.cursor()

        for index, (k, v) in enumerate(tqdm(text_cursor, desc="iter")):  # 遍历sentence
            value_ = msgpack.loads(decompress(text_txn.get(k)), raw=False)  # 获取lmdb数据
            # print(value_['img_fname'])
            sentence = value_['sentence']  # 定位当前句子
            if datasetname in ['COCO2014', 'COCO2014_2023']:
                if datasetname == 'COCO2014_2023':
                    filename = value_['img_fname']

                    brain_value = msgpack.loads(decompress(brain_txn.get(filename.encode('utf-8'))), raw=False)
                    Y = np.append(Y, [brain_value['norm_bb']], axis=0)
                else:
                    brain_value = msgpack.loads(decompress(brain_txn.get(value_['img_fname'].encode('utf-8'))),
                                                raw=False)
                    Y = np.append(Y, [brain_value['norm_bb']], axis=0)

            else:
                brain_value = msgpack.loads(decompress(brain_txn.get(value_['img_fname'].encode('utf-8'))),
                                            raw=False)
                Y = np.append(Y, [brain_value['norm_bb']['data']], axis=0)
            if sentence in sen_embed_dict.keys():  # 如果数据embedding已经保存在文件里
                X = np.append(X, [sen_embed_dict.get(sentence)], axis=0)
            else:
                if embedding_model_name in \
                        ['bert-base-uncased', 'bert-large-uncased', 'bert-base-multilingual-cased',
                         'bert-large-uncased-whole-word-masking']:
                    # # 获取BERT的embedding
                    embedding_list = get_bert_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name == 'roberta-large' or embedding_model_name == 'roberta-base':
                    embedding_list = get_roberta_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name.lower() == "glove":
                    embedding_list = get_glove_embedding(sentence, model).tolist()
                elif embedding_model_name == 'word2vec':
                    embedding_list = get_w2v_embedding(sentence, model).tolist()

                    # model = Sentence2Vec('./models/word2vec.model')
                    # k = get_w2v_embedding(text, model)
                elif embedding_model_name in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                                              'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2',
                                              'albert-xlarge-v2',
                                              'albert-xxlarge-v2']:
                    embedding_list = get_albert_embedding_tensor(sentence, tokenizer, model).tolist()
                    # k = get_albert_embedding_tensor(text, tokenizer, model)

                elif embedding_model_name in ['sup-simcse-bert-base-uncased', 'unsup-simcse-roberta-large',
                                              'sup-simcse-roberta-large']:
                    embedding_list = model.encode(sentence, return_numpy=True, device='cpu', normalize_to_unit=True,
                                                  keepdim=False, batch_size=64,
                                                  max_length=128).tolist()


                elif embedding_model_name == 'sentence-camembert-large':
                    embedding_list = model.encode(sentence).tolist()

                elif embedding_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    # tokenizer = GPT2Tokenizer.from_pretrained(embedding_model_name)
                    # model = GPT2Model.from_pretrained(embedding_model_name)
                    embedding_list = get_gpt_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name in ['brainbert', 'brainbert2.0', 'brainlm2.0']:
                    # 获取brainBERT的embedding
                    sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
                        sentence=sentence, img_feat=None, tokenizer=tokenizer, model=model)
                    embedding_list = pooled_output.detach().squeeze(0).tolist()
                # TODO
                elif embedding_model_name == 'llama2':
                    # 获取brainBERT的embedding
                    embedding_list = get_llama_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name in ['t5', 't5-l']:
                    # 获取brainBERT的embedding
                    embedding_list = get_t5_embedding(sentence, tokenizer, model).tolist()
                elif embedding_model_name == 'bart':
                    # 获取brainBERT的embedding
                    embedding_list = get_bart_embedding(sentence, tokenizer, model).tolist()
                elif embedding_model_name == 'XLM':
                    # 获取brainBERT的embedding
                    embedding_list = get_xlm_embedding(sentence, tokenizer, model).tolist()

                X = np.append(X, [embedding_list], axis=0)
                sen_embed_dict[sentence] = embedding_list
            if _type == 'test_' and X.shape[0] > 1:
                break
        else:
            brain_env.close()
            text_env.close()
            # break
            continue
    if not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
            json.dump(sen_embed_dict, f)
    #         TypeError: Object of type ndarray is not JSON serializable
    X, _, _ = normalization(X)
    Y, _, _ = normalization(Y)

    return X, Y


def preprocess_str_to_num(param):
    preprocessed = []
    _list = param.split(" ")
    for i in _list:
        if 'e' in i:
            preprocessed.append(eval(i.replace(']', '').replace('[', '')))
    return preprocessed


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

    df = pd.read_csv('/Storage2/ying/pyCortexProj/resource/littlePrince/sub_EN057/lppEN_word_information_global.csv',
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


def createDataSet_prince(embedding_model_name, type, method):
    # datasetname = 'prince'
    subj = 'sub_EN057'  # sub_EN0109
    load_dir = "/Storage2/ying/pyCortexProj/"
    # /Storage2/ying/pyCortexProj/resource
    # save_dir = "/home/ying/project/pyCortexProj/"
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
    brain_echo_1_path = load_dir + 'resource/littlePrince/' + subj + '/echo-1-cortex.h5'
    brain_echo_2_path = load_dir + 'resource/littlePrince/' + subj + '/echo-2-cortex.h5'
    brain_echo_3_path = load_dir + 'resource/littlePrince/' + subj + '/echo-3-cortex.h5'
    # if embedding_model_name != 'brainbert':
    #     word_embedding_path = load_dir +'resource/littlePrince/' + subj + '/' + embedding_model_name + '_word_embedding_whole_words.h5'
    # else:
    word_embedding_path = load_dir + 'resource/littlePrince/' + subj + '/' + embedding_model_name + '_word_embedding_whole_words.h5'
    #
    if type != 'pred':
        # 获取脑数据 echo-1
        with h5py.File(brain_echo_1_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        if embedding_model_name == 'brainbert':
            text_embedding1, _, _ = normalization(np.array(loaded_json['brainlm']))
        else:
            text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))
        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        with h5py.File(brain_echo_2_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = np.vstack((brain_cortex, dataset[:]))
            file.close()

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        train_Y = brain_cortex[:int(0.8 * brain_cortex.shape[0]), :]
        train_X = X[:int(0.8 * X.shape[0]), :]
        test_Y = brain_cortex[int(0.8 * X.shape[0]):, :]
        test_X = X[int(0.8 * X.shape[0]):, :]
        train_Y, _, _ = normalization(train_Y)
        train_X, _, _ = normalization(train_X)
        test_Y, _, _ = normalization(test_Y)
        test_X, _, _ = normalization(test_X)
        print("train_X shape:", train_X.shape)
        print("train_Y shape:", train_Y.shape)
        print("test_X shape:", test_X.shape)
        print("test_Y shape:", test_Y.shape)

        return train_Y, train_X, test_Y, test_X

    else:
        # 获取脑数据 echo-3 for test
        with h5py.File(brain_echo_3_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(
                load_dir + 'resource/littlePrince/' + subj + '/' + embedding_model_name + '_word_embedding_whole_words.h5',
                'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        # text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))
        if embedding_model_name == 'brainbert':
            text_embedding1, _, _ = normalization(np.array(loaded_json['brainlm']))
        else:
            text_embedding1, _, _ = normalization(np.array(loaded_json[embedding_model_name]))

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)
        X = np.append(X, text_embedding, axis=0)

        # VAL_Y = brain_cortex[:int(0.2 * brain_cortex.shape[0]), :]
        # VAL_Y, _, _ = normalization(VAL_Y)
        # VAL_X = X[:int(0.2 * X.shape[0]), :]
        # VAL_X, _, _ = normalization(VAL_X)
        # 获取数组的第一维度大小
        size_first_dim = brain_cortex.shape[0]
        # 计算需要提取的数据量
        num_samples = int(size_first_dim * 0.1)
        # 随机选择10%的索引
        selected_indices = np.random.choice(size_first_dim, num_samples, replace=False)
        # 根据选定的索引提取数据
        VAL_Y = brain_cortex[selected_indices, :]
        VAL_X = X[selected_indices, :]
        VAL_Y, _, _ = normalization(VAL_Y)
        VAL_X, _, _ = normalization(VAL_X)

        return VAL_Y, VAL_X


def createDataSet_prince_layer(layer, subj, type, language_type):
    #     prep = 'exp3_layer_'
    # subj = 'sub_FR025' #FR028
    load_dir = "/Storage2/ying/pyCortexProj/"

    H_DIM = 1024

    X = np.empty((0, H_DIM), float)
    brain_echo_1_path = load_dir + 'resource/littlePrince/' + subj + '/echo-1-cortex.h5'
    brain_echo_2_path = load_dir + 'resource/littlePrince/' + subj + '/echo-2-cortex.h5'
    brain_echo_3_path = load_dir + 'resource/littlePrince/' + subj + '/echo-3-cortex.h5'
    if language_type == 'EN':
        word_embedding_path = '/home/ying/project/pyCortexProj/resource/littlePrince/' + subj + '/exp3_' + language_type + '_brainlm_word_embedding_whole_words' + str(
            layer) + '.h5'
    else:
        word_embedding_path = '/home/ying/project/pyCortexProj/resource/littlePrince/' + subj + '/exp2_5095_brainlm_word_embedding_whole_words' + str(
            layer) + '.h5'

    if type != 'pred':
        # 获取脑数据 echo-1
        with h5py.File(brain_echo_1_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        text_embedding1 = (np.array(loaded_json[str(layer)]))
        text_embedding = annotation_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        with h5py.File(brain_echo_2_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = np.vstack((brain_cortex, dataset[:]))
            file.close()

        text_embedding = annotation_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        train_Y = brain_cortex[:int(0.8 * brain_cortex.shape[0]), :]
        train_X = X[:int(0.8 * X.shape[0]), :]
        test_Y = brain_cortex[int(0.8 * X.shape[0]):, :]
        test_X = X[int(0.8 * X.shape[0]):, :]
        train_Y, _, _ = normalization(train_Y)
        train_X, _, _ = normalization(train_X)
        test_Y, _, _ = normalization(test_Y)
        test_X, _, _ = normalization(test_X)
        print("train_X shape:", train_X.shape)
        print("train_Y shape:", train_Y.shape)
        print("test_X shape:", test_X.shape)
        print("test_Y shape:", test_Y.shape)
        return train_Y, train_X, test_Y, test_X

    else:
        # 获取脑数据 echo-3 for test
        with h5py.File(brain_echo_3_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            #  brainlm_word_embedding_whole_words_before.h5
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        text_embedding2 = (np.array(loaded_json[str(layer)]))
        text_embedding = annotation_word_embedding(text_embedding2, timepoint)
        X = np.append(X, text_embedding, axis=0)

        # 获取数组的第一维度大小
        size_first_dim = brain_cortex.shape[0]
        # 计算需要提取的数据量
        num_samples = int(size_first_dim * 0.1)
        # 随机选择10%的索引
        selected_indices = np.random.choice(size_first_dim, num_samples, replace=False)
        # 根据选定的索引提取数据
        VAL_Y = brain_cortex[selected_indices, :]
        VAL_X = X[selected_indices, :]
        VAL_Y, _, _ = normalization(VAL_Y)
        VAL_X, _, _ = normalization(VAL_X)
        print("VAL_X shape:", VAL_X.shape)
        print("VAL_Y shape:", VAL_Y.shape)

        return VAL_Y, VAL_X


def createDataSet_prince_FR(subj, prep, embedding_model_name, type, method, _key):
    # datasetname = 'prince'
    # subj = 'sub_FR025' #FR028
    load_dir = "/Storage2/ying/pyCortexProj/"

    H_DIM = 1024
    if embedding_model_name in ['XLM-RoBERTa', 'bert-base-uncased', 'bert-base-multilingual-cased']:
        H_DIM = 768
    elif embedding_model_name in ['mBERT', 'albert-xlarge-v1', 'albert-xlarge-v2']:
        H_DIM = 2048
    X = np.empty((0, H_DIM), float)
    brain_echo_1_path = load_dir + 'resource/littlePrince/' + subj + '/echo-1-cortex.h5'
    brain_echo_2_path = load_dir + 'resource/littlePrince/' + subj + '/echo-2-cortex.h5'
    brain_echo_3_path = load_dir + 'resource/littlePrince/' + subj + '/echo-3-cortex.h5'
    # TODO word_embedding_path = load_dir + 'resource/littlePrince/'+subj+'/' + embedding_model_name+'_word_embedding_whole_words_'+_key+'.h5' #before after
    #
    # word_embedding_path = '/home/ying/project/pyCortexProj/resource/littlePrince/sub_FR025/5130_brainlm_word_embedding_whole_words_after.h5'
    word_embedding_path = '/Storage2/ying/pyCortexProj/resource/littlePrince/' + subj + '/' + prep + embedding_model_name + '_word_embedding_whole_words_' + _key + '.h5'

    #      测试阶段
    # if _key == 'before' or embedding_model_name != 'brainlm':
    #     word_embedding_path = load_dir + 'resource/littlePrince/'+subj+'/' + embedding_model_name+'_word_embedding_whole_words_'+_key+'.h5' #before after
    #
    # else:
    #
    #     word_embedding_path = load_dir + 'resource/littlePrince/'+subj+'/'+prep+embedding_model_name+'_word_embedding_whole_words_'+_key+'.h5' #before after
    # /home/ying/project/pyCortexProj/resource/littlePrince/sub_FR025/bert-base-multilingual-cased_word_embedding_whole_words_after.h5
    # "/Storage2/ying/pyCortexProj/"
    if type != 'pred':
        # 获取脑数据 echo-1
        with h5py.File(brain_echo_1_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        if embedding_model_name == 'brainbert':
            text_embedding1 = (np.array(loaded_json['brainlm']))
        else:
            text_embedding1 = (np.array(loaded_json[embedding_model_name]))
        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        with h5py.File(brain_echo_2_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = np.vstack((brain_cortex, dataset[:]))
            file.close()

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        train_Y = brain_cortex[:int(0.8 * brain_cortex.shape[0]), :]
        train_X = X[:int(0.8 * X.shape[0]), :]
        test_Y = brain_cortex[int(0.8 * X.shape[0]):, :]
        test_X = X[int(0.8 * X.shape[0]):, :]
        train_Y, _, _ = normalization(train_Y)
        train_X, _, _ = normalization(train_X)
        test_Y, _, _ = normalization(test_Y)
        test_X, _, _ = normalization(test_X)
        # train_Y, _, _ = (train_Y / np.linalg.norm(train_Y, axis=1)[:, np.newaxis])
        # train_X, _, _ = train_X / np.linalg.norm(train_X, axis=1)[:, np.newaxis]
        # test_Y, _, _ = test_Y / np.linalg.norm(test_Y, axis=1)[:, np.newaxis]
        # test_X, _, _ = test_X / np.linalg.norm(test_X, axis=1)[:, np.newaxis]

        return train_Y, train_X, test_Y, test_X

    else:
        # 获取脑数据 echo-3 for test
        with h5py.File(brain_echo_3_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        sentence_path = load_dir + 'resource/littlePrince/' + subj + '/' + prep + embedding_model_name + '_word_embedding_whole_words_' + _key + '.h5'
        with h5py.File(sentence_path, 'r') as hf:
            #  brainlm_word_embedding_whole_words_before.h5
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        if embedding_model_name == 'brainbert':
            text_embedding2 = (np.array(loaded_json['brainlm']))
        else:
            text_embedding2 = (np.array(loaded_json[embedding_model_name]))

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding2, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding2, timepoint)
        X = np.append(X, text_embedding, axis=0)

        # 获取数组的第一维度大小
        size_first_dim = brain_cortex.shape[0]
        # 计算需要提取的数据量
        num_samples = int(size_first_dim * 0.1)
        # 随机选择10%的索引
        selected_indices = np.random.choice(size_first_dim, num_samples, replace=False)
        # 根据选定的索引提取数据
        VAL_Y = brain_cortex[selected_indices, :]
        VAL_X = X[selected_indices, :]

        # VAL_Y = brain_cortex[:int(0.1 * brain_cortex.shape[0]), :]
        # VAL_X = X[:int(0.1 * X.shape[0]), :]
        VAL_Y, _, _ = normalization(VAL_Y)
        VAL_X, _, _ = normalization(VAL_X)

        return VAL_Y, VAL_X


def createDataSet_prince_ZH(subj, prep, embedding_model_name, type, method, _key):
    # datasetname = 'prince'
    # subj = 'sub_FR025' #FR028
    load_dir = "/Storage2/ying/pyCortexProj/"

    H_DIM = 1024
    if embedding_model_name in ['XLM-RoBERTa', 'bert-base-uncased', 'bert-base-multilingual-cased']:
        H_DIM = 768
    elif embedding_model_name in ['mBERT', 'albert-xlarge-v1', 'albert-xlarge-v2']:
        H_DIM = 2048
    X = np.empty((0, H_DIM), float)
    brain_echo_1_path = load_dir + 'resource/littlePrince/' + subj + '/echo-1-cortex.h5'
    brain_echo_2_path = load_dir + 'resource/littlePrince/' + subj + '/echo-2-cortex.h5'
    brain_echo_3_path = load_dir + 'resource/littlePrince/' + subj + '/echo-3-cortex.h5'
    # TODO word_embedding_path = load_dir + 'resource/littlePrince/'+subj+'/' + embedding_model_name+'_word_embedding_whole_words_'+_key+'.h5' #before after
    #
    # word_embedding_path = '/home/ying/project/pyCortexProj/resource/littlePrince/sub_FR025/5130_brainlm_word_embedding_whole_words_after.h5'
    word_embedding_path = '/home/ying/project/pyCortexProj/resource/littlePrince/' + subj + '/' + prep + 'brainlm_word_embedding_whole_words_' + _key + '.h5'

    if type != 'pred':
        # 获取脑数据 echo-1
        with h5py.File(brain_echo_1_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        with h5py.File(word_embedding_path, 'r') as hf:
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        # text_embedding1 = np.array(loaded_json[embedding_model_name])
        if embedding_model_name == 'brainbert':
            text_embedding1 = (np.array(loaded_json['brainlm']))
        else:
            text_embedding1 = (np.array(loaded_json[embedding_model_name]))
        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        with h5py.File(brain_echo_2_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = np.vstack((brain_cortex, dataset[:]))
            file.close()

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding1, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding1, timepoint)

        X = np.append(X, text_embedding, axis=0)

        train_Y = brain_cortex[:int(0.8 * brain_cortex.shape[0]), :]
        train_X = X[:int(0.8 * X.shape[0]), :]
        test_Y = brain_cortex[int(0.8 * X.shape[0]):, :]
        test_X = X[int(0.8 * X.shape[0]):, :]
        train_Y, _, _ = normalization(train_Y)
        train_X, _, _ = normalization(train_X)
        test_Y, _, _ = normalization(test_Y)
        test_X, _, _ = normalization(test_X)

        return train_Y, train_X, test_Y, test_X

    else:
        # 获取脑数据 echo-3 for test
        with h5py.File(brain_echo_3_path, 'r') as file:
            # 读取数据集
            dataset = file[subj]
            # 将数据集转换为NumPy数组
            # data = dataset[:]
            brain_cortex = dataset[:]
            timepoint = brain_cortex.shape[0]
            file.close()
        save_path = load_dir + 'resource/littlePrince/' + subj + '/' + prep + embedding_model_name + '_word_embedding_whole_words_' + _key + '.h5'
        # FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/Storage2/ying/pyCortexProj/resource/littlePrince/sub_CN003/brainlm_word_embedding_whole_words_after.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
        with h5py.File(save_path, 'r') as hf:
            #  brainlm_word_embedding_whole_words_before.h5
            # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
            loaded_json_str = hf[subj][()]
            loaded_json = json.loads(loaded_json_str)
            hf.close()
        if embedding_model_name == 'brainbert':
            text_embedding2 = (np.array(loaded_json['brainlm']))
        else:
            text_embedding2 = (np.array(loaded_json[embedding_model_name]))

        if method == "anno":
            text_embedding = annotation_word_embedding(text_embedding2, timepoint)
        else:
            text_embedding = average_word_embedding(text_embedding2, timepoint)
        X = np.append(X, text_embedding, axis=0)

        # 获取数组的第一维度大小
        size_first_dim = brain_cortex.shape[0]
        # 计算需要提取的数据量
        num_samples = int(size_first_dim * 0.1)
        # 随机选择10%的索引
        selected_indices = np.random.choice(size_first_dim, num_samples, replace=False)
        # 根据选定的索引提取数据
        VAL_Y = brain_cortex[selected_indices, :]
        VAL_X = X[selected_indices, :]

        # VAL_Y = brain_cortex[:int(0.1 * brain_cortex.shape[0]), :]
        # VAL_X = X[:int(0.1 * X.shape[0]), :]
        VAL_Y, _, _ = normalization(VAL_Y)
        VAL_X, _, _ = normalization(VAL_X)

        return VAL_Y, VAL_X


def get_glove_embedding(sentence, model):
    # We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format
    # that will be loaded in the next section
    tokenized_sent = word_tokenize(sentence.lower())

    embs = []
    for t in tokenized_sent:
        try:
            if t in ['waistcoat-']:
                t = 'waistcoat'
            if t == 'waistcoat-pocket':
                t = 'pocket'
            if t == 'cherry-tart':
                t = 'cherry'
            if t == 'sadnwiches':
                t = 'sandwiches'
            emb = model.get_vector(t)
            emb = emb.tolist()
        except Exception as e:
            emb = [0] * 300
            pass
        embs.append(emb)
        continue

    # a = np.array(embs).mean(axis=1)
    # b =  np.array(embs).mean(axis=0)
    # # Show a word embedding
    return np.array(embs).mean(axis=0)


# word2vec
def get_w2v_embedding(sentence, model):
    tokenized_sent = word_tokenize(sentence.lower())

    embs = []
    for t in tokenized_sent:
        emb = model.get_vector(t)
        # print(t)
        # print(emb) 存在问题：大量单词embedding=0
        embs.append(emb.tolist())
    # a = np.array(embs).mean(axis=1)
    # b =  np.array(embs).mean(axis=0)
    # Show a word embedding
    return np.array(embs).mean(axis=0)


def get_brain_bert_attention_output(img_feat, sentence, tokenizer, model):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'][0]
    input_id_list = input_ids.tolist()  # Batch index 0
    input_id_list.extend([104])  # Batch index 0
    input_ids = torch.Tensor(input_id_list).long()
    output = model(brain_feature=img_feat, input_ids=input_ids.unsqueeze(0))

    return output


def get_roberta_embedding_tensor(sentence, tokenizer, model):
    sentence = tokenizer.encode(sentence, padding=False, max_length=512, truncation=True, return_tensors='pt')

    output = model(sentence)

    return output[-1][-1][:, 0, :].detach().squeeze(0)


def get_bert_embedding_tensor(sentence, tokenizer, model):
    text_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True)
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

    res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # bert-base-multilingual-cased;bert-large-uncased-whole-word-masking
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # get cls's output
    k = res[-1].detach().squeeze(0)
    #     print(k.shape)
    return k


def get_albert_embedding_tensor(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output[-1].detach().squeeze(0)


def get_xlm_embedding(sentence, tokenizer, model):
    tokens = tokenizer(sentence, return_tensors="pt")
    outputs = model(**tokens)
    output = outputs.last_hidden_state.mean(dim=1)
    # print(sentence_embedding.shape)
    return output.detach().squeeze(0)


def get_gpt_embedding_tensor(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output[0][:, 0, :].detach().squeeze(0)


def get_llama_embedding_tensor(sentence, tokenizer, model):
    t_input = tokenizer(sentence, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        last_hidden_state = model(**t_input, output_hidden_states=True).hidden_states[-1]
    weights_for_non_padding = t_input.attention_mask * torch.arange(start=1,
                                                                    end=last_hidden_state.shape[
                                                                            1] + 1).unsqueeze(0).to("cuda")

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
    return sentence_embeddings.detach().squeeze(0)


def get_t5_embedding_tensor(sentence, tokenizer, model):
    t_input = tokenizer(sentence, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=t_input["input_ids"],
            attention_mask=t_input["attention_mask"],
            max_length=50,  # 可调整生成的最大长度
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.encoder_hidden_states[-1]  # T5的output
        # outputs = model(**t_input)
        # last_hidden_state = outputs.encoder_hidden_states[-1] #BART的参数名字和别的不一样
        # last_hidden_state = outputs.encoder_last_hidden_state #BART的参数名字和别的不一样

    # 计算权重并聚合嵌入
    weights_for_non_padding = t_input.attention_mask * torch.arange(
        1, last_hidden_state.shape[1] + 1).unsqueeze(0).to("cuda")
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings.detach().squeeze(0)


def get_bart_embedding_tensor(sentence, tokenizer, model):
    t_input = tokenizer(sentence, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**t_input)
        last_hidden_state = outputs.encoder_hidden_states[-1]  # BART的参数名字和别的不一样
        # last_hidden_state = outputs.encoder_last_hidden_state #BART的参数名字和别的不一样

    # 计算权重并聚合嵌入
    weights_for_non_padding = t_input.attention_mask * torch.arange(
        1, last_hidden_state.shape[1] + 1).unsqueeze(0).to("cuda")
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings.detach().squeeze(0)


def get_simcse_embedding_tensor(sentence, model):
    output = model.encode(sentence, return_numpy=True, device='cpu', normalize_to_unit=True,
                          keepdim=False, batch_size=64,
                          max_length=128)
    # output = model(**encoded_input)
    return output[0][:, 0, :].detach().squeeze(0)


if __name__ == "__main__":
    text = "Replace me by any text you'd like."
    model_type = "GloVe"
    k = []
    # Train_Y, Train_X, Test_Y, Test_X = createDataSet_prince(model_type)
    # k1 = get_bert_embedding_tensor('指的都是一個可以用來代表某詞彙。')
    # k2 = get_roberta_embedding_tensor('指的都是一個可以用來代表某詞彙。')
    # glove_input_file = glove_filename
    # if _type == "glove":
    #     word2vec_output_file = 'models/glove.42B.300d' + '.word2vec'
    #     glove2word2vec("models/glove.42B.300d.txt", word2vec_output_file)
    #     # load the Stanford GloVe model
    #     model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    #     # k = get_glove_embedding(text,model)
    #
    # elif _type == 'word2vec':
    #
    #     model = Sentence2Vec('./models/word2vec.model')
    #     k = get_w2v_embedding(text, model)
    #
    # elif _type in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
    #                'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2']:
    #     tokenizer = AlbertTokenizer.from_pretrained(_type)
    #     model = AlbertModel.from_pretrained(_type)
    #     k = get_albert_embedding_tensor(text, tokenizer, model)
    # elif _type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    #     tokenizer = GPT2Tokenizer.from_pretrained(_type)
    #     model = GPT2Model.from_pretrained(_type)
    #     k = get_gpt_embedding_tensor(text, tokenizer, model)
