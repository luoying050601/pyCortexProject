import os
import h5py
import pandas as pd
import json

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
print(Proj_dir)
import sys

test_flag = False
sys.path.append(Proj_dir)
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AlbertTokenizer, AlbertModel
from ridgeRegression.create_word_embedding import get_gpt_embedding_tensor, get_albert_embedding_tensor

proj_dir = "/Storage2/ying/pyCortexProj/"


def is_float(str):
    try:
        float_value = float(str)
        return True
    except ValueError:
        return False


def preprocess_str_to_num(param):
    preprocessed = []
    _list = param.split(" ")
    for i in _list:
        if '0' in i or 'e' in i or is_float(i.replace(']', '').replace('[', '')):
            preprocessed.append(eval(i.replace(']', '').replace('[', '')))
    print(len(preprocessed))
    return preprocessed


if __name__ == '__main__':
    word_embedding = {}
    embedding_model_list = ['albert-xlarge-v1',
                            'albert-xlarge-v2']  # GloVe GPT2 BERT  'albert-xlarge-v1','albert-xlarge-v2'
    for embedding_model_name in embedding_model_list:
        print(embedding_model_name)
        # 读取CSV文件 BERT gpt2         #     TODO 这里加模型
        if embedding_model_name in ["BERT", 'gpt2', 'albert-xlarge-v1', 'albert-xlarge-v2']:
            df = pd.read_csv(proj_dir +
                             'ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_BERT.csv',
                             sep='\t', index_col=0, header=0)
            if embedding_model_name == 'gpt2':
                tokenizer = GPT2Tokenizer.from_pretrained(embedding_model_name)
                model = GPT2Model.from_pretrained(embedding_model_name)
            if embedding_model_name in ['albert-xlarge-v1', 'albert-xlarge-v2']:
                tokenizer = AlbertTokenizer.from_pretrained(embedding_model_name)
                model = AlbertModel.from_pretrained(embedding_model_name)
            model.eval()

        # Glove
        else:
            df = pd.read_csv(proj_dir +
                             'ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_GloVe.csv',
                             sep=',', index_col=0, header=0)
        embedding = []

        for index, row in df.iterrows():
            print(index)  # 输出每行的索引值
            # Glove BERT
            if embedding_model_name in ["BERT", 'GloVe']:  # 有现成的
                embedding.append(preprocess_str_to_num(row[embedding_model_name]))
            # gpt2
            elif embedding_model_name == 'gpt2':
                word = row['word']
                embedding.append(get_gpt_embedding_tensor(word, tokenizer, model).tolist())
            #     TODO 这里加模型
            elif embedding_model_name in ['albert-xlarge-v1', 'albert-xlarge-v2']:  # 需要自己生成
                word = row['word']
                embedding.append(get_albert_embedding_tensor(word, tokenizer, model).tolist())
            if test_flag:
                break

        word_embedding[embedding_model_name] = embedding
        print("saving...")

        with h5py.File(
                proj_dir + 'resource/littlePrince/sub_EN057/' + embedding_model_name + '_word_embedding_whole_words.h5',
                'w') as hf:
            # 将数据写入文件，可以是数组、数据集等
            hf.create_dataset('sub_EN057', data=json.dumps(word_embedding))
            hf.close()

        print("ending...")
