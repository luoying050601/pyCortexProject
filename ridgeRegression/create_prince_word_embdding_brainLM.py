# # /home/ying/project/littlePrinceDatasetProj/resources/annotation/EN/lppEN_word_embeddings_BERT.csv
# # useless
# import pandas as pd
# import torch
# from brainbert_pretrain_model import BrainBertModel
# from create_word_embedding import get_brain_bert_attention_output
# from transformers import BertTokenizer
# import os
#
# # 读取CSV文件
# df = pd.read_csv('/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_BERT.csv',
#                  index_col=0, usecols=[0, 1], header=0)
# # df = df['word']
# # brainBERT_checkpoint = '/home/ying/project/brainLM_SA/output/ckpt/alice_ae_100.0_286230.pt'
# brainBERT_checkpoint = '/Storage2/ying/resources/BrainBertTorch/output/ckpt/bert-large-uncased_5_55.56_205410.pt'
# # checkpoint = {k.replace('module.brainbert.', ''): v for k, v in torch.load(brainBERT_checkpoint).items()}
# model_config = '/Storage2/ying/project/brainBERT/config/brainbert-large.json'
# checkpoint = {k.replace('module.', ''): v for k, v in torch.load(os.path.join(brainBERT_checkpoint)).items()}
# model = BrainBertModel.from_pretrained(model_config, checkpoint)
# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
# # with open('/home/ying/project/littlePrinceDatasetProj/resources/annotation/EN/lppEN_word_embeddings_brainlm.csv', 'w', encoding='utf-8') as f:
# for index, row in df.iterrows():
#         print(index)  # 输出每行的索引值
#         word = row['word']
#         sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
#             sentence=word, img_feat=None, tokenizer=tokenizer, model=model)
#         embedding_list = pooled_output.detach().squeeze(0).tolist()
#         # print(sequence_output)
#         df.loc[index, "brainlm"] = pooled_output.reshape(-1).tolist()
#         for i in range(1, 25):
#             df.loc[index, str(i)] = hidden_states[i].squeeze(0)[1, :].reshape(-1).tolist()
# # df.drop()
# # df.to_csv('/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_brainlm.csv', sep='\t',index=False)
# #     input id : 101 ?? 104 102
# # average ：pooled_output each layer：hidden_state 1～24
import h5py
import os

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
print(Proj_dir)
import sys
from tqdm import tqdm

sys.path.append(Proj_dir)
brainlm_word_embedding = {}
import pandas as pd
import torch
from ridgeRegression.brainbert_pretrain_model import BrainBertModel
from ridgeRegression.create_word_embedding import get_brain_bert_attention_output
from transformers import BertTokenizer
import json

subj = "sub_EN057"
proj_dir = "/home/ying/project/pyCortexProj/"
# import numpy as np

# 读取CSV文件
df = pd.read_csv(proj_dir +
                 'ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_BERT.csv',
                 sep=',', index_col=0, header=0)
# df = df['word']
# /Storage2/ying/resources/BrainBertTorch/output/ckpt/bert-large-uncased_5_55.56_205410.pt
brainBERT_checkpoint = '/Storage2/ying/pyCortexProj/ridgeRegression/models/alice_ae_21.0_1610.pt'
checkpoint = {k.replace('module.brainbert.', ''): v for k, v in torch.load(brainBERT_checkpoint).items()}
model_config = '/Storage2/ying/pyCortexProj/ridgeRegression/models/brainbert-large.json'
model = BrainBertModel.from_pretrained(model_config, checkpoint)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
test_flag = True
brainlm = []

print("loading...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
# for index, row in df.iterrows():
    word = row['word']
    sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
        sentence=word, img_feat=None, tokenizer=tokenizer, model=model)
    embedding_list = pooled_output.detach().squeeze(0).tolist()
    brainlm.append(pooled_output.reshape(-1).tolist())


brainlm_word_embedding["brainlm"] = brainlm

print("saving...")
for k, v in tqdm(brainlm_word_embedding.items()):
    with h5py.File(proj_dir + 'resource/littlePrince/'+subj+'/brainLM21.0_word_embedding_whole_words_' + k + '.h5',
                   'w') as hf:
        # 将数据写入文件，可以是数组、数据集等
        hf.create_dataset(subj, data=json.dumps({k: v}))
        hf.close()

print("ending...")


