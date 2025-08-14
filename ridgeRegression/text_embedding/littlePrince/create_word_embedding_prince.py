import h5py
import os

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
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

proj_dir = "/home/ying/project/pyCortexProj/"
# import numpy as np

# 读取CSV文件
df = pd.read_csv(proj_dir +
                 'ridgeRegression/text_embedding/littlePrince/lppEN_word_embeddings_BERT.csv',
                 sep=',', index_col=0, header=0)
# df = df['word']
# /Storage2/ying/resources/BrainBertTorch/output/ckpt/bert-large-uncased_5_55.56_205410.pt
brainBERT_checkpoint = '/Storage2/ying/pyCortexProj/ridgeRegression/models/alice_ae_0.17674104124408385.pt'
checkpoint = {k.replace('module.brainbert.', ''): v for k, v in torch.load(brainBERT_checkpoint).items()}
model_config = '/Storage2/ying/pyCortexProj/ridgeRegression/models/brainbert-large.json'
model = BrainBertModel.from_pretrained(model_config, checkpoint)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
test_flag = True
brainlm = []
hidden = {}
hidden_1 = []
hidden_2 = []
hidden_3 = []
hidden_4 = []
hidden_5 = []
hidden_6 = []
hidden_7 = []
hidden_8 = []
hidden_9 = []
hidden_10 = []
hidden_11 = []
hidden_12 = []
hidden_13 = []
hidden_14 = []
hidden_15 = []
hidden_16 = []
hidden_17 = []
hidden_18 = []
hidden_19 = []
hidden_20 = []
hidden_21 = []
hidden_22 = []
hidden_23 = []
hidden_24 = []
print("loading...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
# for index, row in df.iterrows():
    word = row['word']
    sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
        sentence=word, img_feat=None, tokenizer=tokenizer, model=model)
    embedding_list = pooled_output.detach().squeeze(0).tolist()
    brainlm.append(pooled_output.reshape(-1).tolist())

    # if test_flag:
    #     break
    # hidden_1.append(hidden_states[1].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_2.append(hidden_states[2].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_3.append(hidden_states[3].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_4.append(hidden_states[4].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_5.append(hidden_states[5].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_6.append(hidden_states[6].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_7.append(hidden_states[7].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_8.append(hidden_states[8].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_9.append(hidden_states[9].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_10.append(hidden_states[10].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_11.append(hidden_states[11].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_12.append(hidden_states[12].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_13.append(hidden_states[13].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_14.append(hidden_states[14].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_15.append(hidden_states[15].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_16.append(hidden_states[16].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_17.append(hidden_states[17].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_18.append(hidden_states[18].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_19.append(hidden_states[19].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_20.append(hidden_states[20].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_21.append(hidden_states[21].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_22.append(hidden_states[22].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_23.append(hidden_states[23].squeeze(0)[1, :].reshape(-1).tolist())
    # hidden_24.append(hidden_states[24].squeeze(0)[1, :].reshape(-1).tolist())

brainlm_word_embedding["brainlm"] = brainlm
# brainlm_word_embedding["1"] = hidden_1
# brainlm_word_embedding["2"] = hidden_2
# brainlm_word_embedding["3"] = hidden_3
# brainlm_word_embedding["4"] = hidden_4
# brainlm_word_embedding["5"] = hidden_5
# brainlm_word_embedding["6"] = hidden_6
# brainlm_word_embedding["7"] = hidden_7
# brainlm_word_embedding["8"] = hidden_8
# brainlm_word_embedding["9"] = hidden_9
# brainlm_word_embedding["10"] = hidden_10
# brainlm_word_embedding["11"] = hidden_11
# brainlm_word_embedding["12"] = hidden_12
# brainlm_word_embedding["13"] = hidden_13
# brainlm_word_embedding["14"] = hidden_14
# brainlm_word_embedding["15"] = hidden_15
# brainlm_word_embedding["16"] = hidden_16
# brainlm_word_embedding["17"] = hidden_17
# brainlm_word_embedding["18"] = hidden_18
# brainlm_word_embedding["19"] = hidden_19
# brainlm_word_embedding["20"] = hidden_20
# brainlm_word_embedding["21"] = hidden_21
# brainlm_word_embedding["22"] = hidden_22
# brainlm_word_embedding["23"] = hidden_23
# brainlm_word_embedding["24"] = hidden_24
print("saving...")
for k, v in tqdm(brainlm_word_embedding.items()):
    with h5py.File(proj_dir + 'resource/littlePrince/sub_EN057/brainLM17.67_word_embedding_whole_words_' + k + '.h5',
                   'w') as hf:
        # 将数据写入文件，可以是数组、数据集等
        hf.create_dataset('sub_EN057', data=json.dumps({k: v}))
        hf.close()

print("ending...")

#   File "/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/littlePrince/create_word_embedding_prince.py", line 116, in <module>
#     hf.create_dataset('sub_EN057', data=json.dumps(brainlm_word_embedding))
#   File "/usr/local/lib/python3.6/dist-packages/h5py/_hl/group.py", line 136, in create_dataset
#     dsid = dataset.make_new_dset(self, shape, dtype, data, **kwds)
#   File "/usr/local/lib/python3.6/dist-packages/h5py/_hl/dataset.py", line 170, in make_new_dset
#     dset_id.write(h5s.ALL, h5s.ALL, data)
#   File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
#   File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
#   File "h5py/h5d.pyx", line 222, in h5py.h5d.DatasetID.write
#   File "h5py/_proxy.pyx", line 163, in h5py._proxy.dset_rw
#   File "h5py/defs.pyx", line 3536, in h5py.defs.H5Tconvert
#   File "h5py/_conv.pyx", line 452, in h5py._conv.str2vlen
#   File "h5py/_conv.pyx", line 116, in h5py._conv.generic_converter
#   File "h5py/_conv.pyx", line 262, in h5py._conv.conv_str2vlen
# ValueError: VLEN strings do not support embedded NULLs
