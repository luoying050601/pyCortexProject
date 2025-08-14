#无效 /home/ying/project/littlePrinceDatasetProj/resources/annotation/EN/lppEN_word_embeddings_BERT.csv
import pandas as pd
import torch
from transformers import BertTokenizer
from transformers import GPT2Tokenizer, GPT2Model
from create_word_embedding import get_gpt_embedding_tensor

# 读取CSV文件
df = pd.read_csv('/home/ying/project/littlePrinceDatasetProj/resources/annotation/EN/lppEN_word_embeddings_BERT.csv',
                 index_col=0, usecols=[0, 1], header=0)
embedding_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(embedding_model_name)
model = GPT2Model.from_pretrained(embedding_model_name)
for index, row in df.iterrows():
        print(index)  # 输出每行的索引值
        word = row['word']
        # sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
        #     sentence=word, img_feat=None, tokenizer=tokenizer, model=model)
        # embedding_list = pooled_output.detach().squeeze(0).tolist()
        embedding_list = get_gpt_embedding_tensor(word, tokenizer, model).tolist()

        # print(sequence_output)
        row["gpt2"] = embedding_list
        # for i in range(1, 25):
        #     row[str(i)] = hidden_states[i].squeeze(0)[1, :].reshape(-1).tolist()
# df.drop()
df.to_csv('/home/ying/project/littlePrinceDatasetProj/resources/annotation/EN/lppEN_word_embeddings_gpt2.csv', sep='\t',index=False)
#     input id : 101 ?? 104 102
# average ：pooled_output each layer：hidden_state 1～24
