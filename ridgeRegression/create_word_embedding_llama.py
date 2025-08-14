import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # GPU
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
sen_embed_dict = {}
_type = "test"
model_name = 'llama2'
test_sentence_dict = json.load(open("/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/COCO2014_2023/albert-base-v1_text_embedding_"+_type+".json", 'r'))
model_id = "/home/ying/llama/models_hf/7B"#
tokenizer = AutoTokenizer.from_pretrained("/home/ying/llama/models_hf/7B", local_files_only=True,
                                          torch_dtype="auto", device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gpus = range(0, n_gpu)

model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
# model = model.cuda()
model.eval()


texts_all = list(test_sentence_dict.keys())


# texts_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
n = 5
for texts in [texts_all[i:i + n] for i in range(0, len(texts_all), n)]:

    t_input = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        last_hidden_state = model(**t_input, output_hidden_states=True).hidden_states[-1]

    weights_for_non_padding = t_input.attention_mask * torch.arange(start=1,
                                                                    end=last_hidden_state.shape[1] + 1).unsqueeze(0).to(
        "cuda")

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
    for i in range(len(texts)):
        print(texts[i])
        sen_embed_dict[texts[i]] = sentence_embeddings[i, :].detach().squeeze(0).tolist()

    # print(t_input.input_ids)
    # print(weights_for_non_padding)
    # print(num_of_none_padding_tokens)
    # print(sentence_embeddings.shape)
    
    # print(b)
with open("/Storage2/ying/pyCortexProj/ridgeRegression/text_embedding/COCO2014_2023/"+model_name+"_text_embedding_"+_type+".json", 'w') as f:
    json.dump(sen_embed_dict, f)