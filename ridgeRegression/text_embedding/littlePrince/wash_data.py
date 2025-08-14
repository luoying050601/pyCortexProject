import json
import h5py

# 从.h5文件中读取JSON对象
with h5py.File('/Storage2/ying/pyCortexProj/resource/littlePrince/sub_EN057/GloVe_word_embedding_whole_words.h5', 'r') as hf:
    # 从'hdf5'格式的数据集中读取字符串，并使用json.loads将其转换回JSON对象
    loaded_json_str = hf['sub_EN057'][()]
    loaded_json = json.loads(loaded_json_str)

# 打印加载后的JSON对象
print(loaded_json['GloVe'])