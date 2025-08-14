import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
print(Proj_dir)
import sys
sys.path.append(Proj_dir)
# 单词 embedding文件读取。
# /home/ying/project/princeProj/resources/annotation/EN/lppEN_word_embeddings_BERT.csv
# 通过ridge regression生成全脑（278244）的像素值。
# 通过mapper获取fMRI活动数据的对应时间的全脑（278244）的像素值
# 计算PC 值。
# 按照177 个annotation去计算每个区块的average，top的区域plot。
# /home/ying/project/princeProj/resources/sub-EN057/func/sub-EN057_task-lppEN_run-15_echo-1_bold.nii.gz
# (64, 64, 33, 286)
import cortex
import nibabel as nib
import h5py
# 注意最后都要迁移到storage2下
save_dir = "/home/ying/project/pyCortexProj/"
# freesurfer_subject_dir = "/home/ying/subjects"
freesurfer_subject_name = "sub_EN057" #sub_CN003 sub_FR025 sub_EN057
condition_str = "echo_" # sub_EN057为 echo_
print(freesurfer_subject_name)
# pycortex_subject_name = "sub_EN057"
xfm = "acl"
def traverse_files(directory, conditional):
    file_list = []
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if conditional in filename and ".json" not in filename:
                file_list.append(os.path.join(foldername, filename))
    return file_list
# cortex.align.automatic(freesurfer_subject_name, 'fullhead2', '/home/ying/project/princeProj/resources/sub-EN057/func/sub-EN057_task-lppEN_run-15_echo-1_bold.nii.gz')

path_to_traverse = '/Storage2/brain_group/dataset/littlePrince/'+freesurfer_subject_name+'/func/'
vol_to_ver_map = cortex.get_mapper(freesurfer_subject_name, xfm, 'line_nearest', recache=True)

print("echo 1 starting")
brain_cortex = []
for file_name in sorted(traverse_files(path_to_traverse, condition_str+"1")):
    print(file_name)
    img = nib.load(file_name)  # the data you want to plot
    volarray = img.get_fdata()  # e.g. (72,71,89) or (72,71,89,194)
    volarray = volarray.transpose(2, 1, 0, 3)  # 4D shape transpose

    for i in range(volarray.shape[3]):
        ver = vol_to_ver_map(cortex.Volume(volarray[:, :, :, i], freesurfer_subject_name, xfm))
        count_zero = sum(1 for element in ver.data if element == 0)
        if count_zero > 0:
            print("零元素的个数为:", count_zero)
        brain_cortex.append(ver.data)
with h5py.File(save_dir+'resource/littlePrince/'+freesurfer_subject_name+'/echo-1-cortex.h5', 'w') as hf:
    # 将数据写入文件，可以是数组、数据集等
    hf.create_dataset(freesurfer_subject_name, data=brain_cortex)
    hf.close()
print("finished")

print("echo 2 starting")
brain_cortex = []
for file_name in sorted(traverse_files(path_to_traverse,  condition_str+"2")):
    print(file_name)
    img = nib.load(file_name)  # the data you want to plot
    volarray = img.get_fdata()  # e.g. (72,71,89) or (72,71,89,194)
    volarray = volarray.transpose(2, 1, 0, 3)  # 4D shape transpose

    for i in range(volarray.shape[3]):
        ver = vol_to_ver_map(cortex.Volume(volarray[:, :, :, i], freesurfer_subject_name, xfm))
        count_zero = sum(1 for element in ver.data if element == 0)
        if count_zero > 0:
            print("零元素的个数为:", count_zero)
        brain_cortex.append(ver.data)

# print(np.array(brain_cortex).shape)
with h5py.File(save_dir+'resource/littlePrince/'+freesurfer_subject_name+'/echo-2-cortex.h5', 'w') as hf:
    # 将数据写入文件，可以是数组、数据集等
    hf.create_dataset(freesurfer_subject_name, data=brain_cortex)#
    hf.close()
print("finished")

print("echo 3 starting")
brain_cortex = []
for file_name in sorted(traverse_files(path_to_traverse,  condition_str+"3")):
    print(file_name)
    img = nib.load(file_name)  # the data you want to plot
    volarray = img.get_fdata()  # e.g. (72,71,89) or (72,71,89,194)
    volarray = volarray.transpose(2, 1, 0, 3)  # 4D shape transpose

    for i in range(volarray.shape[3]):
        ver = vol_to_ver_map(cortex.Volume(volarray[:, :, :, i], freesurfer_subject_name, xfm))
        count_zero = sum(1 for element in ver.data if element == 0)
        if count_zero > 0:
            print("零元素的个数为:", count_zero)
        brain_cortex.append(ver.data)
with h5py.File(save_dir+'resource/littlePrince/'+freesurfer_subject_name+'/echo-3-cortex.h5', 'w') as hf:
    # 将数据写入文件，可以是数组、数据集等
    hf.create_dataset(freesurfer_subject_name, data=brain_cortex)
    hf.close()
print("finished")
#
# file_reading_order = [15, 16, 17, 18, 19, 20, 21, 22, 23]
# file_echo = ['echo-1', 'echo-2', 'echo-3']
#  lmdb.open 数据写入

#
# # 指定路径

# 调用函数获取文件列表
# files = traverse_files(path_to_traverse)
# img = nib.load("/home/ying/project/pycortexProj/venv/share/pycortex/db/sub_EN057/transforms/fullhead2/reference.nii.gz")
# print(img.shape)
# 打印文件列表
# for file in files:
#     example_filename = os.path.join(path_to_traverse, file)
#     img = nib.load(example_filename)
#     print(example_filename)
#     print(img.shape)  # 输出头信息

# example_filename = '/home/ying/project/princeProj/resources/sub-EN057/func/sub-EN057_task-lppEN_run-15_echo-1_bold.nii.gz'
# img = nib.load(example_filename)
# print(example_filename)
# print(img.header['db_name'])  # 输出头信息




# 现在，'data'包含了从HDF5文件中读取的数据



# l_annot = nib.freesurfer.io.read_annot(freesurfer_subject_dir + f'/{freesurfer_subject_name}/label/lh.aparc.annot')
# r_annot = nib.freesurfer.io.read_annot(freesurfer_subject_dir + f'/{freesurfer_subject_name}/label/rh.aparc.annot')
#
# l_annot_proj = copy.deepcopy(l_annot)
# r_annot_proj = copy.deepcopy(r_annot)
# whole = np.concatenate([l_annot_proj[0], r_annot_proj[0]])
# reft = r_annot_proj[0]
# 同一个阶段的数据执行了2-3次。
# 我们可以取所有的echo-1作为rr的训练数据
# 2/3作为验证数据。
# 顺序是 15-23

# def find_index_list(X, id):
#     # l = len(X)
#     # zip_list = zip(*(range(l), X))
#     # id1 = [z[0] for i, z in enumerate(zip_list) if z[1] == 1]
#     # 或者更简单的
#     # id1 = [i for i, x in enumerate(X) if x == 1]
#     return [i for i, x in enumerate(X) if x == id]

# 保存annotation的index dictionary
# annot_dict = {}
# for i in range(0, 36):
#     annot_dict[l_annot[2][i].decode()] = find_index_list(whole, i)
#
# # print(annot_dict)
#
# with open("annot_dict_"+freesurfer_subject_name+".json", "w") as file:
#     json.dump(annot_dict, file)


# for j in range(len(l_annot[1])):  # left brain cortex
#     print(j, l_annot[2][j])
#     for i in range(len(l_annot[0])):
#         if l_annot[0][i] != int(j):
#             l_annot_proj[0][i] = 0
#     for i in range(len(r_annot[0])):
#         if r_annot[0][i] != int(j):
#             r_annot_proj[0][i] = 0
#     tem_l = np.nonzero(l_annot_proj[0])[0]
#     tem_r = np.nonzero(r_annot_proj[0])[0]
#     l_annot_proj[0][tem_l] = np.random.randint(20, size=len(tem_l))
#     r_annot_proj[0][tem_r] = np.random.randint(20, size=len(tem_r))
# min_val = 0
# max_val = max(np.concatenate([l_annot_proj[0], r_annot_proj[0]]))
# ver = cortex.Vertex(np.concatenate([l_annot_proj[0], r_annot_proj[0]]), freesurfer_subject_name, cmap="hot",vmin=min_val, vmax=max_val)
# _ = cortex.quickflat.make_figure(ver, with_colorbar=False)
