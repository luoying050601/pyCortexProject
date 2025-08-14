import json
import scipy.io as scio
import numpy as np

user = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
roi_area = ['LHPPA', 'RHPPA',
            'RHLOC', 'LHLOC',
            'LHEarlyVis', 'RHEarlyVis',
            "LHRSC", 'RHRSC',
            "LHOPA", 'RHOPA']

roi_index = json.load(open(f'/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/coco_dict_index.json', 'r'))
# roi_index = json.load(
#     open(f'/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/caption/imageScene_dict_index.json', 'r'))

ROI_data = {}
for file in user:
    # r_array = [][]
    fileDir = "/Storage/ying/pyCortexProj/ridgeRegression/BOLD5000/ROIs/" + file + '/mat/' + file + '_ROIs_TR34.mat'
    # F3_Dir = "/Storage/ying/pyCortexProj/ridgeRegression/BOLD5000/ROIs/" + file + '/mat/' + file + '_ROIs_TR3.mat'
    # F3_data = scio.loadmat(F3_Dir)  # print(fileDir)
    data = scio.loadmat(fileDir)
    PPA = np.concatenate((data['LHPPA'], data['RHPPA'],
                          np.zeros((data['LHPPA'].shape[0], 370 - data['LHPPA'].shape[1] - data['RHPPA'].shape[1]))),
                         axis=1)
    # np.concatenate((data['LHLOC'], data['RHLOC'],np.zeros((data['LHLOC'].shape[0],1027-data['RHLOC'].shape[1]-data['LHLOC'].shape[1]))), axis=1)
    OPA = np.concatenate((data['LHOPA'], data['RHOPA'],
                          np.zeros((data['LHOPA'].shape[0], 614 - data['RHOPA'].shape[1] - data['LHOPA'].shape[1]))),
                         axis=1)
    EarlyVis = np.concatenate((data['LHEarlyVis'], data['RHEarlyVis'], np.zeros(
        (data['LHEarlyVis'].shape[0], 1218 - data['RHEarlyVis'].shape[1] - data['LHEarlyVis'].shape[1]))), axis=1)
    RSC = np.concatenate((data['LHRSC'], data['RHRSC'],
                          np.zeros((data['LHRSC'].shape[0], 337 - data['RHRSC'].shape[1] - data['LHRSC'].shape[1]))),
                         axis=1)
    LOC = np.concatenate((data['LHLOC'], data['RHLOC'],
                          np.zeros((data['LHLOC'].shape[0], 1027 - data['RHLOC'].shape[1] - data['LHLOC'].shape[1]))),
                         axis=1)

    roi = np.concatenate((PPA, OPA, EarlyVis, RSC, LOC), axis=1)
    ROI_data[file] = roi
    # roi2 =  np.concatenate(PPA,OPA,EarlyVis,RSC,LOC), axis=1)
    # for roi in roi_area:
    #     r_array = r_array + data[roi]
    # print(roi)
coco_data = {}
# is_data = {}
for k, v in ROI_data.items():
    index = roi_index[k]
    coco_value = []
    # is_value = []
    for i in index:
        # is_value.append(v[i])
        coco_value.append(v[i, :])
    print(np.array(coco_value).shape)
    coco_data[k] = np.array(coco_value).tolist()
    # is_data[k] = np.array(is_value).tolist()
    # 【当前run ImageScene图片数量，当前user的ROI合计】
    # (3119, 1685)
    # (3119, 2270)
    # (3121, 3104)
    # (1834, 2787)
    # 【当前runcoco图片数量，当前user的ROI合计】
    # (2135, 1685)
    # (2135, 2270)
    # (2133, 3104)
    # (1274, 2787)
    # print(coco_data)

json_str = json.dumps(coco_data)
# json_str = json.dumps(is_data)
# with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/is_brain_ROI.json', 'w') as json_file:
with open('/Storage/ying/project/ridgeRegression/BOLD5000/ROIs/coco_brain_ROI.json', 'w') as json_file:
    json_file.write(json_str)
    json_file.close()
