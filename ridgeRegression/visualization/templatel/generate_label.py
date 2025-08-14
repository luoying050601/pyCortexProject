# import cortex
import time
import os
# import sys
import cortex
import nibabel as nib
from PIL import Image
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


freesurfer_subject_dir = "/Storage2/brain_group/freesurfer/subjects"
freesurfer_subject_name = "S3002"

Volshape = (72, 96, 96)

# def cortex2Vol(cortex, tvoxels):
#     Vol = np.zeros(np.prod(Volshape))  # 72*96*96の一次元配列
#     Vol[np.array(tvoxels).reshape(-1)] = cortex
#     Vol = np.reshape(Vol, Volshape)
#     return (Vol)
# proj = set()
# tvoxels = mne.read_label(os.path.join(freesurfer_subject_dir, freesurfer_subject_name, 'label', "lh.cortex.label")).vertices
# (64463, 1)
# project_cortex = np.random.rand(tvoxels.shape[0])
surfs = [cortex.polyutils.Surface(*d)
         for d in cortex.db.get_surf(freesurfer_subject_name, "fiducial")]  # flat
num_verts = surfs[0].pts.shape[0] + surfs[1].pts.shape[0]
cortex_data = np.random.rand(num_verts)
min_val = 0
max_val = max(cortex_data)
dv = cortex.Vertex(cortex_data, subject=freesurfer_subject_name, cmap="hot", vmin=min_val,
                   vmax=max_val)
_ = cortex.quickflat.make_figure(dv, with_colorbar=False)

# Traceback (most recent call last):
#   File "/Storage2/ying/pyCortexProj/ridgeRegression/visualization/templatel/generate_label.py", line 38, in <module>
#     vmax=max_val)  # Vertex
#   File "/usr/local/lib/python3.6/dist-packages/cortex/dataset/views.py", line 273, in __init__
#     description=description, **kwargs)
#   File "/usr/local/lib/python3.6/dist-packages/cortex/dataset/braindata.py", line 143, in __init__
#     self._check_size(mask)
#   File "/usr/local/lib/python3.6/dist-packages/cortex/dataset/braindata.py", line 223, in _check_size
#     self._mask, self.mask = _find_mask(nvox, self.subject, self.xfmname)
#   File "/usr/local/lib/python3.6/dist-packages/cortex/dataset/braindata.py", line 556, in _find_mask
#     raise ValueError('Cannot find a valid mask')
# ValueError: Cannot find a valid mask
# mask_([\w]+).nii.gz
plt.axis("off")
timestamps = time.time()
plt.savefig('/Storage2/ying/pyCortexProj/ridgeRegression/visualization/' + str(
        timestamps) + '.png')
plt.show()
plt.close()
