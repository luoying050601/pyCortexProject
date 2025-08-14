# 提取用户大脑皮层的150个ROI对应的vertex index，默认顺序为aparc.a2009s的顺序，从左往右
import os
import json
from random import randint
import mne

# Start a virtual display for headless systems
display_num = randint(100, 200)
os.system(f"Xvfb :{display_num} -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = f":{display_num}"

# Specify subject and the path to the directory where the annotation file is stored
subject = 'sub_EN057'  # Replace with your subject ID
subjects_dir = "/Storage2/brain_group/freesurfer/subjects"  # Replace with the path to your subjects directory
parc = 'aparc.a2009s'  # Define the parcellation scheme


# Function to extract ROI labels and vertices from annotation
def extract_labels_and_vertices(subject, parc, hemi, subjects_dir):
    labels = mne.read_labels_from_annot(subject, parc=parc, hemi=hemi, subjects_dir=subjects_dir)
    return {label.name: label.vertices.tolist() for label in labels}


# Collect ROI index lists for both hemispheres
roi_index_list = {}
roi_index_list.update(extract_labels_and_vertices(subject, parc, 'lh', subjects_dir))
roi_index_list.update(extract_labels_and_vertices(subject, parc, 'rh', subjects_dir))

# Save to a JSON file
output_path = os.path.join("/home/ying/project/pyCortexProj/resource/littlePrince", subject, f"{subject}_roi_indices.json")
with open(output_path, 'w') as json_file:
    json.dump(roi_index_list, json_file, indent=4)

print(f"ROI indices saved to {output_path}")
