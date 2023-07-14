import os
import argparse
import shutil
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()
data_dir = args.data_dir
#--------------------------------------------------------------------------------------------------------#

# 0 Extract landmarks
cmd = f"python 0_get_landmarks.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f' Extracting landmarks from raw images using face-alignment or dlib.\n'
      f' The predicted landmarks will be saved to `{data_dir}/lm_facealignment_dlib`,\n'
      f' the images that resist face alignment detection and dlib detection\n'
      f' will be moved to `data_dir/bad`\n'
)
subprocess.run(cmd, shell=True, check=True)

#--------------------------------------------------------------------------------------------------------#

# 1 Realign images to 2400x2400
cmd = f"python 1_realign.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f' Aligning images to 2400x2400 using the predicted landmarks. \n'
      f' The realigned image  will be saved to `{data_dir}/realign`.        \n'
)
subprocess.run(cmd, shell=True, check=True)


#--------------------------------------------------------------------------------------------------------#

# 2 Predict camera parameter, compute density, save the landmarks predicted by Deep3DFaceRecon_pytorch.
cmd = f"python 2_predict_camera_parameter.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f'Extracting camera parameters using Deep3DFaceRecon_pytorch \n'
      f'In this step, only images with camera density<0.4 will be reserved. \n'
      f'The landmarks predicted by Deep3DFaceRecon_pytorch to \n'
      f'`{data_dir}/realign/lm_5p_pred` and will be used to crop images.\n'
)
subprocess.run(cmd, shell=True, check=True)


#--------------------------------------------------------------------------------------------------------#

