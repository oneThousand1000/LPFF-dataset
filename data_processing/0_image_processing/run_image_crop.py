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

# 3 The 2400x2400 realigned images will be cropped according to EG3D function.
cmd = f"python 3_eg3d_crop.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f' Final EG3D cropped images will be saved to `{data_dir}/realign/crop/eg3d`.\n'
)
subprocess.run(cmd, shell=True, check=True)

#--------------------------------------------------------------------------------------------------------#

# 4 The 2400x2400 realigned images will be cropped according to StyleGAN function.
cmd = f"python 4_stylegan_crop.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f' Final StyleGAN cropped images will be saved to `{data_dir}/realign/crop/stylegan`.\n'
)
subprocess.run(cmd, shell=True, check=True)


#--------------------------------------------------------------------------------------------------------#

# 5 Process camera parameters.
cmd = f"python 5_process_camera_parameter.py --data_dir={data_dir}"
print(f'{cmd} \n'
      f'Final camera parameters will be saved to `{data_dir}/realign/camera_parameters`\n'
      f' (npy files) and `{data_dir}/realign/camera_parameters/camera_parameters.json` \n'
      f'(json file). \n'
)
subprocess.run(cmd, shell=True, check=True)


#--------------------------------------------------------------------------------------------------------#

