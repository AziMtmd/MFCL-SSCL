#script1.sh
clear
echo "Starting shell script .."
pip install tf_slim
#pip install tensorflow
#pip install tensorflow-datasets
#pip install tfds-nightly
python3 run.py
import os
folder_path = '/azi'
for root, dirs, files in os.walk(folder_path):
  for file in files:
    if file.startswith('ckpt'):
      os.remove(os.path.join(root, file))
    if file.startswith('events'):
      os.remove(os.path.join(root, file))
    if file.startswith('checkpoint'):
      os.remove(os.path.join(root, file))
echo "Ending of the script";

cat run.sh
