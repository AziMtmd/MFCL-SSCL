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
        if file.startswith('flags'):
            os.remove(os.path.join(root, file))            
        if file.startswith('result'):
            os.remove(os.path.join(root, file))  
# os.kill(os.getpid(), 9)

import shutil
import os

# Replace 'folder_path' with the path to the folder you want to delete
folder_path = '/content/tensor_data'

# Check if the folder exists
if os.path.exists(folder_path):
    # Delete the folder and its contents
    shutil.rmtree(folder_path)
    print(f"{folder_path} has been deleted.")
else:
    print(f"{folder_path} does not exist.")
