import tensorflow as tf
import numpy as np
from skimage import io
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import pearsonr

# Function to calculate mutual information and normalized mutual information for a single channel
def mutual_information(image1, image2, bins=256):
    # Flatten the images
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Calculate the joint histogram using NumPy
    hist_2d, x_edges, y_edges = np.histogram2d(image1_flat, image2_flat, bins=bins, range=[[0, 255], [0, 255]])
    
    # Convert the histogram to a TensorFlow tensor
    hist_2d = tf.convert_to_tensor(hist_2d, dtype=tf.float32)
    
    # Convert the histogram to probabilities
    joint_prob = hist_2d / tf.reduce_sum(hist_2d)
    
    # Calculate marginal probabilities
    x_marginal_prob = tf.reduce_sum(joint_prob, axis=1)
    y_marginal_prob = tf.reduce_sum(joint_prob, axis=0)
    
    # Calculate the mutual information using sklearn
    mi = mutual_info_score(None, None, contingency=hist_2d.numpy())
    
    # Calculate the normalized mutual information using sklearn
    nmi = normalized_mutual_info_score(image1_flat, image2_flat)
    
    return mi, nmi

# Function to calculate Pearson correlation for a single channel
def pearson_correlation(image1, image2):
    # Flatten the images
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # Calculate Pearson correlation using SciPy
    correlation, _ = pearsonr(image1_flat, image2_flat)
    
    return correlation

mis=0; nmis=0; pers=0
for i in range(10):
  for j in range(10):
    # Load the images using skimage
    image1 = io.imread('/content/nik'+str(j)+'.png')
    image2 = io.imread('/content/nikb'+str(i)+'.png')

    # Ensure the images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Split the images into R, G, B channels
    image1_r, image1_g, image1_b = image1[..., 0], image1[..., 1], image1[..., 2]
    image2_r, image2_g, image2_b = image2[..., 0], image2[..., 1], image2[..., 2]

    # Calculate metrics for each channel
    mi_r, nmi_r = mutual_information(image1_r, image2_r)
    mi_g, nmi_g = mutual_information(image1_g, image2_g)
    mi_b, nmi_b = mutual_information(image1_b, image2_b)

    pearson_r = pearson_correlation(image1_r, image2_r)
    pearson_g = pearson_correlation(image1_g, image2_g)
    pearson_b = pearson_correlation(image1_b, image2_b)

    # Average metrics across channels
    average_mi = (mi_r + mi_g + mi_b) / 3.0
    average_nmi = (nmi_r + nmi_g + nmi_b) / 3.0
    average_pearson = (pearson_r + pearson_g + pearson_b) / 3.0

    print(f"Mutual Information (R): {mi_r}")
    print(f"Mutual Information (G): {mi_g}")
    print(f"Mutual Information (B): {mi_b}")
    print(f"Average Mutual Information: {average_mi}")

    print(f"Normalized Mutual Information (R): {nmi_r}")
    print(f"Normalized Mutual Information (G): {nmi_g}")
    print(f"Normalized Mutual Information (B): {nmi_b}")
    print(f"Average Normalized Mutual Information: {average_nmi}")

    print(f"Pearson Correlation (R): {pearson_r}")
    print(f"Pearson Correlation (G): {pearson_g}")
    print(f"Pearson Correlation (B): {pearson_b}")
    print(f"Average Pearson Correlation: {average_pearson}")
    print('rond', i, j)
    mis=mis+average_mi
    nmis=nmis+average_nmi
    pers=pers+average_pearson
print('mis', mis)
print('nmis', nmis)
print('pers', pers)


from PIL import Image

# Define the coordinates for the cropping box
left = 1008
top = 1952
right = 1350
bottom = 2593
##4_31280nik
##5_31280nik
##6_31280nik
#2_19550nik9
# Open the image file
for j in range(2,7):
  for i in range(10):
    esm="/content/"+str(j)+"_7820nik"+str(i)+".png"
    image_path = esm
    img = Image.open(image_path)

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))

    # Save the cropped image
    output_path = esm
    img_cropped.save(output_path)

  print(f"Cropped image saved to {output_path}")




import matplotlib.pyplot as plt
import numpy as np

# Create data points
x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y1 = [0.879435964, 0.772041462, 0.76, 0.753093858, 0.777272634, 0.820359571, 0.799020191, 0.828176420, 0.828921116, 0.823678286, 0.82]
y2 = [0.819356398, 0.735595931, 0.5, 0.442329244, 0.445884135, 0.552466366, 0.667771164, 0.512706362, 0.605532057, 0.635901363, 0.65]
y3 = [0.892363686, 0.774542635, 0.772, 0.771815673, 0.802712270, 0.817614449, 0.843992111, 0.845087226, 0.834253687, 0.851285629, 0.855]
y4 = [0.855461708, 0.777957257, 0.76, 0.740030975, 0.739243372, 0.789641356, 0.827614284, 0.832840704, 0.778441363, 0.813524999, 0.83]
# Plot the curve
plt.figure(figsize=(5, 3))
plt.plot(x, y1, label='(S-S, C-B)', color='b', linewidth=2, linestyle='-')
plt.plot(x, y2, label='(S-S, M-B)', color='r', linewidth=2, linestyle='-')
plt.plot(x, y3, label='(S-S, C-R)', color='g', linewidth=2, linestyle='-')
plt.plot(x, y4, label='(S-S, M-R)', color='pink', linewidth=2, linestyle='-')
#plt.fill_between(x, y, color='skyblue', alpha=0.4)
#plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2+1, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title('MI', fontsize=16); 
plt.xlabel('Epochs', fontsize=14); plt.ylabel('MI', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.savefig('MI.png', dpi=300, bbox_inches='tight')
plt.show()

x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y5 = [0.228103933, 0.204770860, 0.207, 0.210969056, 0.219764855, 0.223947986, 0.220327106, 0.226250025, 0.224947253, 0.222044661, 0.221]
y6 = [0.215670498, 0.193888730, 0.16, 0.137903527, 0.138865453, 0.164897391, 0.187884403, 0.160525711, 0.175874553, 0.180997424, 0.185]
y7 = [0.231104106, 0.207411893, 0.21, 0.216994274, 0.224837886, 0.223927784, 0.231950101, 0.228348051, 0.225701228, 0.228766458, 0.23]
y8 = [0.223999570, 0.208828875, 0.206, 0.205115589, 0.208729075, 0.215564944, 0.223791858, 0.225194678, 0.214021856, 0.221820797, 0.22]
# Plot the curve
plt.figure(figsize=(5, 3))
plt.plot(x, y5, label='(S-S, C-B)', color='b', linewidth=2, linestyle='-')
plt.plot(x, y6, label='(S-S, M-B)', color='r', linewidth=2, linestyle='-')
plt.plot(x, y7, label='(S-S, C-R)', color='g', linewidth=2, linestyle='-')
plt.plot(x, y8, label='(S-S, M-R)', color='pink', linewidth=2, linestyle='-')
#plt.fill_between(x, y, color='skyblue', alpha=0.4)
#plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2+1, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title('NMI', fontsize=16); 
plt.xlabel('Epochs', fontsize=14); plt.ylabel('NMI', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.savefig('NMI.png', dpi=300, bbox_inches='tight')
plt.show()

x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y9 = [0.197657662, 0.151700734, 0.1518, 0.152233441, 0.145865458, 0.14574933, 0.168335308, 0.17574204, 0.168755885, 0.167423425, 0.166]
y10 = [0.11396199, 0.032357127, 0.0318, 0.030887462, 0.058712009, 0.047480333, 0.052714985, 0.052710766, 0.04816704, 0.038072031, 0.04]
y11 = [0.217047742, 0.168694815, 0.175, 0.195172613, 0.182689776, 0.20221198, 0.20411218, 0.19931103, 0.186594711, 0.180037236, 0.176]
y12 = [0.200723693, 0.189122083, 0.185, 0.182112407, 0.178829644, 0.18601723, 0.183698983, 0.194109563, 0.169843424, 0.175329776, 0.181]
# Plot the curve
plt.figure(figsize=(5, 3))
plt.plot(x, y9, label='(S-S, C-B)', color='b', linewidth=2, linestyle='-')
plt.plot(x, y10, label='(S-S, M-B)', color='r', linewidth=2, linestyle='-')
plt.plot(x, y11, label='(S-S, C-R)', color='g', linewidth=2, linestyle='-')
plt.plot(x, y12, label='(S-S, M-R)', color='pink', linewidth=2, linestyle='-')
#plt.fill_between(x, y, color='skyblue', alpha=0.4)
#plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2+1, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title('Corr', fontsize=16); 
plt.xlabel('Epochs', fontsize=14); plt.ylabel('Corr', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.savefig('Corr.png', dpi=300, bbox_inches='tight')
plt.show()


x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yt1_2= [0.171875, 0.765625, 0.902344, 0.929688, 0.890625, 0.941406, 0.933594, 0.960938, 0.976562, 0.976562, 0.988281]
yt1_3 = [0.140625, 0.195312, 0.414062, 0.367188, 0.382812, 0.292969, 0.25, 0.335938, 0.269531, 0.203125, 0.226562]
yt1_4 = [0.125, 0.402344, 0.628906, 0.507812, 0.484375, 0.5625, 0.765625, 0.875, 0.871094, 0.917969, 0.917969]
yt1_5 = [0.101562, 0.09375, 0.09375, 0.113281, 0.09375, 0.097656, 0.121094, 0.121094, 0.113281, 0.101562, 0.101562]
yt1_6 = [0.113281, 0.085938, 0.09375, 0.078125, 0.105469, 0.09375, 0.101562, 0.109375, 0.101562, 0.101562, 0.101562]
# Plot the curve
plt.figure(figsize=(5, 3))
plt.plot(x, yt1_2, label='(S-S)', color='purple', linewidth=2, linestyle='-')
plt.plot(x, yt1_3, label='(C-B)', color='b', linewidth=2, linestyle='-')
plt.plot(x, yt1_4, label='(M-B)', color='r', linewidth=2, linestyle='-')
plt.plot(x, yt1_5, label='(C-R)', color='g', linewidth=2, linestyle='-')
plt.plot(x, yt1_6, label='(M-R)', color='pink', linewidth=2, linestyle='-')
#plt.fill_between(x, y, color='skyblue', alpha=0.4)
#plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2+1, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title('Corr', fontsize=16); 
plt.xlabel('Epochs', fontsize=14); plt.ylabel('Top-1 Accuracy', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.savefig('Top-1.png', dpi=300, bbox_inches='tight')
plt.show()


x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yt5_2= [0.691406, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
yt5_3 = [0.570312, 0.746094, 0.875, 0.871094, 0.902344, 0.90625, 0.941406, 0.90625, 0.957031, 0.957031, 0.960938]
yt5_4 = [0.523438, 0.84375, 0.972656, 0.976562, 0.984375, 0.976562, 0.992188, 0.996094, 0.996094, 1, 1]
yt5_5 = [0.492188, 0.496094, 0.488281, 0.5, 0.511719, 0.503906, 0.484375, 0.476562, 0.488281, 0.484375, 0.472656]
yt5_6 = [0.460938, 0.429688, 0.46875, 0.542969, 0.519531, 0.523438, 0.550781, 0.523438, 0.570312, 0.574219, 0.574219]
# Plot the curve
plt.figure(figsize=(5, 3))
plt.plot(x, yt5_2, label='(S-S)', color='purple', linewidth=2, linestyle='-')
plt.plot(x, yt5_3, label='(C-B)', color='b', linewidth=2, linestyle='-')
plt.plot(x, yt5_4, label='(M-B)', color='r', linewidth=2, linestyle='-')
plt.plot(x, yt5_5, label='(C-R)', color='g', linewidth=2, linestyle='-')
plt.plot(x, yt5_6, label='(M-R)', color='pink', linewidth=2, linestyle='-')
#plt.fill_between(x, y, color='skyblue', alpha=0.4)
#plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2+1, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.title('Corr', fontsize=16); 
plt.xlabel('Epochs', fontsize=14); plt.ylabel('Top-5 Accuracy', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend()
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.savefig('Top-5.png', dpi=300, bbox_inches='tight')
plt.show()
