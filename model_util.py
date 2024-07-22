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
for j in range(3,7):
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
