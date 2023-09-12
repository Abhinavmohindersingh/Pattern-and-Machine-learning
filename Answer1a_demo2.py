import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from Answer1_demo2 import n_features


def pca_tf(data, num_components):
    """
    Perform PCA using TensorFlow.
    """
    # Compute the mean of the data
    mean = tf.math.reduce_mean(data, axis=0)

    # Center the data
    data_centered = data - mean

    # Singular Value Decomposition
    s, u, v = tf.linalg.svd(data_centered, full_matrices=False)

    # Extract the first 'num_components' principal components
    components = v[:num_components]
    components = tf.transpose(components)

    # Reconstruction of the data using these components
    projected_data = tf.matmul(data_centered, components)
    return projected_data, components, s


# Fetch the labeled faces in the wild dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# Get the shape of the images (for plotting later on)
n_samples, h, w = lfw_people.images.shape

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Convert training and test datasets to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

# Number of components to be used in PCA
n_components = 150

# Apply PCA to the training dataset
X_transformed, eigenfaces, s = pca_tf(X_train, n_components)

# Transform the test dataset using PCA components
X_test_transformed = tf.matmul(X_test - tf.reduce_mean(X_test, axis=0), eigenfaces, transpose_b=True)


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    Helper function to plot a gallery of images
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].numpy().reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


# Plot the eigenfaces
plot_gallery(eigenfaces, ["eigenface %d" % i for i in range(eigenfaces.shape[0])], h, w)

# Compute the explained variance and its cumulative sum
explained_variance = (s ** 2) / (n_samples - 1)
total_var = tf.reduce_sum(explained_variance)
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = tf.math.cumsum(explained_variance_ratio)

# Plot the explained variance
plt.plot(range(n_components), ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# NOTE: sklearn's RandomForest expects numpy arrays, so convert tensors back to numpy arrays
X_transformed_np = X_transformed.numpy()
X_test_transformed_np = X_test_transformed.numpy()

# Train a random forest classifier on the PCA-transformed data
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed_np, y_train)

# Make predictions on the test set
predictions = estimator.predict(X_test_transformed_np)

# Print the classification report
print(classification_report(y_test, predictions, target_names=target_names))
