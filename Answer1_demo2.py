# Import necessary libraries and functions

# Import the function fetch_lfw_people from sklearn.datasets, which provides labeled jpeg images
from sklearn.datasets import fetch_lfw_people

# Import train_test_split to divide the data into training, testing, and validation sets
from sklearn.model_selection import train_test_split

# Import the classification_report for displaying the primary classification metrics
from sklearn.metrics import classification_report

# Import numpy for numerical computations
import numpy as np

# Fetch the dataset and filter to include only those people who have at least 70 images
# Also, the images are resized to reduce resource usage and speed up computation
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Get the shape of the images, providing information about their height and width
n_samples, h, w = lfw_people.images.shape

# Flatten the 2D image data into 1D to facilitate machine learning processing
X = lfw_people.data
n_features = X.shape[1]

# Extract the target labels and their descriptive names
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Display some details about the dataset
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Perform PCA, reducing the data to its first 150 principal components
n_components = 150
mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

# Apply Singular Value Decomposition (SVD)
U, S, V = np.linalg.svd(X_train, full_matrices=False)
components = V[:n_components]

# Convert the principal components back into the original image shape
eigenfaces = components.reshape((n_components, h, w))

# Transform the training and test data into the principal component space
X_transformed = np.dot(X_train, components.T)
X_test_transformed = np.dot(X_test, components.T)

# Visualize the principal components (eigenfaces)
import matplotlib.pyplot as plt

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Visualize images in a grid format."""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

# Calculate and plot the explained variance of the principal components
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = np.cumsum(explained_variance_ratio)
eigenvalueCount = np.arange(n_components)
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# Use a RandomForestClassifier to classify the images based on their principal components
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train)
predictions = estimator.predict(X_test_transformed)

# Calculate and display the classification accuracy
correct = predictions == y_test
total_test = len(X_test_transformed)
print("Total Testing:", total_test)
print("Predictions", predictions)
print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct) / total_test)

# Display a detailed classification report
print(classification_report(y_test, predictions, target_names=target_names))
