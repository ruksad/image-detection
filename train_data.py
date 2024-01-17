import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_and_preprocess_data(data_dir, img_size=(64, 64)):
    images = []
    labels = []

    # Loop through each class (cheque and driving license)
    for label, folder in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, folder)

        # Loop through each image in the class folder
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            # Load and resize the image
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values between 0 and 1

            # Append the image and label to the lists
            images.append(img_array)
            labels.append(label)

    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Replace 'path/to/your/data' with the actual path to your data folders
data_dir = './testImages/dl'
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_dir)

# Print the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
