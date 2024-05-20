NOTE: please upload the required dataset to your notebook and paste the exact path in the 
code to load the dataset to run the model properly.


### Technical Description of the Project

This project involves the development and implementation of a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset contains 70,000 grayscale images of 28x28 pixels each, with 10 different classes representing various types of clothing items.

#### Key Components:

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using TensorFlow/Keras.
   - The images are normalized to have values between 0 and 1.

2. **Model Architecture:**
   - The CNN model consists of several convolutional layers followed by max-pooling layers.
   - After the convolutional layers, fully connected (dense) layers are added.
   - The final layer uses a softmax activation function to output probabilities for the 10 classes.

3. **Training the Model:**
   - The model is compiled with a loss function appropriate for multi-class classification (categorical cross-entropy) and an optimizer (typically Adam).
   - The model is trained on the training set and validated on the validation set.

4. **Evaluation:**
   - The model's performance is evaluated using accuracy metrics on the test set.
   - Incorrectly classified images are visualized to provide insights into the model's performance.

5. **Conversion to OpenVINO:**
   - The trained Keras model is converted to the OpenVINO format for optimized inference on Intel hardware.
   - The conversion process includes installing OpenVINO, running the model optimizer, and handling any directory issues.

6. **Visualization:**
   - The notebook includes visualizations of sample images from the test set, along with their predicted and actual labels.

#### Summary of Code Cells:

1. **Import Libraries:**
   - TensorFlow, Keras, NumPy, Matplotlib, and other necessary libraries are imported.

2. **Load and Preprocess Data:**
   - The Fashion MNIST dataset is loaded and split into training and testing sets.
   - Data normalization is performed.

3. **Define the Model:**
   - A sequential CNN model is defined with multiple layers including Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.

4. **Compile and Train the Model:**
   - The model is compiled with an optimizer, loss function, and evaluation metric.
   - The model is trained on the training data and validated on a subset of the data.

5. **Evaluate the Model:**
   - The trained model is evaluated on the test set.
   - Misclassified images are displayed along with their predicted and actual labels.

6. **Convert Model to OpenVINO:**
   - Instructions and commands for converting the trained Keras model to the OpenVINO format are provided.
   - Installation of OpenVINO and execution of the model optimizer are included.

7. **Clean-up:**
   - Commands to remove temporary files generated during the model conversion process.

This project demonstrates the complete workflow of developing a CNN for image classification, from data preprocessing to model training, evaluation, and optimization for deployment on specific hardware using OpenVINO.
