# Extending Keras: Building a Custom Dense Layer for Image Classification

This project demonstrates an advanced understanding of the Keras API by creating a custom, fully-connected (`Dense`) layer from scratch. To validate its functionality, the custom layer is integrated into a deep learning model and trained to classify a filtered subset of the CIFAR-10 dataset, ultimately achieving **90.17% accuracy** on the test set.

## Problem Statement and Goal of Project

While high-level deep learning frameworks provide a rich set of pre-built layers, the ability to create custom components is essential for research, innovation, and implementing novel architectures. The primary goal of this project is to demonstrate this advanced skill by:

1.  **Implementing a Custom Layer:** Building a `MyDense` layer that inherits from `tf.keras.layers.Layer` and replicates the core functionality of the standard `Dense` layer.
2.  **Model Integration & Training:** Using the custom layer as a building block in a neural network to solve a practical image classification problem.
3.  **Performance Validation:** Training the model with best practices (e.g., callbacks) and rigorously evaluating its performance with accuracy metrics and a confusion matrix.

## Solution Approach

The solution was implemented in a single, well-documented Jupyter Notebook, covering the entire lifecycle from layer creation to model evaluation.

  - **Custom `MyDense` Layer:** The core of the project is a custom Keras layer. It was built by subclassing `tf.keras.layers.Layer` and implementing three essential methods:

      - `__init__()`: Initializes layer-specific parameters, such as the number of output units and the activation function.
      - `build()`: Defines the layer's trainable weights (kernel `w` and bias `b`) using `self.add_weight()`. This method is executed automatically by Keras the first time the layer is called, which allows for flexible input shapes.
      - `call()`: Implements the forward pass logic: `output = activation(input @ w + b)`.

  - **Data Filtering and Preprocessing:** The CIFAR-10 dataset was loaded from `keras.datasets`. To create a focused and manageable problem, the dataset was filtered to include only three classes: **plane, car, and bird**. The labels were then remapped to `0, 1, 2`, and pixel values were normalized to the `[0, 1]` range.

  - **Model Architecture:** A simple yet effective sequential model was constructed to test the custom layer:

    1.  `Flatten` layer to convert the 2D images into 1D vectors.
    2.  `MyDense` (custom layer) with a ReLU activation function.
    3.  A standard `Dense` output layer with 3 units for the final classification.

  - **Training and Callbacks:** The model was compiled with the `Adam` optimizer and `SparseCategoricalCrossentropy` loss. The `model.fit()` method was used for training, enhanced with two key callbacks:

      - `EarlyStopping`: To prevent overfitting by halting training when validation loss stops improving.
      - `ReduceLROnPlateau`: To dynamically adjust the learning rate for finer convergence.

  - **Evaluation:** Performance was thoroughly assessed by plotting accuracy and loss curves for both training and validation sets. A final evaluation on the test set was performed, and a **confusion matrix** was generated to visualize the model's classification performance for each of the three classes.

## Technologies & Libraries

  - **Primary Framework**: TensorFlow 2.10
  - **Core Libraries**: Keras, NumPy
  - **Data Visualization**: Matplotlib
  - **Metrics & Analysis**: scikit-learn

## Description about Dataset

The project uses a modified version of the **CIFAR-10** dataset. The original dataset consists of 60,000 `32x32` color images in 10 classes. For this task, the dataset was filtered to create a 3-class problem, using only the images for **planes, cars, and birds**. This resulted in a focused dataset of 15,000 training images and 3,000 test images.

## Installation & Execution Guide

To run this project locally, please follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/imehranasgari/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: A `requirements.txt` file should be created containing `tensorflow`, `numpy`, `matplotlib`, and `scikit-learn`.)*

3.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

4.  Open and run the cells in the `cumtom_layer_with_limit_class.ipynb` notebook.

## Key Results / Performance

The project successfully demonstrates the ability to create and integrate a custom Keras layer into a functional deep learning model.

  - **Final Test Accuracy:** **90.17%**
  - **Training Stability:** The model shows excellent convergence behavior, with the validation loss closely tracking the training loss, indicating a well-regularized model.
  - **Class Performance:** The confusion matrix reveals high precision and recall across all three classes, validating the model's effectiveness.

## Screenshots / Sample Output

*This file was intentionally created to demonstrate skills in implementing and explaining machine learning models, rather than solely focusing on achieving the highest evaluation metrics. The simple approach is for learning, benchmarking, and illustrating fundamental concepts.*

**Model Architecture Summary**

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 3072)              0         
                                                                 
 my_dense (MyDense)          (None, 64)                196672    
                                                                 
 dense (Dense)               (None, 3)                 195       
                                                                 
=================================================================
Total params: 196,867
Trainable params: 196,867
Non-trainable params: 0
_________________________________________________________________
```

**Training & Validation Performance Curves**

**Test Set Confusion Matrix**

## Additional Learnings / Reflections

Building a custom layer from scratch was an incredibly insightful experience. It solidified my understanding of the fundamental components of a neural network, such as weight and bias initialization (`build` method) and the forward pass computation (`call` method). This project proves that I can move beyond using pre-built library components and extend the Keras framework to implement novel ideas—a critical skill for any advanced machine learning or research role. Manually defining the layer's behavior provides the ultimate flexibility needed to build custom architectures not available in standard libraries.

-----

## 👤 Author

**Mehran Asgari**

  - **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
  - **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

-----

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*