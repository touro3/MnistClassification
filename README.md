# MNIST Classification Project

## Project Overview
This project involves building a machine learning model to classify handwritten digits from the MNIST dataset, which is a popular dataset used in image processing and classification tasks. The MNIST dataset contains 70,000 grayscale images of digits (0-9), each 28x28 pixels in size. The goal of the project is to create a model that can accurately classify these images into one of 10 classes, representing the digits 0 through 9.

The project utilizes Python with the Scikit-learn library for implementing various classification algorithms. We focus on two key strategies for handling multiclass classification problems:
- **One-vs-Rest (OvR)**: This is the default strategy in Scikit-learn for binary classifiers applied to multiclass problems.
- **One-vs-One (OvO)**: Although not used in the main implementation, OvO is discussed as an alternative strategy.

---

## Dataset
- **MNIST**: The MNIST dataset consists of 70,000 labeled images of handwritten digits.
  - **Training Set**: 60,000 images.
  - **Test Set**: 10,000 images.

Each image is represented as a 28x28 matrix, where each value corresponds to the grayscale intensity of a pixel (0 for white and 255 for black). These matrices are flattened into a vector of 784 features for classification.

---

## Algorithms Used

### 1. Stochastic Gradient Descent (SGD) Classifier
- **Algorithm**: Stochastic Gradient Descent is an iterative optimization algorithm used for minimizing an objective function. It is well-suited for large datasets and can be used for both regression and classification tasks.
- **One-vs-Rest (OvR) Strategy**: Scikit-learn automatically uses the One-vs-Rest strategy for multiclass classification when applying binary classifiers like `SGDClassifier`. In this strategy, the classifier creates a separate binary model for each class. For each model, one class is treated as the positive class, and the rest are grouped together as the negative class. During prediction, the model outputs the class with the highest confidence score.
- **Performance**: 
  - This approach is efficient and scales well with large datasets like MNIST.
  - The classifier achieved an average accuracy of **89.7%** using cross-validation.

### 2. Random Forest Classifier
- **Algorithm**: Random Forest is an ensemble method that trains multiple decision trees on different subsets of the data and averages their predictions. It can handle both classification and regression problems and is robust to overfitting.
- **Multiclass Handling**: Random Forest natively supports multiclass classification without needing OvR or OvO strategies. It can assign probabilities to each class based on the majority vote across all decision trees.
- **Use Case**: Although not the primary algorithm used in this project, Random Forest was considered as an alternative due to its robustness and high accuracy in classification tasks.

### 3. One-vs-One (OvO) Strategy
- **Concept**: In One-vs-One, separate classifiers are trained for every possible pair of classes. For MNIST, this would mean training 45 classifiers (one for each pair of digits). The final prediction is made by comparing which class wins the most “duels” across all classifiers.
- **Pros**: This approach can be more accurate when classes are very similar to each other.
- **Cons**: OvO requires significantly more computational resources because of the large number of classifiers that need to be trained.

---

## Steps Followed in the Project

1. **Data Loading**:
   - The MNIST dataset is loaded using Scikit-learn’s `fetch_openml()` function.
   - The dataset is split into a training set (60,000 samples) and a test set (10,000 samples).

2. **Data Preprocessing**:
   - **Feature Scaling**: The data is scaled using `StandardScaler` to ensure that the pixel values (0-255) are standardized. Feature scaling is important for gradient-based optimizers like SGD.

3. **Model Training**:
   - The `SGDClassifier` is trained on the scaled data using the One-vs-Rest strategy.
   - The model is evaluated using **cross-validation** to ensure it generalizes well to unseen data.

4. **Model Evaluation**:
   - The classifier’s accuracy is evaluated using cross-validation, achieving an average accuracy of **89.7%**.
   - The model is further tested by predicting individual samples and examining the confidence scores for each class using the `decision_function`.

5. **Hyperparameter Tuning** (optional):
   - **SGDClassifier**: Various hyperparameters such as `learning_rate`, `eta0`, and `max_iter` can be tuned using GridSearch or manual adjustment to optimize model performance.

---

## Results

- **Accuracy**: The `SGDClassifier` with the One-vs-Rest strategy achieved an accuracy of **89.7%** using cross-validation on the MNIST dataset. This is a solid performance for this baseline model.
- **Predicted Class Example**: For a sample image of a handwritten digit, the model correctly predicted it as **7**, and the confidence score for class 7 was significantly higher than for the other classes.

---

## Conclusion

The MNIST classification project successfully demonstrated the use of **Stochastic Gradient Descent (SGD)** in combination with the **One-vs-Rest** strategy for multiclass classification. This approach is both efficient and effective for large-scale datasets like MNIST. The project also discussed alternative strategies like **One-vs-One** and explored the use of ensemble methods like **Random Forest**.

Future improvements can include hyperparameter tuning, exploring convolutional neural networks (CNNs), or experimenting with more sophisticated ensemble techniques to further boost accuracy. Despite the simplicity of the algorithms used, the results are strong, with nearly 90% accuracy in recognizing handwritten digits.
