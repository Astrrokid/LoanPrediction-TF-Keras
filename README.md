# Loan Repayment Prediction using Neural Networks

This project aims to predict loan repayment using a neural network model built with TensorFlow and Keras. The dataset used contains various features related to loan applications, such as loan amount, interest rate, employment length, etc.

## About LendingClub
[LendingClub](https://www.lendingclub.com/) is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

## Dataset
The dataset used in this project is a sample from a lending club loan dataset. It contains information about loan applicants and whether they repaid the loan or not.

## Code Overview
- `loan_prediction.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model building, training, and evaluation.
- `loan_prediction.h5`: Saved model file after training.

### Data Preprocessing
- Loaded the dataset and performed exploratory data analysis.
- Handled missing values, categorical variables, and feature engineering.
- Split the dataset into training and testing sets.
- Normalized the feature data using MinMaxScaler.

### Model Building
- Built a sequential neural network model using TensorFlow and Keras.
- Added Dense layers with ReLU activation functions and Dropout layers.
- Compiled the model with binary crossentropy loss and Adam optimizer.

### Model Training
- Fitted the model to the training data for 25 epochs with early stopping based on validation loss.
- Monitored model performance using validation data during training.

### Model Evaluation
- Evaluated the model's performance on the test set using classification report and confusion matrix.
- Made predictions on a new customer's data and compared with the actual label.

## Results
The trained neural network model achieved an accuracy of approximately 86% on the test set. It demonstrated the ability to predict loan repayment with reasonable accuracy.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Usage
1. Ensure all required libraries are installed.
2. Run the `loan_prediction.ipynb` Jupyter Notebook to preprocess data, build, train, and evaluate the model.
3. Save the trained model (`loan_prediction.h5`) for future use.

## Author
Nsobundu Chukwudalu C
## License
This project is licensed under the [MIT License](LICENSE).
