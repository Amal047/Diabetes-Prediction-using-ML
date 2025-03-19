# Diabetes-Prediction-using-ML

This project implements a neural network using TensorFlow and Keras to predict diabetes based on the **Pima Indians Diabetes Dataset**. The model is trained using a **deep feedforward neural network** and achieves classification based on patient medical data.

## 📌 Features

- Uses **TensorFlow** and **Keras** for deep learning.
- Implements a **multi-layer perceptron (MLP)** with **ReLU activation**.
- Trains the model using **binary cross-entropy loss** and **Adam optimizer**.
- Loads and processes the **Pima Indians Diabetes Dataset**.
- Saves and reloads the trained model for predictions.

## 📂 Project Structure

```
Diabetes-Prediction-using-ML/
│── pima-indians-diabetes.csv  # Dataset
│── diabetes_prediction.py     # Main script
│── model_diabetes.h5         # Saved model (after training)
│── README.md                 # Documentation
│── venv/                      # Virtual environment (optional)
```

## 🛠 Installation

### 1️⃣ Prerequisites

Ensure you have **Python 3.10 or 3.11** installed. If not, download it from [python.org](https://www.python.org/downloads/).

### 2️⃣ Install Dependencies

Create and activate a virtual environment (recommended):

```sh
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

Then, install the required Python packages:

```sh
pip install tensorflow numpy
```

## 🚀 Usage

### 1️⃣ Run the Training Script

To train the model and save it:

```sh
python diabetes_prediction.py
```

### 2️⃣ Load the Trained Model for Prediction

After training, the model will be saved as `model_diabetes.h5`. You can reload and use it for predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')
x_test = data[760:, :8]
y_test = data[760:, -1]

model = load_model("model_diabetes.h5")
prediction = model.predict(x_test)
predicted_labels = (prediction >= 0.5).astype(int)
print(predicted_labels.flatten())
```

## 📝 Dataset Description

The **Pima Indians Diabetes Dataset** contains **768 samples** with **8 input features** and **1 output label**:

| Feature Name     | Description                                    |
| ---------------- | ---------------------------------------------- |
| Pregnancies      | Number of times pregnant                       |
| Glucose          | Plasma glucose concentration                   |
| BloodPressure    | Diastolic blood pressure (mm Hg)               |
| SkinThickness    | Triceps skin fold thickness (mm)               |
| Insulin          | 2-hour serum insulin (mu U/ml)                 |
| BMI              | Body Mass Index (weight in kg/(height in m)^2) |
| DiabetesPedigree | Diabetes pedigree function (genetic risk)      |
| Age              | Age in years                                   |
| Outcome (Label)  | **0 = No diabetes, 1 = Diabetes**              |

## 📊 Model Architecture

The model consists of multiple dense layers:

- **Input layer:** 8 neurons (corresponding to 8 features).
- **Hidden layers:** 8 layers with `ReLU` activation.
- **Output layer:** 1 neuron with `sigmoid` activation (for binary classification).

## 📈 Training Details

- **Loss function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Batch size:** 10
- **Epochs:** 350

## 📌 Future Improvements

- **Hyperparameter tuning** for better accuracy.
- **Feature engineering** to improve model performance.
- **Deploy the model** as a web application using Flask/Django.

## 📜 License

This project is for educational purposes. Free to use and modify.

---

👨‍💻 **Author:** _ Amal Babu _  
📅 **Last Updated:** March 2025
