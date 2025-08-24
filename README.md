# 🚀 Proactive Fraud Risk Assessment in UPI Transactions

This project is a **full-stack web application** designed to detect fraudulent transactions within India’s **Unified Payments Interface (UPI)** system.  
It leverages a **machine learning backend** to analyze transaction data and a **user-friendly web interface** to display the results, helping identify and flag suspicious activities proactively.

---

## 📖 About The Project

With the rapid adoption of UPI, financial fraud has become increasingly sophisticated.  
Traditional **rule-based systems** often fail to keep up with evolving fraud tactics.  

This project addresses the challenge by implementing an **intelligent system** that uses multiple machine learning models to distinguish between **legitimate and fraudulent transactions**.

👉 The application allows users (financial analysts, researchers, etc.) to:
- Train a model on a dataset  
- Input transaction details via a web interface  
- Receive **instant fraud risk assessments**  

This provides a practical tool for combating financial fraud in the digital payments ecosystem.

---

## ⚙️ Technology Stack

### 🔹 Backend & Machine Learning
- **Python** – Core backend & ML logic  
- **Pandas / NumPy** – Data processing & numerical operations  
- **Scikit-learn** – Preprocessing (StandardScaler) + ML models  
- **XGBoost** – Final fraud detection model with feature importance  
- **TensorFlow / Keras** – CNN-based deep learning experiments  
- **Flask** – Backend framework for serving the web app  
- **Joblib** – Model, scaler & feature persistence  

### 🔹 Frontend & Visualization
- **HTML / CSS / JavaScript** – User interface  
- **Google Charts** – Interactive visualizations  
- **Matplotlib / Seaborn** – Data exploration & plots  

---

## 🛠️ Built With
- **Backend:** Python, Flask  
- **ML:** Scikit-learn, TensorFlow, XGBoost, Pandas, NumPy  
- **Data Exploration:** Jupyter Notebook  
- **Frontend:** HTML, CSS, JS  
- **Visualization:** Matplotlib, Seaborn  

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### ✅ Prerequisites
Make sure you have:
- Python **3.8+**  
- **pip** (Python package installer)  
- A modern web browser (Chrome / Firefox)

---



## 2️⃣ Create & Activate Virtual Environment

> ### Create Environment
> ```bash
> python -m venv venv
> ```
>
> ### Activate Environment
> * **On Windows:**
>     ```bash
>     venv\Scripts\activate
>     ```
> * **On macOS/Linux:**
>     ```bash
>     source venv/bin/activate
>     ```

---

## 3️⃣ Install Dependencies

> Install all required packages using pip:
> ```bash
> pip install numpy pandas flask matplotlib seaborn
> pip install tensorflow xgboost
> pip install scikit-learn==1.6.1
> pip install notebook
> ```

---

## 🧠 Model Training (Generate Required Files)

> Run the Jupyter Notebook to train the model and generate the necessary `.pkl` files.
>
> 1.  **Start the Jupyter server:**
>     ```bash
>     jupyter notebook
>     ```
> 2.  **Run the notebook:**
>     * Open the `SRC` folder and the provided `.ipynb` file.
>     * Run all cells. This will generate the following files:
>         * **`feature_names.pkl`**
>         * **`project_model.pkl`**
>         * **`scaler.pkl`**

---

## ▶️ Run the Flask Web App

> Make sure you are in the project's root directory with the virtual environment (`venv`) activated.
> ```bash
> python app.py
> ```
> You should see the following output in your terminal:
> ```
> Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
> ```

---

## 🌐 Using the Application

> * Open your web browser and navigate to **`http://127.0.0.1:5000`**.
> * Enter the transaction details into the web form to get an instant fraud risk assessment.

---

## 🔮 Future Enhancements

> * **⚡ Real-Time Processing**: Integrate with Kafka for live fraud detection.
> * **🧾 Explainable AI**: Use SHAP / LIME for transparent model predictions.
> * **💻 React Frontend**: Replace Flask templates with a modern JavaScript framework for a better UI/UX.
> * **🔐 Authentication**: Implement a role-based secure login for analysts and administrators.    
