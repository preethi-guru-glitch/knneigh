import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KNN Demo", layout="centered")

st.title("Social Network Ads - KNN Demo")

# 1. Upload CSV
uploaded = st.file_uploader("Upload Social_Network_Ads.csv", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)

    # Basic validation
    required_cols = {"Age", "EstimatedSalary", "Purchased"}
    if not required_cols.issubset(data.columns):
        st.error("CSV must contain columns: Age, EstimatedSalary, Purchased")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # 2. Features & target
    X = data[["Age", "EstimatedSalary"]]
    y = data["Purchased"]

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # 4. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. KNN model
    k = st.slider("Number of neighbors (k)", 1, 30, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    # 6. Accuracy
    acc = model.score(X_test_scaled, y_test)
    st.success(f"Test Accuracy: {acc:.3f}")

    # 7. Prediction section
    st.subheader("Try a New Person")

    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    salary = st.number_input(
        "Estimated Salary", min_value=15000, max_value=200000, value=40000
    )

    if st.button("Predict"):
        new_point = scaler.transform([[age, salary]])
        pred = model.predict(new_point)[0]
        label = "Purchased ✅" if pred == 1 else "Not Purchased ❌"
        st.info(f"Prediction: {label}")

    # 8. Visualization
    st.subheader("KNN Decision Boundary")

    if st.checkbox("Show Decision Boundary"):
        x_min, x_max = X["Age"].min() - 1, X["Age"].max() + 1
        y_min, y_max = (
            X["EstimatedSalary"].min() - 5000,
            X["EstimatedSalary"].max() + 5000,
        )

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.5),
            np.arange(y_min, y_max, 3000),
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        Z = model.predict(grid_scaled).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(xx, yy, Z, alpha=0.3)

        ax.scatter(
            X[y == 0]["Age"],
            X[y == 0]["EstimatedSalary"],
            c="red",
            s=20,
            label="Not Purchased",
        )
        ax.scatter(
            X[y == 1]["Age"],
            X[y == 1]["EstimatedSalary"],
            c="green",
            s=20,
            label="Purchased",
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Estimated Salary")
        ax.legend()
        ax.set_title("KNN Decision Boundary")

        st.pyplot(fig)

else:
    st.info("👆 Please upload the CSV file to begin.")
