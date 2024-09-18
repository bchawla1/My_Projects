import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import base64
from io import BytesIO

# Title of the app
st.title("DS App")

# Sidebar for user input
st.sidebar.header("Upload Dataset")

# File upload for training data
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for training", type=["csv"])

# Display the dataset
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Training Dataset Preview")
    st.write(data.head())

    # Display basic information about the dataset
    st.subheader("Dataset Summary")
    st.write(data.describe())

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Selecting columns for X and y
    st.sidebar.header("Model Configuration")
    target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", options=[col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Data Preprocessing
        st.sidebar.header("Preprocessing Options")
        scale_data = st.sidebar.checkbox("Scale Data", value=False)

        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        st.sidebar.header("Select Models to Train")
        model_options = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Support Vector Regression (SVR)": SVR(),
        }

        # Checkboxes for model selection
        selected_models = [model_name for model_name, model in model_options.items() if st.sidebar.checkbox(f"Train {model_name}", value=False)]

        # Train and evaluate selected models
        model_performance = {}
        best_model = None
        best_mse = float('inf')

        # Prepare lists to store MSE and R² scores for visualization
        mse_scores = []
        r2_scores = []
        model_names = []

        for model_name in selected_models:
            model = model_options[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            model_performance[model_name] = {'model': model, 'mse': mse, 'r2': r2}

            # Track the best model based on MSE
            if mse < best_mse:
                best_mse = mse
                best_model = model

            # Store metrics for visualization
            mse_scores.append(mse)
            r2_scores.append(r2)
            model_names.append(model_name)

            st.write(f"**{model_name}** - MSE: {mse:.2f}, R²: {r2:.2f}")

        # Visualize MSE and R² Scores
        if selected_models:
            st.subheader("Model Performance Comparison")
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))

            # MSE Visualization
            ax[0].barh(model_names, mse_scores, color='skyblue')
            ax[0].set_xlabel("MSE")
            ax[0].set_title("Mean Squared Error of Models")

            # R² Visualization
            ax[1].barh(model_names, r2_scores, color='lightgreen')
            ax[1].set_xlabel("R² Score")
            ax[1].set_title("R² Score of Models")

            st.pyplot(fig)

        # Final Model Selection
        st.subheader("Final Model Selection")
        final_model_name = st.selectbox("Select the final model", list(model_performance.keys()))

        if final_model_name:
            final_model = model_performance[final_model_name]['model']
            st.write(f"Selected Model: **{final_model_name}**")

            # Option to test on new unseen data
            st.sidebar.header("Test on New Unseen Data")
            unseen_file = st.sidebar.file_uploader("Upload a CSV file with unseen data", type=["csv"])

            if unseen_file:
                unseen_data = pd.read_csv(unseen_file)
                X_unseen = unseen_data[feature_columns]
                y_unseen = unseen_data[target_column]

                if scale_data:
                    X_unseen = scaler.transform(X_unseen)

                y_unseen_pred = final_model.predict(X_unseen)
                unseen_mse = mean_squared_error(y_unseen, y_unseen_pred)
                unseen_r2 = r2_score(y_unseen, y_unseen_pred)

                st.subheader("Unseen Data Performance")
                st.write(f"Mean Squared Error on Unseen Data: {unseen_mse:.2f}")
                st.write(f"R² on Unseen Data: {unseen_r2:.2f}")

                # Plotting actual vs predicted values for unseen data
                st.subheader("Actual vs Predicted Values on Unseen Data")
                plt.figure(figsize=(10, 5))
                plt.scatter(y_unseen, y_unseen_pred, color='blue')
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted Values on Unseen Data")
                st.pyplot(plt)

        # Model Export Option
        st.sidebar.header("Export Final Model")
        export_model = st.sidebar.button("Download Final Model")
        
        if export_model and final_model_name:
            model_file = BytesIO()
            joblib.dump(final_model, model_file)
            model_file.seek(0)
            b64 = base64.b64encode(model_file.read()).decode()
            href = f'<a href="data:file/pkl;base64,{b64}" download="{final_model_name}_model.pkl">Download Final Model</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("Please select both target and feature columns.")
else:
    st.info("Please upload a CSV file to get started.")