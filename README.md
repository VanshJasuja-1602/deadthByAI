# DeathByAI — Healthcare Bias Auditor ⚕️

**DeathByAI** is a premium, AI-powered healthcare bias auditing tool designed to ensure fairness in medical prediction models. Built with Streamlit and modern data visualization libraries, it allows users to upload healthcare datasets, detect demographic bias, and compute an overall fairness score.

![DeathByAI-Preview](https://raw.githubusercontent.com/VanshJasuja-1602/deadthByAI/main/assets/preview.png) *(Note: Add your preview image here)*

## 🚀 Features

-   **📂 Dataset Auditing**: Upload any CSV dataset containing healthcare predictions.
-   **⚙️ Configurable Bias Detection**: Choose specific prediction outputs and sensitive demographic columns (e.g., gender, age group, income) for analysis.
-   **📊 Real-time Metrics**: Calculate highest/lowest group rates and disparity percentages instantly.
-   **📈 Advanced Visualizations**:
    -   Interactive Plotly bar charts for prediction rates.
    -   Seaborn heatmaps for deep-dive bias analysis.
-   **🎯 AI Fairness Score**: A proprietary score (0-100) that summarizes the fairness of your model.
-   **🔍 Detailed Breakdown**: Granular view of prediction rates across all demographic segments.

## 🛠️ Tech Stack

-   **Frontend/App Framework**: [Streamlit](https://streamlit.io/)
-   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **Visualizations**: [Plotly](https://plotly.com/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
-   **Styling**: Custom CSS for a premium dark-mode experience.

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/VanshJasuja-1602/deadthByAI.git
    cd deadthByAI
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```
2.  **Upload your data**: Use the sidebar to upload a healthcare CSV file.
3.  **Configure the audit**: Select your prediction column and the sensitive demographic column you want to audit.
4.  **Review the results**: Check the fairness score and visualizations to identify potential biases.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ for a fairer future.*