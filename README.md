
# Fake News Detector

## Overview
Fake News Detector is a machine learning project that classifies news articles as Fake or Real using Natural Language Processing (NLP).

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Dataset
The project uses a dataset containing separate CSV files for fake and real news articles.

## Methodology
1. Load the fake and real news datasets.
2. Assign labels to both categories.
3. Combine the news title and article text.
4. Split the dataset into training and testing sets.
5. Convert text into numerical features using TF-IDF.
6. Train a Logistic Regression classifier.
7. Evaluate the model using accuracy, precision, recall and F1-score.
8. Use Streamlit to provide an interactive interface.

## Model Performance
The model achieved approximately 98.57% accuracy on the held-out test set for this dataset.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the web application:

```bash
streamlit run app.py
```

## Limitations
This model identifies patterns in a labeled dataset. It does not independently verify claims against trusted sources, and its predictions may be incorrect.
## 📸 Application Screenshots

### Home Page
![Home Page](screenshots/home.png)

### News Analysis
![News Analysis](screenshots/show model.png)

### Current News Sources
![Sources](screenshots/footer.png)