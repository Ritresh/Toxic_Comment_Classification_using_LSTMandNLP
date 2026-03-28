# Toxic Comment Classification using Deep Learning (LSTM)

## Project Overview
This project builds a **multi-label toxic comment classification model** using **Natural Language Processing (NLP)** and **Deep Learning**.

The model predicts whether a comment belongs to one or more toxic categories.

### Toxicity Labels
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The goal of this project is to help automatically detect harmful or abusive comments on online platforms.

---

# Dataset

The dataset contains Wikipedia comments labeled for toxicity.

### Files Used
- `train.csv` – Training dataset
- `test.csv` – Test dataset
- `test_labels.csv` – Test labels
- `sample_submission.csv` – Submission format

### Dataset Size
- **159,571 comments**
- **Multi-label classification problem**

Each comment may contain **multiple toxicity labels**.

---

# Exploratory Data Analysis (EDA)

Several analyses were performed to understand the dataset.

### Label Distribution
Visualized how many comments belong to each toxicity class.

### Clean vs Toxic Comments
Identified comments that have **no toxicity labels**.

### Comment Length Analysis
Studied distribution of comment lengths.

### Toxic vs Clean Length Comparison
Compared length of toxic and non-toxic comments.

### Label Correlation
Created a **correlation heatmap** to see relationships between toxicity labels.

### Most Frequent Words
Extracted the **top words appearing in toxic comments and clean comments**.

---

# Feature Engineering

### Text Vectorization

Text comments were converted into numerical form using **TensorFlow TextVectorization**.

Settings used:

- Maximum vocabulary size: `200000`
- Maximum sequence length: `1800`

This process:
- Removes punctuation
- Converts words to tokens
- Maps tokens to integer indices
- Pads sequences to fixed length

---

# Dataset Pipeline

The dataset was converted into a **TensorFlow dataset pipeline**.

Steps used:

- Cache dataset
- Shuffle dataset
- Batch dataset
- Prefetch dataset

### Dataset Split

- **70% Training**
- **10% Validation**
- **20% Testing**

---

# Deep Learning Model

A **Bidirectional LSTM model** was built to classify toxic comments.

### Model Architecture

```
Embedding Layer
↓
Bidirectional LSTM
↓
Dense Layer (ReLU)
↓
Dense Layer (ReLU)
↓
Dense Layer (ReLU)
↓
Output Layer (Sigmoid)
```

### Output

The model predicts **6 probabilities** corresponding to each toxicity label.

### Activation
Sigmoid (for multi-label classification)

### Loss Function
Binary Crossentropy

### Optimizer
Adam

---

# Prediction Example

A function was created to predict toxicity for any input comment.

Example:

```python
predict_comment("You freaking suck! I am going to hit you.")
```

Output:

| Label | Probability | Prediction |
|------|------|------|
| toxic | value | 0/1 |
| severe_toxic | value | 0/1 |
| obscene | value | 0/1 |
| threat | value | 0/1 |
| insult | value | 0/1 |
| identity_hate | value | 0/1 |

---

# Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Natural Language Processing (NLP)

Development Environment:

- Google Colab

---

# Project Workflow

1. Load dataset
2. Perform Exploratory Data Analysis
3. Prepare text data
4. Vectorize comments
5. Build TensorFlow dataset pipeline
6. Train Bidirectional LSTM model
7. Evaluate predictions

---

# Future Improvements

- Train model for more epochs
- Try advanced architectures like **GRU or Transformers**
- Deploy the model as a **web application**

---

# Author

**Ritresh Kumar**  
BCA Student | Aspiring Data Scientist

Skills:
- Python
- Machine Learning
- Data Analysis
- Customer Analytics
- Deep Learning

📧 Email: ritresh273@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/feed/)  
🔗 [GitHub](https://github.com/Ritresh/Ritresh)
