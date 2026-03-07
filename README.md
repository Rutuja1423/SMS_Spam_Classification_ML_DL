# SMS Spam Classification using Machine Learning and Deep Learning

## Project Overview

This project focuses on developing a text classification model to automatically detect whether an SMS message is **Spam** or **Ham (Not Spam)**.

Both **Machine Learning algorithms** and a **Deep Learning model** were implemented and compared to evaluate their performance in spam detection.

The project demonstrates how **Natural Language Processing (NLP)** techniques can be applied to transform unstructured text into numerical features for predictive modeling.

---

## Objectives

* Perform text preprocessing and cleaning
* Convert text into numerical format using NLP techniques
* Train multiple classification models
* Implement a Deep Learning model
* Compare the performance of Machine Learning and Deep Learning approaches

---

## Dataset

The dataset contains labeled SMS messages classified into two categories:

* **Spam** – Unwanted promotional or fraudulent messages
* **Ham** – Legitimate SMS messages

This is a **binary text classification problem** where the goal is to correctly classify each message based on its content.

---

## Tools and Technologies

The following tools and libraries were used in this project:

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Natural Language Processing (NLP)

---

## Machine Learning Models

The following Machine Learning algorithms were implemented:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

These models were trained using **TF-IDF vectorized text features**.

---

## Deep Learning Model

A **Long Short-Term Memory (LSTM)** neural network was implemented to capture sequential patterns and contextual relationships in text data.

---

## Methodology

The project follows these main steps:

1. **Text Preprocessing**

   * Removing punctuation and stopwords
   * Converting text to lowercase
   * Cleaning and normalizing messages

2. **Text Representation**

   * Tokenization
   * TF-IDF vectorization or word embeddings

3. **Model Training**

   * Training Machine Learning models
   * Training the LSTM model

4. **Model Evaluation**

   * Accuracy
   * Confusion Matrix
   * Model comparison

---

## Results

The models achieved **high classification accuracy (approximately 98%)**, demonstrating the effectiveness of NLP techniques in spam detection.

The Deep Learning model showed strong ability to capture contextual information from text.

---

## Conclusion

This project demonstrates practical skills in:

* Natural Language Processing (NLP)
* Machine Learning for text classification
* Deep Learning using LSTM networks
* Model evaluation and comparison

It highlights how both Machine Learning and Deep Learning approaches can be applied to solve real-world spam detection problems.
