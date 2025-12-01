# Spam Email Detector

A simple Machine Learning project to classify emails as **spam** or **ham (legitimate)** using Python. This project demonstrates the use of **text preprocessing**, **feature extraction**, and **Naive Bayes classification** in a practical, real-world application.

---

## 📝 Project Overview

The Spam Email Detector is one of the earliest and most practical AI applications. This project trains a machine learning model to automatically identify whether an email is spam or not. It covers several important concepts in Natural Language Processing (NLP) and Machine Learning:

- **Text preprocessing:** Cleaning raw email text (lowercasing, removing punctuation, whitespace normalization).
- **Feature extraction:** Converting text into numerical features using TF‑IDF vectorization.
- **Classification:** Using the Multinomial Naive Bayes algorithm for spam detection.
- **Evaluation:** Measuring accuracy, precision, recall, and F1-score, along with confusion matrices.

---

## 🛠 Tools & Libraries

- **Python 3**
- **Pandas:** For data handling and manipulation
- **Scikit-learn:** Machine learning framework
- **NLTK:** Natural Language Toolkit for stopwords
- **OS & Re:** For file handling and text preprocessing

---

## 📂 Dataset Structure

The project reads emails from `.txt` files organized in folders:





------------------------------------------------------------------------------------


Project Root/
│
├─ data/
│ ├─ ham/ # Contains legitimate emails
│ │ ├─ email1.txt
│ │ ├─ email2.txt
│ │ └─ ...
│ └─ spam/ # Contains spam emails
│ ├─ spam1.txt
│ ├─ spam2.txt
│ └─ ...



-------------------------------------------------------------------------------------


**Note:** Each `.txt` file should contain a single email message.

---

## ⚡ How It Works

1. The program loads all email files from the `ham` and `spam` directories.
2. Each email is cleaned and converted into numerical features with TF‑IDF vectorizer.
3. A **Multinomial Naive Bayes** classifier is trained on the dataset.
4. Model performance is evaluated with metrics like **accuracy**, **confusion matrix**, and **classification report**.
5. Users can enter a new email in the terminal, and the program will predict whether it is spam or ham.

---

## 💻 Usage

1. Clone this repository:

```bash
git clone https://github.com/KaanOzgurr/Spam_Detector_Machine.git
cd Spam_Detector_Machine



2.Activate your Python virtual environment:

.\.venv_ai11\Scripts\activate





3.Install required dependencies:


pip install -r requirements.txt





4.Run the program:

python spam_email_detector_full.py






5.Enter any email content in the terminal to test:


Enter full email content (or 'q' to quit):
> Congratulations! You won a free iPhone!
⚠️ This email is classified as SPAM.




--------------------------------------------------------------------------

📈 Example Output


Ham emails loaded: 5
Spam emails loaded: 5
Accuracy: 1.0
Confusion Matrix:
 [[1 0]
  [0 1]]
Classification Report:
               precision    recall  f1-score   support
           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1


--------------------------------------------------------------------------




🔍 Learning Outcomes

-- Learn how to preprocess text for machine learning.

-- Understand TF‑IDF vectorization and feature representation.

-- Gain practical experience with Naive Bayes for text classification.

-- Explore evaluation metrics for classification problems.

-- Practice building a terminal-based AI application.


-----------------------------------------------------------------

📚 References

## 📚 References

- [Scikit-learn: Naive Bayes]     (https://scikit-learn.org/stable/modules/naive_bayes.html)
- [NLTK Documentation]            (https://www.nltk.org/)
- [SpamAssassin Public Dataset]   (https://spamassassin.apache.org/old/publiccorpus/)
- [Enron Email Dataset]           (https://www.cs.cmu.edu/~enron/)


----------------------------------------------------------------


💡 Tips

Add more .txt emails in the ham and spam folders to improve model accuracy.

Ensure each file contains only one email.

This project is an excellent starting point for exploring NLP-based AI projects.


---------------------------------------------------------------------





