import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import nltk, re, collections, pickle, os # nltk - Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('ggplot')

# Get the directory where this script is located for resolving relative paths
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
seed = 42

import warnings
warnings.filterwarnings(action = "ignore")
warnings.simplefilter(action = 'ignore', category = Warning)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
nltk.download('punkt_tab')

df = pd.read_csv(SCRIPT_DIR / 'Data' / 'spam.csv', encoding='latin1')
print(df.head())

pd.set_option("display.precision", 3)
pd.options.display.float_format = '{:.3f}'.format

columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
df = df.drop(columns=columns_to_drop)
print(df.head())

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' in s]

    plt.figure(figsize = (12, 5), dpi = 100)
    COLOR = 'gray'

    plt.rc('legend', fontsize = 14)   # legend fontsize
    plt.rc('figure', titlesize = 12)  # fontsize of the figure title

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace = 2, hspace = 2)
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b-o',
                 label = 'Train (' + str(str(format(history.history[l][-1],'.4f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label = 'Valid (' + str(str(format(history.history[l][-1],'.4f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend(facecolor = 'gray', loc = 'best')
    plt.grid(True)
    plt.tight_layout()

    ## Accuracy
    plt.subplot(1, 2, 2)
    plt.subplots_adjust(wspace = 2, hspace = 2)
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b-o',
                 label = 'Train (' + str(format(history.history[l][-1],'.4f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label = 'Valid (' + str(format(history.history[l][-1],'.4f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(facecolor = 'gray', loc = 'best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_conf_matr(conf_matr, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.winter):
  """
  Citation
  ---------
  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

  """
  import itertools

  accuracy = np.trace(conf_matr) / np.sum(conf_matr).astype('float')
  sns.set(font_scale = 1.4)

  plt.figure(figsize = (12, 8))
  plt.imshow(conf_matr, interpolation = 'nearest', cmap = cmap)
  title = '\n' + title + '\n'
  plt.title(title)
  plt.colorbar()

  if classes is not None:
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation = 45)
      plt.yticks(tick_marks, classes)

  if normalize:
      conf_matr = conf_matr.astype('float') / conf_matr.sum(axis = 1)[:, np.newaxis]


  thresh = conf_matr.max() / 1.5 if normalize else conf_matr.max() / 2
  for i, j in itertools.product(range(conf_matr.shape[0]), range(conf_matr.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.2f}%".format(conf_matr[i, j] * 100),
                    horizontalalignment = "center",
                    fontweight = 'bold',
                    color = "white" if conf_matr[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(conf_matr[i, j]),
                    horizontalalignment = "center",
                    fontweight = 'bold',
                    color = "white" if conf_matr[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\n\nAccuracy = {:0.2f}%; Error = {:0.2f}%'.format(accuracy * 100, (1 - accuracy) * 100))
  plt.show()


def plot_words(set, number):
  words_counter = collections.Counter([word for sentence in set for word in sentence.split()]) # finding words along with count
  most_counted = words_counter.most_common(number)
  most_count = pd.DataFrame(most_counted, columns = ["Words", "Amount"]).sort_values(by = "Amount") # sorted data frame
  most_count.plot.barh(x = "Words",
                       y = "Amount",
                       color = "blue",
                       figsize = (10, 15))
  for i, v in enumerate(most_count["Amount"]):
    plt.text(v, i,
             " " + str(v),
             color = 'black',
             va = 'center',
             fontweight = 'bold')

def word_cloud(tag):
  df_words_nl = ' '.join(list(df_spam[df_spam['feature'] == tag]['message']))
  df_wc_nl = WordCloud(width = 600, height = 512).generate(df_words_nl)
  plt.figure(figsize = (13, 9), facecolor = 'k')
  plt.imshow(df_wc_nl)
  plt.axis('off')
  plt.tight_layout(pad = 1)
  plt.show()

df_spam = pd.read_csv(SCRIPT_DIR / 'Data' / 'spam.csv', encoding = 'latin-1')

df_spam = df_spam.filter(['v1', 'v2'], axis = 1)
df_spam.columns = ['feature', 'message']
df_spam.drop_duplicates(inplace = True, ignore_index = True)
print('Number of null values:\n')
print(df_spam.isnull().sum())

print(df_spam['feature'].value_counts())

print(df_spam.shape, df_spam.columns)

plt.figure(figsize = (10, 6))
counter = df_spam.shape[0]
ax1 = sns.countplot(x='feature', data=df_spam)
ax2 = ax1.twinx()                      # Make double axis
ax2.yaxis.tick_left()                 # Switch so the counter's axis is on the right, frequency axis is on the left
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')  # Also switch the labels over
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('frequency, %')


for p in ax1.patches:
  x = p.get_bbox().get_points()[:, 0]
  y = p.get_bbox().get_points()[1, 1]
  ax1.annotate('{:.2f}%'.format(100. * y / counter),
              (x.mean(), y),
              ha = 'center',
              va = 'bottom')

# Use a LinearLocator to ensure the correct number of ticks
ax1.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0, 100)
ax1.set_ylim(0, counter)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)

"""The bar plot visually confirms the imbalance between ham and spam. The percentage labels show what proportion of messages belong to each class, reinforcing that ham dominates the dataset."""

plot_words(df_spam['message'], number = 30)

"""The horizontal bar chart displays the most commonly occurring words across all messages. This output reveals dominant vocabulary patterns and suggests that certain words appear repeatedly in the dataset"""

word_cloud('spam')

"""The spam word cloud highlights words that frequently appear in spam messages. These tend to be promotional or incentive-related terms, showing strong textual patterns associated with spam."""

word_cloud('ham')

"""The ham word cloud shows conversational and everyday language. This contrast with spam words indicates that ham messages are more natural and personal in tone.

##Part-1

##Machine Learning for SPAM classification task
"""

size_vocabulary = 1000
embedding_dimension = 64
trunc_type = 'post'
padding_type = 'post'
threshold = 0.5
oov_token = "<OOV>"
test_size, valid_size = 0.05, 0.2
num_epochs = 20
drop_level = 0.3

print("\t\tStage I. Preliminary actions. Preparing of needed sets\n")
full_df_l = []
lemmatizer = WordNetLemmatizer()
for i in range(df_spam.shape[0]):
    mess_1 = df_spam.iloc[i, 1]
    mess_1 = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', mess_1)
    mess_1 = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', mess_1)
    mess_1 = re.sub('£|\$', 'moneysymb', mess_1)
    mess_1 = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', mess_1)
    mess_1 = re.sub('\d+(\.\d+)?', 'numbr', mess_1)
    mess_1 = re.sub('[^\w\d\s]', ' ', mess_1)
    mess_1 = re.sub('[^A-Za-z]', ' ', mess_1).lower()
    token_messages = word_tokenize(mess_1)
    mess = []
    for word in token_messages:
        if word not in set(stopwords.words('english')):
            mess.append(lemmatizer.lemmatize(word))
    txt_mess = " ".join(mess)
    full_df_l.append(txt_mess)

plot_words(full_df_l, number = 35)

add_df = CountVectorizer(max_features = size_vocabulary)
X = add_df.fit_transform(full_df_l).toarray()
y = df_spam.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (test_size + valid_size), random_state = seed)
print('Number of rows in test set: ' + str(X_test.shape))
print('Number of rows in training set: ' + str(X_train.shape))

"""##II.Naive Bayes Classifier

###Guassian Naive Bayes
"""

print("\t\tStage IIa. Guassian Naive Bayes\n")
class_NBC = GaussianNB().fit(X_train, y_train) # Guassian Naive Bayes
y_pred_NBC = class_NBC.predict(X_test)
print('The first two predicted labels:', y_pred_NBC[0],y_pred_NBC[1], '\n')
conf_m_NBC = confusion_matrix(y_test, y_pred_NBC)
class_rep_NBC = classification_report(y_test, y_pred_NBC)
print('\t\t\tClassification report:\n\n', class_rep_NBC, '\n')
plot_conf_matr(conf_m_NBC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Guassian Naive Bayes')

"""he model achieves 79.43% accuracy and shows very high recall for Spam (0.94), meaning it successfully detects most spam messages. However, its low precision (0.41) indicates many legitimate ham messages are incorrectly classified as spam. Thus, the model behaves as an aggressive spam filter effective at catching spam but prone to false alarms, resulting in a 20.57% error rate.

###Multinomial Naive Bayes
"""

print("\t\tStage IIb. Multinomial Naive Bayes\n")
class_MNB = MultinomialNB().fit(X_train, y_train) # Multinomial Naive Bayes
y_pred_MNB = class_MNB.predict(X_test)
print('The first two predicted labels:', y_pred_MNB[0],y_pred_MNB[1], '\n')
conf_m_MNB = confusion_matrix(y_test, y_pred_MNB)
class_rep_MNB = classification_report(y_test, y_pred_MNB)
print('\t\t\tClassification report:\n\n', class_rep_MNB, '\n')
plot_conf_matr(conf_m_MNB, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Multinomial Naive Bayes')

"""The model achieves 97.29% accuracy with very high precision (0.99) and recall (0.98) for Ham, indicating that legitimate messages are almost always classified correctly. It also detects most spam messages while keeping false positives very low. With only a 2.71% error rate, this model provides a well-balanced and reliable spam filtering performance.

###Decision Tree Classifier
"""

print("\t\tStage III. Decision Tree Classifier\n")
class_DTC = DecisionTreeClassifier(random_state = seed).fit(X_train, y_train)
y_pred_DTC = class_DTC.predict(X_test)
print('The first two predicted labels:', y_pred_DTC[0], y_pred_DTC[1], '\n')
conf_m_DTC = confusion_matrix(y_test, y_pred_DTC)
class_rep_DTC = classification_report(y_test, y_pred_DTC)
print('\t\t\tClassification report:\n\n', class_rep_DTC, '\n')
plot_conf_matr(conf_m_DTC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Decision Tree')

"""The model achieves 95.82% accuracy with strong precision (0.97) and recall (0.98) for Ham, meaning legitimate messages are rarely misclassified. Spam detection is slightly weaker (precision 0.87, recall 0.84), leading to a 4.18% error rate. Overall, the model performs well but is slightly less effective than Multinomial Naive Bayes.

###Logistic Regression
"""

print("\t\tStage IV. Logistic Regression\n")
class_LR = LogisticRegression(random_state = seed, solver = 'liblinear').fit(X_train, y_train)
y_pred_LR = class_LR.predict(X_test)
print('The first two predicted labels:', y_pred_LR[0], y_pred_LR[1], '\n')
conf_m_LR = confusion_matrix(y_test, y_pred_LR)
class_rep_LR = classification_report(y_test, y_pred_LR)
print('\t\t\tClassification report:\n\n', class_rep_LR, '\n')
plot_conf_matr(conf_m_LR, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Logistic Regression')

"""The model achieves 97.68% accuracy with perfect recall (1.00) and high precision (0.98) for Ham, indicating legitimate messages are almost never misclassified. Spam detection remains strong (precision 0.98, recall 0.86). Overall, the model is highly reliable, producing a low 2.32% error rate.

###KNeighbors Classifier
"""

print("\t\tStage V. KNeighbors Classifier\n")
class_KNC = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
y_pred_KNC = class_KNC.predict(X_test)
print('The firs two predicted labels:', y_pred_KNC[0], y_pred_KNC[1], '\n')
conf_m_KNC = confusion_matrix(y_test, y_pred_KNC)
class_rep_KNC = classification_report(y_test, y_pred_KNC)
print('\t\t\tClassification report:\n\n', class_rep_KNC, '\n')
plot_conf_matr(conf_m_KNC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for KNeighbors Classifier')

"""The model achieves 94.43% accuracy with high recall for Ham (0.99), ensuring most legitimate messages are correctly classified. However, Spam recall is lower (0.69), meaning a noticeable portion of spam is missed. This results in a relatively higher 5.57% error rate compared to stronger models.

###Support Vector Classification
"""

print("\t\tStage VI. Support Vector Classification\n")
class_SVC = SVC(probability = True, random_state = seed).fit(X_train, y_train)
y_pred_SVC = class_SVC.predict(X_test)
print('The first two predicted labels:', y_pred_SVC[0], y_pred_SVC[1], '\n')
conf_m_SVC = confusion_matrix(y_test, y_pred_SVC)
class_rep_SVC = classification_report(y_test, y_pred_SVC)
print('\t\t\tClassification report:\n\n', class_rep_SVC, '\n')
plot_conf_matr(conf_m_SVC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for SVC Classifier')

"""###Gradient Boosting Classifier"""

print("\t\tStage VII. Gradient Boosting Classifier\n")
class_GBC = GradientBoostingClassifier(random_state = seed).fit(X_train, y_train)
y_pred_GBC = class_GBC.predict(X_test)
print('The first two predicted labels:', y_pred_GBC[0], y_pred_GBC[1], '\n')
conf_m_GBC = confusion_matrix(y_test, y_pred_GBC)
class_rep_GBC = classification_report(y_test, y_pred_GBC)
print('\t\t\tClassification report:\n\n', class_rep_GBC, '\n')
plot_conf_matr(conf_m_GBC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Gradient Boosting Classifier')

"""The model achieves 97.45% accuracy with perfect recall (1.00) and very high precision (0.99) for Ham, ensuring legitimate messages are almost never misclassified. Spam detection remains strong (precision 0.99, recall 0.83). Overall, it delivers highly reliable performance with a low 2.55% error rate.

###Bagging Classifier
"""

print("\t\tStage VIII. Bagging Classifier + something else\n")
class_BC = BaggingClassifier(random_state = seed).fit(X_train, y_train)
y_pred_BC = class_BC.predict(X_test)
print('The first two predicted labels:', y_pred_BC[0], y_pred_BC[1], '\n')
conf_m_BC = confusion_matrix(y_test, y_pred_BC)
class_rep_BC = classification_report(y_test, y_pred_BC)
print('\t\t\tClassification report:\n\n', class_rep_BC, '\n')
plot_conf_matr(conf_m_BC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Bagging Classifier')

"""The model achieves 96.21% accuracy with high precision (0.97) and recall (0.98) for Ham, meaning legitimate messages are consistently classified correctly. Spam detection is slightly weaker (recall 0.85), leading to a 3.79% error rate. Overall, the model shows strong and stable performance.

| Stage   | Classifier                | Accuracy | Error Rate | Key Interpretation                                                                                   |
| ------- | ------------------------- | -------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| IV & VI | Logistic Regression / SVC | 97.68%   | 2.32%      | Best in Class: Provides the highest overall accuracy and near-perfect protection of legitimate mail. |
| VII     | Gradient Boosting         | 97.45%   | 2.55%      | High Precision: Excellent at ensuring flagged spam is truly junk, though slightly lower spam recall. |
| IIb     | Multinomial Naive Bayes   | 97.29%   | 2.71%      | Balanced: Best middle-ground model with high consistency across both spam and ham classes.           |
| VIII    | Bagging Classifier        | 96.21%   | 3.79%      | Reliable: Strong ham protection but allows about 15% of spam to pass.                                |
| III     | Decision Tree             | 95.82%   | 4.18%      | Solid All-rounder: High performance, but slightly more misclassifications than ensemble methods.     |
| V       | K-Nearest Neighbors       | 94.43%   | 5.57%      | Moderate: High ham recall, but struggles with spam detection (≈31% missed).                          |
| IIa     | Gaussian Naive Bayes      | 79.43%   | 20.57%     | Poor: Trigger-happy model that flags many legitimate messages as spam.                               |

##Part-2
"""

print("Stage I. Preliminary actions. Preparing of needed sets\n")

sentences_new_set = []
labels_new_set = []
for i in range(0, df_spam.shape[0], 1):
    sentences_new_set.append(df_spam['message'][i])
    labels_new_set.append(df_spam['feature'][i])

train_size = int(df_spam.shape[0] * (1 - test_size - valid_size))
valid_bound = int(df_spam.shape[0] * (1 - valid_size))

train_sentences = sentences_new_set[0 : train_size]
valid_sentences = sentences_new_set[train_size : valid_bound]
test_sentences = sentences_new_set[valid_bound : ]

train_labels_str = labels_new_set[0 : train_size]
valid_labels_str = labels_new_set[train_size : valid_bound]
test_labels_str = labels_new_set[valid_bound : ]

print("Stage II. Labels transformations\n")

train_labels = [0] * len(train_labels_str)
for ind, item in enumerate(train_labels_str):
    if item == 'ham':
        train_labels[ind] = 1
    else:
        train_labels[ind] = 0

valid_labels = [0] * len(valid_labels_str)
for ind, item in enumerate(valid_labels_str):
    if item == 'ham':
        valid_labels[ind] = 1
    else:
        valid_labels[ind] = 0

test_labels = [0] * len(test_labels_str)
for ind, item in enumerate(test_labels_str):
    if item == 'ham':
        test_labels[ind] = 1
    else:
        test_labels[ind] = 0

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

"""##Part-3"""

print("Stage III. Tokenization\n")

tokenizer = Tokenizer(num_words = size_vocabulary,
                      oov_token = oov_token,
                      lower = False)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

"""Transforms text messages into numerical sequences by assigning an index to each word."""

train_sequences = tokenizer.texts_to_sequences(train_sentences)
size_voc = len(word_index) + 1
max_len = max([len(i) for i in train_sequences])
train_set = pad_sequences(train_sequences,
                                padding = padding_type,
                                maxlen = max_len,
                                truncating = trunc_type)

valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
valid_set = pad_sequences(valid_sequences,
                               padding = padding_type,
                               maxlen = max_len,
                               truncating = trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_set = pad_sequences(test_sequences,
                               padding = padding_type,
                               maxlen = max_len,
                               truncating = trunc_type)

"""Ensures all message sequences have the same length for efficient model training.

##Part-4 Model building
"""

print("Stage IV. Model building\n")

model = Sequential([
    Embedding(size_voc, embedding_dimension, input_length = max_len),
    Bidirectional(LSTM(100)),
    Dropout(drop_level),
    Dense(20, activation = 'relu'),
    Dropout(drop_level),
    Dense(1, activation = 'sigmoid')
])

"""Defines the neural network architecture used to learn patterns from text sequences.

##Part-5 Model compiling & fitting
"""

print("Stage V. Model compiling & fitting\n")
optim = Adam(learning_rate = 0.0001)

model.compile(loss = 'binary_crossentropy',
              optimizer = optim,
              metrics = ['accuracy'])
model.summary()

"""The model begins with an Embedding layer that transforms each word index into a 64-dimensional dense vector, allowing the model to capture semantic meaning rather than treating words as simple integers. The output shape (None, 189, 64) indicates that each message is represented as a sequence of 189 words, where every word is mapped to a 64-dimensional representation. This layer contains 606,080 parameters, reflecting the size of the learned word vocabulary and embedding space.

Next, a Bidirectional LSTM layer processes these word vectors in both forward and backward directions. This enables the model to understand context from previous and subsequent words simultaneously, which is essential for interpreting sentence meaning. The output shape (None, 200) shows that the layer produces a 200-dimensional feature vector for each message. This layer has 132,000 parameters, allowing it to learn complex sequential patterns.

A Dropout layer follows to reduce overfitting by randomly deactivating a fraction of neurons during training, forcing the model to learn more robust features. Since dropout does not learn weights, it has 0 parameters.

The Dense hidden layer with 20 neurons further refines the extracted features and learns higher-level patterns useful for classification. It contains 4,020 parameters.

Another Dropout layer is applied to provide additional regularization and improve generalization.

Finally, the output Dense layer has a single neuron with a sigmoid activation function, producing a probability between 0 and 1 that represents whether a message is spam or ham. This layer contains 21 parameters.

Overall, the model has 742,121 trainable parameters, all of which are optimized during training. This architecture balances expressive power and regularization, making it well-suited for learning complex patterns in text while avoiding overfitting.
"""

history = model.fit(train_set,
                    train_labels,
                    epochs = num_epochs,
                    validation_data = (valid_set, valid_labels),
                    verbose = 1)

"""The model shows strong and stable learning across 20 epochs.



*   Training accuracy improved from 85.7% to 99.5%.

*   Validation accuracy increased to around 98–99% and remained stable.

*   Both training and validation loss decreased significantly, indicating effective optimization.
*   The gap between training and validation accuracy is very small (≈1%), showing good generalization with no major overfitting.

##Part-6 result visualization
"""

print("Stage VI. Results visualization\n")
plot_history(history)

"""Loss Graph Interpretation

*   Training loss steadily decreases from ~0.45 to 0.0195, showing continuous learning.

*   Validation loss drops sharply in early epochs and stabilizes around 0.07.
*   Slight fluctuations in validation loss after epoch 8 are normal.
*   No major increase in validation loss → No significant overfitting

Accuracy Graph Interpretation



*   Training accuracy increases consistently from ~86% to 99.56%.
*   Validation accuracy improves quickly and stabilizes around 98–99%.
*   Very small gap (~1–1.5%) between training and validation accuracy.
"""

model_score = model.evaluate(test_set, test_labels, batch_size = embedding_dimension, verbose = 1)
print(f"Test accuracy: {model_score[1] * 100:0.2f}% \t\t Test error: {model_score[0]:0.4f}")

"""The model achieved a test accuracy of 98.16% with a test loss of 0.0744, indicating strong performance on completely unseen data. The test accuracy is very close to the training accuracy (99.56%) and validation accuracy (~98–99%), showing only a small gap of around 1–1.5%. This small difference suggests that the model generalizes well and does not suffer from significant overfitting. The low test loss further confirms that the model makes accurate and reliable predictions. Overall, the model is stable, well-trained, and performs effectively on new data.

##Part-7 Model saving & predict checking
"""

M_name = "My_model"

pickle.dump(tokenizer, open(M_name + ".pkl", "wb"))
filepath = M_name + '.h5'
tf.keras.models.save_model(model, filepath, include_optimizer = True, save_format = 'h5', overwrite = True)
print("Size of the saved model :", os.stat(filepath).st_size, "bytes")

y_pred_bLSTM = model.predict(test_set)

y_prediction = [0] * y_pred_bLSTM.shape[0]
for ind, item in enumerate(y_pred_bLSTM):
    if item > threshold:
        y_prediction[ind] = 1
    else:
        y_prediction[ind] = 0

conf_m_bLSTM = confusion_matrix(test_labels, y_prediction)
class_rep_bLSTM = classification_report(test_labels, y_prediction)
print('\t\t\tClassification report:\n\n', class_rep_bLSTM, '\n')
plot_conf_matr(conf_m_bLSTM, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for bLSTM')

"""The confusion matrix shows that the model correctly classified 109 spam messages and 906 ham messages, while misclassifying only 8 spam messages as ham and 11 ham messages as spam. This indicates very strong performance with very few errors. From the classification report, the model achieved 91% precision and 93% recall for spam, resulting in an F1-score of 0.92. For ham messages, the performance is even higher with 99% precision and 99% recall, giving an F1-score of 0.99. The overall accuracy is 98.16%, with an error rate of only 1.84%. Although the dataset is slightly imbalanced (more ham than spam), the model performs well for both classes, showing strong generalization and reliable classification capability."""

# You can change this message (as any short sentence) yourself
message_example = ["Darling, please give me a cup of tea"]

message_example_tp = pad_sequences(tokenizer.texts_to_sequences(message_example),
                                   maxlen = max_len,
                                   padding = padding_type,
                                   truncating = trunc_type)

pred = float(model.predict(message_example_tp))
if (pred > threshold):
    print ("This message is a real text")
else:
    print("This message is a spam message")