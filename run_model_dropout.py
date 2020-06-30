
# -------- Tensorflow packages ------------------ #
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import TimeDistributed, Dense, LSTM, Embedding, Dropout, Bidirectional, GlobalMaxPool1D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils import to_categorical
# -------- other packages ------------------ #
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pickle

np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("wikidata.csv", delimiter = "\t", names = ["Sentence", "Keyword"])
df['Sentence'] = df['Sentence'].astype(str)
df['Keyword'] = df['Keyword'].astype(str)

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

def tag_keywords(all_keywords):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([all_keywords])
	all_keywords = [i for i in tokenizer.word_index.keys()]
	all_keywords = list(set(all_keywords))
	return all_keywords

# sentence cleaning
df['Sentence'] = df['Sentence'].apply(lambda x: x.replace(" – TechCrunch",""))
df['Keyword'] = df['Keyword'].apply(lambda x: tag_keywords(x))

sentence_column = []
keyword_column = []
for index, row in df.iterrows():
	new_keywords = []
	sentence = row['Sentence']
	keywords = row['Keyword']
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([sentence])
	tokens = [i for i in tokenizer.word_index.keys()]
	for i in tokens:
		if i in keywords:
			if not hasNumbers(i):
				new_keywords.append(1)
		else:
			new_keywords.append(0)
	if sum(new_keywords) != 0:
		sentence_column.append(sentence)
		keyword_column.append(new_keywords)


tokenizer = Tokenizer(oov_token = "<UNK>")
tokenizer.fit_on_texts(sentence_column)
X = tokenizer.texts_to_sequences(sentence_column)
X = pad_sequences(X, padding = "post", truncating = "post", maxlen = 30, value = 0)
y = pad_sequences(keyword_column, padding = "post", truncating = "post", maxlen = 30, value = 0)
y = [to_categorical(i, num_classes = 2) for i in y]
embeddings_index = {}
# e_dat is an embedding file(glove)
f = open('e_dat.txt','r')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype = "float32")
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
# Model creation
model = Sequential()
model.add(Embedding(len(word_index) + 1, 100, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(128, return_sequences = True, recurrent_dropout = 0.3)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128, return_sequences = True, recurrent_dropout = 0.1)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(2, activation = "softmax")))
model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train, np.array(y_train), batch_size = 32, epochs = 5, validation_split = 0.1)

# model save
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
pickle.dump(tokenizer, open("tokenizer.pickle","wb"))
print("Saved model to disk")

test_output = model.predict(X_test)
test_output = np.argmax(test_output, axis = -1)


flattened_actual = (np.argmax(np.array(y_test), axis = -1)).flatten()
flattened_output = test_output.flatten()

print(classification_report(flattened_actual, flattened_output))


try:
	while True:
		input_ = input("Type in a headline:")
		new_t = Tokenizer()
		new_t.fit_on_texts([input_])
		tokens = [i for i in new_t.word_index.keys()]
#		print(tokens)
		actual_tokens = new_t.texts_to_sequences([input_])
		inv_map_tokens = {v: k for k, v in new_t.word_index.items()}
		actual_tokens = [inv_map_tokens[i] for i in actual_tokens[0]]
		tokens = actual_tokens
		input_ = tokenizer.texts_to_sequences([input_])
		input_ = pad_sequences(input_, padding = "post", truncating = "post", maxlen = 25, value = 0)
		output = model.predict([input_])
		output = np.argmax(output, axis = -1)
		where_ = np.where(output[0] == 1)[0]
		output_keywords = np.take(tokens, where_)
		output_keywords = [i for i in output_keywords if i not in stop_words]
		output_keywords = list(set(output_keywords))
#		print(tokens)
#		print(output)
		print(output_keywords)
except KeyboardInterrupt:
	pass
