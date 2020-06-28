##------------------------------------ UTILITY PACKAGES --------------------------------------------------- ##
import pickle
import random
from nltk.corpus import stopwords
from util_pt import kwd
stop_words = set(stopwords.words("english"))
import numpy as np
import os
import argparse

##--------------------------------------- TENSORFLOW ----------------------------------------------------##
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

##------------------------------------------ CONFIGS ------------------------------------------------------##
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
tokenizer = pickle.load(open("tokenizer.pickle","rb"))
final_results = []

##------------------------------------------------------------------------------------------------------##
with open('test-sentences.txt') as fil:
	lines = fil.readlines()
	for line in lines:
			input_ = line
			new_t = Tokenizer()
			new_t.fit_on_texts([input_])
			tokens = [i for i in new_t.word_index.keys()]
			actual_tokens = new_t.texts_to_sequences([input_])
			inv_map_tokens = {v: k for k, v in new_t.word_index.items()}
			actual_tokens = [inv_map_tokens[i] for i in actual_tokens[0]]
			tokens = actual_tokens
			txt_kd = kwd(line)
			input_ = tokenizer.texts_to_sequences([input_])
			input_ = pad_sequences(input_, padding = "post", truncating = "post", maxlen = 25, value = 0)
			output = model.predict([input_])
			output = np.argmax(output, axis = -1)
			where_ = np.where(output[0] == 1)[0]
			output_keywords = np.take(tokens, where_)
			output_keywords = txt_kd
			output_keyword = [i for i in output_keywords if i not in stop_words] + txt_kd
			output_keywords = list(set(output_keywords))
			dt = line +'keywords-> ' + ', '.join(output_keywords) + '\n\n'
			final_results.append(dt)


# Storing the output
with open('final_results.csv', 'w') as fil:
	fil.writelines(final_results)

##------------------------------------------------------------------------------------------------------##
