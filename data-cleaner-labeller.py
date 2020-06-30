import re
import textacy
import tqdm
import random
import pandas as pd
import textacy.keyterms
from nltk.tokenize import TweetTokenizer
from multiprocessing import Pool, TimeoutError

from bert_embedding import BertEmbedding
import multiprocessing

manager = multiprocessing.Manager()
shared_list = manager.list()

# -------------------------------------utility functions---------------------------------#

'''
input args : These function takes list of sentences as input
'''

def data_cleaner(l_sentences):
    # clearns the sentences by removing special character
    # extra spaces within setences

    l_cleaned_sentence = []
    for sentence in l_sentences:
        sentence = str(sentence)
        # This takes a sentence, removes special characters, punctuation etc from wikipedia sentence.
        wiki_sentence = re.sub("[\(\[].*?[\)\]]", "", sentence)
        wiki_sentence = re.sub("  ", " ", wiki_sentence)
        wiki_sentence = wiki_sentence.strip('==')
        l_cleaned_sentence.append(wiki_sentence)

    return l_cleaned_sentence


def label_kwd(sentence):

    try:
        txt_doc = textacy.Doc(sentence, lang="en_core_web_sm")
        kwds_sgrank = textacy.keyterms.sgrank(txt_doc, ngrams=(1,2))
        kwds_sgrank = [kwd[0] for kwd in kwds_sgrank]
        kwds_sgrank_str = ', '.join(kwds_sgrank)
        kwds_textrank = textacy.keyterms.textrank(txt_doc)
        kwds_txtrnk = [kwd[0] for kwd in kwds_textrank]
        kw_uni = []
        for kw in kwds_txtrnk:

            if kw not in kwds_sgrank_str:
                kw_uni.append(kw)
        kwds = kwds_sgrank + kwds_txtrnk
        l_kwds = [kwd for kwd in kwds]
        # remove numbers from keywords
        l_kwds = [kwd for kwd in l_kwds if not any(str(v).isdigit() for v in kwd)]
        # removing very small keywords (to reduce noise)
        l_kwds = [kwd for kwd in l_kwds if len(kwd)>3]
        kwds = ','.join(l_kwds)
        final_data = sentence + '\t' + kwds + '\n'

    except:
        final_data = ''

    print(final_data)
    print()
    return final_data


# ------------------------------------- Steps for cleaning -----------------------#

# Data loaded
df = pd.read_csv('wiki-collector-data.csv', delimiter='\t', names=['Sentence', 'Keywords'])

# drop nltk - rake keywords as later I thought to use 
# other method(SGRank) for obtaining label keyword
df = df.drop(['Keywords'], axis=1)

# get the sentences
sentences = [sentences_list[0] for sentences_list in df.values]

tknzr = TweetTokenizer()
# filter short sentences
sentences = [sentence for sentence in sentences if len(tknzr.tokenize(str(sentence))) >4]


# clean the sentences
sentences = data_cleaner(sentences)

# get data with labelled keyword
# labelled_data = label_kwd(sentences)

# multiprocessing
with Pool(processes=4) as pool:
    final_data = pool.map(label_kwd, sentences, chunksize=20)
    if not final_data:
        final_data = final_data.rstrip('\n')

# final data ( lablled and cleaned )
with open('dat_l.csv', 'w') as data_file:
    data_file.writelines(final_data)

