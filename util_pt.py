import textacy
import re
import random
from textacy.extract import named_entities
import textacy.keyterms
from bert_embedding import BertEmbedding
import multiprocessing
manager = multiprocessing.Manager()
shared_list = manager.list()


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



def kwd(sentence):

    try:
        txt_doc = textacy.Doc(sentence, lang="en_core_web_sm")
        kwds_sgrank = textacy.keyterms.sgrank(txt_doc, ngrams=(1,2), n_keyterms=3)
        kwds_sgrank = [kwd[0] for kwd in kwds_sgrank]
        kwds_sgrank_str = ', '.join(kwds_sgrank)
        kwds_textrank = textacy.keyterms.textrank(txt_doc)
        kwds_txtrnk = [kwd[0] for kwd in kwds_textrank]
        kw_uni = []
        for kw in kwds_txtrnk:

            if kw not in kwds_sgrank_str:
                kw_uni.append(kw)

        kwds = kwds_sgrank + kw_uni
        l_kwds = [kwd for kwd in kwds]
        # remove numbers from keywords
        l_kwds = [kwd for kwd in l_kwds if not any(str(v).isdigit() for v in kwd)]
        # removing very small keywords (to reduce noise)
        l_kwds = [kwd for kwd in l_kwds if len(kwd)>3]
        kwds = l_kwds

    except:
        kwds = []

    return kwds




