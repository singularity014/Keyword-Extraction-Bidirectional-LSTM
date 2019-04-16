import wptools
import wikipedia
from rake_nltk import Rake

wikipedia.set_lang("en")

def key_word_extract(text):

	r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
	r.extract_keywords_from_text(text)
	keywords = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.	
	
	for keyword in keywords:
		for chr in keyword:
			if str.isdigit(chr):
				keywords.remove(keyword)
				break
				
				
	keywords = ','.join(keywords)
	data = text +'\t'+keywords+'\n'
		
	return data


for i in range(40000):
	fil = open('Wiki-keyword-data', 'a')
	x = wptools.page(lang='en', silent=True)
	title = str(x.data['title'])
	
	try:
		pg = wikipedia.search(title)[0]
		text = wikipedia.summary(pg, sentences=1)	
		data = key_word_extract(text)
		print(data)
		fil.write(data)
		print()
		#print(y)
		#print()
		#print('---------------------------')
	
	except:
		fil.close()
		print('Error - key not found')
		print()
		print('----------------------------')
				
'''	

def key_word_extract(text):

	r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
	r.extract_keywords_from_text(text)
	keywords = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.	
	
	for keyword in keywords:
		for chr in keyword:
			if str.isdigit:
				keywords.remove(keyword)
				break
				
				
	keywords = ','.join(keywords)
	data = text +'	'+keywords
	
	return data


for i in range(30000):
	x = wptools.page(lang='en', silent=True)
	title = str(x.data['title'])
	
	pg = wikipedia.search(title)[0]
	text = wikipedia.summary(pg, sentences=1)
       		
	data = key_word_extract(text)
	print(data)
	print()
	#print(y)
	#print()
	#print('---------------------------')
'''