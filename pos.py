from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
def AdjectivePolarity(sample):
	sam=word_tokenize(sample)
	tags=nltk.pos_tag(sam)
	adjpol=[]
	TagValue=['RB','RBR','RBS','JJS','JJ','JJR']
	for i in tags:
		if i[1] not in TagValue:
			print(i)
			analysis=TextBlob(i[0])
			sentemp=analysis.sentiment.polarity
			#print(sentemp)
			if sentemp>0:
				adjpol.append("positive")
			elif sentemp<0:
				adjpol.append("negative")
			else:
				adjpol.append("neutral")
	return adjpol