from __future__ import print_function
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pos
def Decision_take(data):
	Negocount=0
	Acccount=0
	nego=[]
	posi=[]
	negotiation=open("words/negative.txt","r")
	accepted=open("words/positive.txt","r")
	ReadLinenego=negotiation.readlines()
	ReadLineAcc=accepted.readlines()
	
	for i in ReadLinenego:
		nego.append(i[:-1])
	for i in ReadLineAcc:
		posi.append(i[:-1])
	for i in data:
		if i in nego:
			Negocount=Negocount+1
		if i in posi:
			Acccount=Acccount+1
	x=['Acceptance','Negotation/decline']
	y=[Acccount,Negocount]
	x_pos = [i for i, _ in enumerate(x)]
	plt.xlabel("Success Rate")
	plt.ylabel("Count Frequency")
	plt.bar(x_pos, y, color='green')
	plt.xticks(x_pos, x)
	plt.show()
	return Acccount,Negocount
			
		
def SentenceBasedAnalysis(datatext):
	Sent_tokens=sent_tokenize(datatext)
	sid = SentimentIntensityAnalyzer()
	negcount=0
	posicount=0
	neucount=0
	count=0;
	temp=0
	for sentence in Sent_tokens:
		 #print(sentence)
		 temp=temp+1
		 ss = sid.polarity_scores(sentence)
		 for k in ss:
				#count=count+1
				if k=="neg":
					negcount=negcount+(ss[k])
				elif k=="neu":
					neucount=neucount+(ss[k])
				elif k=="pos":
					posicount=posicount+(ss[k])
				print('{0}: {1}, '.format(k, ss[k]), end='')
		 print()
	return (negcount/temp),(posicount/temp),(neucount/temp)
def SenTextBlob(datatext):
	sent=sent_tokenize(datatext)
	sentimentvalue=[]
	temp=0
	for i in sent:
		#print(i);
		temp=temp+1
		analysis=TextBlob(i)
		sentemp=analysis.sentiment.polarity
		if(sentemp>0):
			#print("positive")
			sentimentvalue.append("positive")
		if(sentemp==0):
			#print("neutral")
			sentimentvalue.append("neutral")
		else:
			#print("negative")
			sentimentvalue.append("negative")
		
	#print(sentimentvalue)
	posicount=sentimentvalue.count("positive")
	negcount=sentimentvalue.count("negative")
	neucount=sentimentvalue.count("neutral")
	result={'positive':posicount,'negative':negcount,'neutral':neucount}
	labels,values=zip(*result.items())
	# sort your values in descending order
	indSort = np.argsort(values)[::-1]

	# rearrange your data
	labels = np.array(labels)[indSort]
	values = np.array(values)[indSort]

	indexes = np.arange(len(labels))

	bar_width = 0.35

	plt.bar(indexes, values)
	# add labels
	plt.xticks(indexes + bar_width, labels)
	plt.show()
	return posicount,negcount,neucount,temp
def AdjectivePolarity(sample):
	adjpol=[]
	sam=word_tokenize(sample)
	tags=nltk.pos_tag(sam)
	TagValue=['RB','RBR','RBS','JJS','JJ','JJR']
	for i in tags:
		if i[1]  in TagValue:
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
def CleaningStopWords(Person1):
	filtered_data=[]
	stop_words = set(stopwords.words('english'))
	word_tokens=word_tokenize(Person1)
	for word in word_tokens:
		if word not in stop_words:
			filtered_data.append(word)
	return filtered_data
def StemmingWords(filteredText):
	lemmatizer=WordNetLemmatizer()
	stemmedData=[]
	words=['I','.',',','u']
	for i in filteredText:
		if i not in words:
			stemmedData.append(lemmatizer.lemmatize(i))
	return stemmedData
def Frequency_count(countdata):
	counts=dict(Counter(countdata).most_common(10))
	labels, values = zip(*counts.items())
	keys=[]
	keyword_count=[]
	for key in counts:
		keys.append(key)
	# sort your values in descending order
	indSort = np.argsort(values)[::-1]

	# rearrange your data
	labels = np.array(labels)[indSort]
	values = np.array(values)[indSort]

	indexes = np.arange(len(labels))

	bar_width = 0.35

	plt.bar(indexes, values)

	# add labels
	plt.xticks(indexes + bar_width, labels)
	plt.show()
	for i in keys:
		analysis=TextBlob(i)
		sentemp=analysis.sentiment.polarity
		if (sentemp==0):
			keyword_count.append([i,"neutral"])
		elif (sentemp>0):
			keyword_count.append([i,"positive"])
		else:
			keyword_count.append([i,"negative"])
	return keyword_count
			
def PreprocessData(Speaker1,Speaker2,Others):
	#convert list to string
	Speaker1Text=""
	Speaker2Text=""
	for sen in Speaker1:
		Speaker1Text=Speaker1Text+sen
	for sen in Speaker2:
		Speaker2Text=Speaker2Text+sen
	CleanedSpeaker1=CleaningStopWords(Speaker1Text)
	CleanedSpeaker2=CleaningStopWords(Speaker2Text)
	StemmedData1=StemmingWords(CleanedSpeaker1)
	StemmedData2=StemmingWords(CleanedSpeaker2)
	print("Speaker 1 frequency count:")
	countspeak1=Frequency_count(StemmedData1)
	print(countspeak1)
	print("\nSpeaker 2 frequency count:")
	countspeak2=Frequency_count(StemmedData2)
	print(countspeak2)
	print("Determining based on Adjective and adverb of Speaker 1:")
	Adjadv1=AdjectivePolarity(Speaker1Text)
	print(Adjadv1)
	print("Determining based on Adjective and adverb of Speaker 2:")
	Adjadv2=AdjectivePolarity(Speaker2Text)
	print(Adjadv2)
	Positive1,Negative1=Decision_take(StemmedData1)
	print(Positive1,Negative1)
	Positive2,Negative2=Decision_take(StemmedData2)
	print(Positive2,Negative2)
	SentSpeaker1=SentenceBasedAnalysis(Speaker1Text)
	print(SentSpeaker1)
	SentSpeaker2=SentenceBasedAnalysis(Speaker2Text)
	print(SentSpeaker2)
	SentTextBlob1=SenTextBlob(Speaker1Text)
	SentTextBlob2=SenTextBlob(Speaker2Text)
def SeparateSpeakers(Data):
	Speaker1=[]
	Speaker2=[]
	Others=[]
	for i in Data:
		if i.startswith("Speaker 0"):
			Speaker1.append(i[10:-1])
		if i.startswith("Speaker 1"):
			Speaker2.append(i[10:-1])
		else:
			Others.append(i[10:-1])
	PreprocessData(Speaker1,Speaker2,Others)
def OpenFile(name):
	Datafile=open("Text/"+name,'r')
	Datalist=Datafile.readlines()
	SeparateSpeakers(Datalist)

	
name=raw_input("Enter the file name:")
OpenFile(name)

	