#Required libraries and tokenizers
import re
import string
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import numpy as np

#downloading tokenizers
nltk.download('punkt') #punkt slices string into relevant word lists
nltk.download('stopwords')
nltk.download('FreqDist')
nltk.download('averaged_perceptron_tagger')

#--------------------------------------------------------------------------------------------------------------

#absolute file paths of our book's text file
file_path="C:/Users/lenovo/OneDrive/Desktop/NLP-Project-1/aitest.txt"
clean_file_path="C:/Users/lenovo/OneDrive/Desktop/NLP-Project-1/aiclean.txt"
text_file=open(file_path,'r',encoding='utf-8')

#data of txt file is extracted into string
data=text_file.read()
print("text file stored into string")
print(data)

#Preprocessing of string
# removes everything except
# A-Z a-z 0-9 [space character] [newLine] [.] [] () {}
# +-*/ maths
clean_data=re.sub('[^A-Za-z0-9 \n.(){}*-/+%\[\]\'\"]+', '', data)
clean_data=clean_data.lower() #makes all uppercase into lowercase 
print("preprocessing done : Processed string")
print(clean_data)
clean_file=open(clean_file_path,"w")
clean_file.write(clean_data) #stores clean data into a new text file
 

#tokenization
tokens = nltk.word_tokenize(clean_data)
print("Tokens")
print(tokens)

#preprocessing of tokens
#removing punctuations from every tokens
regex = re.compile('[%s]' % re.escape(string.punctuation))
new_review=[]
for token in tokens:
    new_token = regex.sub(u'', token)
    if not new_token == u'':
        new_review.append(new_token)
print("Preprocessed tokens")
print(new_review)
tokens=new_review


#removing stopwords
#stopwords are, frequent letters/words of english alphabet, punctuations and digits
stop_words = set(stopwords.words('english') + list(string.punctuation) + list(string.digits) + list("abcdefghijklmnopqrstuvwxyz"))
filtered_tokens = [w for w in tokens if not w in stop_words]
print("Tokens after removing stopwords")
print(filtered_tokens)

#without filtering : frequency distribution
fdist_filtered = FreqDist(tokens)
fdist_filtered.plot(30,title='Frequency distribution for 30 most common tokens in our text collection (including stopwords and punctuation)')

# filtering : frequency distribution
fdist_filtered = FreqDist(filtered_tokens)
fdist_filtered.plot(30,title='Frequency distribution for 30 most common tokens in our text collection (excluding stopwords and punctuation)')

#without removing stopwords : WORDCLOUD
dictionary=Counter(tokens)
cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(dictionary)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#removing stopwords : WORDCLOUD
dictionary_filtered=Counter(filtered_tokens)
cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(dictionary_filtered)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#pos tagging : TREEBANK
tagged=nltk.pos_tag(filtered_tokens)
print("POS TAGGING")
print(tagged)

#BAR PLOT
# creating the dataset dictionary in acending order
data = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(courses[-10:], values[-10:], color ='maroon',
		width = 0.4)

plt.xlabel("tokens")
plt.ylabel("frequency")
plt.title("Bar plot of Frequency distribution")
plt.show()

