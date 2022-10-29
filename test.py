#absolute file paths of our book's text file
file_path="C:/Users/lenovo/OneDrive/Desktop/nlp_project/aitest.txt"
clean_file_path="C:/Users/lenovo/OneDrive/Desktop/nlp_project/aiclean.txt"
text_file=open(file_path,'r',encoding='utf-8')

#data of txt file is extracted into string
data=text_file.read()
print("text file stored into string")
print(data)

# removes everything except
# A-Z a-z 0-9 [space character] [newLine] [.] [] () {}
# +-*/ maths
import re
import string
clean_data=re.sub('[^A-Za-z0-9 \n.(){}*-/+%\[\]\'\"]+', '', data)
clean_data=clean_data.lower() #makes all uppercase into lowercase 
print("preprocessing done")
print(clean_data)
clean_file=open(clean_file_path,"w")
clean_file.write(clean_data) #stores clean data into a new text file
 
import nltk
nltk.download('punkt') #punkt slices string into relevant word lists
tokens = nltk.word_tokenize(clean_data)
print(tokens)

#removing punctuations from every tokens
regex = re.compile('[%s]' % re.escape(string.punctuation))
new_review=[]
for token in tokens:
    new_token = regex.sub(u'', token)
    if not new_token == u'':
        new_review.append(new_token)
print(new_review)
tokens=new_review

import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('FreqDist')
from nltk.probability import FreqDist
from nltk.corpus import stopwords

#removing stopwords
#stopwords are, frequent letters/words of english alphabet, punctuations and digits
stop_words = set(stopwords.words('english') + list(string.punctuation) + list(string.digits) + list("abcdefghijklmnopqrstuvwxyz"))
filtered_tokens = [w for w in tokens if not w in stop_words]
print(filtered_tokens)

#without filtering : frequency distribution
fdist_filtered = FreqDist(tokens)
fdist_filtered.plot(30,title='Frequency distribution for 30 most common tokens in our text collection (including stopwords and punctuation)')

# filtering : frequency distribution
fdist_filtered = FreqDist(filtered_tokens)
fdist_filtered.plot(30,title='Frequency distribution for 30 most common tokens in our text collection (excluding stopwords and punctuation)')



#removing stopwords : WORDCLOUD
from collections import Counter
from wordcloud import WordCloud
dictionary_filtered=Counter(filtered_tokens)
cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(dictionary_filtered)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#without removing stopwords : WORDCLOUD
dictionary=Counter(tokens)
cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(dictionary)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#pos tagging : TREEBANK
nltk.download('averaged_perceptron_tagger')
tagged=nltk.pos_tag(filtered_tokens)
print(tagged)

print(dictionary)

import numpy as np
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

