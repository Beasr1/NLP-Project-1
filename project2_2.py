import nltk
from nltk.tokenize import word_tokenize,sent_tokenize #for tokenizing words and sentences
nltk.download('punkt') #Punkt Sentence Tokenizer
nltk.download('stopwords')  # list of stopwords by nltk
nltk.download('averaged_perceptron_tagger') # pos tagger by nltk
import string
from nltk.corpus import stopwords 
from collections import Counter #Counter used for getting frequencies of elements
import re
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pylab import rcParams
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import seaborn as sns
import en_core_web_lg
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def ner_preprocess(filename):
    '''Used to preprocess the file for 
    Named entity recognition and performs named entity recognition
    Returns a dataset consisting of entity name and entity type'''
    text_file=open(filename,'r',encoding='utf-8')

    #data of txt file is extracted into string
    f_s=text_file.read()
    f_s=f_s.lower()
    remove_chap="[cC]hapter [0-9]+"
    book=""
    for line in f_s:
        book+=line
    book=re.sub('[^A-Za-z0-9 \n.(){}*-/+%\[\]\'\"]+', '', book)
    book=re.sub(' [0-9]+ ','',book)
    book=book.replace("\n",". ")
    book=book.replace('“','"')
    book=book.replace('”','"')
    book=book.replace('—'," ")
    book=book.replace('_'," ")
    book=book.replace(". .",". ")
    book=book.replace("..",". ")
    book=re.sub(remove_chap,'',book)
    book = re.sub(' +',' ',book)  #For handling multiple spaces
    
    lis=sent_tokenize(book)

    print("BOOK : ",book)
    print("Token : ",lis)
    ent_name=[]
    ent_type=[]
    for sent in lis:
      labl=mdl(sent)
      #print(labl)
      for e in labl.ents:
        #print(e,"\t",e.label_)
        if e.label_=='ORG' or e.label_=='PERSON' or e.label_=='LOC' or e.label_=='GPE' or e.label_=='NORP' or e.label_=='FAC':
              ent_name.append(e.text)
              if(e.label_ not in ['ORG','PERSON','LOC']):
                if(e.label_ == 'NORP'):
                  ent_type.append('ORG')
                elif e.label_=='GPE' or e.label_=='FAC':
                  ent_type.append('LOC')
              else:
                ent_type.append(e.label_)

    lis=list(zip(ent_name,ent_type))
    return pd.DataFrame(lis,columns=['Entity_Name','Entity_Type'])

def confusion_matrix_test(test_pred):
    labDic=[
        ['alice','PERSON'],
        ['bob','PERSON'],
        ['oscar','PERSON'],
        ['european','ORG'],
        ['british','ORG'],
        ['german','ORG'],
        ['alan turing','PERSON'],
        ['german','ORG'],
        ['oscar','PERSON'],
        ['egypt','LOC'],
        ['julius caesar','PERSON'],
        ['the white house','ORG'],
        ['kremlin','ORG'],
        ['alice','PERSON'],
        ['alice','PERSON'],
        ['alice','PERSON'],
        ['bob','PERSON'],
        ['ecdsa','ORG'],
        ['dsa','ORG'],
        ['rsa','PERSON'],
        ['ecdsa','ORG'],
        ['us','LOC'],
        ['american','ORG'],
        ['united states of america','LOC'],
        ['rsa','ORG'],
        ['ecdsa','ORG'],
        ['ecdsa','PERSON'],
        ['rsa','ORG'],
        ['rsa','ORG'],
        ['dsa','ORG'],
        ['ecdsa','ORG'],
        ['rsa','ORG'],
        ['rsa','ORG'],
        ['dsa','ORG'],
        ['rsa','ORG'],
        ['ecdsa','ORG'],
        ['aes','ORG']
    ]
    true_lab=pd.DataFrame(labDic,columns=['Entity_Name','Entity_Type'])
    print("EntityMine",true_lab)

    print(classification_report(true_lab['Entity_Type'],test_pred['Entity_Type']))
    print('Confusion Matrix: \n',confusion_matrix(true_lab['Entity_Type'],test_pred['Entity_Type']))
 
    
mdl=en_core_web_lg.load()

#Entity dataset
file_path="aitest.txt"
ent=ner_preprocess(file_path)
print("Size of entity dataset:",ent.shape)
print("ENTITIES : ",ent)

#Distribution of Entity tags of Frankenstein
ax_f=ent['Entity_Type'].value_counts().plot(kind='bar',figsize=(10,10),title='Entity Tag distribution for BOOK')
ax_f.set_xlabel('Entity_Tags')
ax_f.set_ylabel('Count')
plt.show()


#Entity dataset for random passages stored within extract_text
test_path="extract_text.txt"
entest=ner_preprocess(test_path)
print("Size of entity dataset:",entest.shape)
print("ENTITIES : ",entest)

confusion_matrix_test(entest)
