import nltk
from nltk.tokenize import word_tokenize,sent_tokenize #for tokenizing words and sentences
import string
from nltk.corpus import stopwords 
from collections import Counter #Counter used for getting frequencies of elements
import re #used in re.sub
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pylab import rcParams
import seaborn as sns
import en_core_web_lg
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt') #Punkt Sentence Tokenizer
nltk.download('stopwords')  # list of stopwords by nltk
nltk.download('averaged_perceptron_tagger') # pos tagger by nltk
nltk.download('omw-1.4')

#Extract string data from filePath
def extractString(filename):
    text_file=open(filename,'r',encoding='utf-8')
    data=text_file.read()
    return data

#used for text preprocessing
def preprocess_txt(data):
    #data=data.lower()
    remove_chap="[cC]hapter [0-9]+"
    dataClean=""
    for line in data:
        dataClean+=line
    dataClean=re.sub('[^A-Za-z0-9 \n.(){}*-/+%\[\]\'\"]+', '',  dataClean)
    dataClean=re.sub(' [0-9]+ ','',dataClean)
    dataClean=dataClean.replace("\n",". ")
    dataClean=dataClean.replace('“','"')
    dataClean=dataClean.replace('”','"')
    dataClean=dataClean.replace('—'," ")
    dataClean=dataClean.replace('('," ")
    dataClean=dataClean.replace(')'," ")
    dataClean=dataClean.replace('['," ")
    dataClean=dataClean.replace(']'," ")
    dataClean=dataClean.replace('{'," ")
    dataClean=dataClean.replace('}'," ")
    dataClean=dataClean.replace('_'," ")
    dataClean=dataClean.replace(". .",". ")
    dataClean=dataClean.replace('..','. ')
    dataClean=re.sub(remove_chap,'',dataClean)
    dataClean = re.sub(' +',' ',dataClean)  #For handling multiple spaces
    
    return dataClean

#PART 1
#---------------------------------------------------------------------------------------------------------------------------------
#PART 1 : Stores category of wordnet to which it belongs
def Noun_Verb_Category(data):
    dataClean=preprocess_txt(data)
    
    tokenized = sent_tokenize(dataClean)
    nouns= []
    verbs= []
    for w in tokenized:
        words = nltk.word_tokenize(w)
        tagged = nltk.pos_tag(words)
        for tag in tagged:
            if tag[1][0] == 'N':
                lex = lesk(words,tag[0],'n')
                if lex:
                    nouns.append(lex.lexname())
            elif tag[1][0] == 'V':
                lex = lesk(words,tag[0],'v')
                if lex:
                    verbs.append(lex.lexname())

    return nouns, verbs

# PART 1 : Plots frequency of each category of Nouns and Verbs
def plot_Category(data):

    nouns,verbs = Noun_Verb_Category(data)
    count_nouns=Counter(nouns)
    count_verbs=Counter(verbs)

    #PLOTTING Distribution of Noun TYPES
    df_nouns=pd.DataFrame.from_dict(count_nouns,orient='index')
    ax_nouns=df_nouns.plot(kind='bar',title='Distribution of Noun types')
    ax_nouns.set_xlabel("Noun types")
    ax_nouns.set_ylabel("Count")
    plt.show()
    
    #PLOTTING Distribution of Verb TYPES
    df_verbs=pd.DataFrame.from_dict(count_verbs,orient='index')
    ax_verbs=df_verbs.plot(kind='bar',title='Distribution of Verb Types')
    ax_verbs.set_xlabel("Verb types")
    ax_verbs.set_ylabel("Count")
    plt.show()
#--------------------------------------------------------------------------------------------------------------------------------
#PART 2
#---------------------------------------------------------------------------------------------------------------------------------
#PART 2 : Named entity recognition and performs named entity recognition Returns a dataset consisting of entity name and entity type'''
def ner_preprocess(data):


    data=data.lower()
    dataClean=preprocess_txt(data)
    
    tokenized=sent_tokenize(dataClean)

    print("BOOK (Preprocessed) : ",dataClean)
    print("Token : ",tokenized)
    entity_name=[]
    entity_type=[]
    for sent in tokenized:
      labl=mdl(sent)
      for e in labl.ents:
        if e.label_=='ORG' or e.label_=='PERSON' or e.label_=='LOC' or e.label_=='GPE' or e.label_=='NORP' or e.label_=='FAC':
              entity_name.append(e.text)
              if(e.label_ not in ['ORG','PERSON','LOC']):
                if(e.label_ == 'NORP'):
                  entity_type.append('ORG')
                elif e.label_=='GPE' or e.label_=='FAC':
                  entity_type.append('LOC')
              else:
                entity_type.append(e.label_)

    list_EntityNameType=list(zip(entity_name,entity_type))
    return pd.DataFrame(list_EntityNameType,columns=['Entity_Name','Entity_Type'])

#PART 2 : Entity dataset for random passages stored within extract_text: generate classification report and confusion matrix
def confusion_matrix_test(test_prediction):
    labDict=[
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
    manual=pd.DataFrame(labDict,columns=['Entity_Name','Entity_Type'])
    print("True/Actulal Entity Name and Type manually coded into")
    print(manual)

    print("\n\nClassification Report : ")
    print(classification_report(manual['Entity_Type'],test_prediction['Entity_Type']))
    print('\n\nConfusion Matrix: \n')
    print(confusion_matrix(manual['Entity_Type'],test_prediction['Entity_Type']))
    
#---------------------------------------------------------------------------------------------------------------------------------
#PART 3
#---------------------------------------------------------------------------------------------------------------------------------
#extracts relationship b/w Entities based on Given Relationship Types
def ne_rel(data):
  dataClean=preprocess_txt(data)
  entity_label=[]
  entity_name=[]
  entity_start=[]
  entity_end=[]
  sent_list=sent_tokenize(dataClean)

  bef=0
  for sent in sent_list:
    labl=mdl(sent) #mdl
    for e in labl.ents:
      if e.label_=='PERSON' or e.label_=='ORG' :
        entity_label.append(e.label_)
        entity_name.append(e.text)
        entity_start.append(e.start_char + bef)
        entity_end.append(e.end_char + bef)
    bef= bef+ len(sent)+1

  entity_startend=list(zip(entity_name,entity_start,entity_end))
  entity_startend.sort(key=lambda x:x[1])
  
  pat_list=[r'.*\b[Ss]ynonym\b.*',r'.*\b[Aa]lternative\b.*',r'.*\b[Aa]cronymn\b.*',r'.*\b[Ff]ather\b.*',r'.*\b[Ss]ister\b.*',r'.*\b[Mm]other\b.*',r'.*\b[Cc]ousin\b.*',r'.*\b[Uu]ncle\b.*',r'.*\b[Ww]ife\b.*',r'.*\b[Hh]usband\b.*',r'.*\b[Aa]unt\b.*']
  type_list=["Synonym",'Alternative','Acronymn','Father','Sister','Mother','Cousin','Uncle','Wife','Husband','Aunt']
  relation_list=list(zip(pat_list,type_list))

  print("\n\nEntity Name, Start, End : \n",entity_startend)
  print("\n\nLIST_relation : \n",relation_list)

  print("\nRelations in Book :\n") 
  for i,tup1 in enumerate(entity_startend[:-1]):
    tup2=entity_startend[i+1]
    if tup2[1]-tup1[2]<=100 and str(tup1[0])!=str(tup2[0]):

      sel_text=dataClean[tup1[1]:tup2[2]+1]
      for relation in relation_list:
        reg="\\b"+tup1[0]+"\\b"+relation[0]+"\\b"+tup2[0]+"\\b"
        if re.match(reg,sel_text):
          print(relation[1]+" REL between",tup1[0],"and",tup2[0]+":")
          print(sel_text)
          print("\n")
          pass    
#--------------------------------------------------------------------------------------------------------------------------------
mdl=en_core_web_lg.load()    
data=extractString('aitest.txt')
test_data=extractString("extract_text.txt")

# PART 1 : Noun and Verb Distribution 
rcParams['figure.figsize'] = 15, 15
plot_Category(data)

# PART 2 :Entity set , Precesion and Confusion Matrix
#Recognise all entities
entities=ner_preprocess(data)
print("Size of entity dataset: ",entities.shape)
print("ENTITIES : \n",entities)
print("\n\n")

#Distribution of Entity tags 
ax=entities['Entity_Type'].value_counts().plot(kind='bar',figsize=(10,10),title='Entity Tag distribution for BOOK')
ax.set_xlabel('Entity_Tags')
ax.set_ylabel('Count')
plt.show()


#Entity dataset for random passages stored within extract_text

entity_test=ner_preprocess(test_data)
print("Size of entity dataset:",entity_test.shape)
print("ENTITIES processed from test_data extracted from BOOK (Prediction) : \n",entity_test)

confusion_matrix_test(entity_test)

#PART 3:Entity Relationship
text=extractString("text.txt")
ne_rel(text)


print("/nEND")
