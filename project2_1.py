file_path='aitest.txt'
clean_file_path='aiclean.txt'
text_file=open(file_path,'r',encoding='utf-8')


data=text_file.read()
print("text file stored into string")
print(data)

 
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize #for tokenizing words and sentences
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import pandas as pd



import string
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('FreqDist')
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.wsd import lesk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pylab import rcParams
import re
from collections import Counter #Counter used for getting frequencies of elements


def pre_process_txt(f_s):
    #f_s=f_s.lower()
    remove_chap="[cC]hapter [0-9]+"
    book=""
    for line in f_s:
        book+=line
    book=re.sub('[^A-Za-z0-9 \n.(){}\[\]\'\"]+', '', book)
    book=re.sub(' [0-9]+ ','',book)
    book=book.replace("\n",". ")
    book=book.replace('“','"')
    book=book.replace('”','"')
    book=book.replace('—'," ")
    book=book.replace('('," ")
    book=book.replace(')'," ")
    book=book.replace('['," ")
    book=book.replace(']'," ")
    book=book.replace('{'," ")
    book=book.replace('}'," ")
    book=book.replace('_'," ")
    book=book.replace(". .",". ")
    book=book.replace('..','. ')
    book=re.sub(remove_chap,'',book)
    book = re.sub(' +',' ',book)  #For handling multiple spaces
    
    return book


def Noun_Verb_Category(f_in):
    '''This function does word sense disambiguation and
    stores the category of wordnet to which it belongs'''

    book=""
    for line in f_in:
        book+=line
    
    remove_let ="[lL]etter [0-9]+"
    remove_chap="[cC]hapter [0-9]+"
    remove_nums="[0-9]+"
    book=re.sub(remove_let,'',book)
    book=re.sub(remove_chap,'',book)
    book=re.sub(remove_nums,'',book)
    
    tokenized = sent_tokenize(book)
    noun_list = []
    verb_list = []
    for w in tokenized:
        words = nltk.word_tokenize(w)
        tagged = nltk.pos_tag(words)
        for tag in tagged:
            if tag[1][0] == 'N':
                lex = lesk(words,tag[0],'n')
                if lex:
                    noun_list.append(lex.lexname())
            elif tag[1][0] == 'V':
                lex = lesk(words,tag[0],'v')
                if lex:
                    verb_list.append(lex.lexname())

    return noun_list, verb_list
print("noun list verb list obtained")

def plot_Category(txt):
    '''This function plots frequency of each category
    of noun and verb'''
    noun_list,verb_list = Noun_Verb_Category(txt)
    count_n=Counter(noun_list)
    count_v=Counter(verb_list)
    
    df_n=pd.DataFrame.from_dict(count_n,orient='index')
    ax_n=df_n.plot(kind='bar',title='Distribution of Noun types')
    ax_n.set_xlabel("Noun types")
    ax_n.set_ylabel("Count")
    plt.show()
    print("\n")
    df_v=pd.DataFrame.from_dict(count_v,orient='index')
    ax_v=df_v.plot(kind='bar',title='Distribution of Verb Types')
    ax_v.set_xlabel("Verb types")
    ax_v.set_ylabel("Count")
    plt.show()
    
#Noun and Verb Distribution of Frankenstein
rcParams['figure.figsize'] = 15, 15
plot_Category(data)
print("end")



