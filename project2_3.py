import re
from nltk.tokenize import sent_tokenize, word_tokenize
import en_core_web_lg

def pre_process_txt(f_s):
    '''Reads the text file to a string and preprocesses the text file
    by lowercasing, removing newlines, punctuation, chapter headers, 
    numbers and ordinal dates and does tokenization. Also removes stopwords'''
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

def ne_rel(f_s):
  '''Used to preprocess the file for Named entity relationship and detects relationship
  between PERSON entities.'''

  book=pre_process_txt(f_s)
  ent_label=[]
  ent_name=[]
  ent_start=[]
  ent_end=[]
  sent_list=sent_tokenize(book)

  print(book)
  print(sent_list)
  bef=0
  for sent in sent_list:
    labl=mdl(sent) #mdl
    #print(sent)
    #print(labl, "ents : ",labl.ents)
    for e in labl.ents:
      #print(e,"\t",e.label_)
      if e.label_=='PERSON' or e.label_=='ORG' :
        ent_label.append(e.label_)
        ent_name.append(e.text)
        ent_start.append(e.start_char + bef)
        ent_end.append(e.end_char + bef)
    #print(bef, "book:",book[bef])
    bef= bef+ len(sent)+1

  print(ent_label)
  lis=list(zip(ent_name,ent_start,ent_end))
  lis.sort(key=lambda x:x[1])
  print("LIS : ",lis)
  pat_list=[r'.*\b[Ss]ynonym\b.*',r'.*\b[Aa]lternative\b.*',r'.*\b[Ff]ather\b.*',r'.*\b[Ss]ister\b.*',r'.*\b[Mm]other\b.*',r'.*\b[Cc]ousin\b.*',r'.*\b[Uu]ncle\b.*',r'.*\b[Ww]ife\b.*',r'.*\b[Hh]usband\b.*',r'.*\b[Aa]unt\b.*']
  type_list=["Synonym",'Alternative','Father','Sister','Mother','Cousin','Uncle','Wife','Husband','Aunt']
  rel_list=list(zip(pat_list,type_list))
  print("LIST_rel : ",rel_list)
  for i,tup1 in enumerate(lis[:-1]):
    tup2=lis[i+1]
    if tup2[1]-tup1[2]<=100 and str(tup1[0])!=str(tup2[0]):

      sel_text=book[tup1[1]:tup2[2]+1]
      #print("SEL_TEXT : ",sel_text," tup1 : ",tup1," ::tup2 : ",tup2)
      for rel in rel_list:
        reg="\\b"+tup1[0]+"\\b"+rel[0]+"\\b"+tup2[0]+"\\b"
        #print(reg)
        if re.match(reg,sel_text):
          print(rel[1]+" REL between",tup1[0],"and",tup2[0]+":")
          print(sel_text)
          print("\n")
          pass

mdl=en_core_web_lg.load()

print("running")
file_path="text.txt"
text_file=open(file_path,'r',encoding='utf-8')
t1=text_file.read()
ne_rel(t1)
print("end")
