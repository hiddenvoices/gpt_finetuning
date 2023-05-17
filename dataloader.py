from datasets import load_dataset
dataset = load_dataset("wiki_bio")
print(dataset)
train_dataset=dataset['train']
list1=[]
list2=[]
str1=""
str2="Wikidata ::"
str3='Article ::'
for i in range(2000):
  list1=[]
  list1.append(str2)
  print(i)
  # print(train_dataset['input_text'][i]['table']['column_header'])
  # print(train_dataset['input_text'][i]['table']['content'])
  for key,value in zip(train_dataset['input_text'][i]['table']['column_header'], train_dataset['input_text'][i]['table']['content']):
    str1= str(key) + ':' + " " + str(value)
    list1.append(str1)
    # print(str1)
  list1.append(str3) 
  print(list1) 
  list1.append(train_dataset['target_text'][i])  
  list2.append(list1)
print(list2)

import pandas as pd
df = pd.DataFrame({0: list2})
df.to_csv('wikidata2k.csv')
