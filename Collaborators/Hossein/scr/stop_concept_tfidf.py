import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


folder_name = 'C:/Users/hfani/Documents/NetBeansProjects/semionet/tags/timeseries/SocialNetworkAnalysis/TopicDetection/LDA/Results/Concept/WithStopConcepts/cu'
doc_list = []
label_list = [name for name in os.listdir(folder_name)]
file_list = [folder_name+'/'+name for name in label_list]
for file in file_list:#[:10]
    st = open(file,'r').read()
    doc_list.append(st)
    #print (file)
print ('Found %s documents under the dir %s .....'%(len(doc_list),folder_name))

tfidfvec = TfidfVectorizer(analyzer='word',token_pattern=r"\b[0-9]+\b")
tfidf = tfidfvec.fit_transform(doc_list)#['1 1 1 2', '1 1 1 3', '1 1 1 1 1 25']
weights = np.asarray(tfidf.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tfidfvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).to_csv('C:\Users\hfani\Documents\NetBeansProjects\semionet\\tags\\timeseries\SocialNetworkAnalysis\StopConceptRemoval\\tfidf\\tfidf_values.txt', index=False)