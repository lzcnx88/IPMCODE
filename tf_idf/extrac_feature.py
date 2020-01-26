# dogpic_word=open('dog_129/tf_idf_courp_2.txt','r').readlines()
M = 900
N = 300
mode = 'NoneNN'
dogpic_word=open('dpp_dog_283/3d_courp.txt','r').readlines()
# dogpic_word=open('dog_283/tf_idf_courp_all_38805.txt','r').readlines()
# birdpic_word = open('bird71/features_NnN_900/tf_idf_courp_'+mode+ '_' + str(N) +'.txt', 'r').readlines()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import scipy.io as sio
vectorizer = CountVectorizer(min_df=0, token_pattern=r"\b\w+\b")
transformer=TfidfTransformer()
x = vectorizer.fit_transform(dogpic_word)
tfidf=transformer.fit_transform(x)


# tfidfword = open('dog_283/word_N_400.txt','w')
tfidfword = open('dpp_dog_283/3d_tf_word.txt','w')
# tfidfword = open('bird71/bird_features_/word_all_900.txt','w')
names = vectorizer.get_feature_names()
print(len(names))
for i in range(len(names)):
    tfidfword.write(names[i]+'\n')
tfidfword.close()
print('----------')
res = tfidf.toarray()
print(res.shape)
os.makedirs('dpp_dog_283/3d/')
# os.makedirs('bird71/features_NnN_'+str(M)+'/features_'+ mode + '_' + str(N)+'/')
for i in range(res.shape[0]):
    r = res[i, :]
    # sio.savemat('dog_283/features_NJ_450/features_N_400/'+str(i)+'.mat', {'matrix': r.reshape(1, -1)})
    # sio.savemat('dog_283/features_NnN_'+str(M)+'/features_'+mode+'_'+str(N)+'/' + str(i) + '.mat', {'matrix': r.reshape(1, -1)})
    sio.savemat('dpp_dog_283/3d/' + str(i) + '.mat', {'matrix': r.reshape(1, -1)})
