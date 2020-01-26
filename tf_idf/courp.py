import numpy as np
import scipy.io as sio

root = '/data/zj/re_generate_matrix_32691/click_data_dog_283.mat'
# root = '../bird_71/click_data_bird_71.mat'
data = sio.loadmat(root)
cm = data['click_matrix'].toarray()
print(cm.shape)
query = data['words'].reshape(-1, )
print(query.shape)

# root = '/data/zj/re_generate_matrix_32691/validation_svm/axis/axis_450.mat'
NN = 600
JJ = 300
OTHER = 10
root = '/data/zj/workspace_v2/data_v1/3d_word_index.mat'
# root = '../2019_7_8_3d/axis/axis_ADV_494_OTHER_406.mat'
# root = '../2019_7_8_3d/axis/axis_NnN_'+str(NN)+'_'+str(JJ) +'.mat'
# root = '../2019_7_8_3d/axis/bird_axis_NnN_600_300.mat'
data = sio.loadmat(root)
key_words = data['word_index'].reshape(-1, )
# key_words_N = data['ADV_index']
# key_words_NoneNN = data['OTHER_index']
# key_words_J = data['JJ_index']
# key_words_O = data['OTHER_index']
# key_words = data['all_index']
# print('dog_283/features_NnN_'+str(NN+JJ))

# fN = open('dog_283/919extralExperiment/2d_ADV_OO/tf_idf_courp_ADV_494.txt', 'w')
# fJ = open('dog_283/919extralExperiment/2d_ADV_OO/tf_idf_courp_OO_406.txt', 'w')
# fO = open('dog_283/919extralExperiment/3d_VADO/tf_idf_courp_OO_10.txt', 'w')
f = open('dpp_dog_283/3d_courp.txt', 'w')

for i in range(cm.shape[0]):

    """
    # all
    intersect = np.where(cm[i, :] != 0)[0]

    """
    # subset
    idx = np.where(cm[i, :] != 0)[0]
    intersect = np.intersect1d(key_words, idx)
    # intersectN = np.intersect1d(key_words_N, idx)
    # intersectJ = np.intersect1d(key_words_NoneNN, idx)
    # intersectO = np.intersect1d(key_words_O, idx)
    
    s = ''
    if(len(intersect) == 0):
        pass
    else:
        for e in intersect:
            w = query[e].strip()
            for j in range(int(cm[i, e])):
                s += w+' '
    f.write(s + '\n')

    """
    s = ''
    if(len(intersectN) == 0):
        pass
    else:
        for e in intersectN:
            w = query[e].strip()
            for j in range(int(cm[i, e])):
                s += w+' '
    fN.write(s + '\n')

    s = ''
    if(len(intersectJ) == 0):
        pass
    else:
        for e in intersectJ:
            w = query[e].strip()
            for j in range(int(cm[i, e])):
                s += w+' '
    fJ.write(s + '\n')
    """
#    s = ''
#    if(len(intersectO) == 0):
#        pass
#    else:
#        for e in intersectO:
#            w = query[e].strip()
#            for j in range(int(cm[i, e])):
#                s += w+' '
#    fO.write(s + '\n')


# fN.close()
# fJ.close()
# fO.close()
f.close()