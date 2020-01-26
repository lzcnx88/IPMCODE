import scipy.io as sio
import sys
from generate_axis import generate_axis
import numpy as np
import argparse

with open('../word_type_set.txt', 'r') as f:
    lines = f.readlines()
    NNList = lines[0].split('\n')[0]
    JJList = lines[1].split('\n')[0]
    VVList = lines[2].split('\n')[0]
    OTHERList = lines[3].split('\n')[0]

    NNList = NNList.split(' ')
    JJList = JJList.split(' ')
    VVList = VVList.split(' ')
    OTHERList = OTHERList.split(' ')

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='path')
parser.add_argument('--n', type=int, help='length of axis')
parser.add_argument('--NN', type=int, help='word tag')
parser.add_argument('--JJ', type=int, help='word tag')
parser.add_argument('--VV', type=int, help='word tag')
parser.add_argument('--OTHER', type=int, help='word tag')
parser.add_argument('--ADV', type=int, help='word tag')
args = parser.parse_args()

# TF_IDF_PATH = '../bird_71/endword.txt'
TF_IDF_PATH = '../word_cnt_tag_all.txt'
if(not args.n):
    if(args.NN and args.JJ and args.VV):
        NN_index = generate_axis(TF_IDF_PATH, args.NN, NNList)
        JJ_index = generate_axis(TF_IDF_PATH, args.JJ, JJList)
        VV_index = generate_axis(TF_IDF_PATH, args.VV, VVList)
        NN_labels = [0 for i in range(args.NN)]
        JJ_labels = [0 for i in range(args.JJ)]
        VV_labels = [0 for i in range(args.VV)]
        sio.savemat(args.dir+'axis_NJV_'+str(args.NN)+'_'+str(args.JJ)+'_'+str(args.VV)+'.mat', {'NN_index':NN_index, 'NN_labels':NN_labels, \
                'JJ_index':JJ_index, 'JJ_labels':JJ_labels, \
                'VV_index':VV_index, 'VV_labels':VV_labels})

    elif(args.NN and args.JJ and args.OTHER):
        print('hello')
        OTHERList = VVList + OTHERList
        NN_index = generate_axis(TF_IDF_PATH, args.NN, NNList)
        JJ_index = generate_axis(TF_IDF_PATH, args.JJ, JJList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        NN_labels = [0 for i in range(args.NN)]
        JJ_labels = [0 for i in range(args.JJ)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_NJO_' + str(args.NN) + '_' + str(args.JJ) + '_' + str(args.OTHER) + '.mat',
                    {'NN_index': NN_index, 'NN_labels': NN_labels,
                     'JJ_index': JJ_index, 'JJ_labels': JJ_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })
    
    elif(args.NN and args.VV and args.OTHER):
        print('nvo')
        OTHERList = JJList + OTHERList
        NN_index = generate_axis(TF_IDF_PATH, args.NN, NNList)
        VV_index = generate_axis(TF_IDF_PATH, args.VV, VVList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        NN_labels = [0 for i in range(args.NN)]
        VV_labels = [0 for i in range(args.VV)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_N600V300O10.mat',
                    {'NN_index': NN_index, 'NN_labels': NN_labels,
                     'VV_index': VV_index, 'VV_labels': VV_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })
    
    elif(args.NN and args.ADV and args.OTHER):
        print('nvo')
        ADVList = ['RB', 'RBS', 'RBR']
        OTHERList = JJList + VVList + ['CD', 'MD', 'RP', 'FW', 'EX', 'CC', 'IN'] + ['WRB', 'WDT', 'WP'] + ['DT', 'PDT'] + ['PRP', 'PRP$']
        NN_index = generate_axis(TF_IDF_PATH, args.NN, NNList)
        ADV_index = generate_axis(TF_IDF_PATH, args.ADV, ADVList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        NN_labels = [0 for i in range(args.NN)]
        ADV_labels = [0 for i in range(args.ADV)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_NN600ADV300OO10.mat',
                    {'NN_index': NN_index, 'NN_labels': NN_labels,
                     'ADV_index': ADV_index, 'ADV_labels': ADV_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })

    elif(args.JJ and args.VV and args.OTHER):
        print('jvo')
        OTHERList = NNList + OTHERList
        JJ_index = generate_axis(TF_IDF_PATH, args.JJ, JJList)
        VV_index = generate_axis(TF_IDF_PATH, args.VV, VVList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        JJ_labels = [0 for i in range(args.JJ)]
        VV_labels = [0 for i in range(args.VV)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_J600V300O10.mat',
                    {'JJ_index': JJ_index, 'JJ_labels': JJ_labels,
                     'VV_index': VV_index, 'VV_labels': VV_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })
                                   
    elif(args.JJ and args.ADV and args.OTHER):
        print('jado')
        ADVList = ['RB', 'RBS', 'RBR']
        OTHERList = NNList + VVList + ['CD', 'MD', 'RP', 'FW', 'EX', 'CC', 'IN'] + ['WRB', 'WDT', 'WP'] + ['DT', 'PDT'] + ['PRP', 'PRP$']
        JJ_index = generate_axis(TF_IDF_PATH, args.JJ, JJList)
        ADV_index = generate_axis(TF_IDF_PATH, args.ADV, ADVList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        JJ_labels = [0 for i in range(args.JJ)]
        ADV_labels = [0 for i in range(args.ADV)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_J600ADV300O10.mat',
                    {'JJ_index': JJ_index, 'JJ_labels': JJ_labels,
                     'ADV_index': ADV_index, 'ADV_labels': ADV_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })
                                 
    elif(args.ADV and args.VV and args.OTHER):
        print('advo')
        ADVList = ['RB', 'RBS', 'RBR']
        OTHERList = NNList + JJList + ['CD', 'MD', 'RP', 'FW', 'EX', 'CC', 'IN'] + ['WRB', 'WDT', 'WP'] + ['DT', 'PDT'] + ['PRP', 'PRP$']
        VV_index = generate_axis(TF_IDF_PATH, args.VV, VVList)
        ADV_index = generate_axis(TF_IDF_PATH, args.ADV, ADVList)
        OTHER_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        VV_labels = [0 for i in range(args.VV)]
        ADV_labels = [0 for i in range(args.ADV)]
        OTHER_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_VV600ADV300O10.mat',
                    {'VV_index': VV_index, 'VV_labels': VV_labels,
                     'ADV_index': ADV_index, 'ADV_labels': ADV_labels,
                      'OTHER_index': OTHER_index, 'OTHER_labels': OTHER_labels
                     })

    elif (args.NN and args.OTHER):
        print('nno')
        OTHERList = JJList + VVList + OTHERList
        NN_index = generate_axis(TF_IDF_PATH, args.NN, NNList)
        NoneNN_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        NN_labels = [0 for i in range(args.NN)]
        NoneNN_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_NnN_' + str(args.NN) + '_' + str(args.OTHER) + '.mat',
                    {'NN_index': NN_index, 'NN_labels': NN_labels, \
                     'NoneNN_index': NoneNN_index, 'NoneNN_labels': NoneNN_labels})

    elif (args.JJ and args.OTHER):
        print('jjo')
        OTHERList = NNList + VVList + OTHERList
        JJ_index = generate_axis(TF_IDF_PATH, args.JJ, JJList)
        NoneJJ_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        JJ_labels = [0 for i in range(args.JJ)]
        NoneJJ_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_JJ_' + str(args.JJ) + '_OTHER_' + str(args.OTHER) + '.mat', {
            'JJ_index': JJ_index, 'JJ_labels': JJ_labels,
            'OTHER_index': NoneJJ_index, 'OTHER_labels': NoneJJ_labels
        })

    elif (args.VV and args.OTHER):
        print('vvo')
        OTHERList = NNList + JJList + OTHERList
        VV_index = generate_axis(TF_IDF_PATH, args.VV, VVList)
        NoneVV_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        VV_labels = [0 for i in range(args.VV)]
        NoneVV_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_VV_' + str(args.VV) + '_OTHER_' + str(args.OTHER) + '.mat', {
            'VV_index' : VV_index, 'VV_labels' : VV_labels,
            'OTHER_index' : NoneVV_index, 'OTHER_labels' : NoneVV_labels
        })
    
    elif (args.ADV and args.OTHER):
        print('advo')
        ADVList = ['RB', 'RBS', 'RBR']
        OTHERList = NNList + JJList + VVList + ['CD', 'MD', 'RP', 'FW', 'EX', 'CC', 'IN'] + ['WRB', 'WDT', 'WP'] + ['DT', 'PDT'] + ['PRP', 'PRP$']
        ADV_index = generate_axis(TF_IDF_PATH, args.ADV, ADVList)
        NoneADV_index = generate_axis(TF_IDF_PATH, args.OTHER, OTHERList)
        
        ADV_labels = [0 for i in range(args.ADV)]
        NoneADV_labels = [0 for i in range(args.OTHER)]
        sio.savemat(args.dir + 'axis_ADV_' + str(args.ADV) + '_OTHER_' + str(args.OTHER) + '.mat', {
            'ADV_index' : ADV_index, 'ADV_labels' : ADV_labels,
            'OTHER_index' : NoneADV_index, 'OTHER_labels' : NoneADV_labels
        })

    else:
        exit(0)
else:
    ALL_index = generate_axis(TF_IDF_PATH, args.n)
    ALL_labels = [0 for i in range(len(ALL_index))]
    print('hahaha')
    sio.savemat(args.dir+'bird_axis_all_900.mat', {'all_index': ALL_index, 'all_labels': ALL_labels})
    # with open(TF_IDF_PATH, 'r') as f:
    #     lines = f.readlines()
    # NN_index = []
    # JJ_index = []
    # OTHER_index = []
    # for x in ALL_index:
    #     line = lines[x].split('\n')[0]
    #     tag = line.split(' ')[-1]
    #     if(tag in NNList):
    #         NN_index.append(x)
    #     elif(tag in JJList):
    #         JJ_index.append(x)
    #     else:
    #         OTHER_index.append(x)
    # NN_labels = [0 for i in range(len(NN_index))]
    # JJ_labels = [0 for i in range(len(JJ_index))]
    # OTHER_index = [0 for i in range(len(OTHER_index))]
    # sio.savemat(args.dir+'axis.mat', {'NN_index':NN_index, 'NN_labels':NN_labels, \
    #         'JJ_index':JJ_index, 'JJ_labels':JJ_labels, \
    #         'OTHER_index':OTHER_index, 'OTHER_lables':OTHER_lables})
