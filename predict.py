import sys
import os
import csv
import cPickle as pickle
import glob

import numpy as np
import pandas as pd

from features import extract_features, extract_features2, get_all_features

def load_model(model_dir, verbose=True):
    with open(model_dir, 'rb') as fi:
        m = pickle.load(fi)
    return m

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_data(filename_pairs, filename_info, symmetrize=True):
    df_pairs = parse_dataframe(pd.read_csv(filename_pairs, index_col="SampleID"))
    df_info = pd.read_csv(filename_info, index_col="SampleID")
    features = pd.concat([df_pairs, df_info], axis=1)
    if symmetrize:
        features_inverse = features.copy()
        features_inverse['A'] = features['B']
        features_inverse['A type'] = features['B type']
        features_inverse['B'] = features['A']
        features_inverse['B type'] = features['A type']
        original_index = np.array(zip(features.index, features.index)).flatten()
        features = pd.concat([features, features_inverse])
        features.index = range(0,len(features),2)+range(1,len(features),2)
        features.sort(inplace=True)
        features.index = original_index
        features.index.name = "SampleID"
    return features


def symmetrize_features(ori_features, features, feature_def=None):
    ori_features_inverse = ori_features.copy()
    ori_features_inverse['A'] = ori_features['B']
    ori_features_inverse['A type'] = ori_features['B type']
    ori_features_inverse['B'] = ori_features['A']
    ori_features_inverse['B type'] = ori_features['A type']

    features_inverse = extract_features2(ori_features_inverse, features, feature_def)
    original_index = np.array(zip(features.index, features.index)).flatten()
    features = pd.concat([features, features_inverse])
    features.index = range(0,len(features),2)+range(1,len(features),2)
    features.sort(inplace=True)
    features.index = original_index
    features.index.name = "SampleID"
    return features

def write_predictions(pred_dir, test, predictions):
    writer = csv.writer(open(pred_dir, "w"), lineterminator="\n")
    rows = [x for x in zip(test.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def main():
    if len(sys.argv) < 3:
        print "USAGE: python predict.py input_dir output_dir"
        return -1

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    symmetrize = True

    # Get the file names
    filename_pairs = glob.glob(os.path.join(input_dir, '*_pairs.csv'))
    if len(filename_pairs)!=1:
        print('No or multiple pairs.csv files')
        exit(1)
    filename_pairs = filename_pairs[0]
    filename_info = glob.glob(os.path.join(input_dir, '*_publicinfo.csv'))
    if len(filename_info)!=1:
        print('No or multiple publicinfo.scv files')
        exit(1)
    filename_info = filename_info[0]
    basename = filename_pairs[:-filename_pairs[::-1].index('_')-1]
    if  filename_info[:-filename_info[::-1].index('_')-1] != basename:
        print('Different basenames in publicinfo.csv and pairs.csv files')
        exit(1)

    # Remove the path name
    try:
        dataset = basename[-basename[::-1].index(os.sep):]
    except:
        dataset = basename

    test_ori = read_data(filename_pairs, filename_info, False)

    print "Loading the classifier"
    prog_dir = os.path.dirname(os.path.abspath(__file__))
    amodel  = load_model(os.path.join(prog_dir, 'models', "model2.pkl"))
    if symmetrize:
        ccmodel = load_model(os.path.join(prog_dir, 'models', "ccmodel.pkl"))
        cnmodel = load_model(os.path.join(prog_dir, 'models', "cnmodel.pkl"))
        nnmodel = load_model(os.path.join(prog_dir, 'models', "nnmodel.pkl"))
    else:
        for m in amodel.systems:
            m.symmetrize = symmetrize

    mymodel  = load_model(os.path.join(prog_dir, 'models', "model_t.pkl"))
    mymodel.weights = [0.17275686, 0.1424602, 0.14824986, 0.45374324, 0.08278984]
    mymodel.weights = np.array(mymodel.weights) / sum(mymodel.weights)

    print "Extracting features"
    all_features_clean, used_feature_names = get_all_features()
    test = extract_features(test_ori, all_features_clean)
    test = symmetrize_features(test_ori, test, all_features_clean)
    test = test[['A type', 'B type'] + list(used_feature_names)]

    print "Making predictions"

    aptest  =  amodel.predict(test)
    myptest  =  mymodel.predict(test)
    if symmetrize:
        BINARY      = 0 #"Binary"
        CATEGORICAL = 1 #"Categorical"
        NUMERICAL   = 2 #"Numerical"
        ccfilter = ((test['A type'] != NUMERICAL) & (test['B type'] != NUMERICAL))
        cnfilter = ((test['A type'] != NUMERICAL) & (test['B type'] == NUMERICAL))
        ncfilter = ((test['A type'] == NUMERICAL) & (test['B type'] != NUMERICAL))
        nnfilter = ((test['A type'] == NUMERICAL) & (test['B type'] == NUMERICAL))

        ptest = np.zeros((4,test.shape[0]))
        ccptest = ccmodel.predict(test[ccfilter])
        cnptest = cnmodel.predict(test[cnfilter])
        nnptest = nnmodel.predict(test[nnfilter])

        ptest[0, ccfilter] = ccptest
        ptest[0, cnfilter] = cnptest
        ptest[0, ncfilter] = -cnptest
        ptest[1, nnfilter] = nnptest
        ptest[2, :] = aptest
        ptest[3, :] = myptest

        wopt = [0.80, 1.00, 1.75, 1.75]
        print 'wopt = ', wopt
        predictions = np.dot(wopt, ptest)
    else:
        predictions = aptest

    output_filename = dataset + "_predict.csv"
    print("Writing predictions to " + output_filename)
    submission_dir = os.path.join(output_dir, output_filename)
    if symmetrize:
        write_predictions(submission_dir, test[0::2], predictions[0::2])
    else:
        write_predictions(submission_dir, test, predictions)



if __name__=="__main__":
    main()
