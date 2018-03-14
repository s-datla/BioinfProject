import sys, string, io, os, math
import numpy as np
from collections import Counter

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC123, lcc
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedBaggingClassifier

windowSize = 9
aminoAcids = 'ACDEFGHIKLMNOPQRSTUVWYX'


proteinWeights = [
    89.047678,121.019749,133.037508,147.053158,165.078979,75.032028,155.069477,131.094629,146.105528,131.094629,149.051049,
    132.053492,255.158292,115.063329,146.069142,174.111676,105.042593,119.058243,168.964203,117.078979,204.089878,181.073893,
    143.656733]

def main():
    if len(sys.argv) > 3:
        print("Invalid inputs, please try again")
        print("Expected Format <FORMAT>")
    else:
        global windowSize
        windowSize = int(sys.argv[1])
        sequences,labels = readFasta()
        buildModel(np.asarray(sequences),np.asarray(labels))
        # windows, labels = createWindows(s,l)
        # buildModel(np.asarray(windows),np.asarray(labels))

def readFasta():
    # LABELS: 1 = CYTO, 2 = MITO, 3 = NUCLEUS, 4 = SECRETED
    files = []
    for fa in os.listdir("FASTA"):
        filePath = "FASTA/" + str(fa)
        files += [filePath]
    files = sorted(files)
    print files
    labels = []
    sequences = []
    for i in range(0,len(files)):
        for seq_record in SeqIO.parse(files[i],'fasta'):
            sequences += [processSeq(str(seq_record.seq))]
            # sequences += [oneHotEncode(seq_record)]
            labels += [i]
    l = np.asarray(labels)
    s = np.asarray(sequences)
    print "Distribution of labels = " + str(sorted(Counter(labels).items()))
    np.savez_compressed('data',labels=labels,seqs=sequences)
    return sequences,labels

def oneHotEncode(sequence):
    base = [[0]*21 for _ in range(0,len(sequence))]
    assert(len(base) == len(sequence))
    for i in range(0,len(sequence)):
        pos = aminoAcids.find(sequence[i])
        base[i][pos] = 1
    assert(sum(x.count(1) for x in base) == len(sequence))
    return base

def createWindows(sequences,labels):
    boundary = [0]*20 + [1]
    new_labels = []
    windows = []
    diff = (windowSize-1)/2
    for i in range(0,len(sequences)):
        curr_seq = sequences[i]
        for j in range(0,len(curr_seq)):
            current = [boundary]*max(diff-j,0) + curr_seq[max(j-diff,0):min(j+diff,len(curr_seq))]
            current += [boundary]*(windowSize - len(current))
            assert(len(current) == windowSize)
            new_labels += [labels[i]]
            windows += [current]
    assert(len(windows) == len(new_labels))
    return windows,new_labels

def processSeq(seq):
    prot = ProteinAnalysis(seq)
    seq_length = len(seq)
    # GC_distribution  = list(GC123(seq))
    AA_distribution = getAAPercent(seq)
    isoelectric = prot.isoelectric_point()
    mol_weight = calculateMolecularWeight(seq)
    aroma = prot.aromaticity()
    # instable = prot.instability_index()

    return_vector = [seq_length, mol_weight, aroma, isoelectric] + AA_distribution
    # print seq_length, GC_distribution, mol_weight, aroma, isoelectric
    return return_vector

def calculateMolecularWeight(sequence):
    mol_weight = 0
    for i in range(0,len(sequence)):
        position = aminoAcids.find(sequence[i])
        if (position == -1):
            mol_weight += proteinWeights[22]
        else:
            mol_weight += proteinWeights[position]
    return mol_weight

def getAAPercent(sequence):
    count = [0.0]*23
    for i in range(0,len(sequence)):
        position = aminoAcids.find(sequence[i])
        if (position == -1):
            count[22] += 1
        else:
            count[position] += 1
    return [i / len(sequence) for i in count]


def buildModel(X,y):
    # X = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    print X.shape, y.shape
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaled_train_x = scaler.transform(X)
    X_train,X_test,y_train,y_test = train_test_split(scaled_train_x,y,random_state=19,test_size=0.3)

    logistic = LogisticRegression(solver='lbfgs',max_iter=500)
    bag = BalancedBaggingClassifier(n_estimators=200,random_state=19)
    neural = MLPClassifier(max_iter=500,random_state=19,solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(27,8,4))
    svm = SVC(class_weight='balanced',random_state=19,decision_function_shape='ovo')
    ada = AdaBoostClassifier(n_estimators=100,random_state=19)

    logistic.fit(X_train,y_train)
    bag.fit(X_train,y_train)
    svm.fit(X_train,y_train)
    neural.fit(X_train,y_train)
    ada.fit(X_train,y_train)
    # joblib.dump(bag,'bag.pkl')
    # joblib.dump(scaler,'scaler.pkl')
    y_pred = bag.predict(X_test)
    y_pred2 = svm.predict(X_test)
    y_pred3 = neural.predict(X_test)
    y_pred4 = ada.predict(X_test)
    y_pred5 = logistic.predict(X_test)

    print matthews_corrcoef(y_test,y_pred)
    print matthews_corrcoef(y_test,y_pred2)
    print matthews_corrcoef(y_test,y_pred3)
    print matthews_corrcoef(y_test,y_pred4)
    print matthews_corrcoef(y_test,y_pred5)

    print confusion_matrix(y_test,y_pred)
    print confusion_matrix(y_test,y_pred2)
    print confusion_matrix(y_test,y_pred3)
    print confusion_matrix(y_test,y_pred4)
    print confusion_matrix(y_test,y_pred5)

    print(classification_report_imbalanced(y_test, y_pred))
    print(classification_report_imbalanced(y_test, y_pred2))
    print(classification_report_imbalanced(y_test, y_pred3))
    print(classification_report_imbalanced(y_test, y_pred4))
    print(classification_report_imbalanced(y_test, y_pred5))

if __name__ == "__main__":
    main()
