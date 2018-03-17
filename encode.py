import sys, string, io, os, math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plot

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC123, lcc
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd , Flex


from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split,StratifiedKFold
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
        # global windowSize
        # windowSize = int(sys.argv[1])
        sequences,labels = readFasta()
        num_features = optimalFeatures(np.asarray(sequences),np.asarray(labels))
        evaluateFeatures(np.asarray(sequences),np.asarray(labels), num_features)
        # buildModel(np.asarray(sequences),np.asarray(labels))
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
    species_list = []
    for i in range(0,len(files)):
        for seq_record in SeqIO.parse(files[i],'fasta'):
            species, processed = processSeq(seq_record)
            sequences += [processed]
            species_list += [species]
            # sequences += [oneHotEncode(seq_record)]
            labels += [i]
    l = np.asarray(labels)
    s = np.asarray(sequences)
    labelSpecies(species_list)
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

def processSeq(seq_record):

    ''' Protein features found:
        - Sequence Length
        - Amino Acid Composition (global)
        - Isoelectric Point
        - Molecular Weight (global)
        - Aromacity
    '''

    desc = str(seq_record.description).split('_')
    species = desc[1].split(' ')[0]
    seq = str(seq_record.seq)
    prot = ProteinAnalysis(seq)
    seq_length = len(seq)
    AA_global_dist = getAAPercent(seq,0,seq_length)
    gravy = calculateGravy(seq,0,seq_length)
    flex_global = calculateFlexibility(seq,0,seq_length)
    isoelectric = prot.isoelectric_point()
    mol_global_weight = calculateMolecularWeight(seq,0,seq_length)
    aroma = prot.aromaticity()
    ss_frac = prot.secondary_structure_fraction()
    if (seq_length > 50):
        AA_local_head = getAAPercent(seq,0,50)
        AA_local_tail = getAAPercent(seq,seq_length-50,seq_length)
        mol_local_weight_head = calculateMolecularWeight(seq,0,50)
        mol_local_weight_tail = calculateMolecularWeight(seq,seq_length-50,seq_length)
        flex_localh = calculateFlexibility(seq,0,50)
        flex_localt = calculateFlexibility(seq,seq_length-50,seq_length)
    else:
        AA_local_head = AA_global_dist
        AA_local_tail = AA_global_dist
        mol_local_weight_head = mol_global_weight
        mol_local_weight_tail = mol_global_weight
        flex_localh = flex_global
        flex_localt = flex_global
    return_vector = [seq_length,aroma, isoelectric, mol_global_weight, mol_local_weight_head, mol_local_weight_tail,gravy,flex_global,flex_localh,flex_localt] + \
    AA_global_dist + AA_local_head + AA_local_tail

    # print seq_length, GC_distribution, mol_weight, aroma, isoelectric
    return species, return_vector

def calculateGravy(sequence,start,end):
    total = 0
    for c in sequence[start:end+1]:
        if c in kd:
            total += kd[c]
        else:
            # Average Hydrophobicity score using Doolittle Scale
            total += -0.49
    return total

def calculateFlexibility(sequence,start,end):
    modified_flex = Flex
    modified_flex['X'] = 0.99065
    modified_flex['U'] = 0.99065
    modified_flex['O'] = 0.99065
    window_size = 9
    weights = [0.25, 0.4375, 0.625, 0.8125, 1]
    flex_list = []
    subsequence = sequence[start:end]
    for i in range(0,len(subsequence) - window_size):
        current_score = 0.0
        current_window = subsequence[i:i+windowSize]
        for j in range(window_size // 2):
            current_score += ((modified_flex[subsequence[j]] + modified_flex[subsequence[windowSize-j+1]]) * weights[j])
        current_score += modified_flex[subsequence[windowSize // 2 + 1]]
        flex_list.append(current_score / 5.25)
    return np.mean(flex_list)

def labelSpecies(species_list):
    species_unique = sorted(set(species_list))
    # print species_unique

def calculateMolecularWeight(sequence, start, end):
    mol_weight = 0
    for i in range(start,end):
        position = aminoAcids.find(sequence[i])
        if (position == -1):
            mol_weight += proteinWeights[22]
        else:
            mol_weight += proteinWeights[position]
    return mol_weight

def getAAPercent(sequence, start, end):
    count = [0.0]*23
    for i in range(start,end):
        position = aminoAcids.find(sequence[i])
        if (position == -1):
            count[22] += 1
        else:
            count[position] += 1
    return [i / len(sequence) for i in count]

def optimalFeatures(X,y):
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaled_train_x = scaler.transform(X)
    trees = RandomForestClassifier(n_estimators=100,random_state=19,class_weight='balanced')
    rfecv = RFECV(estimator=trees, step=1, cv=StratifiedKFold(2),scoring='accuracy')
    rfecv.fit(scaled_train_x,y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    plot.figure()
    plot.xlabel("Number of features selected")
    plot.ylabel("Cross validation score (nb of correct classifications)")
    plot.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plot.show()

def evaluateFeatures(X,y, num_features):
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaled_train_x = scaler.transform(X)

    trees = RandomForestClassifier(n_estimators=100,random_state=19,class_weight='balanced')
    selector = RFE(trees,step=1,n_features_to_select=num_features)
    selector.fit(scaled_train_x,y)
    AA_global_labels = []
    AA_localh_labels = []
    AA_localt_labels = []

    for i in range(0,23):
        AA_global_labels += ["Amino Acid Composition (Global) Pos:{}".format(aminoAcids[i])]
        AA_localh_labels += ["Amino Acid Composition (First 50) Pos:{}".format(aminoAcids[i])]
        AA_localt_labels += ["Amino Acid Composition (Last 50) Pos:{}".format(aminoAcids[i])]

    labels = ["Sequence Length","Aromacity","Isoelectricity","Molecular Weight (Global)", "Molecular Weight (First 50)","Molecular Weight (Last 50)","Gravy","Mean Flexibility (Global)","Mean Flexibility (First 50)","Mean Flexibility (Last 50)"] + \
    AA_global_labels + AA_localh_labels + AA_localt_labels
    importances = selector.ranking_
    indices = np.argsort(importances)
    sorted_labels = []
    print len(labels),len(indices)
    for i in range(0,len(indices)):
        sorted_labels += labels[indices[i]]
    print len(importances),len(indices),X.shape[1]
    print("Feature Ranking:")
    for r in range(0,X.shape[1]):
        print "{}: {} ({})".format(r+1,labels[indices[r]],importances[indices[r]])
    f = plot.figure()
    plot.title("Feature importances")
    plot.bar(range(X.shape[1]), importances[indices],color="b", align="center")
    plot.xticks(range(X.shape[1]), sorted_labels,rotation='vertical')
    plot.xlim([-1, X.shape[1]])
    plot.show()


def buildModel(X,y):
    # X = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    print X.shape, y.shape
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaled_train_x = scaler.transform(X)
    X_train,X_test,y_train,y_test = train_test_split(scaled_train_x,y,random_state=19,test_size=0.3)

    bag = BalancedBaggingClassifier(n_estimators=200,random_state=19)
    svm = SVC(class_weight='balanced',random_state=19,decision_function_shape='ovo')
    neural = MLPClassifier(max_iter=500,random_state=19,solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(27,8,4))
    ada = AdaBoostClassifier(n_estimators=100,random_state=19)
    logistic = LogisticRegression(solver='lbfgs',max_iter=500)


    bag.fit(X_train,y_train)
    svm.fit(X_train,y_train)
    neural.fit(X_train,y_train)
    ada.fit(X_train,y_train)
    logistic.fit(X_train,y_train)
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
