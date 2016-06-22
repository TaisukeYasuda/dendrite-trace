
####################### Dendrite Trace Machine Learning ########################
# This is a file that contains the machine learning part of the Dendrite Trace
# program.
################################################################################

import pickle
import os
import numpy
import scipy
import scipy.misc
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

VALUE_PATH = './data/value/'
REWARD_PATH = './data/reward/'
PLUS_PATH = 'positive/'
MINUS_PATH = 'negative/'
CONNECTION_PATH = './data/connection/'
LOG_REG_REWARD_PATH = './data/machine_learning_models/log_reg_reward.txt'
LOG_REG_VALUE_PATH = './data/machine_learning_models/log_reg_value.txt'
TREE_REWARD_PATH = './data/machine_learning_models/tree_reward.txt'
TREE_VALUE_PATH = './data/machine_learning_models/tree_value.txt'
REWARD_PCA_PATH = './data/machine_learning_models/reward_pca.txt'
VALUE_PCA_PATH = './data/machine_learning_models/value_pca.txt'
POS = './data/positive/'
NEG = './data/negative/'
PCA_PATH = './data/machine_learning_models/pca.txt'
LOG_REG_PATH = './data/machine_learning_models/log_reg.txt'
SVM_PATH = './data/machine_learning_models/svm.txt'
TREE_PATH = './data/machine_learning_models/tree.txt'
DATABASE_PATH = './data/database.txt'

def load_connection_data():

    def add_image_data(path,label):
        data,target = list(),list()
        for image_file in os.listdir(path):
            image_path = os.path.join(path,image_file)
            image = scipy.misc.imread(image_path)
            # make into one dimensional array
            image = numpy.ndarray.flatten(image)
            data.append(image)
            target.append(label)
        dataset = dict()
        dataset['data'],dataset['target'] = data,target
        return dataset

    def load_images(main_path):
        main_dataset = dict()
        data = list()
        target = list()
        for i in [0,1]:
            path_list = [MINUS_PATH,PLUS_PATH]
            path = os.path.join(main_path,path_list[i])
            label = i
            dataset = add_image_data(path,label)
            data += dataset['data']
            target += dataset['target']
        main_dataset['data'] = data
        main_dataset['target'] = target
        return main_dataset

    # load images from paths
    reward_dataset = load_images(REWARD_PATH)
    value_dataset = load_images(VALUE_PATH)

    reward_data = reward_dataset['data']
    value_data = value_dataset['data']

    # reduce dimension with principal component analysis
    reward_pca = PCA(n_components=20)
    value_pca = PCA(n_components=20)
    reward_data = reward_pca.fit_transform(reward_data)
    value_data = value_pca.fit_transform(value_data)

    # save pca
    f = open(REWARD_PCA_PATH,'wb')
    pickle.dump(reward_pca,f)
    f = open(VALUE_PCA_PATH,'wb')
    pickle.dump(value_pca,f)

    dataset = dict()
    dataset['reward_data'] = reward_data
    dataset['reward_target'] = reward_dataset['target']
    dataset['value_data'] = value_data
    dataset['value_target'] = value_dataset['target']

    return dataset


def train_connection_ml(dataset):
    # unpack dataset
    reward_data = dataset['reward_data']
    reward_target = dataset['reward_target']
    value_data = dataset['value_data']
    value_target = dataset['value_target']

    # use logistic regression model
    reward_model = LogisticRegression()
    value_model = LogisticRegression()

    # use tree model
    reward_model_tree = DecisionTreeClassifier()
    value_model_tree = DecisionTreeClassifier()

    # fit the models
    reward_model.fit(reward_data,reward_target)
    value_model.fit(value_data,value_target)
    reward_model_tree.fit(reward_data,reward_target)
    value_model_tree.fit(value_data,value_target)

    # save models
    f = open(LOG_REG_REWARD_PATH,'wb')
    pickle.dump(reward_model,f)
    f = open(LOG_REG_VALUE_PATH,'wb')
    pickle.dump(value_model,f)
    f = open(TREE_REWARD_PATH,'wb')
    pickle.dump(reward_model_tree,f)
    f = open(TREE_VALUE_PATH,'wb')
    pickle.dump(value_model_tree,f)

    # test the models
    test_models(reward_model,value_model,reward_model_tree,value_model_tree,
       reward_data,reward_target,value_data,value_target)

def test_models(reward_model,value_model,reward_model_tree,value_model_tree,
    reward_data,reward_target,value_data,value_target):
    print('Reward with Logistic Regression')
    expected = reward_target
    predicted = reward_model.predict(reward_data)
    scores = cv.cross_val_score(reward_model,reward_data,reward_target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))

    print('Value with Logistic Regression')
    expected = value_target
    predicted = value_model.predict(value_data)
    scores = cv.cross_val_score(value_model,value_data,value_target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))

    print('Reward with Tree')
    expected = reward_target
    predicted = reward_model_tree.predict(reward_data)
    scores = cv.cross_val_score(reward_model_tree,reward_data,reward_target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))

    print('Value with Tree')
    expected = value_target
    predicted = value_model_tree.predict(value_data)
    cores = cv.cross_val_score(value_model_tree,value_data,value_target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))


# from course notes
def read_file(path):
    with open(path, "rt") as f:
        return f.read()

def load():
    f = open(DATABASE_PATH,'rb')
    database = pickle.load(f)
    return database


def load_branch_data():

    def add_images(path,label,predict):
        
        data,target = list(),list()
        image_file_list = list()
        for image_file in os.listdir(path):
            image_path = os.path.join(path,image_file)
            image = scipy.misc.imread(image_path)
            # make into one dimensional array
            image = numpy.ndarray.flatten(image)
            # add average pixel value data
            image = numpy.append(image,numpy.average(image))
            # add classifications of surrounding
            length = len('img_')
            i = image_file[length:]
            length = len(i) - 4
            i = int(i[:length])
            predict_list = predict['predict_list_'+str(i)]
            image = numpy.append(image,predict_list)

            data.append(image)
            target.append(label)
        dataset = dict()
        dataset['data'] = numpy.asarray(data)
        dataset['target'] = numpy.asarray(target)
        return dataset

    database = load()

    positive = database['positive']
    negative = database['negative']
    positive = add_images(POS,1,positive)
    negative = add_images(NEG,0,negative)

    data = numpy.vstack((positive['data'],negative['data']))
    target = numpy.hstack((positive['target'],negative['target']))

    # reduce dimension with principal component analysis
    pca = PCA(n_components=20)
    data = pca.fit_transform(data)

    # save pca
    f = open(PCA_PATH,'wb')
    pickle.dump(pca,f)

    dataset = dict()
    dataset['data'] = data
    dataset['target'] = target

    return dataset

def train_branch_ml(dataset):
    # unpack dataset
    data = dataset['data']
    target = dataset['target']

    # use logistic regression model
    log_reg = LogisticRegression()
    # use tree model
    tree = DecisionTreeClassifier()

    # fit the models
    log_reg.fit(data,target)
    tree.fit(data,target)

    # save models
    f = open(LOG_REG_PATH,'wb')
    pickle.dump(log_reg,f)
    f = open(TREE_PATH,'wb')
    pickle.dump(tree,f)

def test():
    dataset = load_branch_data()
    train(dataset)
    data = dataset['data']
    target = dataset['target']

    # load models
    f = open(LOG_REG_PATH,'rb')
    log_reg = pickle.load(f)
    f = open(TREE_PATH,'rb')
    tree = pickle.load(f)

    print('Logistic Regression')
    expected = target
    predicted = log_reg.predict(data)
    scores = cv.cross_val_score(log_reg,data,target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))

    print('Decision Tree')
    expected = target
    predicted = tree.predict(data)
    scores = cv.cross_val_score(tree,data,target,cv=5)
    print(scores)
    print(metrics.confusion_matrix(expected,predicted))
