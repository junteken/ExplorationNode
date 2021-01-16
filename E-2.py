from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn import svm



def train_and_print(clssifier, x_train, y_train, x_test, y_test):
    print(clssifier._estimator_type)
    clssifier.fit(x_train, y_train)
    y_pred = clssifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy = '+str(accuracy))    
    print(classification_report(y_test, y_pred))


def sklearn_sample_practice(data, label):
    x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    label, 
                                                    test_size=0.2, 
                                                    random_state=7)

    print('X_train 개수: ', len(x_train), ', X_test 개수: ', len(x_test))
    
    decision_tree = DecisionTreeClassifier(random_state=32)
    print('Decision tree결과')
    train_and_print(decision_tree,x_train, y_train, x_test, y_test)

    random_forest = RandomForestClassifier(random_state=32)
    print('Random forest결과')
    train_and_print(random_forest,x_train, y_train, x_test, y_test)

    ml = svm.SVC()
    print('SVM 결과')
    train_and_print(ml,x_train, y_train, x_test, y_test)

    sgd_model = SGDClassifier()
    print('SGD Classifier 결과')
    train_and_print(sgd_model,x_train, y_train, x_test, y_test)

'''
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

sklearn_sample_practice(iris_data, iris_label)

#digit문제 시작
print('digits 문제 결과')

digits = load_digits()
digits.keys()
digit_data= digits.data
digit_label= digits.target

sklearn_sample_practice(digit_data, digit_label)
'''
#load wine문제
wine= load_wine()
print(wine.keys())
wine_data= wine.data
wine_label= wine.target
iris_df = pd.DataFrame(data=wine_data, columns=wine.feature_names)
print(iris_df)

sklearn_sample_practice(wine_data, wine_label)
