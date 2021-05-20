from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC

digits = load_digits()
X = digits.data
y = digits.target

plt.figure(figsize=(12, 8))
for i in range(24):
    plt.subplot(4, 6, i + 1)
    plt.imshow(X[i, :].reshape([8, 8]), cmap=plt.cm.get_cmap('gray_r'))
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
print(X_train.shape)
print(X_test.shape)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
expected = y_test

print('Predicted: ', predicted[:20])
print('Expected:  ', expected[:20])
print(f'Score: {knn.score(X_test, y_test):.2%}')

confusion = confusion_matrix(expected, predicted)
print('\nConfusion:\n', confusion, '\n')

names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))


def custom_classifier(classifier, title):
    classifier.fit(X_train, y_train)
    print(title)
    predicted_data = classifier.predict(X_test)
    expected_data = y_test
    print('Predicted: ', predicted_data[:24])
    print('Expected:  ', expected_data[:24])
    print(f'Score: {classifier.score(X_test, y_test):.2%}')


svc_linear = SVC(kernel="linear", C=0.025)
svc_poly = SVC(kernel="poly", C=0.025)
svc_rbf = SVC(kernel="rbf", C=0.025)
svc_sigmoid = SVC(kernel="sigmoid", C=0.025)
bayes = GaussianNB()

custom_classifier(svc_linear, "\nSVC(linear): ")
custom_classifier(svc_poly, "\nSVC(poly):   ")
custom_classifier(svc_rbf, "\nSVC(rbf):    ")
custom_classifier(bayes, "\nBayes: ")


def knn_n():
    knn_2 = KNeighborsClassifier(n_neighbors=2)
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    knn_7 = KNeighborsClassifier(n_neighbors=7)
    custom_classifier(knn_2, "\n2 neighbors: ")
    custom_classifier(knn_3, "\n3 neighbors: ")
    custom_classifier(knn_7, "\n7 neighbors: ")


knn_n()
plt.tight_layout()
plt.show()
