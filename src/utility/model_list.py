from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

models = {
    "random_forest" : RandomForestClassifier(n_estimators=100, random_state=42),
    "naive_bayes": GaussianNB(),
    "svc" : LinearSVC()
}