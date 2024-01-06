from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "random_forest" : RandomForestClassifier(n_estimators=100, random_state=42),
    "naive_bayes": GaussianNB(),
    "svc" : LinearSVC(dual=True),
    "neural_network": MLPClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=0),
    "extra_trees": ExtraTreesClassifier(random_state=0),
    "kneighbors": KNeighborsClassifier(),
}