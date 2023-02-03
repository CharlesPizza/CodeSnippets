from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
def mdl_vs_dummy(mdl, X, y, cv, scoring):
    dummy_clf = DummyClassifier()
    mdl_scores = cross_val_score(mdl, X, y, cv=cv, scoring=scoring)
    dummy_scores = cross_val_score(dummy_clf, X, y, cv=cv, scoring=scoring)
    
    print(f'Model\'s mean score improvement over dummy: {mdl_scores.mean() - dummy_scores.mean()}')
    print(f'Model\'s STD vs dummy STD: {mdl_scores.std()} | {dummy_scores.std()}')
    print(f'Model\'s best impovement over best dummy: {mdl_scores.max() - dummy_scores.max()}')