from sklearn.metrics import recall_score, make_scorer
def false_pos_rate(mdl, X, y, cv):
    false_pos_score = make_scorer(recall_score, pos_label=0)
    dummy_clf = DummyClassifier()
    dummy_scores = cross_val_score(dummy_clf, X, y, cv=cv, scoring='recall')
    mdl_scores = cross_val_score(mdl, X, y, cv=cv, scoring=false_pos_score)
    print(f'FPR - Model: {[1 - i for i in mdl_scores]}')
    print(f'FPR - Dummy: {[1 - i for i in dummy_scores]}')

false_pos_score = make_scorer(recall_score, pos_label=0)