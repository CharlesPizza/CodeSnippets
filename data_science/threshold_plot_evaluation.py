from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve

def threshold_eval(mdl, X, y, cv, method='decision_function'):
    y_scores = cross_val_predict(mdl, X, y, cv=cv, method=method)
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    
    # Threshold eval
    idx_for_90p = (precisions >=0.90).argmax()
    threshold_for_90p = thresholds[idx_for_90p]
    recall = recall_score(y_train_prep.ravel(), (y_scores >= threshold_for_90p))
    
    
    print(f'90% Precision marker: {threshold_for_90p}')
    print(f'Recall at 90% Precision: {recall}')
    # Prepping ROC
    fpr, tpr, roc_thresholds = roc_curve(y, y_scores)
    idx_for_threshold_roc = (roc_thresholds <= threshold_for_90p).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_roc], fpr[idx_for_threshold_roc]
    
    # Plotting PRC, Precision/Recall &, ROC
    fig, axs = plt.subplots(3, 1, figsize=(24,24))
    # Plottin PRC
    axs[0].plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
    axs[0].plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    axs[0].set_xlabel('Threshold', fontsize=16)
    # Plotting Precision/Recall
    axs[1].plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
    axs[1].set_xlabel('Recall', fontsize=16)
    axs[1].set_ylabel('Precision', fontsize=16)
    # Plotting ROC
    axs[2].plot(fpr, tpr, linewidth=2, label='ROC curve')
    axs[2].plot([0,1],[0,1], 'k:', label='Random clf curve')
    axs[2].plot([fpr_90], [tpr_90], 'ko', label='Threshold for 90% precision')
    axs[2].set_xlabel('False Positive Rate', fontsize=16)
    axs[2].set_ylabel('True Positive Rate', fontsize=16)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()
