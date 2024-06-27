import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, f1_score


def plot_evaluation_metrics(y_test, y_pred, model, X_test, model_name):
    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Pastel1')
    plt.savefig(f'plots/confusion_matrix_{model_name}.png', bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/roc_curve_{model_name}.png', bbox_inches='tight')
    plt.close()

    f1 = f1_score(y_test, y_pred)
    print(f'F1-Score: {f1}')

    return f1
