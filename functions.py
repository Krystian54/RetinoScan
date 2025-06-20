import numpy as np


# metryki
def calculate_metrics(cm):
    total_samples = np.sum(cm)
    metrics = {}
    
    # listy do średniej
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []

    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total_samples - (TP + FN + FP)

        accuracy = (TP + TN) / total_samples if total_samples > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        specificity_list.append(specificity)

        metrics[f"Klasa {i}"] = {
            'Dokładność (accuracy)': accuracy,
            'Precyzja (precision)': precision,
            'Czułość (recall)': recall,
            'Współczynnik F1 (F1-score)': f1,
            'Specyficzność (specificity)': specificity
        }

    # makrośrednie
    metrics['Makrośrednia (macro avg)'] = {
        'Dokładność (accuracy)': np.mean(accuracy_list),
        'Precyzja (precision)': np.mean(precision_list),
        'Czułość (recall)': np.mean(recall_list),
        'Współczynnik F1 (F1-score)': np.mean(f1_list),
        'Specyficzność (specificity)': np.mean(specificity_list)
    }

    return metrics
