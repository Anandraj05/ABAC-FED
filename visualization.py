import matplotlib.pyplot as plt
import numpy as np


def visualize_results(methods, precisions, recalls, f1_scores, accuracies):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.2

    ax.bar(x - 1.5 * width, precisions, width, label='Precision', color='b', alpha=0.7)
    ax.bar(x - 0.5 * width, recalls, width, label='Recall', color='g', alpha=0.7)
    ax.bar(x + 0.5 * width, f1_scores, width, label='F1 Score', color='r', alpha=0.7)


    ax.set_xticks(x)
    ax.set_xticklabels(methods, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics Comparison')
    ax.legend(loc='lower right')

    for i, v in enumerate(precisions):
        ax.text(i - 1.5 * width, v + 0.01, str(round(v, 2)), ha='center', color='black', fontweight='bold')
    for i, v in enumerate(recalls):
        ax.text(i - 0.5 * width, v + 0.01, str(round(v, 2)), ha='center', color='black', fontweight='bold')
    for i, v in enumerate(f1_scores):
        ax.text(i + 0.5 * width, v + 0.01, str(round(v, 2)), ha='center', color='black', fontweight='bold')


    ax.set_ylim(0, max(precisions + recalls + f1_scores ) + 0.1)


    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)


    fig.patch.set_facecolor('white')
    ax.set_facecolor('lightgray')
    plt.tight_layout()
    plt.show()