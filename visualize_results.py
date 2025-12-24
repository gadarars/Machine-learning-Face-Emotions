import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os

# Load cached data
X_test_hog = np.load("X_test_hog.npy")
y_test = np.load("y_test.npy")

# Load best model
model = joblib.load("models/best_emotion_model.pkl")

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create results folder
os.makedirs("results", exist_ok=True)

# =====================================
# 1. CONFUSION MATRIX
# =====================================
print("Generating confusion matrix...")
y_pred = model.predict(X_test_hog)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotions, yticklabels=emotions,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - XGBoost (Best Model)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/confusion_matrix.png")
plt.close()

# =====================================
# 2. MODEL COMPARISON BAR CHART
# =====================================
print("Generating model comparison chart...")

models_names = ['XGBoost', 'KNN', 'Random\nForest', 'MLP', 'SVM', 'Logistic\nReg', 'Decision\nTree']
accuracies = [0.4994, 0.4671, 0.4632, 0.4565, 0.4447, 0.4409, 0.3266]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]

plt.figure(figsize=(12, 6))
bars = plt.bar(models_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 0.6)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/model_comparison.png")
plt.close()

# =====================================
# 3. PCA COMPARISON CHART
# =====================================
print("Generating PCA comparison chart...")

pca_components = ['Full HOG\n(900)', '50', '100', '200', '300']
pca_accuracies = [0.4447, 0.5425, 0.5486, 0.5417, 0.5293]
pca_colors = ['#e74c3c' if i == 0 else '#2ecc71' if acc == max(pca_accuracies) else '#3498db'
              for i, acc in enumerate(pca_accuracies)]

plt.figure(figsize=(10, 6))
bars = plt.bar(pca_components, pca_accuracies, color=pca_colors, edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('PCA Feature Selection Impact on Accuracy', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Number of PCA Components', fontsize=12)
plt.ylim(0, 0.65)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/pca_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/pca_comparison.png")
plt.close()

# =====================================
# 4. METRICS COMPARISON (Precision, Recall, F1)
# =====================================
print("Generating metrics comparison chart...")

models = ['XGBoost', 'KNN', 'RF', 'MLP', 'SVM', 'LR', 'DT']
precision = [0.4977, 0.4650, 0.4774, 0.4573, 0.4313, 0.4216, 0.3279]
recall = [0.4994, 0.4671, 0.4632, 0.4565, 0.4447, 0.4409, 0.3266]
f1_score = [0.4904, 0.4505, 0.4402, 0.4565, 0.4333, 0.4256, 0.3270]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e67e22')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Precision, Recall, and F1-Score Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/metrics_comparison.png")
plt.close()

print("\n" + "="*50)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*50)
print("Check the 'results/' folder for:")
print("  - confusion_matrix.png")
print("  - model_comparison.png")
print("  - pca_comparison.png")
print("  - metrics_comparison.png")