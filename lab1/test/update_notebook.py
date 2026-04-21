#!/usr/bin/env python3
import json

# Load notebook
with open('zz1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 5: Meta-model tuning (index 4)
meta_model_code = """# ─── Phase 4: Optimized Stacking ────────────────────────────────────────────────
print("\\n[4/5] Training optimized stacking meta-model...")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

print('  Testing extended meta-model regularization parameters...')
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
test_matrix = np.column_stack([test_preds[n] for n in model_names])

best_c = 0.1
best_meta_ba = 0.0
# Extended C values with tighter ranges
c_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

for c_val in c_values:
    oof_stack_temp = np.zeros(len(y_train))
    ba_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(meta_skf.split(oof_matrix, y_train)):
        meta = LogisticRegression(
            class_weight='balanced', max_iter=3000, random_state=42, C=c_val, 
            solver='lbfgs', tol=1e-4
        )
        meta.fit(oof_matrix[tr_idx], y_train[tr_idx])
        oof_stack_temp[val_idx] = meta.predict_proba(oof_matrix[val_idx])[:, 1]
        
        fold_best_ba = 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            ba = balanced_accuracy_score(y_train[val_idx], (oof_stack_temp[val_idx] >= t).astype(int))
            fold_best_ba = max(fold_best_ba, ba)
        ba_scores.append(fold_best_ba)
    
    mean_ba = np.mean(ba_scores)
    if mean_ba > best_meta_ba:
        best_meta_ba = mean_ba
        best_c = c_val
    print(f'    C={c_val:<4}: mean BA={mean_ba:.5f}')

print(f'  ✓ Best meta-model C: {best_c} (mean BA: {best_meta_ba:.5f})')"""

nb['cells'][4]['source'] = meta_model_code

# Update Cell 7: Threshold search (index 6) - use fine-grained search
threshold_code = """# ─── Phase 5: Find Best Threshold with Multi-Metric Optimization ─────────────
print("\\n[5/5] Optimizing threshold with multi-metric search...")
print('  Threshold |    BA    |   F1    | Precision | Recall | Pred 1s')
print('  ' + '-'*70)

best_ba, best_threshold = 0.0, 0.5
best_f1, best_threshold_f1 = 0.0, 0.5
best_result = {}

# FINE-GRAINED threshold search (0.001 step) in sweet spot range
for t in np.arange(0.40, 0.50, 0.001):
    preds = (oof_stack >= t).astype(int)
    ba = balanced_accuracy_score(y_train, preds)
    f1 = f1_score(y_train, preds)
    prec = precision_score(y_train, preds, zero_division=0)
    rec = recall_score(y_train, preds, zero_division=0)
    pred_1s = preds.sum()
    
    marker = ''
    if ba > best_ba:
        best_ba = ba
        best_threshold = t
        marker += ' ← BA'
        
    if f1 > best_f1:
        best_f1 = f1
        best_threshold_f1 = t
        if '← BA' not in marker:
            marker += ' ← F1'
    
    if t % 0.01 < 0.0011:  # Print every 0.01 for readability
        print(f'    {t:.3f}   | {ba:.5f} | {f1:.5f} | {prec:.5f}  | {rec:.5f} | {pred_1s:5d}{marker}')
    
    best_result[round(t, 3)] = {'ba': ba, 'f1': f1, 'prec': prec, 'rec': rec, 'pred_1s': pred_1s}

print('  ' + '='*70)
print(f'\\n✓ Best BA threshold:  {best_threshold:.3f} (BA={best_ba:.5f})')
print(f'✓ Best F1 threshold:  {best_threshold_f1:.3f} (F1={best_f1:.5f})')

# Apply threshold to test
test_classes = (test_stack >= best_threshold).astype(int)

print('\\n' + '='*70)
print('FINAL PREDICTIONS WITH OPTIMIZED THRESHOLD')
print('='*70)
print(f'Threshold used:        {best_threshold:.4f}')
print(f'OOF BA achieved:       {best_ba:.5f}')
print(f'OOF F1 at this thresh: {best_result[round(best_threshold, 3)]["f1"]:.5f}')
print(f'OOF Precision:         {best_result[round(best_threshold, 3)]["prec"]:.5f}')
print(f'OOF Recall:            {best_result[round(best_threshold, 3)]["rec"]:.5f}')
print(f'Predicted Class 1s:    {best_result[round(best_threshold, 3)]["pred_1s"]}')

print(f'\\nTest Set Predictions:')
print(f'  Class 0: {(test_classes==0).sum():,} samples ({100*(test_classes==0).sum()/len(test_classes):.1f}%)')
print(f'  Class 1: {(test_classes==1).sum():,} samples ({100*(test_classes==1).sum()/len(test_classes):.1f}%)')

print(f'\\nConfidence Distribution (test set):')
min_conf = test_stack.min()
max_conf = test_stack.max()
mean_conf = test_stack.mean()
std_conf = test_stack.std()
print(f'  Min probability: {min_conf:.4f}')
print(f'  Max probability: {max_conf:.4f}')
print(f'  Mean probability: {mean_conf:.4f}')
print(f'  Std probability: {std_conf:.4f}')
print(f'  Median probability: {np.median(test_stack):.4f}')"""

nb['cells'][6]['source'] = threshold_code

# Save modified notebook
with open('zz1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ Updated zz1.ipynb:")
print("  - Cell 5: Extended C parameter tuning (0.001 to 3.0)")
print("  - Cell 7: Fine-grained threshold search (0.001 step in 0.40-0.50 range)")
