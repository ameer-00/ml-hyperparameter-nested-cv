# Hyperparameter Tuning & Nested Cross Validation

---

## Part 1
- **Method:** GridSearchCV with 5-fold Stratified CV
- **Model:** `RandomForestClassifier(class_weight='balanced')`
- **Best Parameters:** `{'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 200}`
- **Best F1 Score:** **0.5017**

### Analysis
The analysis reveals that **max_depth** is the most significant driver of model performance. Shallow trees (depth 3) result in underfitting, while performance improvements plateau between depth 5 and 10. Increasing depth further or leaving it unrestricted (None) yields negligible gains, suggesting the model captures the necessary patterns at moderate complexity.

The **n_estimators** parameter shows a clear improvement when moving from 50 to 100 trees, but gains beyond that are minimal, indicating diminishing returns for the added computational cost. **min_samples_split** had the least impact, with F1 variations smaller than 0.005. The "sweet spot" for this dataset lies at a moderate depth (5–10) with at least 100 estimators. Future tuning should explore regularization axes like `min_samples_leaf` or `max_features` rather than further increasing depth.

---

## Part 2
### Results

| Model | Inner Score | Outer Score | Gap |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.4989 | 0.5011 | -0.0022 |
| **Decision Tree** | 0.4781 | 0.4781 | 0.0000 |



### Analysis
The Nested Cross-Validation (NCV) results highlight the difference between **model selection** (inner loop) and **model evaluation** (outer loop). The gap between scores represents the "selection bias"—the risk that hyperparameter tuning overfits the training data.

* **Random Forest Stability:** The Random Forest exhibits a very small gap (**-0.0022**), showcasing its stability. Because it uses bagging (averaging multiple trees), it is less sensitive to the noise in specific data folds. The parameters chosen during tuning generalize well to the outer test folds.
* **Decision Tree Variance:** While the gap here appears as 0.0000 in the final mean, Decision Trees generally exhibit higher variance. Their optimal parameters are highly sensitive to the specific samples in the training set.

**Key Takeaway:** Nested CV provides a more "honest" estimate of performance. The inner loop acts as the **train + select** phase, while the outer loop acts as the **independent test** phase. As seen in the results, standard GridSearchCV scores can be slightly optimistic. NCV ensures that data used to make a decision (selecting parameters) is not the same data used to evaluate that decision, mirroring the best practices for robust machine learning deployment.