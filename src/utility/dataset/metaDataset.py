import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import KFold
from torch import logit
from utility.model.modelTraining_embeddings import CNN1D, FocalLoss
from utility.model.modelTraining_feature import ModelTraining

class MetaDataset:
    @staticmethod
    def build_single_message_meta(probs_feature, probs_cnn):
        probs = np.hstack([probs_feature, probs_cnn])

        df = pd.DataFrame(
            probs,
            columns=[f"probs_feature{i}" for i in range(probs_feature.shape[1])] + 
                    [f"probs_cnn{i}" for i in range(probs_cnn.shape[1])]
        )

        return MetaDataset.enhance_meta_dataset(df)
    
    @staticmethod
    def build_simple_meta_dataset(output_name, model_choice, config, feature_train_dataset, embeddings_train_dataset, n_folds):
        # Initialising k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Storage for out-of-fold predictions
        all_meta_features = []
        all_labels = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(feature_train_dataset)):
            print(f"\nFold {fold+1}/{n_folds}")

                # Split data for this fold
            feature_fold_train = feature_train_dataset.iloc[train_idx].reset_index(drop=True)
            feature_fold_val = feature_train_dataset.iloc[val_idx].reset_index(drop=True)
            
            embeddings_fold_train = embeddings_train_dataset.iloc[train_idx].reset_index(drop=True)
            embeddings_fold_val = embeddings_train_dataset.iloc[val_idx].reset_index(drop=True)

            # Train feature model on this fold
            feature_train_data  = feature_fold_train.drop("message_id", axis=1)
            fold_feature_model = ModelTraining(f"{output_name}_fold{fold+1}", model_choice, feature_train_data , config, True)
            fold_feature_model.train(plot_results=False)

            # Train CNN model on this fold
            embeddings_train_data  = embeddings_fold_train.drop("message_id", axis=1)
            fold_cnn = CNN1D(embeddings_train_data, output_name=f"{output_name}_fold{fold+1}", retrain=True)
            fold_cnn.train_and_evaluate(criterion=FocalLoss(alpha=.5, gamma=4), plot_results=False)

            # Generate predictions for validation data
            feature_val_data  = feature_fold_val.drop(columns=['user', 'message_id'])
            embeddings_val_data = embeddings_fold_val.drop(columns=['user', 'message_id'])
            
            # Get probabilities from both models
            probs_feature = fold_feature_model.get_model().predict_proba(feature_val_data)
            probs_cnn = fold_cnn.predict_proba_batch(embeddings_val_data)

            # Combine probabilities to form meta-features
            X_meta_fold = np.hstack([probs_feature, probs_cnn])
            y_meta_fold = feature_fold_val['user'].values

            # Store these meta-features and labels
            all_meta_features.append(X_meta_fold)
            all_labels.append(y_meta_fold)

        # Combine all fold results
        X_meta_full = np.vstack(all_meta_features)
        y_meta_full = np.concatenate(all_labels)

        # Create dataframe for meta-learner
        n_feature_classes = probs_feature.shape[1]
        n_cnn_classes = probs_cnn.shape[1]

        df_meta = pd.DataFrame(
            X_meta_full,
            columns=[f"probs_feature{i}" for i in range(n_feature_classes)] +
                    [f"probs_cnn{i}" for i in range(n_cnn_classes)]
        )
        df_meta['user'] = y_meta_full

        return df_meta

    
    @staticmethod
    def enhance_meta_dataset(df_meta):
        # Make a copy to avoid modifying the original
        enhanced_df = df_meta.copy()

        # Get the number of classes for each model
        feature_cols = [col for col in df_meta.columns if col.startswith('probs_feature')]
        cnn_cols = [col for col in df_meta.columns if col.startswith('probs_cnn')]

        n_feature_classes = len(feature_cols)
        n_cnn_classes = len(cnn_cols)

        # Extract probability arrays for easier processing
        probs_feature = df_meta[feature_cols].values
        probs_cnn = df_meta[cnn_cols].values

        # 1. Per-model summary statistics
        # Mean probability
        enhanced_df['mean_prob_feature'] = np.mean(probs_feature, axis=1)
        enhanced_df['mean_prob_cnn'] = np.mean(probs_cnn, axis=1)
        
        # Standard deviation and variance
        enhanced_df['std_prob_feature'] = np.std(probs_feature, axis=1)
        enhanced_df['var_prob_feature'] = np.var(probs_feature, axis=1)
        enhanced_df['std_prob_cnn'] = np.std(probs_cnn, axis=1)
        enhanced_df['var_prob_cnn'] = np.var(probs_cnn, axis=1)
        
        # Range (max - min)
        enhanced_df['range_prob_feature'] = np.max(probs_feature, axis=1) - np.min(probs_feature, axis=1)
        enhanced_df['range_prob_cnn'] = np.max(probs_cnn, axis=1) - np.min(probs_cnn, axis=1)
        
        # 2. Cross-model comparisons
        # Combine all probabilities
        all_probs = np.hstack([probs_feature, probs_cnn])
        
        # Overall mean and std across both models
        enhanced_df['mean_prob_all'] = np.mean(all_probs, axis=1)
        enhanced_df['std_prob_all'] = np.std(all_probs, axis=1)
        
        # Absolute difference between the two models' means
        enhanced_df['abs_mean_diff'] = np.abs(enhanced_df['mean_prob_feature'] - enhanced_df['mean_prob_cnn'])
        
        # 3. Prediction-based features
        # Predicted class from each model
        enhanced_df['pred_feature'] = np.argmax(probs_feature, axis=1)
        enhanced_df['pred_cnn'] = np.argmax(probs_cnn, axis=1)
        
        # Disagreement flag (1 if models predict different classes, 0 otherwise)
        enhanced_df['model_disagreement'] = (enhanced_df['pred_feature'] != enhanced_df['pred_cnn']).astype(int)
        
        # Margin features (difference between top probability and second-best)
        # For each sample, we need to find the top two probabilities
        feature_margins = []
        cnn_margins = []
        
        for i in range(len(df_meta)):
            # Sort probabilities in descending order
            feature_sorted = np.sort(probs_feature[i])[::-1]
            cnn_sorted = np.sort(probs_cnn[i])[::-1]
            
            # Margin = top probability - second best
            feature_margin = feature_sorted[0] - feature_sorted[1] if len(feature_sorted) > 1 else feature_sorted[0]
            cnn_margin = cnn_sorted[0] - cnn_sorted[1] if len(cnn_sorted) > 1 else cnn_sorted[0]
            
            feature_margins.append(feature_margin)
            cnn_margins.append(cnn_margin)
        
        enhanced_df['margin_feature'] = feature_margins
        enhanced_df['margin_cnn'] = cnn_margins


        # Probability entropy
        # For each sample, compute entropy of probability distribution
        feature_entropy = []
        cnn_entropy = []
        
        for i in range(len(df_meta)):
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            feature_entropy.append(entropy(probs_feature[i] + eps))
            cnn_entropy.append(entropy(probs_cnn[i] + eps))
        
        enhanced_df['entropy_feature'] = feature_entropy
        enhanced_df['entropy_cnn'] = cnn_entropy

        # 4. Calibration-inspired transforms
        # Logit of top probability
        enhanced_df['logit_max_prob_feature'] = logit(
            np.clip(np.max(probs_feature, axis=1), 0.001, 0.999)  # Clip to avoid infinite values
        )
        enhanced_df['logit_max_prob_cnn'] = logit(
            np.clip(np.max(probs_cnn, axis=1), 0.001, 0.999)
        )

        # 5. Additional advanced features
        # Confidence score - difference between highest and average probability
        enhanced_df['confidence_feature'] = np.max(probs_feature, axis=1) - enhanced_df['mean_prob_feature']
        enhanced_df['confidence_cnn'] = np.max(probs_cnn, axis=1) - enhanced_df['mean_prob_cnn']
        
        # Model certainty - ratio of top probability to sum of all probabilities
        enhanced_df['certainty_feature'] = np.max(probs_feature, axis=1) / np.sum(probs_feature, axis=1)
        enhanced_df['certainty_cnn'] = np.max(probs_cnn, axis=1) / np.sum(probs_cnn, axis=1)

        # Probability of top 2 classes combined (indicating confidence in top candidates)
        top2_feature = []
        top2_cnn = []
        
        for i in range(len(df_meta)):
            # Sort probabilities in descending order
            feature_sorted = np.sort(probs_feature[i])[::-1]
            cnn_sorted = np.sort(probs_cnn[i])[::-1]
            
            top2_feature.append(feature_sorted[0] + feature_sorted[1] if len(feature_sorted) > 1 else feature_sorted[0])
            top2_cnn.append(cnn_sorted[0] + cnn_sorted[1] if len(cnn_sorted) > 1 else cnn_sorted[0])
        
        enhanced_df['top2_prob_feature'] = top2_feature
        enhanced_df['top2_prob_cnn'] = top2_cnn
        
        # Add weighted average probabilities - gives more weight to the more accurate model
        # Feature model accuracy (70%) / CNN accuracy (63%)
        feature_weight = 0.7 / (0.7 + 0.63)  # Proportional to accuracy
        cnn_weight = 0.63 / (0.7 + 0.63)
        
        for i in range(min(n_feature_classes, n_cnn_classes)):
            # Assuming class labels are aligned between models
            enhanced_df[f'weighted_prob_{i}'] = (
                feature_weight * df_meta[f'probs_feature{i}'] + 
                cnn_weight * df_meta[f'probs_cnn{i}']
            )
        
        # Confidence ratio between models - which model is more confident?
        enhanced_df['confidence_ratio'] = np.max(probs_feature, axis=1) / np.max(probs_cnn, axis=1)
        
        return enhanced_df
