import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
import operator
from estimator_base import *
from features_base import *

BINARY      = "Binary"
CATEGORICAL = "Categorical"
NUMERICAL   = "Numerical"

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name in self.features:
            extractor.fit(X[feature_name].values[:,np.newaxis], y)

    def transform(self, X):
        return X[self.features].as_matrix()

    def fit_transform(self, X, y=None):
        return self.transform(X)

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T

def get_all_features():
    all_features = [
        ('Max', 'A', SimpleTransform(max)),
        ('Max', 'B', SimpleTransform(max)),
        ('Min', 'A', SimpleTransform(min)),
        ('Min', 'B', SimpleTransform(min)),
        ('Numerical', 'A type', SimpleTransform(lambda x: int(numerical(x)))),
        ('Numerical', 'B type', SimpleTransform(lambda x: int(numerical(x)))),
        ('Sub', ['Numerical[A type]','Numerical[B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Numerical[A type],Numerical[B type]]', SimpleTransform(abs)),

        ('Number of Samples', 'A', SimpleTransform(len)),
        ('Log', 'Number of Samples[A]', SimpleTransform(np.log)),

        ('Number of Unique Samples', 'A', SimpleTransform(count_unique)),
        ('Number of Unique Samples', 'B', SimpleTransform(count_unique)),
        ('Max', ['Number of Unique Samples[A]','Number of Unique Samples[B]'], MultiColumnTransform(max)),
        ('Min', ['Number of Unique Samples[A]','Number of Unique Samples[B]'], MultiColumnTransform(min)),
        ('Sub', ['Number of Unique Samples[A]','Number of Unique Samples[B]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Number of Unique Samples[A],Number of Unique Samples[B]]', SimpleTransform(abs)),

        ('Log', 'Number of Unique Samples[A]', SimpleTransform(np.log)),
        ('Log', 'Number of Unique Samples[B]', SimpleTransform(np.log)),
        ('Max', ['Log[Number of Unique Samples[A]]','Log[Number of Unique Samples[B]]'], MultiColumnTransform(max)),
        ('Min', ['Log[Number of Unique Samples[A]]','Log[Number of Unique Samples[B]]'], MultiColumnTransform(min)),
        ('Sub', ['Log[Number of Unique Samples[A]]','Log[Number of Unique Samples[B]]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]', SimpleTransform(abs)),

        ('Ratio of Unique Samples', 'A', SimpleTransform(count_unique_ratio)),
        ('Ratio of Unique Samples', 'B', SimpleTransform(count_unique_ratio)),
        ('Max', ['Ratio of Unique Samples[A]','Ratio of Unique Samples[B]'], MultiColumnTransform(max)),
        ('Min', ['Ratio of Unique Samples[A]','Ratio of Unique Samples[B]'], MultiColumnTransform(min)),
        ('Sub', ['Ratio of Unique Samples[A]','Ratio of Unique Samples[B]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]', SimpleTransform(abs)),

        ('Normalized Value', ['A','A type'], MultiColumnTransform(normalize)),
        ('Normalized Value', ['B','B type'], MultiColumnTransform(normalize)),
        ('Count Value', ['A','A type'], MultiColumnTransform(count_value), ['Normalized Value[A,A type]']),
        ('Count Value', ['B','B type'], MultiColumnTransform(count_value), ['Normalized Value[B,B type]']),
        ('DisSeq', ['A','A type'], MultiColumnTransform(discrete_seq)),
        ('DisSeq', ['B','B type'], MultiColumnTransform(discrete_seq)),
        ('DisProb', ['A','A type'], MultiColumnTransform(discrete_probability), ['DisSeq[A,A type]']),
        ('DisProb', ['B','B type'], MultiColumnTransform(discrete_probability), ['DisSeq[B,B type]']),

        ('Normalized Entropy Baseline', ['A','A type'], MultiColumnTransform(normalized_entropy_baseline), ['Normalized Value[A,A type]']),
        ('Normalized Entropy Baseline', ['B','B type'], MultiColumnTransform(normalized_entropy_baseline), ['Normalized Value[B,B type]']),
        ('Max', ['Normalized Entropy Baseline[A,A type]','Normalized Entropy Baseline[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Normalized Entropy Baseline[A,A type]','Normalized Entropy Baseline[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Normalized Entropy Baseline[A,A type]','Normalized Entropy Baseline[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]', SimpleTransform(abs)),

        ('Normalized Entropy', ['A','A type'], MultiColumnTransform(normalized_entropy), ['Count Value[A,A type]']),
        ('Normalized Entropy', ['B','B type'], MultiColumnTransform(normalized_entropy), ['Count Value[B,B type]']),
        ('Max', ['Normalized Entropy[A,A type]','Normalized Entropy[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Normalized Entropy[A,A type]','Normalized Entropy[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Normalized Entropy[A,A type]','Normalized Entropy[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]', SimpleTransform(abs)),

        ('IGCI', ['A','A type','B','B type'], MultiColumnTransform(igci), ['Normalized Value[A,A type]', 'Normalized Value[B,B type]']),
        ('IGCI', ['B','B type','A','A type'], MultiColumnTransform(igci), ['Normalized Value[B,B type]', 'Normalized Value[A,A type]']),
        ('Sub', ['IGCI[A,A type,B,B type]','IGCI[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[IGCI[A,A type,B,B type],IGCI[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Gaussian Divergence', ['A','A type'], MultiColumnTransform(gaussian_divergence), ['Count Value[A,A type]']),
        ('Gaussian Divergence', ['B','B type'], MultiColumnTransform(gaussian_divergence), ['Count Value[B,B type]']),
        ('Max', ['Gaussian Divergence[A,A type]','Gaussian Divergence[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Gaussian Divergence[A,A type]','Gaussian Divergence[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Gaussian Divergence[A,A type]','Gaussian Divergence[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Gaussian Divergence[A,A type],Gaussian Divergence[B,B type]]', SimpleTransform(abs)),

        ('Uniform Divergence', ['A','A type'], MultiColumnTransform(uniform_divergence), ['Count Value[A,A type]']),
        ('Uniform Divergence', ['B','B type'], MultiColumnTransform(uniform_divergence), ['Count Value[B,B type]']),
        ('Max', ['Uniform Divergence[A,A type]','Uniform Divergence[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Uniform Divergence[A,A type]','Uniform Divergence[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Uniform Divergence[A,A type]','Uniform Divergence[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]', SimpleTransform(abs)),

        ('Discrete Entropy', ['A','A type'], MultiColumnTransform(discrete_entropy), ['DisProb[A,A type]']),
        ('Discrete Entropy', ['B','B type'], MultiColumnTransform(discrete_entropy), ['DisProb[B,B type]']),
        ('Max', ['Discrete Entropy[A,A type]','Discrete Entropy[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Discrete Entropy[A,A type]','Discrete Entropy[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Discrete Entropy[A,A type]','Discrete Entropy[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]', SimpleTransform(abs)),

        ('Normalized Discrete Entropy', ['A','A type'], MultiColumnTransform(normalized_discrete_entropy), ['Discrete Entropy[A,A type]', 'Number of Unique Samples[A]']),
        ('Normalized Discrete Entropy', ['B','B type'], MultiColumnTransform(normalized_discrete_entropy), ['Discrete Entropy[B,B type]', 'Number of Unique Samples[B]']),
        ('Max', ['Normalized Discrete Entropy[A,A type]','Normalized Discrete Entropy[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Normalized Discrete Entropy[A,A type]','Normalized Discrete Entropy[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Normalized Discrete Entropy[A,A type]','Normalized Discrete Entropy[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]', SimpleTransform(abs)),

        ('Discrete Joint Entropy', ['A','A type','B','B type'], MultiColumnTransform(discrete_joint_entropy), ['DisSeq[A,A type]', 'DisSeq[B,B type]']),
        ('Normalized Discrete Joint Entropy', ['A','A type','B','B type'], MultiColumnTransform(normalized_discrete_joint_entropy), ['Discrete Joint Entropy[A,A type,B,B type]']),
        ('Discrete Conditional Entropy', ['A','A type','B','B type'], MultiColumnTransform(discrete_conditional_entropy), ['Discrete Joint Entropy[A,A type,B,B type]', 'Discrete Entropy[B,B type]']),
        ('Discrete Conditional Entropy', ['B','B type','A','A type'], MultiColumnTransform(discrete_conditional_entropy), ['Discrete Joint Entropy[A,A type,B,B type]', 'Discrete Entropy[A,A type]']),
        ('Discrete Mutual Information', ['A','A type','B','B type'], MultiColumnTransform(discrete_mutual_information), ['Discrete Joint Entropy[A,A type,B,B type]', 'Discrete Entropy[A,A type]', 'Discrete Entropy[B,B type]']),
        ('Normalized Discrete Mutual Information', ['Discrete Mutual Information[A,A type,B,B type]','Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]'], MultiColumnTransform(operator.div)),
        ('Normalized Discrete Mutual Information', ['Discrete Mutual Information[A,A type,B,B type]','Discrete Joint Entropy[A,A type,B,B type]'], MultiColumnTransform(operator.div)),
        ('Adjusted Mutual Information', ['A','A type','B','B type'], MultiColumnTransform(adjusted_mutual_information), ['DisSeq[A,A type]', 'DisSeq[B,B type]']),

        ('Polyfit', ['A','A type','B','B type'], MultiColumnTransform(fit)),
        ('Polyfit', ['B','B type','A','A type'], MultiColumnTransform(fit)),
        ('Sub', ['Polyfit[A,A type,B,B type]','Polyfit[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Polyfit[A,A type,B,B type],Polyfit[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Polyfit Error', ['A','A type','B','B type'], MultiColumnTransform(fit_error)),
        ('Polyfit Error', ['B','B type','A','A type'], MultiColumnTransform(fit_error)),
        ('Sub', ['Polyfit Error[A,A type,B,B type]','Polyfit Error[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Polyfit Error[A,A type,B,B type],Polyfit Error[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Normalized Error Probability', ['A','A type','B','B type'], MultiColumnTransform(normalized_error_probability), ['DisSeq[A,A type]', 'DisSeq[B,B type]', 'DisProb[A,A type]', 'DisProb[B,B type]']),
        ('Normalized Error Probability', ['B','B type','A','A type'], MultiColumnTransform(normalized_error_probability), ['DisSeq[B,B type]', 'DisSeq[A,A type]', 'DisProb[B,B type]', 'DisProb[A,A type]']),
        ('Sub', ['Normalized Error Probability[A,A type,B,B type]','Normalized Error Probability[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Normalized Error Probability[A,A type,B,B type],Normalized Error Probability[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Conditional Distribution Entropy Variance', ['A','A type','B','B type'], MultiColumnTransform(fit_noise_entropy), ['DisSeq[A,A type]', 'DisSeq[B,B type]', 'DisProb[A,A type]']),
        ('Conditional Distribution Entropy Variance', ['B','B type','A','A type'], MultiColumnTransform(fit_noise_entropy), ['DisSeq[B,B type]', 'DisSeq[A,A type]', 'DisProb[B,B type]']),
        ('Sub', ['Conditional Distribution Entropy Variance[A,A type,B,B type]','Conditional Distribution Entropy Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Conditional Distribution Skewness Variance', ['A','A type','B','B type'], MultiColumnTransform(fit_noise_skewness), ['DisSeq[A,A type]', 'DisProb[A,A type]']),
        ('Conditional Distribution Skewness Variance', ['B','B type','A','A type'], MultiColumnTransform(fit_noise_skewness), ['DisSeq[B,B type]', 'DisProb[B,B type]']),
        ('Sub', ['Conditional Distribution Skewness Variance[A,A type,B,B type]','Conditional Distribution Skewness Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Conditional Distribution Kurtosis Variance', ['A','A type','B','B type'], MultiColumnTransform(fit_noise_kurtosis), ['DisSeq[A,A type]', 'DisProb[A,A type]']),
        ('Conditional Distribution Kurtosis Variance', ['B','B type','A','A type'], MultiColumnTransform(fit_noise_kurtosis), ['DisSeq[B,B type]', 'DisProb[B,B type]']),
        ('Sub', ['Conditional Distribution Kurtosis Variance[A,A type,B,B type]','Conditional Distribution Kurtosis Variance[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]', SimpleTransform(abs)),

        ('DisSeq2', ['A','A type'], MultiColumnTransform(discrete_seq2)),
        ('DisSeq2', ['B','B type'], MultiColumnTransform(discrete_seq2)),
        ('DisProb2', ['A','A type'], MultiColumnTransform(discrete_probability2), ['DisSeq2[A,A type]']),
        ('DisProb2', ['B','B type'], MultiColumnTransform(discrete_probability2), ['DisSeq2[B,B type]']),
        ('Conditional Distribution Similarity', ['A','A type','B','B type'], MultiColumnTransform(conditional_distribution_similarity), ['DisSeq2[A,A type]','DisProb2[A,A type]','DisProb2[B,B type]']),
        ('Conditional Distribution Similarity', ['B','B type','A','A type'], MultiColumnTransform(conditional_distribution_similarity), ['DisSeq2[B,B type]','DisProb2[B,B type]','DisProb2[A,A type]']),
        ('Sub', ['Conditional Distribution Similarity[A,A type,B,B type]','Conditional Distribution Similarity[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Conditional Distribution Similarity[A,A type,B,B type],Conditional Distribution Similarity[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Moment21', ['A','A type','B','B type'], MultiColumnTransform(moment21), ['Normalized Value[A,A type]', 'Normalized Value[B,B type]']),
        ('Moment21', ['B','B type','A','A type'], MultiColumnTransform(moment21), ['Normalized Value[B,B type]', 'Normalized Value[A,A type]']),
        ('Sub', ['Moment21[A,A type,B,B type]','Moment21[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Moment21[A,A type,B,B type],Moment21[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Abs', 'Moment21[A,A type,B,B type]', SimpleTransform(abs)),
        ('Abs', 'Moment21[B,B type,A,A type]', SimpleTransform(abs)),
        ('Sub', ['Abs[Moment21[A,A type,B,B type]]','Abs[Moment21[B,B type,A,A type]]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]', SimpleTransform(abs)),

        ('Moment31', ['A','A type','B','B type'], MultiColumnTransform(moment31), ['Normalized Value[A,A type]', 'Normalized Value[B,B type]']),
        ('Moment31', ['B','B type','A','A type'], MultiColumnTransform(moment31), ['Normalized Value[B,B type]', 'Normalized Value[A,A type]']),
        ('Sub', ['Moment31[A,A type,B,B type]','Moment31[B,B type,A,A type]'], MultiColumnTransform(operator.sub)),
        ('Abs','Sub[Moment31[A,A type,B,B type],Moment31[B,B type,A,A type]]', SimpleTransform(abs)),

        ('Abs','Moment31[A,A type,B,B type]', SimpleTransform(abs)),
        ('Abs','Moment31[B,B type,A,A type]', SimpleTransform(abs)),
        ('Sub', ['Abs[Moment31[A,A type,B,B type]]','Abs[Moment31[B,B type,A,A type]]'], MultiColumnTransform(operator.sub)),
        ('Abs','Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]', SimpleTransform(abs)),

        ('Skewness', ['A','A type'], MultiColumnTransform(normalized_skewness), ['Normalized Value[A,A type]']),
        ('Skewness', ['B','B type'], MultiColumnTransform(normalized_skewness), ['Normalized Value[B,B type]']),
        ('Sub', ['Skewness[A,A type]','Skewness[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Skewness[A,A type],Skewness[B,B type]]', SimpleTransform(abs)),

        ('Abs', 'Skewness[A,A type]', SimpleTransform(abs)),
        ('Abs', 'Skewness[B,B type]', SimpleTransform(abs)),
        ('Max', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(max)),
        ('Min', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(min)),
        ('Sub', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]', SimpleTransform(abs)),

        ('Kurtosis', ['A','A type'], MultiColumnTransform(normalized_kurtosis), ['Normalized Value[A,A type]']),
        ('Kurtosis', ['B','B type'], MultiColumnTransform(normalized_kurtosis), ['Normalized Value[B,B type]']),
        ('Max', ['Kurtosis[A,A type]','Kurtosis[B,B type]'], MultiColumnTransform(max)),
        ('Min', ['Kurtosis[A,A type]','Kurtosis[B,B type]'], MultiColumnTransform(min)),
        ('Sub', ['Kurtosis[A,A type]','Kurtosis[B,B type]'], MultiColumnTransform(operator.sub)),
        ('Abs', 'Sub[Kurtosis[A,A type],Kurtosis[B,B type]]', SimpleTransform(abs)),

        ('Pearson R', ['A','A type','B','B type'], MultiColumnTransform(correlation), ['Normalized Error Probability[A,A type,B,B type]','Normalized Error Probability[B,B type,A,A type]']),
        ('HSIC', ['A','A type','B','B type'], MultiColumnTransform(normalized_hsic), ['Pearson R[A,A type,B,B type]']),
        ('Abs', 'Pearson R[A,A type,B,B type]', SimpleTransform(abs))
        ]

    all_features = [fea if len(fea) > 3 else (fea[0], fea[1], fea[2], []) for fea in all_features]

    used_feature_names = set(selected_direction_categorical_features + selected_direction_cn_features + selected_direction_numerical_features
                             +selected_independence_categorical_features + selected_independence_cn_features + selected_independence_numerical_features
                             +selected_symmetric_categorical_features + selected_symmetric_cn_features + selected_symmetric_numerical_features
                             +selected_onestep_categorical_features + selected_onestep_cn_features + selected_onestep_numerical_features)
    used_feature_names_add = used_feature_names
    all_features_clean = []
    for feature_prefix, column_names, extractor, aux_column_names in reversed(all_features):
        if not type(column_names) is list:
            column_names = [column_names]

        feature_name = feature_prefix + '[' + ','.join(column_names) + ']'
        if len([x for x in used_feature_names_add if feature_name in x]) > 0:
            used_feature_names_add = set(list(used_feature_names_add) + column_names + aux_column_names)

    for feature_prefix, column_names, extractor, aux_column_names in all_features:
        if type(column_names) is list:
            feature_name = feature_prefix + '[' + ','.join(column_names) + ']'
        else:
            feature_name = feature_prefix + '[' + column_names + ']'

        if len([x for x in used_feature_names_add if feature_name in x]) > 0:
            all_features_clean.append((feature_prefix, column_names, extractor, aux_column_names))

    return all_features_clean, used_feature_names


def extract_features(X, features=None, y=None):
    if features is None:
        features, _ = get_all_features()
    X = X.copy()
    X['A type'] = pd.Series.apply(X['A type'], lambda x: 0 if x == BINARY else 1 if x == CATEGORICAL else 2 if x == NUMERICAL else np.nan)
    X['B type'] = pd.Series.apply(X['B type'], lambda x: 0 if x == BINARY else 1 if x == CATEGORICAL else 2 if x == NUMERICAL else np.nan)
    for feature_name, column_names, extractor, aux_column_names in features:
        if not type(column_names) is list:
            column_names = [column_names]

        feature_name = feature_name + '[' + ','.join(column_names) + ']'
        if (feature_name[0] == '+') or (feature_name not in X.columns):
            if feature_name[0] == '+':
                feature_name = feature_name[1:]

            column_names = column_names + aux_column_names
            if len(column_names) > 1:
                tmp = extractor.fit_transform(X[column_names], y)
            else:
                tmp = extractor.fit_transform(X[column_names[0]], y)

            X[feature_name] = tmp

    return X

def get_sym_col(col):
    col = col.replace('A type', '<>').replace('B type', 'A type').replace('<>', 'B type')
    col = col.replace('[A', '<>').replace('[B', '[A').replace('<>', '[B')
    col = col.replace(',A,', '<>').replace(',B,', ',A,').replace('<>', ',B,')
    return col

def extract_features2(X, X_inv, features=None, y=None): #, used_feature_names=used_feature_names):
    symmetric_feature_names = ['HSIC[A,A type,B,B type]', 'Pearson R[A,A type,B,B type]', 'Discrete Joint Entropy[A,A type,B,B type]', 'Adjusted Mutual Information[A,A type,B,B type]']
    if features is None:
        features, _ = get_all_features()
    X = X.copy()
    X['A type'] = pd.Series.apply(X['A type'], lambda x: 0 if x == BINARY else 1 if x == CATEGORICAL else 2 if x == NUMERICAL else np.nan)
    X['B type'] = pd.Series.apply(X['B type'], lambda x: 0 if x == BINARY else 1 if x == CATEGORICAL else 2 if x == NUMERICAL else np.nan)
    for feature_name, column_names, extractor, aux_column_names in features:
        if not type(column_names) is list:
            column_names = [column_names]

        feature_name = feature_name + '[' + ','.join(column_names) + ']'
        if feature_name in symmetric_feature_names:
            X[feature_name] = X_inv[feature_name]
            continue

        if get_sym_col(feature_name) in X_inv.columns:
            X[feature_name] = X_inv[get_sym_col(feature_name)]
            continue

        if (feature_name[0] == '+') or (feature_name not in X.columns):
            if feature_name[0] == '+':
                feature_name = feature_name[1:]

            column_names = column_names + aux_column_names
            if len(column_names) > 1:
                tmp = extractor.fit_transform(X[column_names], y)
            else:
                tmp = extractor.fit_transform(X[column_names[0]], y)

            X[feature_name] = tmp

    return X
