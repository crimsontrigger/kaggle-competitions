from base import Feature, get_arguments, generate_features

import pandas as pd
import numpy as np 
from scipy.stats import skew

from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier

from sklearn.decomposition import PCA

from base import *

def replace_columns(train,test,class_name,cols_len):
    columns = ["{}{}".format(class_name,i) for i in range(cols_len)]
    train.columns = columns
    test.colummns = columns

    return train,test

class select_features(Feature):
    def create_features(self):
        select = get_leak_columns()
        self.train = train[select]
        self.test = test[select]

class subsets_statics(Feature):
    def create_features(self):
         extra_columns = get_extra_columns()
         i = 0
         for cols in extra_columns:
            i += 1 
            tmp_train = train[cols].replace({0:np.nan})
            tmp_test = test[cols].replace({0:np.nan})

            for df,self_df in [(tmp_train,self.train),(tmp_test,self.test)]:
                self_df["subset{}_count_not0".format(i)] = df.count(axis=1)
                self_df["subset{}_sum".format(i)] = df.sum(axis=1)
                self_df["subset{}_var".format(i)] = df.var(axis=1)
                self_df["subset{}_median".format(i)] = df.median(axis=1)
                self_df["subset{}_mean".format(i)] = df.mean(axis=1)
                self_df["subset{}_std".format(i)] = df.std(axis=1)
                self_df["subset{}_max".format(i)] = df.max(axis=1)
                self_df["subset{}_min".format(i)] = df.min(axis=1)
                self_df["subset{}_skew".format(i)] = df.skew(axis=1)
                self_df["subset{}_kurtosis".format(i)] = df.kurtosis(axis=1)

             
class leak_columns_statics(Feature):
    def create_features(self):
        cols = get_leak_columns()
        tmp_train = train[cols].replace({0:np.nan})
        tmp_test = test[cols].replace({0:np.nan})

        for df,self_df in [(tmp_train,self.train),(tmp_test,self.test)]:
            self_df["leak_columns_count_not0"] = df.count(axis=1)
            self_df["leak_columns_sum"] = df.sum(axis=1)
            self_df["leak_columns_var"] = df.var(axis=1)
            self_df["leak_columns_median"] = df.median(axis=1)
            self_df["leak_columns_mean"] = df.mean(axis=1)
            self_df["leak_columns_std"] = df.std(axis=1)
            self_df["leak_columns_max"] = df.max(axis=1)
            self_df["leak_columns_min"] = df.min(axis=1)
            self_df["leak_columns_skew"] = df.skew(axis=1)
            self_df["leak_columns_kurtosis"] = df.kurtosis(axis=1)

        del(tmp_train)
        del(tmp_test)

class statics(Feature):
    def create_features(self):
        #train=train.replaceにするとtrainがローカル変数になってしまう
        tmp_train = train.replace({0:np.nan})
        tmp_test = test.replace({0:np.nan})

        for df,self_df in [(tmp_train,self.train),(tmp_test,self.test)]:
            self_df["count_not0"] = df.count(axis=1)
            self_df["sum"] = df.sum(axis=1)
            self_df["var"] = df.var(axis=1)
            self_df["median"] = df.median(axis=1)
            self_df["mean"] = df.mean(axis=1)
            self_df["std"] = df.std(axis=1)
            self_df["max"] = df.max(axis=1)
            self_df["min"] = df.min(axis=1)
            self_df["skew"] = df.skew(axis=1)
            self_df["kurtosis"] = df.kurtosis(axis=1)

        del(tmp_train)
        del(tmp_test)

class rfc_label_5(Feature):
    def get_rfc(self):
        return RandomForestClassifier(
        n_estimators=100,
        max_features=0.5,
        max_depth=None,
        max_leaf_nodes=270,
        min_impurity_decrease=0.0001,
        random_state=123,
        n_jobs=-1
    )

    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_class)
        
        for i_class in range(self.n_class):
            if i_class + 1 == self.n_class:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels

    def create_features(self):
        self.n_class = 5
        self.cv = 5
        estimator = self.get_rfc()
        estimators = []
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(estimator))
        
        
        for tr_idx,_ in cv.split(train,y_labels):
            estimators.append(
                clone(estimator).fit(train.loc[tr_idx], y_labels[tr_idx])
            )
        train_prob = np.zeros([train.shape[0],self.n_class])
        train_pred = np.zeros(train.shape[0])

        test_prob = np.zeros([test.shape[0],self.n_class])
        test_pred = np.zeros(test.shape[0])

        cv = check_cv(self.cv, classifier=is_classifier(estimator))
        for estimator, (_, te_idx) in zip(estimators, cv.split(train)):
            train_prob[te_idx] = estimator.predict_proba(train.loc[te_idx])
            train_pred[te_idx] = estimator.predict(train.loc[te_idx])

        for estimator, (_, te_idx) in zip(estimators, cv.split(test)):
            test_prob[te_idx] = estimator.predict_proba(test.loc[te_idx])
            test_pred[te_idx] = estimator.predict(test.loc[te_idx])
        
        tmp_train = pd.DataFrame(train_prob)
        tmp_test  = pd.DataFrame(test_prob)
        tmp_train["class_pred"] =  np.array([train_pred]).T
        tmp_test["class_pred"] = np.array([test_pred]).T
        
        columns = ["{}_prob".format(i) for i in range(self.n_class)] + ["class_pred"]
        tmp_train.columns = columns
        tmp_test.column = columns

        self.train = tmp_train
        self.test =  tmp_test
        

class timespan(Feature):
    def create_features(self):
        cols = get_leak_columns()
        for time in [10,20,30,len(cols)]:
            for df,self_df in [(train,self.train),(test,self.test)]:
                tmp_df = df[cols[:time]].replace({0:np.nan})
                self_df["mean_{}".format(time)] = tmp_df.mean(axis=1)
                self_df["sum_{}".format(time)]  = tmp_df.sum(axis=1)
                self_df["diff_mean_{}".format(time)] = tmp_df.diff(axis=1).mean(axis=1)
                self_df["min_{}".format(time)] = tmp_df.min(axis=1)
                self_df["max_{}".format(time)] = tmp_df.max(axis=1)
                self_df["count_not_0_{}".format(time)] = tmp_df.count(axis=1)
 
class RandomProjection(Feature):
    def create_features(self):
        n_com = 100
        transformer = random_projection.SparseRandomProjection(n_components = n_com)

        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["RandomProjection{}".format(i) for i in range(n_com)]
        self.train.columns = columns
        self.test.columns = columns


class Principal_Component_Analysis(Feature):
    def create_features(self):
        n_com = 100

        pca = PCA()

        pca.fit(train)
        self.train = pd.DataFrame(pca.transform(train))
        self.test = pd.DataFrame(pca.transform(test))

        t = pd.DataFrame(pca.explained_variance_ratio_)

        for i in range(1,len(t)):
            t.loc[i] += t.loc[i-1]
            if t.loc[i].values > 0.5:#need tune
                end = i 
                break
        del(t)

        self.train = self.train[self.train.columns[:end]]
        self.test = self.test[self.test.columns[:end]]

    
        columns = ["PCA{}".format(i) for i in range(end)]
        self.train.columns = columns
        self.test.columns = columns
        

class tSVD(Feature):
    def create_features(self):
        n_com = 100

        transformer =TruncatedSVD(n_components = n_com)
        
        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["TruncatedSVD{}".format(i) for i in range(n_com)]
        self.train.columns = columns
        self.test.columns = columns


if __name__ == '__main__':
    args = get_arguments()

    train,test = get_input()
    #train_leak,test_leak = get_leak_df()

    #train = train[train_leak["compiled_leak"] == 0]
    #test = test[test_leak["compiled_leak"] == 0]
    y = train["target"]
    train = train.drop(["ID","target"],axis=1)
    test = test.drop("ID",axis=1)
    
    generate_features(globals(), args.force)

