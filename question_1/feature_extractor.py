import pandas as pd
import numpy as np

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

def derive_tc_icf_prereqs(train_data, labels, unique_labels):
    class_term_freq = np.array([dict() for category in unique_labels])
    term_to_class_dict = dict()
    for index, category in enumerate(unique_labels):
        samples_in_category  = train_data[labels == category]
        count_vect = CountVectorizer(min_df=5, stop_words='english')
        count_features = count_vect.fit_transform(samples_in_category)

        # sum_of_other_terms = np.sum(count_features.toarray().sum(axis=1))
        # count_features = count_features.toarray()/sum_of_other_terms
        count_features = count_features.toarray().sum(axis=0)
        # count_features = count_features / count_features.sum(axis=0)
        class_term_freq[category]= dict(zip(count_vect.get_feature_names(),count_features)) 
                # inverse class frequence computation

        number_of_classes = unique_labels

        for dict_key in count_vect.get_feature_names():
            if dict_key not in term_to_class_dict:
                term_to_class_dict[dict_key] =[category]
            else:
                term_to_class_dict[dict_key].append(category)

    return class_term_freq,term_to_class_dict

def derivetficf_and_reduce_features(class_term_freq,term_to_class_dict,unique_labels):
    tf_icf_scores = np.array([dict() for category in unique_labels])
    selected_terms_for_each_class = np.array([dict() for category in unique_labels])
    for index, category in enumerate(unique_labels):
        terms_with_frequencies = class_term_freq[category]
        tf_icf = dict()
        for term in terms_with_frequencies.keys():
            icf = np.log(len(unique_labels)/len(term_to_class_dict[term]))
            tf_icf[term] = terms_with_frequencies[term] * icf
        tf_icf_scores[category] = tf_icf
    for  index, category in enumerate(unique_labels):
        terms_with_frequencies = class_term_freq[category]

        tf_icf_for_class = tf_icf_scores[category]
        terms = tf_icf_for_class.keys()
        tf_icf_scores_with_terms = sorted(list(zip(tf_icf_for_class.values(),terms)),reverse=True)[:top_k]
        top_terms = [value[1] for value in tf_icf_scores_with_terms]
        count_values = [terms_with_frequencies[term] for term in top_terms]
        selected_terms_for_each_class[category] = dict(zip(top_terms,count_values))

    return selected_terms_for_each_class



if __name__=="__main__":
    top_k = int(input("enter value of k for top k feature selection in each class"))
    train_80_20 = pd.read_csv("train_80_20_split.csv")
    test_80_20 = pd.read_csv("test_80_20_split.csv")
    LE = preprocessing.LabelEncoder()
    train_80_20['label'] = LE.fit_transform(train_80_20['label'].values)
    train_80_20 = train_80_20[train_80_20.notnull()]
    train_80_20['text'] = train_80_20.text.astype(str)
    class_term_freq, term_to_class_dict = derive_tc_icf_prereqs(train_80_20['text'].values,train_80_20['label'].values, list(set(train_80_20['label'].values)))
    print("class_term_freq",class_term_freq[0]['jpeg'])
    selected_features = derivetficf_and_reduce_features(class_term_freq,term_to_class_dict,list(set(train_80_20['label'].values)))
    joblib.dump(selected_features,"selected_features_80_20_k_{}".format(top_k))
    train_70_30 = pd.read_csv("train_70_30_split.csv")
    test_70_30 = pd.read_csv("test_70_30_split.csv")
    LE = preprocessing.LabelEncoder()
    train_70_30['label'] = LE.fit_transform(train_70_30['label'].values)
    train_70_30 = train_70_30[train_70_30.notnull()]
    train_70_30['text'] = train_70_30.text.astype(str)
    class_term_freq, term_to_class_dict = derive_tc_icf_prereqs(train_70_30['text'].values,train_70_30['label'].values, list(set(train_70_30['label'].values)))
    print("class_term_freq",class_term_freq[0]['jpeg'])
    selected_features = derivetficf_and_reduce_features(class_term_freq,term_to_class_dict,list(set(train_70_30['label'].values)))
    joblib.dump(selected_features,"selected_features_70_30_k_{}".format(top_k))
    train_50_50 = pd.read_csv("train_50_50_split.csv")
    test_50_50 = pd.read_csv("test_50_50_split.csv")
    LE = preprocessing.LabelEncoder()
    train_50_50['label'] = LE.fit_transform(train_50_50['label'].values)
    train_50_50 = train_50_50[train_50_50.notnull()]
    train_50_50['text'] = train_50_50.text.astype(str)
    class_term_freq, term_to_class_dict = derive_tc_icf_prereqs(train_50_50['text'].values,train_50_50['label'].values, list(set(train_50_50['label'].values)))
    print("class_term_freq",class_term_freq[0]['jpeg'])
    selected_features = derivetficf_and_reduce_features(class_term_freq,term_to_class_dict,list(set(train_50_50['label'].values)))
    joblib.dump(selected_features,"selected_features_50_50_k_{}".format(top_k))