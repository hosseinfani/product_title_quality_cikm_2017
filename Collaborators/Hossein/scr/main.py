# coding: utf-8
import sys
import getopt
import numpy as np
import traceback

from time import time
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, LinearRegression, RandomizedLogisticRegression, RandomizedLasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif, RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD

from xgboost import XGBClassifier #http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

from dal import Dal
from text_corpus import TextCorpus
from brand import Brand
import utils

random_seed = 7881
np.random.seed(random_seed)

def load_features(file):
    return np.load(file)

def extract_features(products, brands, words=None, chars=None, tofile=None):
    t_0 = time()
    char_ngram_range = (1, 1)
    word_ngram_range = (1, 1)
    if words is None and chars is None:
        tc_title_desc = TextCorpus(products['Title'].append(products['Description']),
                                   char_ngram_range=char_ngram_range,
                                   word_ngram_range=word_ngram_range)
        words = tc_title_desc.words
        chars = tc_title_desc.chars
    tc_title = TextCorpus(products['Title'],
                          words=words,
                          word_ngram_range=word_ngram_range,
                          chars=chars,
                          char_ngram_range=char_ngram_range
                          )
    tc_desc = TextCorpus(products['Description'],
                         words=words,
                         word_ngram_range=word_ngram_range,
                         chars=chars,
                         char_ngram_range=char_ngram_range
                         )
    features = sparse.csr_matrix(
               [tc_title.getLengths()] +
               [tc_title.getLengthsByTerm()] +
               [tc_title.getSpecialCharStat()[0]] +
               [tc_title.hasSpecialChar()] +
               [tc_title.getUpperCharStat()[0]] +
               [tc_title.getNounStat()] +
               [tc_title.getVerbStat()] +
               [tc_title.getAdjectiveStat()] +
               [tc_title.hasNumber()] +
               [tc_title.getNumberStat()] +
               [tc_title.hasTamilChar()] +
               [tc_title.hasChineseChar()] +
               [tc_title.getNonEnglishCharStat()[0]] +
               [tc_title.getColorStat()[0]] +
               [tc_title.getBrandStat(brands)] +
               [tc_title.getSyllableStat()] +
               [tc_title.getPolysyllabStat()] +
               [tc_title.getAvgLetterPerWord()] +
               [tc_title.getAvgSentencePerWord()] +
               [tc_title.getAvgSyllablePerWord()] +
               [tc_title.getColemanLiauIndex()] +
               [tc_title.getDaleChallReadabilityScore()] +
               [tc_title.getAutomatedReadabilityIndex()] +
               [tc_title.getDifficultWordsStat()] +
               [tc_title.getFleschReadingEase()] +
               [tc_title.getFleschKincaidGrade()] +
               [tc_title.getGunningFog()] +
               [tc_title.getLexiconStat()] +
               [tc_title.getLinsearWriteFormula()] +
               [tc_title.getSmogIndex()] +
               [tc_title.getTextStandardLevel()] +
               [products["CountryId"].tolist()] +
               [products["SkuIdPrefix"].tolist()] +
               [products["CategoryId"].tolist()] +
               [products["SubCategoryId"].tolist()] +
               [products["SubSubCategoryId"].tolist()] +
               [products["DelivaryTypeId"].tolist()] +
               [products["AdjustedPrice"].tolist()] +
               [tc_desc.getLengths()] +
               [tc_desc.getLengthsByTerm()] +
               [tc_desc.getSpecialCharStat()[0]] +
               [tc_desc.hasSpecialChar()] +
               [tc_desc.getUpperCharStat()[0]] +
               [tc_desc.getNounStat()] +
               [tc_desc.getVerbStat()] +
               [tc_desc.getAdjectiveStat()] +
               [tc_desc.hasNumber()] +
               [tc_desc.getNumberStat()] +
               [tc_desc.hasTamilChar()] +
               [tc_desc.hasChineseChar()] +
               [tc_desc.getNonEnglishCharStat()[0]] +
               [tc_desc.getColorStat()[0]] +
               [tc_desc.getBrandStat(brands)] +
               [tc_desc.getSyllableStat()] +
               [tc_desc.getPolysyllabStat()] +
               [tc_desc.getAvgLetterPerWord()] +
               [tc_desc.getAvgSentencePerWord()] +
               [tc_desc.getAvgSyllablePerWord()] +
               [tc_desc.getColemanLiauIndex()] +
               [tc_desc.getDaleChallReadabilityScore()] +
               [tc_desc.getAutomatedReadabilityIndex()] +
               [tc_desc.getDifficultWordsStat()] +
               [tc_desc.getFleschReadingEase()] +
               [tc_desc.getFleschKincaidGrade()] +
               [tc_desc.getGunningFog()] +
               [tc_desc.getLexiconStat()] +
               [tc_desc.getLinsearWriteFormula()] +
               [tc_desc.getSmogIndex()] +
               [tc_desc.getTextStandardLevel()]
                )

    features = sparse.csr_matrix(sparse.hstack((
                tc_title.getCharStat()[0],
                tc_title.getTermStat()[0],
                tc_title.getTfIdF(),
                tc_desc.getCharStat()[0],
                tc_desc.getTermStat()[0],
                tc_desc.getTfIdF(),
                features.transpose())))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_basic.npz', features)
        np.savez_compressed(tofile.split('.')[0] + '_basic_dict.npz', words=words, chars=chars)

    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)

    features = sparse.csr_matrix(sparse.hstack((
        tc_title.getCharBM25(),
        tc_title.getTermBM25(),
        tc_desc.getCharBM25(),
        tc_desc.getTermBM25())))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_bm25.npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    dim = 5
    win = 1
    #doc2vec
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(tc_title.getEmbeddingsByChar(dim, win)),
        tc_title.getEmbeddingsByTerm(dim, win),
        tc_desc.getEmbeddingsByChar(dim, win),
        tc_desc.getEmbeddingsByTerm(dim, win))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_d2v(50_1).npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)



    # w2v_sum
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(tc_title.getEmbeddingsByChar(dim, win, op='sum')),
        tc_title.getEmbeddingsByTerm(dim, win, op='sum'),
        tc_desc.getEmbeddingsByChar(dim, win, op='sum'),
        tc_desc.getEmbeddingsByTerm(dim, win, op='sum'))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_w2v(50_1_sum).npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    #w2v_avg
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(tc_title.getEmbeddingsByChar(dim, win, op='avg')),
        tc_title.getEmbeddingsByTerm(dim, win, op='avg'),
        tc_desc.getEmbeddingsByChar(dim, win, op='avg'),
        tc_desc.getEmbeddingsByTerm(dim, win, op='avg'))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_w2v(' + str(dim) + '_' + str(win) + '_avg).npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    lsi = TruncatedSVD(n_components=dim, n_iter=10, algorithm='randomized')
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(lsi.fit_transform(tc_title.getCharStat()[0])),
        lsi.fit_transform(tc_title.getTermStat()[0]),
        lsi.fit_transform(tc_desc.getCharStat()[0]),
        lsi.fit_transform(tc_desc.getTermStat()[0]))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_lsi(' + str(dim) + ').npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    pca = PCA(n_components=dim)
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(pca.fit_transform(tc_title.getCharStat()[0].toarray())),
        pca.fit_transform(tc_title.getTermStat()[0].toarray()),
        pca.fit_transform(tc_desc.getCharStat()[0].toarray()),
        pca.fit_transform(tc_desc.getTermStat()[0].toarray()))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_pca(' + str(dim) + ').npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    tsne = TSNE(n_components=dim)
    features = sparse.csr_matrix(sparse.hstack((
        sparse.csr_matrix(tsne.fit_transform(tc_title.getCharStat()[0].toarray())),
        tsne.fit_transform(tc_title.getTermStat()[0].toarray()),
        tsne.fit_transform(tc_desc.getCharStat()[0].toarray()),
        tsne.fit_transform(tc_desc.getTermStat()[0].toarray()))))

    if tofile is not None:
        utils.save_sparse_csr(tofile.split('.')[0] + '_tsne(' + str(dim) + ').npz', features)
    print("Feature's shape (data size, feature size): (%s, %s)" % features.shape)


    return features, words, chars

def select_features(X, y):
    # variance
    print 'variance'
    var = VarianceThreshold(threshold=0.1).fit(X)
    for v in var.variances_:
        print v
    # X = var.transform(X)

    #univariate
    chi2_test = SelectKBest(chi2, k=10).fit(np.abs(X), y)
    print 'chi2'
    for i, s in enumerate(chi2_test.scores_):
        print('%f, %f, %d' % (s, chi2_test.pvalues_[i], chi2_test.get_support()[i]))

    print 'f_classif' #f-test: only linear dependency
    f_test = SelectKBest(f_classif, k=10).fit(X, y)
    for i, s in enumerate(f_test.scores_):
        print('%f, %f, %d' % (s, f_test.pvalues_[i], f_test.get_support()[i]))

    print 'mutual_info_classif'  #mutual information: any kind of dependency between variables
    mi = SelectKBest(mutual_info_classif, k=10).fit(X, y)
    for i, s in enumerate(mi.scores_):
        print('%f, %d' % (s, mi.get_support()[i]))

    # #recursive eleimination
    # print 'logistic reg'
    # rf = RFECV(estimator=LogisticRegression(), step=1, cv=StratifiedKFold(10), scoring='accuracy', n_jobs=-1).fit(X, y)
    # for i, r in enumerate(rf.ranking_):
    #     print('%d, %d' % (r, rf.get_support()[i]))
    #
    # print 'linear reg'
    # rf = RFECV(estimator=LinearRegression(), step=1, cv=StratifiedKFold(10), scoring='accuracy', n_jobs=-1).fit(X, y)
    # for i, r in enumerate(rf.ranking_):
    #     print('%d, %d' % (r, rf.get_support()[i]))

    # Decision tree: tends to use all features, tends to overfit
    # GaussianNB, rbfsvm, polysvm,sigmoidsvm cannot be used. Only linear estimators! https://www.researchgate.net/publication/220637867_Feature_selection_for_support_vector_machines_with_RBF_kernel
    # print 'linear SVM'
    # rf = RFECV(estimator=SVC(kernel='linear'), step=1, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1).fit(X, y)
    # for i, r in enumerate(rf.ranking_):
    #     print('%d, %d' % (r, rf.get_support()[i]))

    # # stability selection
    # print 'random logistic reg'
    # rlr = RandomizedLogisticRegression().fit(X, y)
    # for s in rlr.scores_:
    #     print('%f' % s)
    #
    # print 'random lasso'
    # rl = RandomizedLasso().fit(X, y)
    # for s in rl.scores_:
    #     print('%f' % s)

    return X

def learn_kf(X, y, model, folds=10): #better change it to use cross_val_score() function
    t_0 = time()

    scores = []
    if folds < 2:
        model.fit(X, y)
        score = mean_squared_error(model.predict_proba(X)[:, 1], y) ** 0.5
        scores.append(score)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
        for k, (trainIdx, validationIdx) in enumerate(skf.split(X, y)):
            X_train = X[trainIdx, :]
            y_train = y[trainIdx]
            X_validation = X[validationIdx, :]
            y_validation = y[validationIdx]

            model.fit(X_train, y_train)
            score = mean_squared_error(model.predict_proba(X_validation)[:, 1], y_validation) ** 0.5
            scores.append(score)
            # print("Fold #%d RMSE: %f" % (k, score))

    print("train time: %0.3fs" % (time() - t_0))
    print("%s[%d-Fold: (avg=%f, std=%f)]" % (type(model).__name__, folds, np.mean(scores), np.std(scores)))
    return np.asarray(scores), model

if __name__ == "__main__":

    #input parameter parsing
    feature_file_2_load = None
    feature_file_2_save = None
    cores = -1
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:i:o:", ["feature_file_2_load=", "feature_file_2_save="])
    except getopt.GetoptError:
        print 'main.py -i <feature_file_2_load> -o <feature_file_2_save>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -i <feature_file_2_load> -o <feature_file_2_save>'
            sys.exit()
        elif opt in ("-i", "--feature_file_2_load"):
            feature_file_2_load = arg
        elif opt in ("-o", "--feature_file_2_save"):
            feature_file_2_save = arg

    #loading data
    h = Dal()
    product_file_train = '../../Dataset/training/data_train.csv'
    clarity_file = '../../Dataset/training/clarity_train.labels'
    conciseness_file = '../../Dataset/training/conciseness_train.labels'
    products_train = h.readData(product_file_train, clarity_file, conciseness_file)
    train_size = products_train.shape[0]


    # product_file_validation = '../../Dataset/validation/data_valid.csv'
    # products_validation = h.readData(product_file_validation, None, None)

    product_file_validation = '../../Dataset/testing/data_test.csv'
    products_validation = h.readData(product_file_validation, None, None)
    valid_size = products_validation.shape[0]

   # brand_file = '../../Collaborators/Tam/brands_from_lazada_portal.txt'
    brand_file = '../../Collaborators/Hossein/brands_by_category.txt'
    brands = Brand(brand_file).getBrands();

    y_clear = products_train['IsClear'].as_matrix()
    y_concise = products_train['IsConcise'].as_matrix()

    products = products_train.append(products_validation)
    #loading or extracting features
    if feature_file_2_save is not None:
        [X, words, chars] = extract_features(products=products, brands=brands, tofile=feature_file_2_save)#'features_title_desc_attributes.npz')
        # [XX, _, _] = extract_features(products=products_validation, brands=brands, words=words, chars=chars,
        #                                         tofile=feature_file_2_save.split('.')[0] + '_valid.npz')
        exit()
    elif feature_file_2_load is not None:
        X = utils.load_sparse_csr(feature_file_2_load)
        # XX = utils.load_sparse_csr(feature_file_2_load.split('.')[-2] + '_valid.npz')
        loaded = np.load(feature_file_2_load.split('.')[-2] + '_dict.npz')
        words = loaded['words'].tolist()
        chars = loaded['chars'].tolist()
    else:
        [X, words, chars] = extract_features(products=products_train, brands=brands)
        # [XX, _, _] = extract_features(products=products_validation, brands=brands, words=words, chars=chars)

    XX = X[train_size:, :]
    X = X[:train_size, :]

    feature_file_2_load = 'features_train_test/features_title_desc_attributes_word(1_1)_char(1_1)_lsi(5).npz'
    lsi = utils.load_sparse_csr(feature_file_2_load)[:, :10]
    feature_file_2_load = 'features_train_test/features_title_desc_attributes_word(1_1)_char(1_1)_pca(5).npz'
    pca = utils.load_sparse_csr(feature_file_2_load)[:, :10]
    feature_file_2_load = 'features_train_test/features_title_desc_attributes_word(1_1)_char(1_1)_d2v(5_1).npz'
    d2v = utils.load_sparse_csr(feature_file_2_load)[:, :10]
    feature_file_2_load = 'features_train_test/features_title_desc_attributes_word(1_1)_char(1_1)_w2v(5_1_avg).npz'
    w2v_avg = utils.load_sparse_csr(feature_file_2_load)[:, :10]
    feature_file_2_load = 'features_train_test/features_title_desc_attributes_word(1_1)_char(1_1)_w2v(5_1_sum).npz'
    w2v_sum = utils.load_sparse_csr(feature_file_2_load)[:, :10]
    reduced_X = sparse.csr_matrix(sparse.hstack((lsi, pca, d2v, w2v_avg, w2v_sum)))
    # reduced_XX = utils.load_sparse_csr(feature_file_2_load.split('.')[-2] + '_valid.npz')[:, :10]
    reduced_XX = reduced_X[train_size:, :]
    reduced_X = reduced_X[:train_size, :]

    #selecting features
    feature_set = {}

    # i_th feature = X[:, 2 * (len(chars) + 2 * len(words)) + i - 1]
    # feature_set['sample'] = [2, 35, 14]

    # Clarity
    # feature_set['all_reduced'] = xrange(2, 70)
    chi2_top = [2,           11, 17,         24,     26, 27, 35, 36, 37, 39        ] #top10 in chi2 for clarity
    ftest_top = [2, 3, 7, 10, 11, 17,             25,     27, 35, 36                ] #top10 in f-test for clarity
    mi_top = [                     22, 23,         26,     35, 36, 37, 39, 57, 61] #top10 in mi for clarity
    # feature_set['union_chi_f_mi_all_reduced'] = list(set(chi2_top).union(set(ftest_top)).union(set(mi_top)))
    # feature_set['union_chi_f'] = list(set(feature_idx_1).union(set(feature_idx_2)))
    # feature_set['union_chi_mi'] = list(set(feature_idx_1).union(set(feature_idx_3)))
    # feature_set['union_f_mi'] = list(set(feature_idx_2).union(set(feature_idx_3)))
    # feature_set['inters_chi_f'] = list(set(feature_idx_1).intersection(set(feature_idx_2)))
    # feature_set['inters_chi_mi'] = list(set(feature_idx_1).intersection(set(feature_idx_3)))
    # feature_set['inters_f_mi'] = list(set(feature_idx_2).intersection(set(feature_idx_3)))
    # feature_set['inters_chi_f_mi'] = list(set(feature_idx_1).intersection(set(feature_idx_2)).intersection(set(feature_idx_3)))
    #
    # # Conciseness
    chi2_top = [39, 2, 17, 37, 16, 26, 7, 29, 3, 30]  # top10 in chi2 for conciseness
    ftest_top = [17, 2, 7, 3, 29, 25, 30, 16, 38]  # top10 in f-test for conciseness
    mi_top = [39, 22, 2, 26, 17, 23, 7, 3, 29, 20]  # top10 in mi for conciseness
    feature_set['union_chi_f_mi_all_reduced2'] = list(set(chi2_top).union(set(ftest_top)).union(set(mi_top)))
    # feature_set['union_chi_f'] = list(set(feature_idx_1).union(set(feature_idx_2)))
    # feature_set['union_chi_mi'] = list(set(feature_idx_1).union(set(feature_idx_3)))
    # feature_set['union_f_mi'] = list(set(feature_idx_2).union(set(feature_idx_3)))
    # feature_set['inters_chi_f'] = list(set(feature_idx_1).intersection(set(feature_idx_2)))
    # feature_set['inters_chi_mi'] = list(set(feature_idx_1).intersection(set(feature_idx_3)))
    # feature_set['inters_f_mi'] = list(set(feature_idx_2).intersection(set(feature_idx_3)))
    # feature_set['inters_chi_f_mi'] = list(set(feature_idx_1).intersection(set(feature_idx_2)).intersection(set(feature_idx_3)))

    for k, v in feature_set.iteritems():
        print ('feature set %s' % k)
        X_clear = None
        X_validation = None
        for i in v:
            idx = 2 * (len(chars) + 2 * len(words)) + i - 1 - 1 #the last '-1' is for compatibility with excel sheet
            if X_clear is None:
                X_clear = X[:, idx]
                X_validation = XX[:, idx]
            else:
                X_clear = sparse.hstack((X_clear, X[:, idx]))
                X_validation = sparse.hstack((X_validation, XX[:, idx]))

        # print 'clarity...'
        # select_features(X_clear, y_clear)
        # print 'conciseness...'
        # select_features(X_clear, y_concise)
        #
        # exit()
        # Multi-label classification. Note the in validation or test, these true labels are not available
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, sparse.csr_matrix(y_clear).transpose())))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, sparse.csr_matrix(y_concise).transpose())))
        # We have to add to the X_validation the predicted IsConcise as a feature
        # predicted_conciseness_file = './submission/submission_4/conciseness_valid.predict'
        # predicted_conciseness_file = 'BestTamOnConciseness.txt'
        # with open(predicted_conciseness_file) as f:
        #     concisenesses = map(float, f)
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, sparse.csr_matrix([int(round(v)) for v in concisenesses]).transpose())))

        # ngram title char
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, :len(chars)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, :len(chars)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, :len(chars)])))
        #
        # # ngram title term
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars):len(words)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars):len(words)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, len(chars):len(words)])))
        #
        # # ngram title tfidf
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars)+len(words):len(chars)+2*len(words)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars)+len(words):len(chars)+2*len(words)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, len(chars)+len(words):len(chars)+2*len(words)])))
        #
        # # ngram desc char
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars)+2*len(words):2*len(chars)+2*len(words)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, len(chars)+2*len(words):2*len(chars)+2*len(words)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, len(chars)+2*len(words):2*len(chars)+2*len(words)])))
        #
        # # ngram desc term
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, 2*len(chars)+2*len(words):2*len(chars)+3*len(words)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, 2*len(chars)+2*len(words):2*len(chars)+3*len(words)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, 2*len(chars)+2*len(words):2*len(chars)+3*len(words)])))
        #
        # # ngram desc tfidf
        # X_concise = sparse.csr_matrix(sparse.hstack((X_clear, X[:, 2*len(chars)+3*len(words):2*len(chars)+4*len(words)])))
        # X_clear = sparse.csr_matrix(sparse.hstack((X_clear, X[:, 2*len(chars)+3*len(words):2*len(chars)+4*len(words)])))
        # X_validation = sparse.csr_matrix(sparse.hstack((X_validation, XX[:, 2*len(chars)+3*len(words):2*len(chars)+4*len(words)])))

        # reduced features
        X_concise = sparse.csr_matrix(sparse.hstack((X_clear, reduced_X)))
        X_clear = sparse.csr_matrix(sparse.hstack((X_clear, reduced_X)))
        X_validation = sparse.csr_matrix(sparse.hstack((X_validation, reduced_XX)))

        # X_concise = sparse.csr_matrix(X_clear)
        # X_clear = sparse.csr_matrix(X_clear)
        # X_validation = sparse.csr_matrix(X_validation)

        #learning by different models
        models = [
                  LogisticRegression(n_jobs=cores, max_iter=1000, random_state=random_seed),
                  RandomForestClassifier(n_jobs=cores, n_estimators=1000, max_depth=15, random_state=random_seed),  #
                  XGBClassifier(n_jobs=cores, n_estimators=1000, max_depth=15, learning_rate=0.1, random_state=random_seed),
                  # KNeighborsClassifier(n_neighbors=2, random_state=random_seed),  #
                  # DecisionTreeClassifier(max_depth=15, random_state=random_seed),  #
                  # MLPClassifier(alpha=0.001, random_state=random_seed),
                  # AdaBoostClassifier(n_estimators=100, random_state=random_seed),
                  # QuadraticDiscriminantAnalysis(random_state=random_seed),
                  # MultinomialNB(),
                  # GaussianNB(),
                  # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, random_state=random_seed),
                  # SVC(probability=True, kernel="linear", random_state=random_seed),#uses libsvm which is too slow (quadratic to the number of samples)
                              # LinearSVC(random_state=random_seed) #does not have probability
                  ]
        for model in models:
            try:
                if (isinstance(model, GaussianProcessClassifier) or isinstance(model, GaussianNB) or isinstance(model, QuadraticDiscriminantAnalysis)) and (not isinstance(X_clear, np.ndarray)):
                    X_clear = X_clear.toarray()
                    X_concise = X_concise.toarray()
                    X_validation = X_validation.toarray()

                # print(type(model).__name__ + '_clarity ...')
                # [_, _] = learn_kf(X_clear, y_clear, model, 10)
                # [scores, model] = learn_kf(X_clear, y_clear, model, 0)
                # predicted_results = model.predict_proba(X_validation)[:, 1]
                # utils.write_submission(k + '_' + type(model).__name__ + '_clarity_validation.predict', predicted_results)

                print(type(model).__name__ + '_conciseness ...')
                [_, _] = learn_kf(X_concise, y_concise, model, 10)
                [scores, model] = learn_kf(X_concise, y_concise, model, 0)

                # for multi-lable classification
                # predicted_results = model.predict(X_validation)
                # utils.write_submission(type(model).__name__ + '_conciseness_validation_label.predict', predicted_results)

                predicted_results = model.predict_proba(X_validation)[:, 1]
                utils.write_submission(k + '_' + type(model).__name__ + '_conciseness_validation.predict', predicted_results)
            except:
                traceback.print_exc(file=sys.stdout)

        multioutput_models = [
                  RandomForestClassifier(n_jobs=cores, n_estimators=1000, random_state=random_seed),
                  #KNeighborsClassifier(n_neighbors=2, random_state=random_seed),
                  #DecisionTreeClassifier(max_depth=15, random_state=random_seed),
                  ]
        for model in multioutput_models:
            try:
                X = X_clear
                y = np.asarray([y_clear, y_concise]).transpose()

                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
                clear_scores = []
                concise_scores = []
                for k, (trainIdx, validIdx) in enumerate(skf.split(X, y_clear)):#stratified based on clarity
                    X_train = X[trainIdx, :]
                    y_train = y[trainIdx, :]
                    X_valid = X[validIdx, :]
                    y_valid = y[validIdx, :]

                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_valid)
                    clear_scores.append(mean_squared_error(y_pred[0][:, 1], y_valid[:, 0]) ** 0.5)
                    concise_scores.append(mean_squared_error(y_pred[1][:, 1], y_valid[:, 1]) ** 0.5)
                    print("Fold #%d RMSE(clear,concise): (%f, %f)" % (k, clear_scores[-1], concise_scores[-1]))
                print("%s[%d-Fold Clear: (avg=%f, std=%f)]" % (type(model).__name__, 10, np.mean(clear_scores), np.std(clear_scores)))
                print("%s[%d-Fold Concise: (avg=%f, std=%f)]" % (type(model).__name__, 10, np.mean(concise_scores), np.std(concise_scores)))

                model.fit(X, y)
                y_pred = model.predict_proba(X)

                print('MultiOutput_' + type(model).__name__ + '_clarity ...')
                score_clear = mean_squared_error(y_pred[0][:, 1], y_clear) ** 0.5
                print("%s[%d-Fold: (avg=%f, std=%f)]" % ('MultiOutput_' + type(model).__name__, 0, score_clear, 0))

                print('MultiOutput_' + type(model).__name__ + '_conciseness ...')
                score_concise = mean_squared_error(y_pred[1][:, 1], y_concise) ** 0.5
                print("%s[%d-Fold: (avg=%f, std=%f)]" % ('MultiOutput_' + type(model).__name__, 0, score_concise, 0))

                y_validation = model.predict_proba(X_validation)
                utils.write_submission('MultiOutput_' + type(model).__name__ + '_clarity_validation.predict', y_validation[0][:, 1])
                utils.write_submission('MultiOutput_' + type(model).__name__ + '_conciseness_validation.predict', y_validation[1][:, 1])

            except:
                traceback.print_exc(file=sys.stdout)


