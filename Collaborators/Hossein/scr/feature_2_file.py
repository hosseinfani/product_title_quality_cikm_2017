import utils
import numpy as np
from scipy import sparse
from sklearn.externals import joblib

#root to this code is: /Google Drive/CIKM_AnalytiCup_2017/Code/CIKM_AnalytiCup_2017/
filename = 'features/features_title_desc_attributes_1_3_word_1_5_char.npz';
X = utils.load_sparse_csr(filename)
XX = utils.load_sparse_csr(filename.split('.')[-2] + '_valid.npz')
loaded = np.load(filename.split('.')[-2] + '_dict.npz')
words = loaded['words'].tolist()
chars = loaded['chars'].tolist()

feature_names = [
'title.getLengths',
'title.getLengthsByTerm',
'title.getSpecialCharStat',
'title.hasSpecialChar',
'title.getUpperCharStat',
'title.getNounStat',
'title.getVerbStat',
'title.getAdjectiveStat',
'title.hasNumber',
'title.getNumberStat',
'title.hasTamilChar',
'title.hasChineseChar',
'title.getNonEnglishCharStat',
'title.getColorStat',
'title.getBrandStat(brands)',
'title.getSyllableStat',
'title.getPolysyllabStat',
'title.getAvgLetterPerWord',
'title.getAvgSentencePerWord',
'title.getAvgSyllablePerWord',
'title.getColemanLiauIndex',
'title.getDaleChallReadabilityScore',
'title.getAutomatedReadabilityIndex',
'title.getDifficultWordsStat',
'title.getFleschReadingEase',
'title.getFleschKincaidGrade',
'title.getGunningFog',
'title.getLexiconStat',
'title.getLinsearWriteFormula',
'title.getSmogIndex',
'title.getTextStandardLevel',
'products.CountryId',
'products.SkuIdPrefix',
'products.CategoryId',
'products.SubCategoryId',
'products.SubSubCategoryId',
'products.DelivaryTypeId',
'products.AdjustedPrice',
'desc.getLengths',
'desc.getLengthsByTerm',
'desc.getSpecialCharStat',
'desc.hasSpecialChar',
'desc.getUpperCharStat',
'desc.getNounStat',
'desc.getVerbStat',
'desc.getAdjectiveStat',
'desc.hasNumber',
'desc.getNumberStat',
'desc.hasTamilChar',
'desc.hasChineseChar',
'desc.getNonEnglishCharStat',
'desc.getColorStat',
'desc.getBrandStat(brands)',
'desc.getSyllableStat',
'desc.getPolysyllabStat',
'desc.getAvgLetterPerWord',
'desc.getAvgSentencePerWord',
'desc.getAvgSyllablePerWord',
'desc.getColemanLiauIndex',
'desc.getDaleChallReadabilityScore',
'desc.getAutomatedReadabilityIndex',
'desc.getDifficultWordsStat',
'desc.getFleschReadingEase',
'desc.getFleschKincaidGrade',
'desc.getGunningFog',
'desc.getLexiconStat',
'desc.getLinsearWriteFormula',
'desc.getSmogIndex',
'desc.getTextStandardLevel',
]

X_train = None
X_validation = None
for i, name in enumerate(feature_names):
    idx = 2 * (len(chars) + 2 * len(words)) + i - 1
    if X_train is None:
        X_clear = X[:, idx]
        X_validation = XX[:, idx]
    else:
        X_train = sparse.hstack((X_clear, X[:, idx]))
        X_validation = sparse.hstack((X_validation, XX[:, idx]))

feat_file = 'features/4Tam/features_all'
print('saving %s' % feat_file)
joblib.dump(X_train, feat_file + '.trn')
joblib.dump(X_validation, feat_file + '.tst')


