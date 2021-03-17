# import mysql.connector
# from mysql.connector import errorcode
# from config import Config
from csv import reader

import pandas as pd
import numpy as np
import pylab as plt
import re
import pandas as pd
import matplotlib as mpl
import io

from text_corpus import TextCorpus
from brand import Brand

# class MysqlConnection:
#     cnx = None
#
#     def getHandle(self):
#         if MysqlConnection.cnx is None:
#             f = file('dal.cfg')
#             cfg = Config(f)
#             try:
#                 cnx = mysql.connector.connect(user=cfg.user,
#                                               password=cfg.password,
#                                               host=cfg.host,
#                                               port=cfg.port,
#                                               database=cfg.database,
#                                               raise_on_warnings=cfg.raise_on_warnings,
#                                               use_pure=cfg.use_pure)
#             except mysql.connector.Error as err:
#                 if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
#                     print("Something is wrong with your user name or password")
#                 elif err.errno == errorcode.ER_BAD_DB_ERROR:
#                     print("Database does not exist")
#                 else:
#                     print(err)
#             else:
#                 cnx.autocommit = False
#                 MysqlConnection.cnx = cnx
#
#         return MysqlConnection.cnx


class Dal:
    def __init__(self, database=False):
        # if database:
        #     self.cnx = MysqlConnection().getHandle()
        self.columns = ["Country", "SkuId", "Title", "Category", "SubCategory", "SubSubCategory", "Description", "Price", "DelivaryType"];

    # def importData(self, product_file, clarity_file, conciseness_file):
    #     """Imports data to mysql. Supposed to be run once. TODO: support for validation set"""
    #     cursor = self.cnx.cursor()
    #
    #     if clarity_file is not None:
    #         clarities = open(clarity_file).read().split()
    #     if conciseness_file is not None:
    #         concisenesses = open(conciseness_file).read().split()
    #
    #     with open(product_file) as infile:
    #         i = 0
    #         for line in reader(infile):
    #             if line[0] == 'my':
    #                 country = 1
    #             elif line[0] == 'ph':
    #                 country = 2
    #             elif line[0] == 'sg':
    #                 country = 3
    #             else:
    #                 country = 0
    #             skuId = line[1]
    #             title = line[2]
    #             category = line[3]
    #             subCategory = line[4]
    #             subSubCategory = line[5] if line[5] != 'NA' else None
    #             description = line[6] if line[5] != 'NULL' else None
    #             price = line[7]
    #             if line[8] == 'local':
    #                 delivaryType = 1
    #             elif line[8] == 'international':
    #                 delivaryType = 2
    #             elif line[8] == 'NA':
    #                 delivaryType = None
    #             else:
    #                 delivaryType = None
    #
    #             isClear = None
    #             isConcise = None
    #             if clarity_file is not None:
    #                 isClear = True if int(clarities[i]) else False
    #             if conciseness_file is not None:
    #                 isConcise = True if int(concisenesses[i]) else False
    #
    #             data_category = (category, None, None)
    #             add_category = ("INSERT IGNORE INTO Categories (Title, Parent, GrandParent) "
    #                             "VALUES (%s, %s, %s);")
    #             cursor.execute(add_category, data_category)
    #             self.cnx.commit()
    #
    #             cursor.execute("SELECT Id FROM Categories WHERE Title = %s;", (category,))
    #             category_Id = cursor.fetchone()[0]
    #             data_subcategory = (subCategory, category_Id, None)
    #             cursor.execute(add_category, data_subcategory)
    #             self.cnx.commit()
    #
    #             if subSubCategory is not None:
    #                 cursor.execute("SELECT Id FROM Categories WHERE Title = %s;", (subCategory,))
    #                 subcategory_Id = cursor.fetchone()[0]
    #                 data_subsubcategory = (subSubCategory, subcategory_Id, category_Id)
    #                 cursor.execute(add_category, data_subsubcategory)
    #
    #             data_product = (skuId, title, description, country, category, subCategory, subSubCategory, price, delivaryType, isClear, isConcise)
    #             add_product = ("INSERT INTO Products (SkuId, Title, Description, Country, Category, SubCategory, SubSubCategory, Price, DelivaryType, IsClear, IsConcise)"
    #                            "VALUES(%s,%s,%s,%s,(SELECT Id FROM Categories WHERE Title = %s),(SELECT Id FROM Categories WHERE Title = %s),(SELECT Id FROM Categories WHERE Title = %s),%s,%s,%s,%s);")
    #             cursor.execute(add_product, data_product)
    #             self.cnx.commit()
    #             i = i + 1
    #             print i
    #
    #     cursor.close()

    # def fetchData(self, doCleansing=True):
    #     """Fetches data from mysql."""
    #     products = pd.read_sql_query('SELECT * FROM Products', con=MysqlConnection().cnx)
    #     #products = pd.read_sql_table(table_name='Products', con=MysqlConnection().cnx)
    #     return self._cleanseData(products) if doCleansing else products
    #
    def readData(self, product_file, clarity_file, conciseness_file, sample=None, doCleansing=True):
        """Read data to memory."""
        products = pd.read_csv(product_file,
                               encoding='utf-8',
                               names=self.columns)
        if sample is not None:
            products = products.sample(sample)
        if clarity_file is not None:
            clarities = pd.read_csv(clarity_file, names=["IsClear"])
            products = products.join(clarities)
        if conciseness_file is not None:
            concisenesses = pd.read_csv(conciseness_file, names=["IsConcise"])
            products = products.join(concisenesses)
        return self._cleanseData(products) if doCleansing else products

    def findBrands(self, products, filename):
        gp = products.groupby(by=[products.SkuId.str[:3], 'Category'])
        titles = gp.apply(lambda x: ' '.join(x.Title.str.split().str[0]))
        counts = gp.size()
        tc = TextCorpus(titles)
        tf = tc.getTermStat()[0].toarray()
        max_idxes = tf.argmax(axis=1)
        brands = list(set([(tc.inv_words[idx], titles.keys()[i][1], tf[i][idx]) for i, idx in enumerate(max_idxes) if len(tc.inv_words[idx]) > 1 ]))
        # brands = list(set([(float(counts[i])/tf[i][idx], tc.inv_words[idx]) for i, idx in enumerate(max_idxes)]))
        # brands = list(set([tc.inv_words[idx] for i, idx in enumerate(max_idxes)]))
        brands.sort()
        f = io.open(file=filename, mode='w', encoding='utf-8')
        for b in brands:
            f.write("%s,\"%s\",%d\n" % b)
            # f.write("%s\n" % b)
        f.close()
        return brands

    def augmentDataByColor(self, products, color_filename, out_filename):
        # need the titles and color set
        tc_title = TextCorpus(products['Title'].tolist())
        # colors = mpl.colors.cnames.keys()
        colors = [line.split(':')[0] for line in open(color_filename).readlines() if int(line.split(':')[1]) > 50]
        newtitles = tc_title.permute(colors)
        newproducts = pd.DataFrame(columns=self.columns + ["IsClear","IsConcise"])
        for i, (index, row) in enumerate(products.filter(self.columns + ["IsClear","IsConcise"], axis=1).iterrows()):
            for j, newtitle in enumerate(newtitles[i]):
                newrow = row.copy()
                newrow['Title'] = newtitle
                newrow['SkuId'] = row['SkuId'] + '_' + str(j)
                newproducts = newproducts.append(newrow, ignore_index=True)
        newproducts.to_csv(out_filename, header=True, index=False, encoding='utf-8')

    def augmentDataByBrand(self, products, brand_filename, out_filename):
        b = Brand(brand_filename)
        # for each category in categories:
        # need the titles and brands of each category
        newproducts = pd.DataFrame(columns=self.columns + ["IsClear", "IsConcise"])
        for c in b.getCategories():
            titles = products.loc[products['Category'] == c]
            tc_title = TextCorpus(titles['Title'].tolist())
            newtitles = tc_title.permute(b.getTopBrandsByCategory(c, top_n=5))
            for i, (index, row) in enumerate(titles.filter(self.columns + ["IsClear", "IsConcise"], axis=1).iterrows()):
                for j, newtitle in enumerate(newtitles[i]):
                    newrow = row.copy()
                    newrow['Title'] = newtitle
                    newrow['SkuId'] = row['SkuId'] + '_' + str(j)
                    newproducts = newproducts.append(newrow, ignore_index=True)
        newproducts.to_csv(out_filename, header=True, index=False, encoding='utf-8')


    def _cleanseData(self, products):

        # Encoding categorical features: ValueError: could not convert string to float
        categories = products['Category'].append(products['SubCategory']).append(products['SubSubCategory']).unique().tolist()
        products['CategoryId'] = products.apply(lambda row: categories.index(row['Category']) if row['Category'] in categories and row['Category'] is not np.nan else None, axis=1)
        products['SubCategoryId'] = products.apply(lambda row: categories.index(row['SubCategory']) if row['SubCategory'] in categories and row['SubCategory'] is not np.nan else None, axis=1)
        products['SubSubCategoryId'] = products.apply(lambda row: categories.index(row['SubSubCategory']) if row['SubSubCategory'] in categories and row['SubSubCategory'] is not np.nan else None, axis=1)

        delivary_types =  products['DelivaryType'].unique().tolist()
        products['DelivaryTypeId'] = products.apply(lambda row: delivary_types.index(row['DelivaryType']) if row['DelivaryType'] in delivary_types and row['DelivaryType'] is not np.nan else None, axis=1)

        countries = products['Country'].unique().tolist()
        products['CountryId'] = products.apply(lambda row: countries.index(row['Country']) if row['Country'] in countries and row['Country'] is not np.nan else None, axis=1)

        skuid_prefix = list(set([s[0:1] for s in products['SkuId']]))
        products['SkuIdPrefix'] = products.apply(lambda row: skuid_prefix.index(row['SkuId'][0:1]), axis=1)

        # Imputing missing values before building an estimator
        # from sklearn.preprocessing import Imputer
        # Imputer(missing_values=np.nan, strategy='mean', axis=0)
        # imp.fit(X)
        products.loc[products['DelivaryType'].isnull(), 'DelivaryTypeId'] = -1; #products['DelivaryType'].mean or most_frequent
        products.loc[products['SubSubCategory'].isnull(), 'SubSubCategoryId'] = -1;#products['SubSubCategory'].mean or most_frequent
        products.loc[products['Description'].isnull(), 'Description'] = ''

        products['Description'] = products.apply(lambda row: self._removeHtmlTags(row['Description']), axis=1)
        products['AdjustedPrice'] = products.apply(self._adjustPrice, axis=1)
        return products

    def _removeHtmlTags(self, text):
        # from BeautifulSoup import BeautifulSoup
        # return BeautifulSoup(text).text

        # import xml.etree.ElementTree as et
        # return ''.join(et.fromstring(text).itertext())

        # the test may be not well-formed html
        # '%<li%>%' AND '%<br%>%' AND '%<tr%>%'
        return re.sub('<[^>]*>', '', text)

    def _adjustPrice(self, product):
        # normalize to sg's currency
        if product['Country'] == 'my' or product['Country'] == 1:
            return 0.32 * product['Price']
        elif product['Country'] == 'ph' or product['Country'] == 2:
            return 0.028 * product['Price']
        else:
            return product['Price']

if __name__ == '__main__':

    product_file = '../../Dataset//training/data_train.csv'
    clarity_file = '../../Dataset//training/clarity_train.labels'
    conciseness_file = '../../Dataset/training/conciseness_train.labels'
    product_file_validation = '../../Dataset/validation/data_valid.csv'

    # h = Dal(database=True)
    # h.importData(product_file, clarity_file, conciseness_file)
    # h.importData(product_file_validation, None, None)
    # exit()
    # products = h.fetchData()

    h = Dal()
    products_train = h.readData(product_file, clarity_file, conciseness_file)
    products_validation = h.readData(product_file_validation, None, None)
    # print products.loc[products['Country'] == 'my']

    sample = 1000
    products_train = h.readData(product_file, clarity_file, conciseness_file, sample=sample, doCleansing=False)
    products_validation = h.readData(product_file_validation, None, None, sample=int(sample * 0.3), doCleansing=False)
    # color_file = '../../Collaborators/Hossein/color_freq.txt'
    # h.augmentDataByColor(products_train, color_file, 'data_train_augmented_by_color.csv')
    # h.augmentDataByColor(products_validation, color_file, 'data_validation_augmented_by_color.csv')

    # h.findBrands(products_train.append(products_validation), 'brands_by_category.txt')
    brand_file = '../../Collaborators/Hossein/brands_by_category_count.txt'
    h.augmentDataByBrand(products_train, brand_file, 'data_train_augmented_by_brand.csv')
    h.augmentDataByBrand(products_validation, brand_file, 'data_validation_augmented_by_brand.csv')

def findColorCounts():
    all = products_train.append(products_validation)
    colors = mpl.colors.cnames.keys()
    color_counts = {c:0 for c in colors}
    for i, r in all.iterrows():
        for c in colors:
            if c in r['Title']:
                color_counts[c] = color_counts[c] + 1
    import operator

    color_counts = sorted(color_counts.items(), key=operator.itemgetter(1), reverse=True)
    for k, v in color_counts:
        if v > 0:
            print "%s: %d" % (k,v)


def findBrandByCategoryCounts():
    pass