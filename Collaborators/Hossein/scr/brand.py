import pandas as pd
class Brand:
    def __init__(self, brand_filename):
        self.brands = pd.read_csv(brand_filename, names = ['Title', 'Category', 'Count'], encoding='utf-8', dtype=str, quotechar="\"")

    def getBrands(self):
        return set(self.brands['Title'])

    def getCategories(self):
        return set(self.brands['Category'])

    def getBrandsByCategory(self, category):
        return set(self.brands.loc[self.brands['Category'] == category, 'Title'])

    def getCategoriesByBrand(self, brand):
        return set(self.brands.loc[self.brands['Title'] == brand, 'Category'])

    def getTopBrandsByCategory(self, category, top_n=10):
        r = self.brands.loc[self.brands['Category'] == category, ['Title', 'Count']]
        return set(r.sort_values('Count', ascending=False).head(top_n)['Title'])

if __name__ == '__main__':
    #
    # brand_file = 'brands.txt'
    # brand_file = '../../Collaborators/Hossein/brands_by_category.txt'
    brand_file = '../../Collaborators/Hossein/brands_by_category_count.txt'
    b = Brand(brand_file)
    print b.getTopBrandsByCategory('Computers & Laptops')
    # print b.getBrands()
    # print b.getCategories()
    # print b.getBrandsByCategory('Computers & Laptops')
    # print b.getCategoriesByBrand('dell')
    # for c in b.getCategoriesByBrand('sony'):
    #     print c, b.getBrandsByCategory(c)

    # b.brands.to_csv('dataframe_2_csv_test.txt', columns=['Title', 'Category'], header=False, index=False, encoding='utf-8')

    # for c in b.getCategories():
    #     print "%s has %d brands" % (c, len(b.getBrandsByCategory(c)))
    # pass