from datetime import datetime

class Cleaning:
    
    def __init__(self):
        pass
    
    def remove_na(self,data):
        print('Number of Na values:\n{}'.format(data.isna().sum()))
        return data.dropna().reset_index(drop=True)
    
    def drop_dup(self,data):
        print('\nNumber of duplicated rows:',sum(data.duplicated()))     
        return data.drop_duplicates()
    
    def remove_nonpos(self,data,colname):
        for n in range(len(colname)):
            print('\nNon-positive values in '+colname[n]+' :',sum(data[colname[n]]<=0),'rows')
            data =  data.loc[data[colname[n]]>0,:]
        return data
    
    def remove_nonint(self,data,colname):
        for n in range(len(colname)):
            print('\nNon-integer values in '+colname[n]+' :',sum([int(i) != i for i in data[colname[n]]]),'rows')
            data = data.loc[[int(i) == i for i in data[colname[n]]],:]
        return data
    
    def check_dtype(self,data):
        print('\nFeature Data Types: ')
        for i in range(data.shape[1]):
            print(data.columns[i],type(data.iloc[0,i]))
    
    def change_dtype(self,data,colname,dtype,dt_format = "%m/%d/%Y %H:%M"):
        if dtype == 'int':
            for n in range(len(colname)):
                data[colname[n]] = [int(i) for i in data[colname[n]]]
                print('\nChanged '+colname[n]+' to integer type')
            
        if dtype == 'str':
            for n in range(len(colname)):
                data[colname[n]] = [str(i) for i in data[colname[n]]]
                print('\nChanged '+colname[n]+' to string type')
        
        if dtype == 'datetime':
            for n in range(len(colname)):
                data[colname[n]] = [datetime.strptime(i, dt_format) for i in data[colname[n]]] 
                print('\nChanged '+colname[n]+' to datetime type')
        
        return data
    
class FeatEng:
    
    def __init__(self):
        pass
    
    def features(self,data):
        df2 = data.groupby(['CustomerID']).nunique().iloc[:,:2]
        df2.columns = ['NoOfInvoices', 'NoOfUniqueItems']

        df2['QuantityPerInvoice'] = data.groupby(['CustomerID','InvoiceNo']).sum().groupby(level=0).sum()['Quantity']/df2['NoOfInvoices']

        df_spending = data.groupby(['CustomerID','InvoiceNo','StockCode']).sum()
        df_spending['Spending'] = df_spending['Quantity']*df_spending['UnitPrice']
        df2['SpendingPerInvoice'] = df_spending.groupby(level=0).sum()['Spending']/df2['NoOfInvoices']

        df2['TotalQuantity'] = data.groupby(['CustomerID']).count()['Quantity']

        df2['UniqueItemsPerInvoice'] = df2['NoOfUniqueItems']/df2['NoOfInvoices']

        df2['UnitPriceMean'] = data.groupby(['CustomerID','InvoiceNo','StockCode']).sum().groupby(level=0).mean()['UnitPrice']
        df2['UnitPriceStd'] = data.groupby(['CustomerID','InvoiceNo','StockCode']).sum().groupby(level=0).std()['UnitPrice']
        
        return df2
    
    
    
    
        