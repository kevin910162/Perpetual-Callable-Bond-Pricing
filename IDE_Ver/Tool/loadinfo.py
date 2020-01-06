#import requests

import pandas as pd

class LoadInfo:
    def __init__(self,date):
        self.date=date
        #self.YC_tw=YC_tw
    
    def CorporateCurve(self):
        COCurve = pd.read_excel('.\data_source\COCurve.{}-C.xls'.format(self.date), sheet_name='COCurve')
        try:
            COC_df = COCurve.rename(columns=COCurve.iloc[1]).drop(COCurve.index[[0, 1, 6]]).set_index('到期年限')
        except KeyError:
            COC_df = COCurve.rename(columns=COCurve.iloc[1]).drop(COCurve.index[[0, 1, 6]]).set_index('到期年限Residual Month/Year')
        COC_df.index.rename(name='Rated', inplace=True)
        self.COC_twAA = COC_df.loc['twAA']
        #print(COC_twAA)
    
    def GovernmentCurve(self):
        YCurve = pd.read_excel('.\data_source\Curve.{}-C.xls'.format(self.date), sheet_name='含息殖利率曲線')
        YC_df = YCurve.set_index('Tenor').drop(['Bond Code'], axis=1).dropna()
        self.YC_tw = YC_df.iloc[:,0]/100
    
    def HullMonteCarlo(self):
        self.YCDF = pd.DataFrame(self.YC_tw)
        self.YCDF.columns = ['Interest Rate']
        self.YCDF['MT_dt'] = [ 2, 5, 10, 20, 30 ]
        
