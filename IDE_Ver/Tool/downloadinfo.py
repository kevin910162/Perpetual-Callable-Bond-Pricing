import requests
import os

class DownLoadInfo:
    def __init__(self,date):
        self.date=date
        try:
            os.makedirs('data_source')
        except:
            pass
    
    def DownloadCurve(self):
        url='https://www.tpex.org.tw/storage/bond_zone/tradeinfo/govbond/{}/{}/Curve.{}-C.xls'.format(self.date[0:4],self.date[0:6],self.date)
        with open('data_source/{}'.format(url.split('/')[-1]),'wb') as fileW:
            fileW.write(requests.get(url).content)
        
    
    def DownloadCOC(self):
        url='https://www.tpex.org.tw/storage/bond_zone/tradeinfo/govbond/{}/{}/COCurve.{}-C.xls'.format(self.date[0:4],self.date[0:6],self.date)
        with open('data_source/{}'.format(url.split('/')[-1]),'wb') as fileW:
            fileW.write(requests.get(url).content)