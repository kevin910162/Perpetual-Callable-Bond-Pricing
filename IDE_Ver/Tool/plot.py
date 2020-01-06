import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Plot:
    def __init__(self):
        self.test='i'
        #self.YC_tw=YC_tw
    
    def DrawCubicSpline(self,x,y):
        plt.title('Cubic Spline Yield Curve', size=20 )
        plt.plot(x, y)
        plt.grid(linestyle='--', linewidth=0.5)
    
    def DrawThetaApproximation(self,x):
        plt.figure(figsize=(8, 4))
        plt.title('Theta Approximation', size=20)
        plt.xlabel('Maturity(Days)',size=15)
        plt.ylabel('Long Term Short Rate * reversion speed',size=15)
        plt.grid(linestyle='--', linewidth=1.2)
        plt.plot(x, linewidth=2)
    
    def DrawHullWhite(self,x):
        plt.figure(figsize=(10, 6))
        plt.title("Hull-White Short Rate Simulation", size=15)
        for i in range(150):
            plt.plot(x[i, :], lw=0.8, alpha=0.6)
    
    def DrawDiscountPath(self,result):
        print('Numbers of Call :', result['Counts'])
        ### Path Examination
        plt.figure(figsize=(12, 6))
        plt.title('Discount Path', size=15)
        for i in range(100):    
            plt.plot(result['V-Matrix'][i,:])
        print('The Mean Price of LSM : ', np.mean( result['V-Matrix'][:,0]) )
        print('The Median Price of LSM : ', np.median( result['V-Matrix'][:,0]) )
        print('The S.D of Price with LSM : ', np.std( result['V-Matrix'][:,0]))
    
        plt.figure(figsize=(12, 6))
        plt.title('Bond Price Distribution Plot', size=15)
        plt.xlabel('Price')
        sns.distplot(result['V-Matrix'][:,0], hist=True, kde=False, 
                     bins=50, color = 'blue',
                     hist_kws={'edgecolor':'red'})

