#import requests
from scipy import interpolate
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


class Functions:
    def __init__(self,YC_tw,COC_twAA,Setting):
        self.YC_tw=YC_tw
        self.COC_twAA=COC_twAA
        self.Maturity_Selected = ['1年', '2年', '3年', '5年', '10年']
        self.COC_MT = [1, 2, 3, 5, 10]
        self.Setting=Setting
    
    def Interpolation_YC(self,x):
        x_points = [ 2, 5, 10, 20, 30 ]
        y_points = list(self.YC_tw)
        tck = interpolate.splrep(x_points, y_points)
        return interpolate.splev(x, tck)
    
    def CubicSplineInterpolation(self):
        ix = 0
        self.YC_tw_value = np.zeros(len(self.COC_MT))
        self.time_len = np.zeros(len(self.COC_MT))
        for x in self.COC_MT :
            self.YC_tw_value[ix], self.time_len[ix] = self.Interpolation_YC(x), x
            ix += 1
    
    def CreditSpread(self):
        self.COC_value = self.COC_twAA.loc[self.Maturity_Selected]
        self.Cr_Spread = list(self.COC_value) - self.YC_tw_value
        self.df_Cr_Spread = pd.DataFrame(self.Cr_Spread, columns=['Credit Spread'])
        self.df_Cr_Spread['Maturity'] = list(self.Maturity_Selected)
        self.df_Cr_Spread.set_index('Maturity')
    
    def LambdaEstimation(self):
        self.PD=(1-np.exp(-np.multiply(self.Cr_Spread, self.COC_MT))) / (self.Setting['LGD'])  
    
    def loss_func(self,lamb) : 
        E_PD = list( (1 - np.exp(-lamb*i)) for i in self.COC_MT)
        MSE = ( np.array(self.PD) - np.array(E_PD) )**2
        return np.sum(MSE)
    
    def YTM_fit(self,YTM, days_df, reversion_speed, volatility, length, days_count ) : 
        coeff = np.polyfit( days_df, YTM, 3 ) 
        Y = np.zeros(length * days_count)
        for i, x in zip( list(range(days_count*length)), 
                        [ t/days_count for t in range(1, days_count*length+1)] ) :  
            zero = coeff[3] * reversion_speed
            first = 2 * coeff[2] * (1 + reversion_speed * x)  
            second = 3 * coeff[1] * (2 + reversion_speed * x) * x
            third = 4 * coeff[0] * (3 + reversion_speed * x) * x**2
            epsilon = volatility**2 / ( 2 * reversion_speed ) * (1 - np.exp(-reversion_speed * x)) 
            Y[i] = zero + first + second + third + epsilon                   
        return Y , coeff
    
    def MC_Hull(self,I0, K, theta, sigma, steps) :
        dt = 1 / steps    
        rates = [ I0 ]
        for i in range(steps):
            dr = (theta[i] - K*rates[-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
            rates.append(rates[-1] + dr)
        return  rates
    
    def InterestRatePath(self,Init_Rate,rv_spd,THETA,vol,timesteps):
        self.paths = self.MC_Hull(I0=Init_Rate, K=rv_spd, theta=THETA, sigma=vol, steps=timesteps)
        for i in range(self.Setting['num_paths']-1):
            #print(i)
            self.paths = np.vstack(( self.paths, self.MC_Hull( I0=Init_Rate, K=rv_spd, theta=THETA, sigma=vol,steps=timesteps) ))
    
    def Decision(self):
        call_or_not = np.zeros(self.Setting['timesteps'])
        for i in range(int(self.Setting['days_per_yr']/2), self.Setting['timesteps'], self.Setting['days_per_yr']):
            if i < self.Setting['days_per_yr']*10 : 
                call_or_not[i] = 0
            else : 
                call_or_not[i] = 1
        
        cp_or_not = np.zeros(self.Setting['timesteps'])
        for i in range(int(self.Setting['days_per_yr']/2), self.Setting['timesteps'], self.Setting['days_per_yr']):
            cp_or_not[i] = 1
        
        self.decision = np.concatenate((call_or_not, cp_or_not)).reshape(2, self.Setting['timesteps'])
        self.temp_path = np.delete(self.paths, 0, axis=1)
    
    def discount_func(self,prev_val, rt, dr_prob, rc, redem, cp, s ) :
        value = cp + np.exp(-(rt + s)*self.Setting['dt'] ) * ( (1 - dr_prob)*prev_val + dr_prob*rc*redem )
        return value
    
    def EstimatedLiquidity(self,dr_prob):
        init_lq_sprd = 0.001
        est_lq_sprd = float(fsolve(self.full_disc_solve, init_lq_sprd,dr_prob))
        return est_lq_sprd
    
    def full_disc(self,lq_sprd,dr_prob):
        V = np.zeros((self.Setting['num_paths'], self.Setting['timesteps'])) 
        ###################################################################################################
        value = self.Setting['Face_Val']
        counts = 0
    
        for t in range(self.Setting['timesteps']-1, -1, -1) :
            # callable : no /  coupon : no
            if list(self.decision[:, t]) == [0.0, 0.0] :  
                value = self.discount_func(value, self.temp_path[:, t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][0], s=lq_sprd )
                V[:, t] = value 
    
            # callable : no /  coupon : yes
            elif list(self.decision[:, t]) == [0.0, 1.0] : 
                value = self.discount_func( value, self.temp_path[:,t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][1], s=lq_sprd )
                V[:, t] = value 
    
            # callable : yes /  coupon : no
            elif list(self.decision[:, t]) == [1.0, 0.0] : 
                value = self.discount_func( value, self.temp_path[:,t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][0], s=lq_sprd )
                V[:, t] = value
                itm = np.greater( V[:, t], self.Setting['Redempt_Price'])
                V_itm = np.compress( itm == 1, V[:, t] )
                r_itm = np.compress( itm == 1, self.temp_path[:, t] )
                if len(r_itm) == 0 : 
                    continue
                else : 
                    rg = np.polyfit( r_itm, np.log(V_itm), 2 )
                    H = np.exp( np.polyval(rg, r_itm ) )
                    V[itm == True][:, t] = np.where( H > self.Setting['Redempt_Price'], self.Setting['Redempt_Price'], H )
                    counts += 1
    
            # callable : yes /  coupon : yes
            else :                         
                value = self.discount_func(value, self.temp_path[:, t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][2], s=lq_sprd )
                V[:, t] = value
                itm = np.greater(V[:, t], self.Setting['Redempt_Price'])
                V_itm = np.compress( itm == 1, V[:, t] )
                r_itm = np.compress( itm == 1, self.temp_path[:, t] )
                if len(r_itm) == 0 : 
                    continue
                else :            
                    rg = np.polyfit( r_itm, np.log(V_itm), 2 )
                    H = np.exp(np.polyval( rg, r_itm ) )
                    V[itm == True][:, t] = np.where( H > self.Setting['Redempt_Price'], self.Setting['coupon'][2]+self.Setting['Redempt_Price'], H )
                    counts += 1
        result = {'V-Matrix':V, 'Counts':counts}
        return result     
    
    def full_disc_solve(self,lq_sprd,dr_prob):
        #  Create Empty Matrix for Present Value 
        V = np.zeros((self.Setting['num_paths'], self.Setting['timesteps'])) 
        ###################################################################################################
        value = self.Setting['Face_Val']
        counts = 0
    
        for t in range(self.Setting['timesteps']-1, -1, -1) :
            # callable : no /  coupon : no
            if list(self.decision[:, t]) == [0.0, 0.0] :  
                value = self.discount_func(value, self.temp_path[:, t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][0], s=lq_sprd )
                V[:, t] = value 
    
            # callable : no /  coupon : yes
            elif list(self.decision[:, t]) == [0.0, 1.0] : 
                value = self.discount_func( value, self.temp_path[:,t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][1], s=lq_sprd )
                V[:, t] = value 
    
            # callable : yes /  coupon : no
            elif list(self.decision[:, t]) == [1.0, 0.0] : 
                value = self.discount_func( value, self.temp_path[:,t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][0], s=lq_sprd )
                V[:, t] = value
                itm = np.greater( V[:, t], self.Setting['Redempt_Price'])
                V_itm = np.compress( itm == 1, V[:, t] )
                r_itm = np.compress( itm == 1, self.temp_path[:, t] )
                if len(r_itm) == 0 : 
                    continue
                else : 
                    rg = np.polyfit( r_itm, np.log(V_itm), 2 )
                    H = np.exp( np.polyval(rg, r_itm ) )
                    V[itm == True][:, t] = np.where( H > self.Setting['Redempt_Price'], self.Setting['Redempt_Price'], H )
            
            # callable : yes /  coupon : yes
            else :                         
                value = self.discount_func(value, self.temp_path[:, t], dr_prob, self.Setting['reco_rate'], 
                                      self.Setting['Redempt_Price'], self.Setting['coupon'][2], s=lq_sprd )
                V[:, t] = value
                itm = np.greater(V[:, t], self.Setting['Redempt_Price'])
                V_itm = np.compress( itm == 1, V[:, t] )
                r_itm = np.compress( itm == 1, self.temp_path[:, t] )
                if len(r_itm) == 0 : 
                    continue
                else :            
                    rg = np.polyfit( r_itm, np.log(V_itm), 2 )
                    H = np.exp(np.polyval( rg, r_itm ) )
                    V[itm == True][:, t] = np.where( H > self.Setting['Redempt_Price'], self.Setting['coupon'][2]+self.Setting['Redempt_Price'], H )
        
        return np.mean(V[:,0])-1000.0    
        
        
        
        
        
        
        
        
        
        
        
    