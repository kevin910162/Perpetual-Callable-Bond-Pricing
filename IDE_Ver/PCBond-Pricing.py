from Tool.downloadinfo import DownLoadInfo
from Tool.functions import Functions
from Tool.loadinfo import LoadInfo
from Tool.plot import Plot
from scipy.optimize import minimize_scalar
import numpy as np


def DownLoadData(date):
    DLI=DownLoadInfo(date)
    DLI.DownloadCOC()
    DLI.DownloadCurve()

def LoadData(date):
    LI=LoadInfo(date)
    LI.CorporateCurve()
    LI.GovernmentCurve()
    return LI

def CubicSplineInterpolation_YC_tw(LI,Setting):
    FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
    FC.CubicSplineInterpolation()
    PL=Plot()
    PL.DrawCubicSpline(FC.time_len,FC.YC_tw_value)

def TableofCreditSpread(LI,Setting):
    FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
    FC.CubicSplineInterpolation()
    FC.CreditSpread()
    print(FC.df_Cr_Spread)

def MSEminimization_LambdaEstimation(LI,Setting):  
    FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
    FC.CubicSplineInterpolation()
    FC.CreditSpread()
    FC.LambdaEstimation()
    result = minimize_scalar(FC.loss_func)
    est_lambda = result.x
    print('Default Probablity List :', FC.PD)
    print('Estimated Lambda :', est_lambda)
    print('Estimated Default Probability:', 1 - np.exp(-float(est_lambda)))

def Get_dr_prob(LI,Setting):
    FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
    FC.CubicSplineInterpolation()
    FC.CreditSpread()
    FC.LambdaEstimation()
    result = minimize_scalar(FC.loss_func)
    est_lambda = result.x
    dr_prob = 1 - np.exp(-float(est_lambda)*dt)
    return dr_prob

def YCDF(LI):
    LI.HullMonteCarlo()
    print(LI.YCDF)    

def ThetaApproximation(LI,Setting):
     FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
     LI.HullMonteCarlo()
     THETA , coef = FC.YTM_fit(LI.YCDF['Interest Rate'],LI.YCDF['MT_dt'],rv_spd,vol,yr_length,days_per_yr)
     PL=Plot()
     PL.DrawThetaApproximation(THETA)   

def HullWhiteShortRateSimulation(LI,Setting):
     FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
     LI.HullMonteCarlo()
     THETA , coef = FC.YTM_fit(LI.YCDF['Interest Rate'],LI.YCDF['MT_dt'],rv_spd,vol,yr_length,days_per_yr)
     FC.InterestRatePath(Init_Rate,rv_spd,THETA,vol,timesteps)
     print('Interest Rate Path Shape : ', FC.paths.shape)
     PL=Plot()
     PL.DrawHullWhite(FC.paths)

def ImplementingLSM(LI,Setting):
     FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
     LI.HullMonteCarlo()
     THETA , coef = FC.YTM_fit(LI.YCDF['Interest Rate'],LI.YCDF['MT_dt'],rv_spd,vol,yr_length,days_per_yr)    
     FC.InterestRatePath(Init_Rate,rv_spd,THETA,vol,timesteps)
     FC.Decision()
     PL=Plot()
     PL.DrawDiscountPath(FC.full_disc(0.0,Get_dr_prob(LI,Setting)))

def SolutionForLiquidityFactor(LI,Setting):
     FC=Functions(LI.YC_tw,LI.COC_twAA,Setting)
     LI.HullMonteCarlo()
     THETA , coef = FC.YTM_fit(LI.YCDF['Interest Rate'],LI.YCDF['MT_dt'],rv_spd,vol,yr_length,days_per_yr)    
     FC.InterestRatePath(Init_Rate,rv_spd,THETA,vol,timesteps)
     FC.Decision()
     PL=Plot()
     est_lq_sprd=FC.EstimatedLiquidity(Get_dr_prob(LI,Setting))
     PL.DrawDiscountPath(FC.full_disc(est_lq_sprd,Get_dr_prob(LI,Setting)))


# #### Settings - 1
num_paths = 2000
yr_length = 30
days_per_yr = 360
timesteps = days_per_yr * yr_length
dt = 1 / days_per_yr # Time Step 

### LGD=50% 
LGD = 0.5

Face_Val = 1000
Redempt_Price = 1000 
coupon = [ 0, Face_Val*0.0345, Face_Val*0.0445] 

# Default LAMBDA
reco_rate = 0.5 # Recovery Rate

## Initial Setting for Rate Simulation
Init_Rate = 0.026 # 基準利率
rv_spd = 0.05
vol = 0.025


Setting={'num_paths':num_paths, 'timesteps':timesteps, 'Face_Val':Face_Val, 'days_per_yr':days_per_yr, 
'dt':dt, 'reco_rate':reco_rate, 'Redempt_Price':Redempt_Price, 'coupon':coupon,'LGD':LGD}




#print(LoadData('20170623').COC_twAA)
#print(LoadData('20170622').COC_twAA)
#print(LoadData('20170623').YC_tw)
#print(LoadData('20170622').YC_tw)

DownLoadData('20170623')
LD=LoadData('20170623')


CubicSplineInterpolation_YC_tw(LD,Setting)
TableofCreditSpread(LD,Setting)
MSEminimization_LambdaEstimation(LD,Setting)
YCDF(LD)
ThetaApproximation(LD,Setting)
HullWhiteShortRateSimulation(LD,Setting)
ImplementingLSM(LD,Setting)
SolutionForLiquidityFactor(LD,Setting)

















