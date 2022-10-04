import torch
import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import os
import random
from collections import deque
import sys
from utils.Scalers import *

class OTGym_v0(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(OTGym_v0,self).__init__()
        self.newRecord = {'tradePosition':0.0,'deposit':0.0,
                          'pL':0.0,'roi':0.0,'runningPL':0.0,'runningRoi':0.0,'totalRealizedPL':0.0,'totalRealizedROI':0.0,
                         'daysOpen':0.0,'sequencesOpen':0.0} 

        self.idx_list = []
        self.totalroi = []
        self.totalpl = []
        self.sqz_open = []
        self.tradeCount = 0.0
        self.current_step = 0.0
        self.historicalWindow = 5
        self.totalRealizedPL = 0.0
        self.totalRealizedROI = 0.0
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if 'cuda' in device.type:
        #     from google.colab import drive
        #     drive.mount('/content/drive',force_remount=False)
        #     self.data_directory = '/content/drive/Othercomputers/My MacBook Pro/DRLOT/Data.nosync/SPY/stableBaselines/'
        # else:
        #     self.data_directory = 'Data.nosync/SPY/stableBaselines/'
        self.data_directory = '/content/drive/Othercomputers/My MacBook Pro/DRLOT/Data.nosync/SPY/stableBaselines/'
        
        self.closed = False
        
        # if 'cuda' in device.type:
        self.minmax = MinMax3()
        # else:
            # self.minmax = MinMax2(self.data_directory,reset=False)

        self.min_deposit = self.minmax.minimum[self.minmax.columns.index('deposit_mark')]
        self.max_deposit = self.minmax.maximum[self.minmax.columns.index('deposit_mark')]
        self.min_tradePosition = 0.0
        self.max_tradePosition = 1.0
        self.min_pL = -25.0
        self.max_pL = 25.0
        self.min_Roi = -1.0
        self.max_Roi = 10.0
        self.min_sqz = 10.0
        self.max_sqz = 445.0
        self.min_days_open = self.minmax.minimum[self.minmax.columns.index('daysToExpiration_front')]
        self.max_days_open = self.minmax.maximum[self.minmax.columns.index('daysToExpiration_front')]

        self.files = sorted(glob.glob(self.data_directory+'*'))
        self.fileUsage = {x:1 for x in self.files}
        self.subFileUsage = {x:{} for x in self.files}
        self.width = pd.read_parquet(glob.glob(self.data_directory+'*')[0]).shape[-1] + len(self.newRecord)
        self.observation_space = Box(low=-1, high=1, shape=(self.historicalWindow*self.width,))
        self.action_space = Discrete(2)   

        self.seed()

        ## not sure if I should reset in the __init__
        self.reset()
        
    def _next_observation(self,):
        self.scaler_trade_record()
        
        ts_a = pd.DataFrame(self.trade_state_list_scaled).to_numpy(dtype=np.float32)
        self.obs = np.concatenate((self.minmax.scale(self.dflocked[0:self.historicalWindow]), ts_a), axis=1)
        
        # ts_t = torch.Tensor([[np.float32(v) for k,v in ts.items()] for ts in self.trade_state_list_scaled])
        # self.obs = torch.concat((self.minmax.scale(self.dflocked[self.current_step-self.historicalWindow:self.current_step]),ts_t),dim=1)
        
    def _take_action(self,):
        self.reward = 0.0
        if self.action == 1:
            if self.trade_record['tradePosition'] == 0:
                self.trade_record['tradePosition'] = 1
                self.tradeCount +=1
                self.db4exp = self.dflocked.iloc[self.current_step]['daysToExpiration_front']
                self.trade_record['deposit'] = round(random.uniform(self.dflocked.iloc[self.current_step]['bid_back'],self.dflocked.iloc[self.current_step]['ask_back']) - \
                                                     random.uniform(self.dflocked.iloc[self.current_step]['bid_front'],self.dflocked.iloc[self.current_step]['ask_front']),2)
                if self.trade_record['deposit'] < 0.01:
                    self.trade_record['deposit']=0.01

            elif self.trade_record['tradePosition'] == 1:
                self.trade_record['sequencesOpen'] += 1
                self.trade_record['daysOpen'] = self.db4exp - self.dflocked.iloc[self.current_step]['daysToExpiration_front']
                self.trade_record['pL'] = np.float32(self.dflocked.iloc[self.current_step]['deposit_mark'] - self.trade_record['deposit'])
                self.trade_record['roi'] = np.float32(self.trade_record['pL']/self.trade_record['deposit'])
                self.trade_record['runningPL'] = np.float32(self.trade_record['pL'] + self.trade_record['totalRealizedPL'])
                self.trade_record['runningRoi'] = np.float32(self.trade_record['roi'] + self.trade_record['totalRealizedROI'])
                
                self.reward += np.float32(self.trade_record['pL'])
                if self.done[self.current_step] == True:
                    self.reward -= 1000
        
        elif self.action == 0:
            if self.trade_record['tradePosition'] == 1:
                self.sqz_open.append(self.trade_record['sequencesOpen'])
                self.trade_record['tradePosition'] = 0
                self.trade_record['pL'] = round(random.uniform(self.dflocked.iloc[self.current_step]['bid_back'],self.dflocked.iloc[self.current_step]['ask_back']) - \
                                                random.uniform(self.dflocked.iloc[self.current_step]['bid_front'],self.dflocked.iloc[self.current_step]['ask_front']),2) - self.trade_record['deposit']                
                
                self.trade_record['roi'] = self.trade_record['pL']/self.trade_record['deposit']    
                self.trade_record['totalRealizedPL'] +=  self.trade_record['pL']                 
                self.trade_record['totalRealizedROI'] +=  self.trade_record['roi']                 
                self.trade_record['deposit'] = 0

                self.db4exp = 0
                self.closed = True

            elif self.trade_record['tradePosition'] == 0:
                pass  
            
        self.reward += np.float32(self.trade_record['totalRealizedPL'])
            
    def scaler_trade_record(self,):
        self.trade_state_list_scaled =[{key: value for key, value in x.items()} for x in self.trade_state_list] #self.trade_state_list.copy()
        for d in self.trade_state_list_scaled:
            d['tradePosition'] = (((d['tradePosition'] - self.min_tradePosition) / (self.max_tradePosition - self.min_tradePosition))*2)-1   
            d['deposit'] = (((d['deposit'] - self.min_deposit) / (self.max_deposit - self.min_deposit))*2)-1
            d['pL'] = (((d['pL'] - self.min_pL) / (self.max_pL - self.min_pL))*2)-1
            d['roi'] = (((d['roi'] - self.min_Roi ) / (self.max_Roi - self.min_Roi ))*2)-1
            d['runningPL'] = (((d['runningPL'] - self.min_pL) / (self.max_pL - self.min_pL))*2)-1    
            d['runningRoi'] = (((d['runningRoi'] - self.min_Roi ) / (self.max_Roi - self.min_Roi ))*2)-1    
            d['daysOpen'] = (((d['daysOpen'] - self.min_days_open) / (self.max_days_open - self.min_days_open))*2)-1    
            d['sequencesOpen'] = (((d['sequencesOpen'] - self.min_sqz ) / (self.max_sqz  - self.min_sqz ))*2)-1    

    def step(self, action):    
        self.current_step +=1
        self.action = action
        
        self._take_action()
                
        self.trade_state_list.append({key: value for key, value in self.trade_record.items()})
        self._next_observation()
        
        # display(pd.DataFrame(self.dflocked.iloc[self.current_step]).T[['WD','HOUR','daysToExpiration_front','bid_front','mark_front','ask_front','bid_back','mark_back','ask_back','deposit_mark','openInterest_front','openInterest_back','delta_front','delta_back']])
        # display(pd.DataFrame(self.trade_state_list))
        
        if self.closed == True:
            self.trade_record['pL'] = 0.0
            self.trade_record['roi'] = 0.0
            self.trade_record['daysOpen'] = 0.0
            self.trade_record['sequencesOpen'] = 0.0
            self.closed = False
        
        if self.done[self.current_step] == True:
            self.trade_record['runningPL'] = 0.0
            self.trade_record['runningRoi'] = 0.0 
            self.trade_record['totalRealizedPL'] = 0.0
            self.trade_record['totalRealizedROI'] = 0.0
            
        return self.obs.flatten().flatten(), round(self.reward,4) , self.done[self.current_step], {}

    def reset(self):
        self.reward = 0.0
        self.trade_record = {key: value for key, value in self.newRecord.items()} 
        self.db4exp = 0
        self.trade_state_list = deque(maxlen=5)
        [self.trade_state_list.append({key: value for key, value in self.newRecord.items()} ) for _ in range(5)]
        self.current_step = self.historicalWindow - 1
        
        if len(self.idx_list) == 0:
            p_ = sum([v for k,v in self.fileUsage.items()])
            self.file = np.random.choice(list(self.fileUsage.keys()),p=[(v/p_) for k,v in self.fileUsage.items()])
            for k,v in self.fileUsage.items():
                if k != self.file:
                    self.fileUsage[k]+=1
            self.df = pd.read_parquet(self.file)
            self.idx_list = list(self.df.index.unique())

        self.idx = np.random.choice(self.idx_list)
        self.dflocked = self.df.loc[self.idx].sort_values(['daysToExpiration_front','HOUR'],ascending=[False,True])  
        self.idx_list.remove(self.idx)
        
        self.scaler_trade_record()
        
        ts_a = pd.DataFrame(self.trade_state_list_scaled).to_numpy(dtype=np.float32)
        self.obs = np.concatenate((self.minmax.scale(self.dflocked[0:self.historicalWindow]), ts_a), axis=1)
        
        # ts_t = torch.Tensor([[np.float32(v) for k,v in ts.items()] for ts in self.trade_state_list_scaled])
        # self.obs = torch.concat((self.minmax.scale(self.dflocked[0:self.historicalWindow]),ts_t),dim=1)
        
        self.done = [False for _ in range(self.dflocked.shape[0]-1)]+[True]
        self.max_steps = len(self.dflocked)
        return self.obs.flatten().flatten()
        
    def render(self, mode='human', close=False):
        if len(self.totalpl)==0: tplmean = np.mean(self.totalpl)
        else:tplmean = 0
            
        if len(self.totalpl)==0: troimean = np.mean(self.totalroi)   
        else:troimean = 0
            
        print(f'Average PL: {round(tplmean,2)}, Average ROI: {round(troimean,2)}, Total Trades {self.tradeCount}')
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close (self):
        pass
