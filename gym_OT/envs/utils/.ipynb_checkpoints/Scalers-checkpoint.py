import numpy as np
import pandas as pd
import os , glob , ujson
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import torch


class MinMax3():
    def __init__(self,stock='SPY'):
        
        self.eps = np.finfo(np.float32).eps
        path_ = f'/content/drive/Othercomputers/My MacBook Pro/DRLOT/{stock}_minmax.json'
        if (os.path.exists(path_) == True):
            minmax_loaded = ujson.load(open(path_,'r'))

            print('Loading Scalers')
            self.minimum = minmax_loaded['minimum']
            self.maximum = minmax_loaded['maximum']
            self.columns = minmax_loaded['columns']
        else:
            print('Minmax json not found!')
        
    def min_max_calculator(self,filename):
        df = pd.read_parquet(filename)    
        return (df.min(),df.max())
        
    def scale(self,df,to_tensor = True):
        
        
        df = (df - pd.Series(np.array(self.minimum),self.columns)) + self.eps
        df = ((df / ((pd.Series(np.array(self.maximum),self.columns)  - pd.Series(np.array(self.minimum),self.columns))+self.eps))*2)-1
        
        
        if to_tensor == True:
            if len(df)==1:
                return torch.Tensor(df.values.astype(np.float32)).unsqueeze(0)
            else:
                return torch.Tensor(df.values.astype(np.float32))
        else:
            return df
        


class MinMax2():
    def __init__(self,data_folder,stock='SPY',reset=False):
        self.eps = np.finfo(np.float32).eps
        if (os.path.exists(f'{stock}_minmax.json') == True) and (reset == False):
            df = pd.read_parquet(np.random.choice(glob.glob(data_folder+'*')))
            minmax_loaded = ujson.load(open(f'{stock}_minmax.json','r'))
            if list(df.columns) != list(minmax_loaded['columns']):
                print('Column Mismatch, resetting')
                reset=True
            else:
                self.minimum = minmax_loaded['minimum']
                self.maximum = minmax_loaded['maximum']
                self.columns = minmax_loaded['columns']

        if (reset == True) or (os.path.exists(f'{stock}_minmax.json') == False):
            minmax = Parallel(n_jobs=8)(delayed(self.min_max_calculator)(filename) for filename in tqdm(glob.glob(data_folder+'*')))
            self.minimum = pd.DataFrame(list(zip(*minmax))[0]).min(axis=0)
            self.maximum = pd.DataFrame(list(zip(*minmax))[1]).max(axis=0)
            self.columns = tuple(i[0] for i in (self.maximum.items()))
            ujson.dump({'minimum' :list(self.minimum), 'maximum':list(self.maximum),'columns':tuple(i[0] for i in (self.maximum.items()))},open(f'{stock}_minmax.json','w'))

    def min_max_calculator(self,filename):
        df = pd.read_parquet(filename)    
        return (df.min(),df.max())
        
    def scale(self,df,to_tensor = True):
        df = (df - pd.Series(np.array(self.minimum),self.columns)) + self.eps
        df = ((df / ((pd.Series(np.array(self.maximum),self.columns)  - pd.Series(np.array(self.minimum),self.columns))+self.eps))*2)-1
        
        if to_tensor == True:
            if len(df)==1:
                return torch.Tensor(df.values.astype(np.float32)).unsqueeze(0)
            else:
                return torch.Tensor(df.values.astype(np.float32))
        else:
            return df


        
# class MinMax():
#     def __init__(self,stock='SPY',reset=False):
        
#         self.eps = np.finfo(np.float32).eps
        
#         if (os.path.exists('SPY_minmax.json') == True) and (reset == False):
#             df = pd.read_parquet(np.random.choice(glob.glob('Data.nosync/SPY/Calendar/train/*')))
#             minmax_loaded = ujson.load(open('SPY_minmax.json','r'))
            
#             if list(df.columns) != list(minmax_loaded['columns']):
#                 print('Column Mismatch, resetting')
#                 reset=True
#             else:
#                 print('Loading Scalers')
#                 self.minimum = minmax_loaded['minimum']
#                 self.maximum = minmax_loaded['maximum']
#                 self.columns = minmax_loaded['columns']


#         if (reset == True) or (os.path.exists('SPY_minmax.json') == False):
#             minmax = Parallel(n_jobs=8)(delayed(self.min_max_calculator)(filename) for filename in tqdm(glob.glob('Data.nosync/SPY/Calendar/train/*')))
#             self.minimum = pd.DataFrame(list(zip(*minmax))[0]).min(axis=0)
#             self.maximum = pd.DataFrame(list(zip(*minmax))[1]).max(axis=0)
#             self.columns = tuple(i[0] for i in (self.maximum.items()))
#             ujson.dump({'minimum' :list(self.minimum), 'maximum':list(self.maximum),'columns':tuple(i[0] for i in (self.maximum.items()))},open('SPY_minmax.json','w'))


        
#     def min_max_calculator(self,filename):
#         df = pd.read_parquet(filename)    
#         return (df.min(),df.max())
        
#     def scale(self,df,to_tensor = True):
        
        
#         df = (df - pd.Series(np.array(self.minimum),self.columns)) + self.eps
#         df = df / ((pd.Series(np.array(self.maximum),self.columns)-pd.Series(np.array(self.minimum),self.columns))+self.eps)
        
#         #reverses the scaling
#         #X_scaled = X_std * (pd.Series(np.array(self.maximum),self.columns) - pd.Series(np.array(self.minimum),self.columns))+pd.Series(np.array(self.minimum),self.columns)
        
#         if to_tensor == True:
#             if len(df)==1:
#                 return torch.Tensor(df.values.astype(np.float32)).unsqueeze(0)
#             else:
#                 return torch.Tensor(df.values.astype(np.float32))
#         else:
#             return df
        
#     def value_scale(self,val,col,reverse=False):
#         assert col in self.columns, 'Column does not exist'
#         col_index = list(self.columns).index(col)
#         if reverse == False:
#             val = (val - self.minimum[col_index])+self.eps
#             val = val/((self.maximum[col_index] - self.minimum[col_index])+self.eps)
#             return val
#         else:
#             val = (val*((self.maximum[col_index] - self.minimum[col_index])+self.eps))+self.minimum[col_index]
#             return round(float(val),3)
