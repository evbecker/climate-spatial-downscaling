
import numpy as np


def RMSE(pred,target):
    return np.sqrt(np.mean((pred-target)**2))

def PCorrelation(pred,target):
    m_p=np.mean(pred)
    m_t=np.mean(target)
    return np.sum((pred-m_p)*(target-m_t))/(np.sqrt(np.sum((pred-m_p)**2))*np.sqrt(np.sum((target-m_t)**2)))
