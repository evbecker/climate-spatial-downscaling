
import numpy as np


def RMSE(pred,target):
    return np.sqrt(np.mean((pred-target)**2))

def PCorrelation(pred,target):
    m_p=np.mean(pred)
    m_t=np.mean(target)
    return np.sum((pred-m_p)*(target-m_t))/(np.sqrt(np.sum((pred-m_p)**2))*np.sqrt(np.sum((target-m_t)**2))+0.00001)

def max_per_init(regions=['nwus','neus','seus','swus'],years=1):
    dic_pred=np.zeros((len(regions),years,12,31))
    dic_target=np.zeros((len(regions),years,12,31))
    regions_dic={}
    i=0
    for region in regions:
        regions_dic[region]=i
        i+=1
    return dic_pred,dic_target,regions_dic

def name_parser(name):
    p=name.find('-')
    region=name[p+1:p+5]
    name=name[p+1:]
    p=name.find('-')
    name=name[p+1:]
    p=name.find('-')
    name=name[p+1:]
    year=int(name[:4])
    month=int(name[5:7])
    day=int(name[8:10])
    return region,year,month,day

def max_per_calculate(pred,target,name,dic_pred,dic_target,regions_dic,base_year=2000):
    region,year,month,day=name_parser(name)
    pred_sum=np.sum(pred)
    target_sum=np.sum(target)
    dic_pred[regions_dic[region],year-base_year,month-1,day-1]+=pred_sum
    dic_target[regions_dic[region],year-base_year,month-1,day-1]+=target_sum
    return dic_pred,dic_target

def max_per(dic_pred,dic_target):
    dic_target=np.max(dic_target,axis=-1)
    dic_pred=np.max(dic_pred,axis=-1)
    dif=dic_target-dic_pred
    return np.mean(dif),np.std(dif)

