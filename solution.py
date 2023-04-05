import pandas as pd
import numpy as np

from scipy.stats import norm, t


chat_id = 156287560 # Ваш chat ID, не меняйте название переменной
'''
def confidence_interval(alpha, arr):
    n = len(arr)
    t_alpha_2 = t.ppf(1 - alpha/2, n-1)
    s = np.std(arr, ddof=1)
    lambda_ = s/np.mean(arr)
    a = np.mean(arr) - t_alpha_2*s/np.sqrt(n*(n-1))*np.sqrt(lambda_+1/4*n*lambda_**2)
    b = np.mean(arr) + t_alpha_2*s/np.sqrt(n*(n-1))*np.sqrt(lambda_+1/4*n*lambda_**2)
    return (a, b)
'''
def solution(p: float, x: np.array) -> tuple:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    alpha = 1 - p
    #loc = x.mean()
    #scale = np.sqrt(np.var(x)) / np.sqrt(len(x))
    #return loc - scale * norm.ppf(1 - alpha / 2), \
    #       loc - scale * norm.ppf(alpha / 2)
    distances_list = x
    time = 14
    n = len(distances_list)
    #error_distribution = np.random.exponential(1, size = len(x))
    #error_distribution = np.full(len(x),1/2) - error_distribution
    #distances_with_error = distances_list +error_distribution
    # x = x_0 + v_0*t + a*t^2/2  ===> a*t^2 = 2*x ===> 2*x/t^2
    acceleration_list = (np.array(x)*2)/(time**2)
    # задаем распределение ошибки измерения
    #error_distribution = lambda x: 1/2 - np.exp(1)
    # преобразуем данные с учетом ошибки измерения
    #acceleration_with_error = acceleration_list + error_distribution(acceleration_list)
   

    loc = acceleration_list.mean()
    scale = np.sqrt(np.var(acceleration_list)) / np.sqrt(len(acceleration_list))
    return loc - scale * norm.ppf(1 - alpha / 2), \
           loc - scale * norm.ppf(alpha / 2)
    '''
    mean = np.mean(acceleration_with_error)
    std_error = np.std(acceleration_with_error, ddof=1) / np.sqrt(n)
    t_val = abs(t.ppf(alpha / 2, n - 1))
    z_val = abs(norm.ppf(alpha / 2))
    interval = t_val * std_error if n > 30 else z_val * std_error
    upper_bound = np.exp(mean + interval)
    lower_bound = np.exp(mean - interval)
    return (upper_bound, lower_bound)
    '''


#p = 0.95
#arr = [100.5, 105.2, 101.1, 99.8, 97.3, 103.7, 98.6, 104.8, 99.9]
#print(solution(p, arr))
#solution(p, arr)
