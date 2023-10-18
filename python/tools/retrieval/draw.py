import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy.stats

def normalize_data(data):
    score_max = data[0]
    score_min = data[0]
    for i in data:
        score_max = max(score_max, i)
        score_min = min(score_min, i)
    diff = score_max - score_min
    for i in range(len(data)):
        data[i] = (data[i] - score_min) / diff
    return data

def draw_distribute(data, name="test", num=1):
    plt.figure(num)
    print(name)
    n, bins, patches = plt.hist(x=data, bins=20, density=True, color='steelblue', edgecolor='black', alpha=0.5)
    mu = np.mean(data)
    sigma = np.std(data)
    print('mu: %f   sigma:  %f' % (mu, sigma))
    print(len(data))
    print("max: {},  min: {}".format(max(data), min(data)))
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel(f'{name} score')
    plt.ylabel('frequency* class width')

if __name__ == '__main__':
    data_list = ['load_mmlu','load_truthfulqa', 'load_bbq', 'load_gsm8k']
    num = 0
    for i in data_list:
        trec_path = f"/home/lzy/nips/data/platypus/match/trec/run.bigbench.{i}.txt"
        data = []
        with open(trec_path, 'r') as f:
            for j in f:
                _, _, _, _, score, _ = j.split()
                data.append(float(score))
        num += 1
        draw_distribute(data, name=f"bigbench_{i}_all", num=num)
        #data = normalize_data(data)
        #draw_distribute(data, name=f"{i}_all_min_max", num=2)
        
        plt.savefig(f"/home/lzy/nips/data/platypus/match/pic/bigbench_{i}.png")


    

# flan_load_mmlu_all                                                                                                         
# mu: 23.977620   sigma:  24.546617                                                                                          
# 14033423                                                                                                                   
# max: 534.730774,  min: 0.0169                                                                                              
# flan_load_truthfulqa_all                                                                                                   
# mu: 10.323494   sigma:  3.561863                                                                                           
# 817000                                                                                                                     
# max: 36.645401,  min: 2.483297
# flan_load_bbq_all                                                                                                          
# mu: 21.841430   sigma:  8.976598                                                                                           
# 1000000                                                                                                                    
# max: 68.875603,  min: 6.814795                                                                                             
# flan_load_gsm8k_all                                                                                                        
# mu: 33.238144   sigma:  11.592124

# bigbench_load_mmlu_all                                       
# mu: 18.324720   sigma:  20.137359
# 14000166                                                     
# max: 334.83139,  min: 0.1687
# bigbench_load_truthfulqa_all                                 
# mu: 7.849502   sigma:  2.759725
# 817000                                                       
# max: 44.2356,  min: 1.891589
# bigbench_load_bbq_all        
# mu: 20.506145   sigma:  13.802776
# 1000000                                                      
# max: 207.050903,  min: 4.9059
# bigbench_load_gsm8k_all      
# mu: 25.338115   sigma:  9.845980        
# 1319000
# max: 176.345306,  min: 8.0451