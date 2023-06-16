import numpy as np
import matplotlib.pyplot as plt
import random

c = 0.8             # reward sensitivity
d = 0.3             #punishment sensitivity
init_Ps = np.full(2, 0.5)
gamma = 0.1         #reversal probability
Ps = np.zeros(2)

log_lik = []
action = []

#设置正确action
right_act = np.concatenate((np.ones(80), np.zeros(20),np.ones(20),np.zeros(20),np.ones(20)))
#随机获得假数据
n = 1  # 每次试验的次数
p = 0.5  # 成功的概率
size = len(right_act)  # 数组的长度
action = np.random.binomial(n, p, size)
print(action)

#获得真实的rewards，当被试每一项都正确答对时，仍然有对应概率获得±1的奖赏
reward = np.zeros(size)
# 前80个一致时的奖励
reward[:80] = np.where(right_act[:80] == action[:80], np.random.choice([1, -1], size=80, p=[0.75, 0.25]), np.random.choice([1, -1], size=80, p=[0.25, 0.75]))
# 80-100个一致时的奖励
reward[80:100] = np.where(right_act[80:100] == action[80:100], np.random.choice([1, -1], size=20, p=[0.2, 0.8]), np.random.choice([1, -1], size=20, p=[0.8, 0.2]))
# 100-120个一致时的奖励
reward[100:120] = np.where(right_act[100:120] == action[100:120], np.random.choice([1, -1], size=20, p=[0.8, 0.2]), np.random.choice([1, -1], size=20, p=[0.2, 0.8]))
# 120-140个一致时的奖励
reward[120:140] = np.where(right_act[120:140] == action[120:140], np.random.choice([1, -1], size=20, p=[0.2, 0.8]), np.random.choice([1, -1], size=20, p=[0.8, 0.2]))
# 140-160个一致时的奖励
reward[140:160] = np.where(right_act[140:160] == action[140:160], np.random.choice([1, -1], size=20, p=[0.8, 0.2]), np.random.choice([1, -1], size=20, p=[0.2, 0.8]))


print(reward)