import numpy as np
import matplotlib.pyplot as plt
##设定的gamma、c、d都是较为理想的值（意思是能让均值分布接近于real_traj）
c = 0.9             # reward sensitivity
d = 0.95             # punishment sensitivity
init_Ps = np.full(2, 0.5)
gamma = 0.05         # reversal probability
size = 160
num_subjects = 100
real_traj = [0.75] * 80 + [0.2] * 20 + [0.8] * 20 + [0.2] * 20 + [0.8] * 20
#设置正确action
right_act = np.concatenate((np.zeros(80), np.ones(20),np.zeros(20),np.ones(20),np.zeros(20)))

# Generate data for 100 subjects
all_actions = np.zeros((num_subjects, size))
all_rewards = np.zeros((num_subjects, size))

##subject loop
for subject in range(num_subjects):
    Ps = np.zeros(2)
    action = np.zeros(size)
    reward = np.zeros(size)
    ##trial loop
    for t in range(size):
        if t == 0:
            Ps = init_Ps
        else:
            Ps[0] = Ps[0] * (1 - gamma) + Ps[1] * gamma
            Ps[1] = 1 - Ps[0]

        action[t] = np.random.choice([0, 1], p=Ps)
        #在不同的trial有不同的reward prob，前80个正确获得并正确反馈的概率为0.75，后80个为0.8
        #注意：在这里real_traj也就是action prob的体现就在于先前设定的right_act，已经设置了正确答案。
        #因此可以直接将正确结果和fake action进行比较。而不用在每个转换概率的试次阶段内设置0.75-0.2-0.8-0.2-0.8
        if t < 80:
            reward[t] = np.where(right_act[t] == action[t], np.random.choice([1, -1], p=[0.75, 0.25]), np.random.choice([1, -1], p=[0.25, 0.75]))
        else:
            reward[t] = np.where(right_act[t] == action[t], np.random.choice([1, -1], p=[0.8, 0.2]), np.random.choice([1, -1], p=[0.2, 0.8]))
            
        if reward[t] == 1:
            if action[t] == 1:
                    P_O_S1 = 0.5 * c
                    P_O_S2 = 0.5 * (1 - c)
            else:
                    P_O_S1 = 0.5 * (1 - c)
                    P_O_S2 = 0.5 * c
        else:
            if action[t] == 1:
                P_O_S1 = 0.5 * (1 - d)
                P_O_S2 = 0.5 * d
            else:
                P_O_S1 = 0.5 * d
                P_O_S2 = 0.5 * (1 - d)

        new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * (1-Ps[1]))
        Ps[0] = new_Ps0
        Ps[1] = 1 - new_Ps0
   
    all_actions[subject] = action


 #计算每个被试在每个trial上的action均值
mean_actions = np.mean(all_actions, axis=0)


# Plot combined trajectory and mean action values as line plots
##generate fake data后画图
# 绘制action的轨迹图
plt.plot(mean_actions,'b-')
plt.plot(real_traj,'r-')

# 设置图表标题和坐标轴标签
plt.title("Trajectory")
plt.xlabel("Trial")
plt.ylabel("Action")

# 显示图形
plt.savefig('compare')
