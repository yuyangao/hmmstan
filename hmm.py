import numpy as np
import matplotlib.pyplot as plt


choice = np.loadtxt("demo_files/action.txt", dtype=int)
reward = np.loadtxt("demo_files/rewards.txt", dtype=int)

c = 0.8             # reward sensitivity
d = 0.3             #punishment sensitivity
init_Ps = np.full(2, 0.5)
gamma = 0.1         #reversal probability
Ps = np.zeros(2)

log_lik = []
action = []

#整个过程就是prior-likelihood-posterior

##计算 transition probability
#-------------------------------------------------#
#          P(St|St-1) = (1-gamma    gamma  )      #
#                       (gamma      1-gamma)      #
#-------------------------------------------------#

for t in range(160):
    if t == 0:
        Ps = init_Ps
    else:
        Ps[0] = Ps[0] * (1 - gamma) + Ps[1] * gamma
        Ps[1] = 1 - Ps[0]

    log_lik.append(np.log(Ps[choice[t] - 1]))
    action.append(np.random.choice([0,1],p=Ps))  # 使用相同的 Ps 生成动作，就是generate fake_data
    
##得到St之后,根据不同的sensitivity计算观察到Ot的概率
#-------------------------------------------------#
#          P(Ot|St) = 0.5 * (c     1-c)           #
#                           (1-d     d)           #
#-------------------------------------------------#
    if reward[t] == 1:
        if choice[t] == 1:
            P_O_S1 = 0.5 * c
            P_O_S2 = 0.5 * (1 - c)
        else:
            P_O_S1 = 0.5 * (1 - c)
            P_O_S2 = 0.5 * c
    else:
        if choice[t] == 1:
            P_O_S1 = 0.5 * (1 - d)
            P_O_S2 = 0.5 * d
        else:
            P_O_S1 = 0.5 * d
            P_O_S2 = 0.5 * (1 - d)
##计算St的posteri并作为下一次循环的prior
#-------------------------------------------------#
#                    P(Ot|St)*P(St)               #
#         P(St) = ————————————————————            #
#                  ΣSt P(Ot|St)*P(St)             #
#-------------------------------------------------#
    new_Ps0 = (P_O_S1 * Ps[0]) / (P_O_S1 * Ps[0] + P_O_S2 * (1-Ps[1]))
    Ps[0] = new_Ps0
    Ps[1] = 1 - new_Ps0

print(len(action), len(log_lik))

##generate fake data后画图
# 绘制action的轨迹图
real_traj = [0.75]*80 + [0.2]*20 + [0.8]*20 + [0.2]*20 +[0.8]*20 
plt.plot(action,'b-')
plt.plot(real_traj,'r-')

# 设置图表标题和坐标轴标签
plt.title("Trajectory")
plt.xlabel("Trial")
plt.ylabel("Action")

# 显示图形
plt.savefig('compare')

