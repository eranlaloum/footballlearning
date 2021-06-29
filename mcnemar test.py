import numpy as np
import csv
from scipy.special import comb
from scipy.stats import chi2, binom
alpha=0.05
a=0
b=0
c=0
d=0
learning_rewards=[]
policy_rewards=[]
with open('test_learning.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        learning_rewards.append([float(val) for val in row][5])

with open('test_policy.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        policy_rewards.append([float(val) for val in row][5])

loop=len(policy_rewards)
for i in range(loop):

    if learning_rewards[i]==policy_rewards[i] and learning_rewards[i]==1:
        a+=1
    elif policy_rewards[i]>learning_rewards[i]:
        b+=1
    elif learning_rewards[i] > policy_rewards[i]:
        c+=1
    else:
        d+=1

table=np.array([[a,b],[c,d]])
print(table)
print("ho: there is no difference between the learning and the policy")
print("h1: there is a difference between the learning and the policy")
###two side test###
'''
x2_statistic=((table[0][1]-table[1][0])**2)/(table[0][1]+table[1][0])
print(f"x2_statistic: {x2_statistic}")
x2_critic= chi2.ppf(1-alpha,df=1)
print(f"x2_critic: {x2_critic}")
if x2_statistic<x2_critic:
    print(f"x2_statistic: {x2_statistic} is smaller than x2_critic: {x2_critic}, do not reject ho: there is no difference between policy and laerning")
else:
    print(
        f"x2_statistic: {x2_statistic} is bigger than x2_critic: {x2_critic}, reject ho: there is a difference between policy and laerning")
'''
###one side test###
x2_statistic = (np.absolute(table[0][1] - table[1][0]) - 1) ** 2 / (table[0][1] + table[1][0])
p_value = chi2.sf(x2_statistic, 1)
print(f"x2_statistic {x2_statistic}")
print(f"pvalue {p_value}")
i = table[0][1]
print(i)
n = table[1][0] + table[0][1]
print(n)
i_n = np.arange(i + 1, n + 1)
print(i_n)

p_value_exact = 1 - np.sum(comb(n, i_n) * 0.5 * i_n * (1 - 0.5) * (n - i_n))
p_value_exact *= 2

mid_p_value = p_value_exact - binom.pmf(table[0][1], n, 0.5)

print('p-value Exact: ', p_value_exact)
print('Mid p-value: ', mid_p_value)


