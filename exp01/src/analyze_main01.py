import pandas as pd 
import matplotlib.pyplot as plt 

df_test = pd.read_csv('results_test.csv')
print(len(df_test))
df_test.dropna(inplace=True)
print(len(df_test))
df_val = pd.read_csv('results_val.csv')
print(len(df_val))
df_val.dropna(inplace=True)
print(len(df_val))

# x1_arr = []
# x2_arr = []
# a = 0
# for i in range(len(df_test)):
#     x1_arr.append(a)
#     x2_arr.append(a+1)
#     a = a + 2

plt.figure(figsize=(10,5))
plt.subplot(211)
plt.ylabel('Testing\nFrequency')
# plt.bar(range(len(df_test)),df_test['l1']/df_test['l2'])
plt.hist(df_test['l1']/df_test['l2'],bins=300)

plt.subplot(212)
plt.xlabel('L1/L2')
plt.ylabel('Validation\nFrequency')
# plt.bar(range(len(df_val)),df_val['l1']/df_val['l2'])
plt.hist(df_val['l1']/df_val['l2'],bins=300)

plt.savefig('analyze_main01.png',bbox_inches='tight')