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
# plt.ylabel('Testing\nFrequency')
# plt.bar(range(len(df_test)),df_test['l1']/df_test['l2'])
# plt.hist(df_test['superior']-df_test['inferior'],bins=300)
df_test_left = df_test[df_test['left_right']=='LEFT']
df_test_right = df_test[df_test['left_right']=='RIGHT']
plt.bar([0],[len(df_test_left)])
plt.bar([1],[len(df_test_right)])
plt.xticks([0,1],['LEFT','RIGHT'])
plt.ylabel('Testing\nNumber of samples')


plt.subplot(212)
# plt.xlabel('Superior Rim - Inferior Rim')
# plt.ylabel('Validation\nFrequency')
# plt.bar(range(len(df_val)),df_val['l1']/df_val['l2'])
# plt.hist(df_val['superior']-df_val['inferior'],bins=300)
df_val_left = df_val[df_val['left_right']=='LEFT']
df_val_right = df_val[df_val['left_right']=='RIGHT']
plt.bar([0],[len(df_val_left)])
plt.bar([1],[len(df_val_right)])
plt.xticks([0,1],['LEFT','RIGHT'])
plt.ylabel('Validation\nNumber of samples')

plt.savefig('analyze_main01.png',bbox_inches='tight')