# -*- coding: utf-8 -*-
#"""
#<Script Info>
#
#데이터 분포 보여주기
#"""

import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd


import math_mnist
mnist = math_mnist.read_data_sets()

batch_size = 100
df_labels = math_mnist.read_categories()

total_batch = int(mnist.train.num_examples / batch_size)

category_lists = []
for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    for ys in batch_ys:
        category_lists.append(math_mnist.get_category_char(ys,df_labels))

category_dict={}
for lst in category_lists:
    try: category_dict[lst] += 1
    except: category_dict[lst]=1

#Sort by values
category_dict = OrderedDict(sorted(category_dict.items(),key=lambda t: t[1], reverse=True))

#for Visualization
group_data = list(category_dict.values())
group_names = list(category_dict.keys())
group_mean = numpy.mean(group_data)

fig, ax = plt.subplots(figsize=(10,40))

y_pos = numpy.arange(len(group_names))
ax.barh(y_pos, group_data, align='center', color='green', ecolor='black', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(group_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('counts')
ax.set_title('numbers of data')

plt.show()

#Creating a Table
df = pd.DataFrame(list(category_dict.items()),columns=['symbols','counts'])
df = df.sort_values(['counts','symbols'],ascending=False)
print(df)
#df.to_excel('dict1.xlsx')
