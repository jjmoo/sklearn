#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import *
import inspect
import matplotlib.pyplot as plt

print(abspath(inspect.getsourcefile(lambda: 0)))
print(join(dirname(abspath(inspect.getsourcefile(lambda: 0))), 'data/test.combined.txt'))

with open('data/test.combined.txt') as fr:
    start_tag = '<review id='
    label_param = 'label="'
    end_tag = '</review>'
    comments = []
    labels = []
    comment = ''
    label = -1
    for line in fr.readlines():
        line = line.strip()
        if 0 == len(line):
            continue
        elif line.__contains__(start_tag):
            label_index = line.find(label_param)
            if label_index >= 0:
                label = int(line[label_index + len(label_param)])
        elif line.__contains__(end_tag):
            comments.append(comment)
            labels.append(label)
            comment = ''
        else:
            if len(comment) > 0:
                comment += '\n'
            comment += line
    print(comments[4], labels[4])


_, axes = plt.subplots(1, 2, sharey='row', figsize=(15, 6))
axes[0].hist(0, 50, density=1)
axes[1].hist(0, 50, density=1)
plt.show()

plt.hist()

