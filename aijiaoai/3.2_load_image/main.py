#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '../../data/sample_1.jpg')
img = plt.imread(path)
print(img.shape)

plt.imshow(img)
plt.show()
