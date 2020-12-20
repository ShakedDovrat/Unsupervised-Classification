import numpy as np
import pandas as pd

pd = pd.read_csv(r"E:\code\dl-course\Unsupervised-Classification\output\cifar-10\pretext\temp.txt", delim_whitespace=True)


celeb_a_attr_path = r"E:\datasets\celeb-a\celeba\list_attr_celeba.txt"
attr_pd = pd.read_csv(celeb_a_attr_path, delim_whitespace=True, skiprows=1)

m = attr_pd['Male'].values > 0
print(m.mean())
