import numpy as np
import sys, os


def loadmusk(dirpath):
    # bag
    bags = []
    with open(os.path.join(dirpath, 'musk1-bag.data'), 'r') as f:
        data = f.read().split(os.linesep)

        n = 0
        bagnum = 0

        while n < len(data):
            bag = []
            while True:
                if n == len(data):
                    break
                tmp = data[n].split(' ')
                header = tmp[0]
                if bagnum == int(float(header)):
                    bag.append(np.array(tmp[1:]))
                    n += 1
                else:
                    break
            #print(np.array(bag).shape)
            bags.append(np.array(bag, dtype=float))
            bagnum += 1


    # label
    labels = []
    with open(os.path.join(dirpath, 'musk1-label.data'), 'r') as f:
        data = f.read().split(os.linesep)

        for baglabel in data:
            labels.append(float(baglabel))

    labels = np.array(labels, dtype=float)

    return bags, labels