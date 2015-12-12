import numpy
import pickle

pkl_file = open('data_l.pkl', 'rb')

data1 = pickle.load(pkl_file)

bestresults = numpy.array(list(map(lambda x: x[1], data1)))

import matplotlib.pyplot as plt

fig = plt.figure()
# plt.yscale('log')

labels = [0.75, 0.375, 0.25, 0.75/4, 0.75/5, 0.75/6, 0.75/7, 0.75/8, 0.75/9, 0.075]

z = []
for i in range(10):
    seq = bestresults[i: 101: 10]
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in numpy.linspace(0, 1, 10)]
    
    q = []

    for j in range(10):
        q.append(seq[j].flatten())
        '''if 0 == j:
            plt.plot(seq[j].flatten(), color=colors[i], label='T(initial)=' + str(labels[i]))
        else:
            plt.plot(seq[j].flatten(), color=colors[i])
    '''
    qm = []
    qe = []
    offsets = [999]
    for offs in offsets:
        qv = list(map(lambda x: x[offs], q))
        qm.append(numpy.mean(qv))
        qe.append(numpy.std(qv))
    z.append((qm[0]))
    plt.errorbar([labels[i]], qm, numpy.sqrt(qe))
    
plt.plot(labels, z, 'k')

plt.xlabel('t')
plt.ylabel('E')
#plt.legend()
fig.show()

input("exit>")