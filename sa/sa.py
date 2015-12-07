# coding: utf-8
# In[1]:
import os
import sys
import tsplib

dataset = "tsplib/"
index = 0
files = sorted(next(os.walk(dataset))[2])
for file in files:
    end="\t\t"
    if index % 4 == 3:
        end="\r\n"
    print(str(index).zfill(3), '\t', file, end=end) 
    index+=1
print()    
problem = int(input('problem number>'))
vertices, labels = tsplib.parse("tsplib/" + files[problem])
V = len(vertices)
print("Vertices: ", V)

# In[2]:
import numpy
z = numpy.array([[complex(c[1][0], c[1][1]) for c in vertices.items()]])
distances = numpy.array(numpy.round(abs(z.T - z)), dtype=numpy.float32, order='F')

# In[3]:
# The order given by the TSPLib format is near optimal,
# so we shuffle it to a random state to make the problem more interesting
state = numpy.array(list(range(V)), dtype=numpy.uint16)
import random
random.seed(0)
random.shuffle(state)

result = 0.
for i in range(V):
    result += distances[state[i], state[(1 + i) % V]]
print('mu=', numpy.mean(distances), ' E[d]=', V * numpy.mean(distances), ' r=', result)


# In[4]:
import numba
import numbapro
from numbapro.cudalib import curand

numba.cuda.profile_start()
cycles = 10
import optimise
sa2opt = optimise.CreateKernel(V, cycles, 0.99)

# precautions: 64/1024, 48/96
threads = min(512, numba.cuda.get_current_device().MAX_THREADS_PER_BLOCK)
blocks = 96 // 2
operations = blocks * threads
results = numpy.repeat(numpy.array([result], dtype=numpy.float32), operations)
###############################################################################
iterations = 10000
initial_exponent = 0.7
delta = 0.25

best = (result, state, initial_exponent)

for i in range(iterations):    
    exponent = best[2]
    high = numpy.max(0.000001, exponent + delta - 1./operations)
    low = numpy.max(0.000001, exponent - delta)
    temperatures = numpy.array(numpy.linspace(low, high, num=operations), dtype=numpy.float32, order='F')

    configuration = best[1].reshape((1, V))
    configuration = numpy.repeat(configuration, operations, axis=0)
    results = numpy.repeat(numpy.array([best[0]], dtype=numpy.float32), operations)
    d_distances = numba.cuda.to_device(distances)

    stream = numba.cuda.stream()
    with stream.auto_synchronize():
        entropy = operations * cycles * 3
        d_uniform = numba.cuda.device_array(entropy, dtype=numpy.float32, stream=stream)
        generator = curand.PRNG(curand.PRNG.MRG32K3A, stream=stream, seed=0)
        generator.uniform(d_uniform)

        execute = sa2opt[(blocks, 1), (threads, 1), stream]

        d_results = numba.cuda.to_device(results, stream=stream)
        d_temperatures = numba.cuda.to_device(temperatures, stream=stream)
        d_configuration = numba.cuda.to_device(configuration, stream=stream)
        
        execute(d_results, d_distances, d_uniform, d_configuration, d_temperatures)

        d_results.to_host(stream=stream)
        d_temperatures.to_host(stream=stream)
        d_configuration.to_host(stream=stream)
    
    local = numpy.min(results)
    first = numpy.where(results == local)[0][0]
    print((numpy.min(results),numpy.mean(results), exponent))
    sequence = configuration[first, :]
    
    if local < best[0]:
        best = (local, sequence, numpy.min(exponent * 0.98, 0.9))#, temperatures[first]))
        last = numpy.where(results == numpy.max(results))[0]
        

    else:
        # should restart all
        break

###############################################################################
import matplotlib.pyplot as plt

figure = plt.figure()

sequence = list(zip(*list(map(lambda v: vertices[v], best[1]))))
plt.plot(sequence[0], sequence[1])
plt.scatter(sequence[0], sequence[1])
figure.show()





    
numba.cuda.profile_stop()

input("exit>")
sys.exit(0)