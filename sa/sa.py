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
problem = 0
if os.getenv('PROFILE') is None:
    problem = int(input('problem number>'))
vertices, labels = tsplib.parse("tsplib/" + files[problem])
V = len(vertices)
print("Vertices: ", V)

# In[2]:
import numpy
z = numpy.array([[complex(c[1][0], c[1][1]) for c in vertices.items()]])
distances = numpy.array(numpy.round(abs(z.T - z)), dtype=numpy.float32, order='F')


# in the worst case we swap the smallest route with the largest
smallest = 2 * numpy.min(distances[numpy.nonzero(distances)])
largest = 2 * numpy.max(distances[numpy.nonzero(distances)])
domain = largest - smallest

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
cycles = V


# precautions: 64/768, 48/96
threads = min(256, numba.cuda.get_current_device().MAX_THREADS_PER_BLOCK)
blocks = 96 // 4
operations = blocks * threads
#results = numpy.repeat(numpy.array([result], dtype=numpy.float32), operations)
results = numba.cuda.pinned_array(operations, dtype=numpy.float32, order='F')
for o in range(operations):
    results[o] = result

import optimise
sa2opt = optimise.CreateKernel(V, cycles, domain, 0.995)

###############################################################################
iterations = 100
initial_exponent = 0.7
delta = 0.28
best = (result, state, initial_exponent)
configuration = numba.cuda.pinned_array((operations,V), dtype=numpy.uint16, order='F')
stream = numba.cuda.stream()

d_distances = numba.cuda.to_device(distances)

@numba.jit 
def Copy(config, energy):
    for o in range(operations):
        configuration[o, :] = config
Copy(best[1], best[0])
    
d_results = numba.cuda.to_device(results, stream=stream)
d_configuration = numba.cuda.to_device(configuration, stream=stream)

output = numba.cuda.pinned_array((V,), dtype=numpy.uint16, order='F')
for i in range(V):
        output[i] = state[i]

entropy = operations * cycles * 3
d_uniform = numba.cuda.device_array(entropy, dtype=numpy.float32, stream=stream)
generator = curand.PRNG(curand.PRNG.MRG32K3A, stream=stream, seed=0)

for i in range(iterations):    
    
    # every iteration only set temperatures
    exponent = best[2]
    high = numpy.max(0.000001, exponent + delta - 1./operations)
    low = numpy.max(0.000001, exponent - delta)
    temperatures = numpy.array(numpy.linspace(low, high, num=operations), dtype=numpy.float32, order='F')

    with stream.auto_synchronize():
        generator.uniform(d_uniform)

        execute = sa2opt[(blocks, ), (threads, ), stream]

        d_output = numba.cuda.to_device(output, stream)
        d_temperatures = numba.cuda.to_device(temperatures, stream=stream)
        
        execute(d_results, d_distances, d_uniform, d_configuration, d_temperatures, d_output)
        d_output.to_host(stream=stream)
    
    local = 0.
    for i in range(V):
        local += distances[output[i], output[(1 + i) % V]]

    adaptive = min(0.99, local/best[0])
    print(local, exponent)
    if local < best[0] and len(set(output)) == V:
        assert len(set(output)) == V, 'state corrupted, check for cuda errors'
        best = (local, output, exponent * adaptive)
    else:
        if len(set(output)) == V:
            best = (local, output, exponent * 0.98)
        else:
            best = (best[0], best[1], exponent * 0.98)
        #break



numba.cuda.profile_stop()
###############################################################################
if os.getenv('PROFILE') is None:
    import matplotlib.pyplot as plt

    figure = plt.figure()

    sequence = list(zip(*list(map(lambda v: vertices[v], best[1]))))
    plt.plot(sequence[0], sequence[1], 'b')
    plt.plot([sequence[0][0], sequence[0][-1]], [sequence[1][0], sequence[1][-1]], 'b')

    plt.scatter(sequence[0], sequence[1])
    figure.show()


    input("exit>")
print('done')
sys.exit(0)