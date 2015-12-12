import numba 
import numpy 

def CreateKernel(V, cycles, domain, cooling=0.99):

    largest = numpy.finfo(numpy.float32).max

    @numba.cuda.jit('void(f4[:], f4[:,:], f4[:], uint16[:,:], f4[:], uint16[:])', target='gpu')
    def _sa2opt(energy, distances, uniform, configuration, temperatures, input):
        index = numba.cuda.grid(1)
        temperature = temperatures[index]
        
        state = numba.cuda.local.array((V, ), dtype=numba.uint16)
        lowest = energy[index]
        local = numba.cuda.local.array((V, ), dtype=numba.uint16)

        for i in range(V):
            state[i] = input[i]
            local[i] = input[i]

        for c in range(cycles):
            r = ((c + index) << 1)
            b = numpy.uint16(uniform[r + 0] * V)
            d = 1 + numpy.uint16(uniform[r + 1] * (V // 2 - 0.))
            e = (b + d) % V

            delta = -distances[state[(V+b-1)%V], state[b]]  -distances[state[e], state[(e+1)%V]]  + distances[state[(V+b-1)%V], state[e]]  + distances[state[b], state[(e+1)%V]]

            if (delta < 0.) or ( ((energy[index] + delta) / (lowest)) * uniform[r + 2] < temperature):
                for s2 in range((1+d) // 2):
                    first = (b + s2) % V
                    second = (V + e - s2) % V
                    t = state[first]
                    state[first] = numpy.uint16(state[second])
                    state[second] = t
                energy[index] += delta

                if energy[index] < lowest:
                    lowest = energy[index]
                    for i in range(V):
                        local[i] = state[i]

            temperature *= cooling

        energy[index] = lowest
        for i1 in range(V):
            configuration[index, i1] = local[i1] 

        numba.cuda.syncthreads()

        minimum = 0
        value = largest
        for o in range(1, numba.cuda.gridsize(1)):
            if energy[o] < value:
                value = energy[o]
                minimum = o

        if index == minimum:
            for i2 in range(V):
                input[i2] = configuration[minimum, i2]
        else:
            new = 0.
            for i3 in range(V):
                configuration[index, i3] = configuration[minimum, i3]
                new += distances[configuration[minimum, i3], configuration[minimum, (i3+1)%V]]
            energy[index] = new


    return _sa2opt



def CreateKernelAnalysis(V, cycles, domain, cooling=0.99):
    '''
    A version that traces all energy and temperature values at each iteration
    '''
    largest = numpy.finfo(numpy.float32).max
    @numba.cuda.jit('void(f4[:], f4[:,:], f4[:], uint16[:,:], f4[:], uint16[:], f4[:,:,:], uint16)', target='gpu')
    def a_sa2opt(energy, distances, uniform, configuration, temperatures, input, trace, iteration):
        index = numba.cuda.grid(1)
        temperature = temperatures[index]
        
        state = numba.cuda.local.array((V, ), dtype=numba.uint16)
        lowest = energy[index]
        local = numba.cuda.local.array((V, ), dtype=numba.uint16)

        for i in range(V):
            state[i] = input[i]
            local[i] = input[i]

        for c in range(cycles):
            r = ((c + index) << 1)
            b = numpy.uint16(uniform[r + 0] * V)
            d = 1 + numpy.uint16(uniform[r + 1] * (V // 2 - 0.))
            e = (b + d) % V

            delta = -distances[state[(V+b-1)%V], state[b]]  -distances[state[e], state[(e+1)%V]]  + distances[state[(V+b-1)%V], state[e]]  + distances[state[b], state[(e+1)%V]]

            trace[iteration, index, c] = energy[index] + delta

            if (delta < 0.) or ( ((energy[index] + delta) / (lowest)) * uniform[r + 2] < temperature):
                for s2 in range((1+d) // 2):
                    first = (b + s2) % V
                    second = (V + e - s2) % V
                    t = state[first]
                    state[first] = numpy.uint16(state[second])
                    state[second] = t
                energy[index] += delta

                if energy[index] < lowest:
                    lowest = energy[index]
                    for i in range(V):
                        local[i] = state[i]

            
            temperature *= cooling

        energy[index] = lowest
        for i1 in range(V):
            configuration[index, i1] = local[i1] 

        numba.cuda.syncthreads()

        minimum = 0
        value = largest
        for o in range(1, numba.cuda.gridsize(1)):
            if energy[o] < value:
                value = energy[o]
                minimum = o

        if index == minimum:
            for i2 in range(V):
                input[i2] = configuration[minimum, i2]
        else:
            new = 0.
            for i3 in range(V):
                configuration[index, i3] = configuration[minimum, i3]
                new += distances[configuration[minimum, i3], configuration[minimum, (i3+1)%V]]
            energy[index] = new


    return a_sa2opt