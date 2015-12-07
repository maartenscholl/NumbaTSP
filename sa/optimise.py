import numba 
import numpy 

def CreateKernel(V, cycles, cooling=0.99):
    @numba.cuda.jit('void(f4[:], f4[:,:], f4[:], uint16[:,:], f4[:], uint16[:], uint16[:])', target='gpu')
    def _sa2opt(debug, distances, uniform, configuration, temperatures, b_, e_):
        index = numba.cuda.grid(1)
        temperature = temperatures[index]

        state = numba.cuda.local.array((V, ), dtype=numba.uint16)
        for i in range(V):
            state[i] = configuration[index, i]
        delta = 0.
        for c in range(cycles):
            r = ((c + index) << 1)
            b = numpy.uint16(uniform[r + 0] * V)
            d = 2 + numpy.uint16(uniform[r + 1] * (V // 2 - 0.))
            e = (b + d) % V

            delta = -distances[state[(V+b-1)%V], state[b]]  -distances[state[e], state[(e+1)%V]]  + distances[state[(V+b-1)%V], state[e]]  + distances[state[b], state[(e+1)%V]]   #new - debug[index]

            b_[index] = b
            e_[index] = e

            if (delta < 0.) or (uniform[r + 2] < temperature):
                for s2 in range((1+d) // 2):
                    first = (b + s2) % V
                    second = (V + e - s2) % V
                    t = state[first]
                    state[first] = numpy.uint16(state[second])
                    state[second] = t
                temperatures[index] = temperature
                debug[index] += delta
            temperature *= cooling
            


    return _sa2opt