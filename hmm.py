import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """

    N = len(A)
    M = len(B[0])
    T = len(O)
    calculated_alphas = np.zeros((T,N))  

    for t in range(T):
        if t == 0 : #initialization 
            calculated_alphas[0, :] = pi * B[:, O[0]] # a_0[j] = pi * B[j][O_0] 
        else:
            for j in range(N): # recursion
                total = 0
                for x_i, y_i in zip(calculated_alphas[t - 1], A[:, j]): # summation
                    total += x_i * y_i
                    
                calculated_alphas[t, j] = B[j, O[t]] * total

    prob = sum(calculated_alphas[len(calculated_alphas)-1])

    return prob, calculated_alphas.transpose()




def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    N = len(A)
    M = len(B[0])
    T = len(O)
    sigmas = np.zeros((T,N))
    sequence = np.zeros((T))
    prev = np.zeros((T - 1, N))
    for t in range(T):
        if t == 0 : #initialization 
            
            sigmas[0, :] = pi * B[:, O[0]] # a_0[j] = pi * B[j][O_0] 
        else:
            for j in range(N): #recursion
                maximum = -float("inf")
                i = 0
                index = 0
                # highest probability at t
                for x_i, y_i in zip(sigmas[t - 1], A[:, j]):
                    if x_i * y_i > maximum:
                        maximum = x_i * y_i
                        index = i
                    i += 1
                # keeping pointer back to the winning state to find the
                # most likely state sequence
                prev[t - 1, j] = index
                sigmas[t, j] = B[j, O[t]] * maximum # sigma for t at j


    last_state = np.argmax(sigmas[T - 1, :])
    sequence[0] = last_state
    index = 1

    for i in range(T - 2, -1, -1):  # Backtrack from last observation.
        sequence[index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)] # Most likely state saved
        index += 1

    sequence = sequence[::-1] # Since backtracked to sequence, we will reverse it
    return sequence, sigmas.transpose()