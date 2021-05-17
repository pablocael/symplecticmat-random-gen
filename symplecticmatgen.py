import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200


def isSymplecticMat(M, O):

    R = np.matmul(np.matmul(M.T, O), M)
    return np.allclose(R, O, rtol=1e-5, atol=1e-5)

def generateSymplecticMat(size):

    A = np.random.randn(size*size).reshape(size, size)
    Ainv = np.linalg.inv(A.T)

    D = np.zeros((2*size, 2*size))
    D[:size, :size] = A
    D[-size:, -size:] = Ainv

    B = np.random.randn(size*size).reshape(size, size)
    B = np.matmul(B, B.T) # multiply a matrix by its transpose to generate a symmetric matrix

    N = np.zeros((2*size, 2*size))
    N[:size, :size] = np.identity(size)
    N[-size:, -size:] = np.identity(size)
    N[:size, -size:] = B

    O = np.zeros((2*size, 2*size))
    O[:size, -size:] = np.identity(size)
    O[-size:, :size] = -np.identity(size)
    return np.matmul(O, np.matmul(D, N)), O

if __name__ == '__main__':
    M, O = generateSymplecticMat(8)
    print(M)
    print('Is symplectic?', isSymplecticMat(M, O))