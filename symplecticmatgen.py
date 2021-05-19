import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

# sets a seed for reproducibility
np.random.seed(42)

"""
    Checks if a matrix is symplectic by applying the definition
    O is Omega matrix [[0, I(n)], [-I(n), 0]], where I(n) is nxn identity matrix
"""
def isSymplecticMat(M, O):
    R = np.matmul(np.matmul(M.T, O), M)
    return np.allclose(R, O, rtol=1e-5, atol=1e-5)

"""
    Generates a random gaussian symplectic matrix of size 2*size by 2*size
    Reference: https://en.wikipedia.org/wiki/Symplectic_matrix
"""
def generateSymplecticMat(size):
    A = np.random.randn(size*size).reshape(size, size)
    Ainv = np.linalg.inv(A.T)

    # N matrix as in reference (see reference above function signature)
    D = np.zeros((2*size, 2*size))
    D[:size, :size] = A
    D[-size:, -size:] = Ainv

    # B matrix block
    B = np.random.randn(size*size).reshape(size, size)
    B = np.matmul(B, B.T) # multiply a matrix by its transpose to generate a symmetric matrix

    # N matrix as in reference (see reference above function signature)
    N = np.zeros((2*size, 2*size))
    N[:size, :size] = np.identity(size)
    N[-size:, -size:] = np.identity(size)
    N[:size, -size:] = B

    # Omega Matrix
    O = np.zeros((2*size, 2*size))
    O[:size, -size:] = np.identity(size)
    O[-size:, :size] = -np.identity(size)
    return np.matmul(O, np.matmul(D, N)), O

def getAverageEigenValues(M):
    eigValues, _ = np.linalg.eig(M)

    return eigValues.real.mean()

if __name__ == '__main__':
    # generate thousands of random symplectic matrices and check if they are correct
    numMatricesToGenerate = 1000
    avgEigenValues = []
    for i in range(numMatricesToGenerate):
        # choose a random matrix size
        matrixSize = np.random.randint(8,100)
        M, O = generateSymplecticMat(matrixSize)
        if not isSymplecticMat(M, O):
            print('ERROR, some generated matrix is not symplectic:', M)
        avgEigenValues.append(getAverageEigenValues(M))


    print(f'All {numMatricesToGenerate} generated symplectic matrices are correct!')

    print('plotting average eigen values histogram...')
    plt.hist(avgEigenValues, density=1)
    plt.show()

