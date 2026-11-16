import numpy as np

def heisenberg_xxz(n, J=1.0, Delta=0.7, h=0.8):
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def kron_chain(ops):
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out
        
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(n-1):
        for P, c in [(X, J), (Y, J), (Z, J*Delta)]:
            ops = [I]*n
            ops[i] = P
            ops[i+1] = P
            H += c * kron_chain(ops)
            
    for i in range(n):
        ops = [I]*n
        ops[i] = Z
        H += h * kron_chain(ops)
        
    return H
