# Parameter validation using Sage (Neo App. B.10)
from sage.all import *

# Parameters matching SECURE_PARAMS
q = 2**64 - 2**32 + 1
n = 54
k = 16
d = 32
b = 2
sigma = 3.2
beta = 3

# MSIS bound (rough estimate)
msis = k * d * log(q, 2) + log(2 * sigma * sqrt(n * k), 2) * d - log(b, 2)
print("MSIS lambda ~", msis)
assert msis > 128

# RLWE bound (rough estimate)
rlwe = n * log(q, 2) - log(sigma**2 * n, 2)
print("RLWE lambda ~", rlwe)
assert rlwe > 128
