import dace

M = dace.symbol("M")
N = dace.symbol("N")

@dace.program
def vecadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]

@dace.program
def matadd(A: dace.float64[M,N], B: dace.float64[M,N], C: dace.float64[M,N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = A[i, j] + B[i, j]

# Represents single read multiple write
@dace.program
def shiftright0(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:(N-1)]:
        B[i+1] = A[i] + 1

# Represents single read multiple write
@dace.program
def shiftright1(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N]:
        B[i] = A[i-1] + 1

# Represents single write multiple read
@dace.program
def shiftleft0(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:(N-1)]:
        B[i] = A[i+1] + 1

# Represents single write multiple read
@dace.program
def shiftleft1(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N]:
        B[i-1] = A[i] + 1

programs = [vecadd]