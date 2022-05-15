import sys
import timeit
from math import factorial

import recopt
from bootstrap import bootstrap

sys.setrecursionlimit(10 ** 9)

n = 10 ** 6
mod = 10 ** 9 + 7
ans = factorial(n) % mod

def fact(n):
    if n == 0:
        return 1
    return n * fact(n-1) % mod

@bootstrap
def bootstrap_fact(n):
    if n == 0:
        yield 1
    r = yield bootstrap_fact(n-1)
    yield r * n % mod

@recopt.optrec('fact')
def run_fact(n, mod):
    def fact(n):
        if n == 0:
            return 1
        r = fact(n-1)
        return n * r % mod
    return fact(n)

def direct():
    assert fact(n) == ans

def with_bootstrap():
    assert bootstrap_fact(n) == ans

def with_recopt():    
    assert run_fact(n, mod) == ans

def run(f):
    print('{:<16}{:.3f}'.format(f.__name__, timeit.timeit(f, number=5) / 5))

run(direct)
run(with_bootstrap)
run(with_recopt)

# $ pypy3.9 bench_fact.py 
# direct          0.379
# with_bootstrap  0.464
# with_recopt     0.213
