import sys
import timeit

import recopt
from bootstrap import bootstrap

sys.setrecursionlimit(10 ** 9)

n = 10 ** 6
g = [[i+1] for i in range(n-1)] + [[]] # path graph
ans = sum(range(n))

def dfs(u):
    s = u
    for v in g[u]:
        s += dfs(v)
    return s

@bootstrap
def bootstrap_dfs(u):
    s = u
    for v in g[u]:
        s += yield bootstrap_dfs(v)
    yield s

@recopt.optrec('dfs')
def run_dfs(g, u):
    def dfs(u):
        s = u
        for v in g[u]:
            r = dfs(v)
            s += r
        return s
    return dfs(u)

def direct():
    assert dfs(0) == ans

def with_bootstrap():
    assert bootstrap_dfs(0) == ans

def with_recopt():    
    assert run_dfs(g, 0) == ans

def run(f):
    print('{:<16}{:.3f}'.format(f.__name__, timeit.timeit(f, number=5) / 5))

run(direct)
run(with_bootstrap)
run(with_recopt)

# $ pypy3.9 bench_dfs.py
# direct          0.517
# with_bootstrap  0.453
# with_recopt     0.341
