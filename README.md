# recopt

Optimize recursion in [PyPy](https://www.pypy.org/download.html)

---

```py
@recopt.optrec('fact') # <--- this
def run_fact(n, mod):
    def fact(n):
        if n == 0:
            return 1
        r = fact(n-1)
        return n * r % mod
    return fact(n)
```

**What is this?**  
Recursion in PyPy is very slow (why? I don't know). This function transforms the [AST](https://docs.python.org/3/library/ast.html) of a recursive function to remove the recursive calls while preserving the behavior.

**How does it work?**  
A function call is about executing a function and resuming where we left off once the called function is done. This is achieved here by splitting the body of a recursive function into multiple functions, such that resuming means executing another function. The functions to resume are maintained on a stack.

There are some restrictions on what is supported.
* The recursive function must have positional arguments only.
* Recursive call statements must be simple calls `f()` or assignments `x = f()`.
* The only control flow statements supported are `if`, `while`, `for`.

**How fast is it?**  
See [`bench_fact.py`](bench_fact.py) and [`bench_dfs.py`](bench_dfs.py).

**Tell me more about how to use it**  
Don't.
