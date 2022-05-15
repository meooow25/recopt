import ast
import inspect
import sys
import textwrap
from typing import Callable, List, Set

runner_stmts = ast.parse('''
recopt_stack = []
recopt_result = None
recopt_running = False

def recopt_push(f, *args):
    recopt_stack.append((f, args))

def recopt_run_stack():
    nonlocal recopt_running
    recopt_running = True
    while recopt_stack:
        f, args = recopt_stack.pop()
        f(*args)
    recopt_running = False
    return recopt_result
''').body

assert_not_running_stmt = ast.parse('assert not recopt_running, "already running"').body[0]
run_stack_stmt = ast.parse('return recopt_run_stack()').body[0]

def name_load(id: str) -> ast.Name:
    return ast.Name(id, ast.Load())

def name_store(id: str) -> ast.Name:
    return ast.Name(id, ast.Store())

def simple_arguments(args: List[str]) -> ast.arguments:
    if sys.version_info < (3, 8):
        return ast.arguments(
            args=[ast.arg(arg=arg) for arg in args],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        )
    return ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg) for arg in args],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        )

def simple_func_def(name: str, args: List[str], body: List[ast.AST]) -> ast.FunctionDef:
    return ast.FunctionDef(name=name, args=simple_arguments(args), body=body, decorator_list=[])

def simple_call(func: str, args: List[str]) -> ast.Call:
    return ast.Call(func=name_load(func), args=[name_load(arg) for arg in args], keywords=[])

def get_param_names(arguments: ast.arguments) -> List[str]:
    return (
        ([] if sys.version_info < (3, 8) else [a.arg for a in arguments.posonlyargs]) +
        [a.arg for a in arguments.args]
    )

def visit_iwf(stmts: List[ast.AST], f: Callable):
    for stmt in stmts:
        if isinstance(stmt, (ast.If, ast.While, ast.For)):
            visit_iwf(stmt.body, f)
            visit_iwf(stmt.orelse, f)
        else:
            f(stmt)

def get_not_locals(stmts: List[ast.AST]) -> Set[str]:
    not_local = set()
    def one(stmt):
        nonlocal not_local
        if isinstance(stmt, (ast.Nonlocal, ast.Global)):
            not_local |= set(stmt.names)
    visit_iwf(stmts, one)
    return not_local

def get_assigns(stmts: List[ast.AST]) -> Set[str]:
    # Take common assignments from the two branches for ifs
    # Assume while/for loops don't introduce any assignments
    def one(stmt):
        if isinstance(stmt, ast.Assign):
            return {tgt.id for tgt in stmt.targets if isinstance(tgt, ast.Name)}
        if isinstance(stmt, ast.FunctionDef):
            return {stmt.name}
        if isinstance(stmt, ast.If):
            return many(stmt.body) & many(stmt.orelse)
        return {}
    def many(stmts):
        return {x for stmt in stmts for x in one(stmt)}
    return many(stmts)

def replace_break_continue(stmts: List[ast.AST], break_rep: ast.AST, continue_rep: ast.AST) -> List[ast.AST]:
    def one(stmt):
        if isinstance(stmt, ast.Break):
            return [break_rep, ast.Return()]
        if isinstance(stmt, ast.Continue):
            return [continue_rep, ast.Return()]
        if isinstance(stmt, ast.If):
            return [ast.If(stmt.test, many(stmt.body), many(stmt.orelse))]
        return [stmt]
    def many(stmts):
        return [x for stmt in stmts for x in one(stmt)]
    return many(stmts)

def replace_return(stmts: List[ast.AST]) -> List[ast.AST]:
    def one(stmt):
        if isinstance(stmt, ast.Return) and stmt.value:
            return [ast.Assign([name_store('recopt_result')], stmt.value), ast.Return()]
        if isinstance(stmt, ast.If):
            return [ast.If(stmt.test, many(stmt.body), many(stmt.orelse))]
        if isinstance(stmt, ast.While):
            return [ast.While(stmt.test, many(stmt.body), many(stmt.orelse))]
        if isinstance(stmt, ast.For):
            return [ast.For(stmt.target, stmt.iter, many(stmt.body), many(stmt.orelse))]
        return [stmt]
    def many(stmts):
        return [x for stmt in stmts for x in one(stmt)]
    return many(stmts)

def get_top_name(name):
    return name + '_top'

def optimize_recursion_ast(scope_ast: ast.FunctionDef, func_names: List[str]) -> ast.FunctionDef:

    def is_relevant_call_stmt(stmt: ast.AST):
        def check(expr: ast.AST):
            return (isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id in func_names)
        return isinstance(stmt, (ast.Assign, ast.Expr)) and check(stmt.value)

    def has_recursive_call(stmts: List[ast.AST]) -> bool:
        has = False
        def one(stmt):
            nonlocal has
            has |= is_relevant_call_stmt(stmt)
        visit_iwf(stmts, one)
        return has

    def transform_func(f_ast: ast.FunctionDef) -> List[ast.FunctionDef]:
        not_local_vars = get_not_locals(f_ast.body)

        name_counter = 0
        funcs = []

        def nxt_name(base: str) -> str:
            nonlocal name_counter
            name_counter += 1
            return f_ast.name + '_' + base + '_' + str(name_counter)

        def make_fn_return(name: str, args: List[str], stmts: List[ast.AST]) -> ast.Return:
            assert len(stmts) >= 1
            if len(stmts) == 1:
                # If the only statement is a function call, don't make a new fn
                stmt = stmts[0]
                if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) or
                    isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)):
                    return stmt
            fn_def = simple_func_def(name, args, [ast.Nonlocal(['recopt_result'])] + stmts)
            funcs.append(fn_def)
            return ast.Expr(simple_call('recopt_push', [fn_def.name] + args))

        def split(stmts: List[ast.AST], local_vars: List[str]) -> List[ast.AST]:
            stmts1 = []

            def get_cur_local_vars():
                assigns = get_assigns(stmts1) - not_local_vars
                return sorted(set(local_vars) | assigns)

            for j in range(len(stmts)):
                stmt = stmts[j]
                if isinstance(stmt, ast.If) and has_recursive_call([stmt]):
                    cur_local_vars = get_cur_local_vars()
                    assigns = get_assigns([stmt])
                    cur_if_local_vars = sorted(set(cur_local_vars) | assigns)

                    nxt_ret_stmt = make_fn_return(
                        nxt_name('next'),
                        cur_if_local_vars,
                        split(stmts[j+1:], cur_if_local_vars),
                    )

                    stmts1.append(
                        ast.If(
                            stmt.test,
                            split(stmt.body + [nxt_ret_stmt], cur_local_vars),
                            split(stmt.orelse + [nxt_ret_stmt], cur_local_vars),
                        )
                    )
                    return stmts1

                if isinstance(stmt, ast.While) and has_recursive_call([stmt]):
                    cur_local_vars = get_cur_local_vars()

                    nxt_ret_stmt = make_fn_return(
                        nxt_name('next'),
                        cur_local_vars, # Assume loops dont make any local vars
                        split(stmts[j+1:], cur_local_vars),
                    )

                    while_fn_name = nxt_name('while_body')
                    while_fn_ret = ast.Expr(simple_call('recopt_push', [while_fn_name] + cur_local_vars))
                    while_fn_body = [
                        ast.If(
                            test=stmt.test,
                            body=split(
                                replace_break_continue(stmt.body, nxt_ret_stmt, while_fn_ret),
                                cur_local_vars,
                            ),
                            orelse=split(stmt.orelse + [nxt_ret_stmt], cur_local_vars),
                        )
                    ]

                    stmts1.append(make_fn_return(while_fn_name, cur_local_vars, while_fn_body))
                    return stmts1

                if isinstance(stmt, ast.For) and has_recursive_call([stmt]):
                    cur_local_vars = get_cur_local_vars()

                    nxt_ret_stmt = make_fn_return(
                        nxt_name('next'),
                        cur_local_vars, # Assume loops dont make any local vars
                        split(stmts[j+1:], cur_local_vars),
                    )

                    it_var = nxt_name('for_it')
                    cur_for_local_vars = sorted(set(cur_local_vars) | {it_var})
                    for_fn_name = nxt_name('for_body')
                    for_fn_ret = ast.Expr(simple_call('recopt_push', [for_fn_name] + cur_for_local_vars))
                    for_body_stmts = [
                        ast.Try(
                            body=[ast.Assign([stmt.target], simple_call('next', [it_var]))],
                            handlers=[
                                ast.ExceptHandler(
                                    type=name_load('StopIteration'),
                                    body=split(stmt.orelse + [nxt_ret_stmt], cur_local_vars),
                                )
                            ],
                            orelse=split(
                                replace_break_continue(stmt.body, nxt_ret_stmt, for_fn_ret) + [for_fn_ret],
                                cur_for_local_vars,
                            ),
                            finalbody=[],
                        )
                    ]

                    stmts1.append(
                        ast.Assign(
                            [name_store(it_var)],
                            ast.Call(func=name_load('iter'), args=[stmt.iter], keywords=[]),
                        )
                    )
                    stmts1.append(make_fn_return(for_fn_name, cur_for_local_vars, for_body_stmts))
                    return stmts1

                if is_relevant_call_stmt(stmt):
                    if isinstance(stmt, ast.Expr):
                        down_stmts = []
                    elif isinstance(stmt, ast.Assign):
                        down_stmts = [ast.Assign(stmt.targets, name_load('recopt_result'))]
                    else:
                        raise ValueError('unreachable')
                    down_stmts += stmts[j+1:]

                    cur_local_vars = get_cur_local_vars()
                    nxt_ret_stmt = make_fn_return(
                        nxt_name('next'),
                        cur_local_vars,
                        split(down_stmts, set(cur_local_vars)),
                    )

                    assert isinstance(stmt.value, ast.Call)
                    assert not stmt.value.keywords, 'only positional args supported'
                    call_stmt = ast.Expr(
                        ast.Call(
                            func=name_load('recopt_push'),
                            args=[name_load(get_top_name(stmt.value.func.id))] + stmt.value.args,
                            keywords=[],
                        )
                    )

                    stmts1.append(nxt_ret_stmt)
                    stmts1.append(call_stmt)
                    return stmts1

                stmts1.append(stmt)

            if not stmts1:
                stmts1.append(ast.Pass())
            return stmts1

        assert not (f_ast.args.vararg or f_ast.args.kwonlyargs or f_ast.args.kwarg), 'only positional args supported'
        param_names = get_param_names(f_ast.args)
        top_ret = make_fn_return(
            get_top_name(f_ast.name),
            param_names,
            split(replace_return(f_ast.body), param_names),
        )

        make_fn_return(
            f_ast.name,
            param_names,
            [assert_not_running_stmt, top_ret, run_stack_stmt],
        )

        return funcs

    def transform_all(scope: ast.FunctionDef) -> ast.FunctionDef:
        stmts1 = runner_stmts[:]
        for stmt in scope.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name in func_names:
                fn_defs = transform_func(stmt)
                stmts1 += fn_defs
            else:
                stmts1.append(stmt)
        return ast.FunctionDef(
            name=scope.name,
            args=scope.args,
            body=stmts1,
            decorator_list=[],
        )

    return transform_all(scope_ast)

def optimize_recursion(scope_func, func_names: List[str]):
    scope_ast = ast.parse(textwrap.dedent(inspect.getsource(scope_func)), filename='<recopt>').body[0]
    assert isinstance(scope_ast, ast.FunctionDef)
    
    scope_ast_optimized = optimize_recursion_ast(scope_ast, func_names)

    scope_ast_module = ast.Module([scope_ast_optimized], type_ignores=[])
    ast.fix_missing_locations(scope_ast_module)
    lcls = {}
    exec(compile(scope_ast_module, filename='<recopt>', mode='exec'), globals(), lcls)
    scope_func_optimized = lcls[scope_ast.name]
    if sys.version_info > (3, 9):
        scope_func_optimized.src = ast.unparse(scope_ast_optimized)
    return scope_func_optimized

def optrec(*func_names: str):
    def wrapper(scope_func):
        return optimize_recursion(scope_func, func_names)
    return wrapper
