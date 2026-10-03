"""Static top-to-bottom check of a notebook: walks statements in execution order (including inside
for/if/with/try blocks) and flags names used before any earlier code defines them. Free names
inside functions must be defined somewhere in the notebook by the time... (checked loosely)."""
import ast, builtins, json, sys

def targets(node):
    # every name bound anywhere inside node: assignments, loop/with targets, imports,
    # except-handler names, nested def/class names, function arguments
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, ast.arg): out.add(n.arg)
    return out

def loads(expr):
    out = []
    def visit(n):
        if isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # comprehension/lambda: its own vars are local; check only iterables of the first generator
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                local = {c.id for c in ast.walk(n) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}  # incl. nested comprehensions
                for sub in ast.walk(n):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in local:
                        out.append((sub.id, sub.lineno))
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load): out.append((n.id, n.lineno))
        for c in ast.iter_child_nodes(n): visit(c)
    if expr is not None: visit(expr)
    return out

class Walker:
    def __init__(self, known): self.known, self.problems = set(known), []
    def use(self, expr, cell):
        for name, line in loads(expr):
            if name not in self.known: self.problems.append(f"cell {cell}, line {line}: '{name}' used before definition")
    def block(self, stmts, cell):
        for s in stmts: self.stmt(s, cell)
    def stmt(self, s, cell):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in s.decorator_list: self.use(d, cell)
            self.known.add(s.name); return
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names: self.known.add((a.asname or a.name).split(".")[0])
            return
        if isinstance(s, ast.Assign):
            self.use(s.value, cell)
            for t in s.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load): self.use(n, cell)
                    if isinstance(n, (ast.Subscript, ast.Attribute)): self.use(n.value if isinstance(n, ast.Attribute) else n, cell)
                self.known |= targets(t)
            return
        if isinstance(s, (ast.AugAssign, ast.AnnAssign)):
            self.use(s.value, cell); self.known |= targets(s.target); return
        if isinstance(s, (ast.For, ast.AsyncFor)):
            self.use(s.iter, cell); self.known |= targets(s.target)
            self.block(s.body, cell); self.block(s.orelse, cell); return
        if isinstance(s, ast.While):
            self.use(s.test, cell); self.block(s.body, cell); self.block(s.orelse, cell); return
        if isinstance(s, ast.If):
            self.use(s.test, cell); self.block(s.body, cell); self.block(s.orelse, cell); return
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                self.use(item.context_expr, cell)
                if item.optional_vars is not None: self.known |= targets(item.optional_vars)
            self.block(s.body, cell); return
        if isinstance(s, ast.Try):
            self.block(s.body, cell)
            for h in s.handlers:
                if h.type is not None: self.use(h.type, cell)
                if h.name: self.known.add(h.name)
                self.block(h.body, cell)
            self.block(s.orelse, cell); self.block(s.finalbody, cell); return
        if isinstance(s, ast.Global):
            self.known |= set(s.names); return
        for child in ast.iter_child_nodes(s):
            if isinstance(child, ast.expr): self.use(child, cell)

def parse_cells(spec):
    # "0-3,7,10-12" -> [0, 1, 2, 3, 7, 10, 11, 12]
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def check(path, cells_spec=None):
    nb = json.load(open(path, encoding="utf-8"))
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    if cells_spec:
        cells = [cells[i] for i in parse_cells(cells_spec)]   # simulate running only these code cells
    trees = [ast.parse("\n".join("pass" if l.lstrip().startswith("!") else l for l in s.splitlines())) for s in cells]
    w = Walker(set(dir(builtins)) | {"get_ipython", "display"})
    for i, t in enumerate(trees): w.block(t.body, i)
    # names used inside functions must be defined somewhere in the notebook
    every = set(w.known)
    for i, t in enumerate(trees):
        top_fns = [n for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        top_fns += [n for st in t.body if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    for n in ast.walk(st) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in top_fns:
            local = targets(fn)   # outermost function: covers nested defs, closures, local imports
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in every and n.id not in local:
                    w.problems.append(f"cell {i}: '{n.id}' used inside {fn.name}() but never defined")
    return sorted(set(w.problems))

if __name__ == "__main__":
    # Usage: check_notebook_order.py NB.ipynb [NB2.ipynb ...] [--cells 0-3,7]
    #   --cells simulates a partial run (e.g. "setup + merge only"); indices count code cells from 0.
    args, spec = sys.argv[1:], None
    if "--cells" in args:
        i = args.index("--cells"); spec = args[i + 1]; args = args[:i] + args[i + 2:]
    for p in args:
        probs = check(p, spec)
        print(f"{p.split('/')[-1]}: " + ("OK" if not probs else f"{len(probs)} problem(s)"))
        for x in probs: print("   ", x)
