"""
Function calling framework — lightweight tool execution for the LLM.

How it works:
  1. add_tools_to_prompt()   injects available tools into the system prompt
  2. extract_tool_call()     parses LLM output for <tool_call>...</tool_call>
  3. execute_tool()          runs the tool and returns a result string
  4. The chat endpoint calls these helpers and loops if a tool was invoked.

Available tools (safe, no side effects):
  - get_current_datetime()
  - calculate(expression)
  - get_book_list()          — requires books_dir
  - get_book_info(title)     — searches Qdrant metadata
"""
import re
import ast
import math
import json
import datetime
import logging
import os
import operator as _op

logger = logging.getLogger("functions")

# ── Tool definitions (sent to LLM in system prompt) ──────────────────────────
TOOL_DEFINITIONS = """
الأدوات المتاحة لك (استخدمها عند الحاجة):

<tool_call>{"name": "get_current_datetime", "args": {}}</tool_call>
  → يُرجع التاريخ والوقت الحالي.

<tool_call>{"name": "calculate", "args": {"expression": "2 + 2 * 10"}}</tool_call>
  → يُرجع نتيجة عملية حسابية آمنة.

<tool_call>{"name": "get_book_list", "args": {}}</tool_call>
  → يُرجع قائمة الكتب المتاحة في المكتبة.

<tool_call>{"name": "get_book_info", "args": {"title": "اسم الكتاب"}}</tool_call>
  → يُرجع معلومات عن كتاب معين (عدد الصفحات، التاريخ).

قواعد استخدام الأدوات:
- ضع وسم <tool_call> مباشرة في ردك عند الحاجة.
- انتظر النتيجة قبل الإكمال — لا تخترع النتائج.
- إن لم تحتج أداة، أجب مباشرة بدون <tool_call>.
"""

_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# ── Safe AST-based math evaluator (replaces eval) ────────────────────────────
_SAFE_OPS = {
    ast.Add:  _op.add,   ast.Sub:  _op.sub,
    ast.Mult: _op.mul,   ast.Div:  _op.truediv,
    ast.Mod:  _op.mod,   ast.Pow:  _op.pow,
    ast.UAdd: _op.pos,   ast.USub: _op.neg,
    ast.FloorDiv: _op.floordiv,
}
_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": math.pow, "floor": math.floor,
    "ceil": math.ceil, "log": math.log, "sin": math.sin,
    "cos": math.cos, "tan": math.tan,
}
_SAFE_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_MAX_NUM = 1e15   # prevent gigantic intermediate values


def _safe_eval_node(node):
    """Recursively evaluate an AST node — only safe numeric ops allowed."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, complex)):
            raise ValueError("غير مسموح")
        if abs(node.value) > _MAX_NUM:
            raise ValueError("الرقم كبير جداً")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise ValueError(f"متغير غير معروف: {node.id}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("عملية غير مدعومة")
        left  = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Guard against DoS exponents
        if isinstance(node.op, ast.Pow) and abs(right) > 300:
            raise ValueError("الأس كبير جداً")
        result = op_fn(left, right)
        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            raise ValueError("النتيجة خارج النطاق")
        return result
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("عملية غير مدعومة")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("دالة غير مسموح")
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"دالة غير معروفة: {node.func.id}")
        args = [_safe_eval_node(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"بنية AST غير مدعومة: {type(node).__name__}")


def add_tools_to_prompt(system_prompt: str, enable: bool = True) -> str:
    if not enable:
        return system_prompt
    return system_prompt + "\n\n" + TOOL_DEFINITIONS


def extract_tool_call(text: str) -> dict | None:
    m = _TOOL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def strip_tool_call(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_get_current_datetime(**_) -> str:
    now = datetime.datetime.now()
    weekdays = ["الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"]
    months   = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
                "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]
    return (f"{weekdays[now.weekday()]}، {now.day} {months[now.month - 1]} {now.year}"
            f" — الساعة {now.strftime('%H:%M')}")


def _tool_calculate(expression: str = "", **_) -> str:
    expr = expression.strip()
    if not expr or len(expr) > 200:
        return "خطأ: تعبير فارغ أو طويل جداً"
    # Handle ^ as ** (common user expectation: 2^3 → 8)
    expr = expr.replace('^', '**')
    try:
        tree   = ast.parse(expr, mode='eval')
        result = _safe_eval_node(tree)
        # Format: hide float decimals when result is a whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression.strip()} = {result}"
    except (SyntaxError, ValueError) as ex:
        return f"خطأ: {ex}"
    except Exception as ex:
        return f"خطأ في الحساب: {ex}"


def _tool_get_book_list(books_dir: str = "", **_) -> str:
    if not books_dir or not os.path.isdir(books_dir):
        return "لا يمكن الوصول للمكتبة حالياً"
    pdfs = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]
    if not pdfs:
        return "المكتبة فارغة"
    return "الكتب المتاحة:\n" + "\n".join(f"• {p[:-4]}" for p in sorted(pdfs))


def _tool_get_book_info(title: str = "", scroll_fn=None, **_) -> str:
    if not title:
        return "يرجى تحديد اسم الكتاب"
    if scroll_fn is None:
        return "الميزة غير متاحة حالياً"
    try:
        data = scroll_fn()
        metas = data.get("metadatas", [])
        book_metas = [m for m in metas
                      if m.get("book_title", "").lower() == title.lower()
                      and not m.get("raptor_level")]
        if not book_metas:
            return f"لم يُعثر على الكتاب: {title}"
        pages = {m.get("page", 0) for m in book_metas}
        return (f"الكتاب: {title}\n"
                f"عدد الصفحات المفهرسة: {len(pages)}\n"
                f"عدد الفقرات: {len(book_metas)}")
    except Exception as e:
        return f"خطأ: {e}"


# ── Tool dispatcher ───────────────────────────────────────────────────────────

_TOOL_MAP = {
    "get_current_datetime": _tool_get_current_datetime,
    "calculate":            _tool_calculate,
    "get_book_list":        _tool_get_book_list,
    "get_book_info":        _tool_get_book_info,
}


def execute_tool(call: dict, *, books_dir: str = "", scroll_fn=None) -> str:
    """Execute a parsed tool call and return a result string."""
    name = call.get("name", "")
    args = call.get("args", {})
    fn   = _TOOL_MAP.get(name)
    if fn is None:
        return f"أداة غير معروفة: {name}"
    try:
        return fn(**args, books_dir=books_dir, scroll_fn=scroll_fn)
    except Exception as e:
        logger.warning(f"Tool {name} failed: {e}")
        return f"فشل تنفيذ الأداة {name}: {e}"
