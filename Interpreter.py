from ast_new import *
from SymbolTable import *
from Memory import *
from Exceptions import *
from visit import *
import sys
import operator
import numpy as np

sys.setrecursionlimit(10000)


def multiply(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dot(b)
    else:
        return a * b


class Interpreter(object):
    def __init__(self):
        self.memory_stack = MemoryStack()

        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
        }

        self.assignment_dict = {
            '=': lambda x, y: y,
            '+=': lambda x, y: x + y,
            '-=': lambda x, y: x - y,
            '*=': lambda x, y: x * y,
            '/=': lambda x, y: x / y,
        }

        self.relation_dict = {
            '>': lambda x, y: x > y,
            '>=': lambda x, y: x >= y,
            '<': lambda x, y: x < y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
        }

        self.matrix_dict = {
            '.+': (lambda x, y: (np.matrix(x) + np.matrix(y)).tolist()),
            '.-': (lambda x, y: (np.matrix(x) - np.matrix(y)).tolist()),
            '.*': (lambda x, y: np.multiply(np.array(x), np.array(y)).tolist()),
            './': (lambda x, y: np.divide(np.array(x), np.array(y)).tolist())
        }

        self.unary_dict = {
            'NEGATE': lambda x: -x,
            'TRANSPOSE' : lambda x: x.T
        }


    @on('node')
    def visit(self, node):
        pass

    @when(Program)
    def visit(self, node):
        try:
            node.instructions.accept(self)
        except ReturnValueException as e:
            if e.value:
                print("\nRETURN: ")
                print(e.value)

    @when(Instructions)
    def visit(self, node):
        for instructon in node.instructions:
            instructon.accept(self)

    @when(IntNum)
    def visit(self, node):
        return node.value

    @when(FloatNum)
    def visit(self, node):
        return node.value

    @when(Variable)
    def visit(self, node):
        return self.memory_stack.get(node.name)

    @when(Ref)
    def visit(self, node):
        var = node.variable.accept(self)
        for index in node.indexes:
            var = var[index.accept(self)]
        return var

    @when(BinExpr)
    def visit(self, node):
        left = node.left.accept(self)
        right = node.right.accept(self)
        return self.operations[node.op](left, right)

    @when(Assignment)
    def visit(self, node):
        value = node.right.accept(self)
        if isinstance(node.left, Ref):
            var = node.left.variable.accept(self)
            indexes = list()
            for index in node.left.indexes:
                indexes.append(index.accept(self))
            var[tuple(indexes)] = value

        elif isinstance(node.left, Variable):
            self.memory_stack.insert(node.left.name, value)

    @when(AssignmentAndExpr)
    def visit(self, node):
        left = node.left.accept(self)
        right = node.right.accept(self)
        value = self.assignment_dict[node.op](left, right)
        if isinstance(node.left, Ref):
            var = node.left.variable.accept(self)
            indexes = list()
            for index in node.left.indexes:
                indexes.append(index.accept(self))
            var[tuple(indexes)] = value
        elif isinstance(node.left, Variable):
            self.memory_stack.set(node.left.name, value)

    @when(RelExpr)
    def visit(self, node):
        left = node.left.accept(self)
        right = node.right.accept(self)
        return self.relation_dict[node.op](left, right)

    @when(UnaryExpr)
    def visit(self, node):
        arg = node.arg.accept(self)
        return self.unary_dict[node.op](arg)

    @when(For)
    def visit(self, node):
        self.memory_stack.push(Memory("for"))

        range = node.range1.accept(self)
        name = node.variable.name
        self.memory_stack.insert(name, range[0])
        while self.memory_stack.get(name) < range[1]:
            try:
                node.instruction.accept(self)
            except ContinueException:
                self.memory_stack.set(name, self.memory_stack.get(name) + 1)
                continue
            except BreakException:
                break
            self.memory_stack.set(name, self.memory_stack.get(name) + 1)

        self.memory_stack.pop()

    @when(While)
    def visit(self, node):
        self.memory_stack.push(Memory("while"))

        while (node.condition.accept(self)):
            try:
                node.instruction.accept(self)
            except ContinueException:
                continue
            except BreakException:
                break

        self.memory_stack.pop()

    @when(Range)
    def visit(self, node):
        start = node.start.accept(self)
        end = node.end.accept(self)
        return (start, end)

    @when(If)
    def visit(self, node):
        cond = node.condition.accept(self)
        if node.condition.accept(self):
            node.instruction.accept(self)
        elif node.else_instruction:
            node.else_instruction.accept(self)

    @when(Return)
    def visit(self, node):
        if node.value:
            value = node.value.accept(self)
            raise ReturnValueException(value)
        else:
            raise ReturnValueException()

    @when(Print)
    def visit(self, node):
        expressions = list()
        for expr in node.expressions:
            to_print = expr.accept(self)
            print(to_print, end=' ')
        print()

    @when(Continue)
    def visit(self, node):
        raise ContinueException()

    @when(Break)
    def visit(self, node):
        raise BreakException()

    @when(Vector)
    def visit(self, node):
        coors = list()
        for coor in node.coordinates:
            coors.append(coor.accept(self))
        return np.array(coors)

    @when(Eye)
    def visit(self, node):
        dim_1 = node.dim_1.accept(self)
        if node.dim_2:
            dim_2 = node.dim_2.accept(self)
        else:
            dim_2 = dim_1
        return np.eye(dim_1, dim_2)

    @when(Zeros)
    def visit(self, node):
        dim_1 = node.dim_1.accept(self)
        if node.dim_2:
            dim_2 = node.dim_2.accept(self)
        else:
            dim_2 = dim_1
        return np.zeros((dim_1, dim_2))

    @when(Ones)
    def visit(self, node):
        dim_1 = node.dim_1.accept(self)
        if node.dim_2:
            dim_2 = node.dim_2.accept(self)
        else:
            dim_2 = dim_1
        return np.ones((dim_1, dim_2))

    @when(String)
    def visit(self, node):
        return node.value[1:-1]