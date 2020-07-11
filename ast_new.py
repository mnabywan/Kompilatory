class Node(object):
    def __init__(self):
        self.line = 0
        self.column = 0

    def accept(self, visitor):
        return visitor.visit(self)

class Program(Node):
    def __init__(self, instructions):
        self.instructions = instructions


class Instructions(Node):
    def __init__(self, instructions = []):
        self.instructions = instructions


class IntNum(Node):
    def __init__(self, value):
        self.value = value


class FloatNum(Node):
    def __init__(self, value):
        self.value = value

class String(Node):
    def __init__(self, value):
        self.value = value


class Variable(Node):
    def __init__(self, name):
        self.name = name

class Ref(Node):
    def __init__(self, variable, indexes):
        self.variable = variable
        self.indexes = indexes

class BinExpr(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class Assignment(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class AssignmentAndExpr(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class RelExpr(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class UnaryExpr(Node):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

class For(Node):
    def __init__(self, variable, range1, instruction):
        self.variable = variable
        self.range1 = range1
        self.instruction = instruction

class While(Node):
    def __init__(self, condition, instruction):
        self.condition = condition
        self.instruction = instruction


class If(Node):
    def __init__(self, condition, instruction, else_instruction=None):
        self.condition = condition
        self.instruction = instruction
        self.else_instruction = else_instruction

class Range(Node):
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Print(Node):
    def __init__(self, expressions):
        self.expressions = expressions

class Continue(Node):
    def __init__(self):
        pass

class Break(Node):
    def __init__(self):
        pass

class Return(Node):
    def __init__(self, value=None):
        self.value = value


class Vector(Node):
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def int_dims(self):
        result = (len(self.coordinates), )
        if isinstance(self.coordinates[0], Vector):
            result += self.coordinates[0].int_dims()
        return result

class MatrixInit(Node):
    def __init__(self, dim_1, dim_2=None):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
    def int_dims(self):
        if isinstance(self.dim_1, IntNum) and (not self.dim_2 or isinstance(self.dim_2, IntNum)):
            dim1 = self.dim_1.value
            if self.dim_2:
                dim2 = self.dim_2.value
            else:
                dim2 = self.dim_1.value
            return (dim1, dim2)
        else:
            return None

class Eye(MatrixInit):
    pass

class Zeros(MatrixInit):
    pass

class Ones(MatrixInit):
    pass

class Error(Node):
    def __init__(self):
        pass