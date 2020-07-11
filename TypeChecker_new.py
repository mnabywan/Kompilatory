from SymbolTable import *
from ast_new import *
from collections import defaultdict
import functools
import operator
import copy


# types
INVALID = 'invalid'
INTEGER = 'int'
FLOAT = 'float'
BOOL = 'bool'
STRING = 'string'
VECTOR = 'vector'
MATRIX = 'matrix'

#  operators
arithmetic = ['+', '-', '*', '/']
assign = ['+=', '-=', '*=', '/=']
matrix = ['.+', '.-', '.*', './']
relational = ['>', '>=', '<', '<=', '==', '!=']

class NodeVisitor(object):

    def visit(self, node, table=None):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, table)

    def generic_visit(self, node, table):  # Called if no explicit visitor function exists for a node.
        if isinstance(node, list):
            for elem in node:
                self.visit(elem)
        else:
            for child in node.children:
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, Node):
                            self.visit(item)
                elif isinstance(child, Node):
                    self.visit(child)


class TypeChecker(NodeVisitor):
    def __init__(self):
        self.error = False
        self.operation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 'unknown')))

        self.operation_results['+'][INTEGER][INTEGER] = INTEGER
        self.operation_results['+'][FLOAT][INTEGER] = FLOAT
        self.operation_results['+'][FLOAT][FLOAT] = FLOAT
        self.operation_results['+'][INTEGER][FLOAT] = FLOAT
        self.operation_results['+'][STRING][STRING] = STRING



        self.operation_results['-'][INTEGER][INTEGER] = INTEGER
        self.operation_results['-'][FLOAT][INTEGER] = FLOAT
        self.operation_results['-'][FLOAT][FLOAT] = FLOAT
        self.operation_results['-'][INTEGER][FLOAT] = FLOAT
        self.operation_results['-'][STRING][STRING] = STRING



        self.operation_results['*'][INTEGER][INTEGER] = INTEGER
        self.operation_results['*'][FLOAT][INTEGER] = FLOAT
        self.operation_results['*'][FLOAT][FLOAT] = FLOAT
        self.operation_results['*'][INTEGER][FLOAT] = FLOAT

        self.operation_results['*'][INTEGER][VECTOR] = VECTOR
        self.operation_results['*'][FLOAT][VECTOR] = VECTOR
        self.operation_results['*'][VECTOR][INTEGER] = VECTOR
        self.operation_results['*'][VECTOR][FLOAT] = VECTOR
        self.operation_results['*'][MATRIX][MATRIX] = MATRIX
        self.operation_results['*'][MATRIX][VECTOR] = MATRIX
        self.operation_results['*'][VECTOR][VECTOR] = MATRIX

        self.operation_results['/'][INTEGER][INTEGER] = FLOAT
        self.operation_results['/'][FLOAT][FLOAT] = FLOAT
        self.operation_results['/'][INTEGER][FLOAT] = FLOAT
        self.operation_results['/'][FLOAT][INTEGER] = FLOAT
        self.operation_results['/'][VECTOR][INTEGER] = VECTOR
        self.operation_results['/'][VECTOR][FLOAT] = VECTOR

        self.operation_results['+='][INTEGER][INTEGER] = INTEGER
        self.operation_results['+='][FLOAT][INTEGER] = FLOAT
        self.operation_results['+='][FLOAT][FLOAT] = FLOAT
        self.operation_results['+='][INTEGER][FLOAT] = FLOAT
        self.operation_results['+='][STRING][STRING] = STRING



        self.operation_results['-='][INTEGER][INTEGER] = INTEGER
        self.operation_results['-='][FLOAT][INTEGER] = FLOAT
        self.operation_results['-='][FLOAT][FLOAT] = FLOAT
        self.operation_results['-='][INTEGER][FLOAT] = FLOAT
        self.operation_results['-='][STRING][STRING] = STRING

        self.operation_results['*='][INTEGER][INTEGER] = INTEGER
        self.operation_results['*='][FLOAT][INTEGER] = FLOAT
        self.operation_results['*='][FLOAT][FLOAT] = FLOAT
        self.operation_results['*='][INTEGER][FLOAT] = FLOAT

        self.operation_results['*='][INTEGER][VECTOR] = VECTOR
        self.operation_results['*='][FLOAT][VECTOR] = VECTOR
        self.operation_results['*='][VECTOR][INTEGER] = VECTOR
        self.operation_results['*='][VECTOR][FLOAT] = VECTOR
        self.operation_results['*='][MATRIX][MATRIX] = MATRIX
        self.operation_results['*='][MATRIX][VECTOR] = MATRIX
        self.operation_results['*='][VECTOR][VECTOR] = MATRIX


        self.operation_results['/='][INTEGER][INTEGER] = FLOAT
        self.operation_results['/='][FLOAT][FLOAT] = FLOAT
        self.operation_results['/='][INTEGER][FLOAT] = FLOAT
        self.operation_results['/='][FLOAT][INTEGER] = FLOAT
        self.operation_results['/='][VECTOR][INTEGER] = VECTOR
        self.operation_results['/='][VECTOR][FLOAT] = VECTOR


        self.operation_results['.+'][VECTOR][VECTOR] = VECTOR
        self.operation_results['.+'][MATRIX][MATRIX] = MATRIX
        self.operation_results['.+'][MATRIX][INTEGER] = MATRIX
        self.operation_results['.+'][VECTOR][INTEGER] = VECTOR
        self.operation_results['.+'][INTEGER][MATRIX] = MATRIX
        self.operation_results['.+'][INTEGER][VECTOR] = VECTOR
        self.operation_results['.+'][MATRIX][FLOAT] = MATRIX
        self.operation_results['.+'][VECTOR][FLOAT] = VECTOR
        self.operation_results['.+'][FLOAT][MATRIX] = MATRIX
        self.operation_results['.+'][FLOAT][VECTOR] = VECTOR


        self.operation_results['.-'][VECTOR][VECTOR] = VECTOR
        self.operation_results['.-'][MATRIX][MATRIX] = MATRIX
        self.operation_results['.-'][MATRIX][INTEGER] = MATRIX
        self.operation_results['.-'][VECTOR][INTEGER] = VECTOR
        self.operation_results['.-'][INTEGER][MATRIX] = MATRIX
        self.operation_results['.-'][INTEGER][VECTOR] = VECTOR
        self.operation_results['.-'][MATRIX][FLOAT] = MATRIX
        self.operation_results['.-'][VECTOR][FLOAT] = VECTOR
        self.operation_results['.-'][FLOAT][MATRIX] = MATRIX
        self.operation_results['.-'][FLOAT][VECTOR] = VECTOR

        self.operation_results['.*'][VECTOR][VECTOR] = VECTOR
        self.operation_results['.*'][MATRIX][MATRIX] = MATRIX
        self.operation_results['.*'][MATRIX][INTEGER] = MATRIX
        self.operation_results['.*'][VECTOR][INTEGER] = VECTOR
        self.operation_results['.*'][INTEGER][MATRIX] = MATRIX
        self.operation_results['.*'][INTEGER][VECTOR] = VECTOR
        self.operation_results['.*'][MATRIX][FLOAT] = MATRIX
        self.operation_results['.*'][VECTOR][FLOAT] = VECTOR
        self.operation_results['.*'][FLOAT][MATRIX] = MATRIX
        self.operation_results['.*'][FLOAT][VECTOR] = VECTOR

        self.operation_results['./'][VECTOR][VECTOR] = VECTOR
        self.operation_results['./'][MATRIX][MATRIX] = MATRIX
        self.operation_results['./'][MATRIX][INTEGER] = MATRIX
        self.operation_results['./'][VECTOR][INTEGER] = VECTOR
        self.operation_results['./'][INTEGER][MATRIX] = MATRIX
        self.operation_results['./'][INTEGER][VECTOR] = VECTOR
        self.operation_results['./'][MATRIX][FLOAT] = MATRIX
        self.operation_results['./'][VECTOR][FLOAT] = VECTOR
        self.operation_results['./'][FLOAT][MATRIX] = MATRIX
        self.operation_results['./'][FLOAT][VECTOR] = VECTOR

        self.operation_results['=='][INTEGER][INTEGER] = BOOL
        self.operation_results['=='][FLOAT][INTEGER] = BOOL
        self.operation_results['=='][INTEGER][FLOAT] = BOOL
        self.operation_results['=='][FLOAT][FLOAT] = BOOL
        self.operation_results['=='][STRING][STRING] = BOOL
        self.operation_results['=='][VECTOR][VECTOR] = BOOL
        self.operation_results['=='][MATRIX][MATRIX] = BOOL

        self.operation_results['!='][INTEGER][INTEGER] = BOOL
        self.operation_results['!='][FLOAT][INTEGER] = BOOL
        self.operation_results['!='][INTEGER][FLOAT] = BOOL
        self.operation_results['!='][FLOAT][FLOAT] = BOOL
        self.operation_results['!='][STRING][STRING] = BOOL
        self.operation_results['!='][VECTOR][VECTOR] = BOOL
        self.operation_results['!='][MATRIX][MATRIX] = BOOL

        self.operation_results['<'][INTEGER][INTEGER] = BOOL
        self.operation_results['<'][FLOAT][INTEGER] = BOOL
        self.operation_results['<'][INTEGER][FLOAT] = BOOL
        self.operation_results['<'][FLOAT][FLOAT] = BOOL
        self.operation_results['<'][STRING][STRING] = BOOL
        self.operation_results['<'][VECTOR][VECTOR] = BOOL
        self.operation_results['<'][MATRIX][MATRIX] = BOOL

        self.operation_results['>'][INTEGER][INTEGER] = BOOL
        self.operation_results['>'][FLOAT][INTEGER] = BOOL
        self.operation_results[''][INTEGER][FLOAT] = BOOL
        self.operation_results['>'][FLOAT][FLOAT] = BOOL
        self.operation_results['>'][STRING][STRING] = BOOL
        self.operation_results['>'][VECTOR][VECTOR] = BOOL
        self.operation_results['>'][MATRIX][MATRIX] = BOOL

        self.operation_results['<='][INTEGER][INTEGER] = BOOL
        self.operation_results['<='][FLOAT][INTEGER] = BOOL
        self.operation_results['<='][INTEGER][FLOAT] = BOOL
        self.operation_results['<='][FLOAT][FLOAT] = BOOL
        self.operation_results['<='][STRING][STRING] = BOOL
        self.operation_results['<='][VECTOR][VECTOR] = BOOL
        self.operation_results['<='][MATRIX][MATRIX] = BOOL

        self.operation_results['>='][INTEGER][INTEGER] = BOOL
        self.operation_results['>='][FLOAT][INTEGER] = BOOL
        self.operation_results['>='][INTEGER][FLOAT] = BOOL
        self.operation_results['>='][FLOAT][FLOAT] = BOOL
        self.operation_results['>='][STRING][STRING] = BOOL
        self.operation_results['>='][VECTOR][VECTOR] = BOOL
        self.operation_results['>='][MATRIX][MATRIX] = BOOL

    # def are_types_correct(self, left_type, operator, right_type):




    def handle_error(self, message):
        self.error = True
        print(message)

    def visit_Program(self, node, table):
        symbolTable = SymbolTable(None, 'global')
        self.visit(node.instructions, symbolTable)

    def visit_Instructions(self, node, table):
        for instruction in node.instructions:
            self.visit(instruction, table)

    def visit_IntNum(self, node, table):
        return INTEGER

    def visit_FloatNum(self, node, table):
        return FLOAT

    def visit_Variable(self, node, table):
        symbol = table.get(node.name)
        if symbol:
            return symbol.type
        else:
            print('Line {}: {} is used but not declared'.format(node.line, node.name))
            return INVALID

    def visit_Reference(self, node, table):
        symbol = table.get(node.variable.name)
        types = [self.visit(i, table) for i in node.indexes]

        for t in types:
            if t != INVALID and t != INTEGER:
                print('Line {}: index is not integer'.format(node.line))
                return INVALID

        if not symbol:
            print('Line {}: {} is used but not declared'.format(node.line, node.variable.name))
            return INVALID
        if symbol.type == INVALID:
            return INVALID
        elif symbol.type != VECTOR and symbol.type != MATRIX:
            self.handle_error('Line {}: Reference to: {}'.format(node.line, symbol.type))
            return INVALID

        if len(node.indexes) > 2:
            print('Line {}: {} reference to too many indexes in {}'.format(node.line, node.variable.name, symbol.type))
        if symbol.type == 'vector' and len(node.indexes) != 1:
            print('Line {}: {} reference to too many indexes in {}'.format(node.line, node.variable.name, symbol.type))

        return 'float'

    def get_matrix_vector_dims(self, matrix, table):
        dims = None
        matrix = copy.deepcopy(matrix)
        if isinstance(matrix, Variable):
            matrix = table.get(matrix.name)
            if not matrix:
                return None
            matrix = matrix.value
        if isinstance(matrix, Vector):
            dims = matrix.int_dims()
        elif isinstance(matrix, MatrixInit) and matrix.int_dims():
            dims = matrix.int_dims()
        return dims

    def visit_BinExpr(self, node, table):
        left_type = self.visit(node.left, table)
        right_type = self.visit(node.right, table)
        op = node.op
        if left_type == INVALID or right_type == INVALID:
            return INVALID
        result = self.operation_results[op][left_type][right_type]
        if result == INVALID:
            print(
                'Line {}: Operation {} unsupported between types: {} and {}'.format(node.line, op, left_type,
                                                                                    right_type))
            return INVALID

        # check if opreration can be performed on matrixes (vectors) with this dimensions
        if (right_type == 'matrix' or right_type == 'vector') and (left_type == 'matrix' or left_type == 'vector'):
            right = node.right
            left = node.left
            right_dims = self.get_matrix_vector_dims(right, table)
            left_dims = self.get_matrix_vector_dims(left, table)
            if op in ('.+', '.-', '.*', './'):
                if right_dims and left_dims:
                    if right_dims != left_dims:
                        print(
                            'Line {}: Elementwise operation between arrays of wrong dimensions: {} and {}'.format(
                                node.line, left_dims, right_dims))
                        return INVALID
            if op == '*':
                if right_dims and left_dims:
                    if left_dims[1] != right_dims[0]:
                        print(
                            'Line {}: Multiplication between matrixes of wrong dimensions: {} and {}'.format(node.line,
                                                                                                             left_dims,
                                                                                                             right_dims))
                        return INVALID

        return result

    def visit_Assignment(self, node, table):
        right_type = self.visit(node.right, table)
        if isinstance(node.left, Ref):
            self.visit(node.left, table)
            if right_type != INTEGER and right_type != FLOAT and right_type != INVALID:
                print('Line {}: Assignment to reference {} of wrong type: {}'.format(node.line, node.left.variable.name,
                                                                                   right_type))
        elif isinstance(node.left, Variable):
            table.put(node.left.name, VariableSymbol(node.left.name, right_type, node.right))

    def visit_AssignmentAndExpr(self, node, table):
        left_type = self.visit(node.left, table)
        right_type = self.visit(node.right, table)
        result = self.operation_results[node.op][node.left][node.right]
        if result == INVALID:
            print('Line {}: Operation {} unsupported between types: {} and {}'.format(node.line, op, left_type,
                                                                                    right_type))
            return

        left_dims = self.get_matrix_vector_dims(node.left, table)
        right_dims = self.get_matrix_vector_dims(node.right, table)
        if node.op == '*=':
            if (left_type == MATRIX and (right_type == MATRIX or right_type == VECTOR)):
                if right_dims and left_dims:
                    if left_dims[1] != right_dims[0]:
                        print(
                            'Line {}: Multiplication between matrixes of wrong dimensions: {} and {}'.format(node.line,
                                                                                                             left_dims,
                                                                                                             right_dims))
                        return

    def visit_RelExpr(self, node, table):
        left_type = self.visit(node.left, table)
        right_type = self.visit(node.right, table)
        op = node.op
        if left_type == INVALID or right_type == INVALID:
            return BOOL
        result = self.operation_results[op][left_type][right_type]
        if result == INVALID:
            print('Line {}: Operation {} unsupported between types: {} and {}'.format(node.line, op, left_type,
                                                                                    right_type))
        return BOOL

    # def visit_UnaryExpr(self, node, table):
    #     unary_type = self.visit(node.arg, table)
    #     op = node.op
    #     if unary_type == INVALID:
    #         return INVALID
    #     return_type = self.unary_types[op][unary_type]
    #     if return_type == INVALID:
    #         self.handle_error('Line {}: Operation {} unsupported on type {}'.format(node.line, op, unary_type))
    #     return return_type

    def visit_For(self, node, table):
        symbol_table = SymbolTable(table, 'for')
        range_type = self.visit(node.range1, table)
        symbol_table.put(node.variable.name, VariableSymbol(node.variable.name, range_type, None))
        self.visit(node.instruction, symbol_table)

    def visit_While(self, node, table):
        symbol_table = SymbolTable(table, 'while')
        self.visit(node.condition, table)
        self.visit(node.instruction, symbol_table)

    def visit_If(self, node, table):
        self.visit(node.condition, table)
        self.visit(node.instruction, table)
        if node.else_instruction:
            self.visit(node.else_instruction, table)

    def visit_Range(self, node, table):
        start_type = self.visit(node.start, table)
        end_type = self.visit(node.end, table)
        if start_type == INVALID or end_type == INVALID:
            return INVALID
        elif start_type != INTEGER or end_type != INTEGER:
            print('Line {}: Range of unsupported types: {} and {}'.format(node.line, start_type, end_type))
            return INVALID
        return INTEGER

    def visit_Return(self, node, table):
        if node.value:
            return self.visit(node.value, table)

    def visit_Print(self, node, table):
        for expr in node.expressions:
            self.visit(expr, table)

    def visit_Continue(self, node, table):
        if table.name == 'global':
            print('Line {}: Continue from global scope'.format(node.line))

    def visit_Break(self, node, table):
        if table.name == 'global':
            print('Line {}: Break from global scope'.format(node.line))

    def visit_Vector(self, node, table):
        coor_types = [self.visit(t, table) for t in node.coordinates]
        if self.operation_results.__contains__(INVALID):
            return INVALID
        if not len(set(coor_types)) == 1:
            self.handle_error('Line {}: Vector initialization with different types'.format(node.line))
            return INVALID
        coor_type = coor_types[0]
        if coor_type == INTEGER or coor_type == FLOAT:
            return VECTOR
        if coor_type == VECTOR:
            if not len(set([len(l.coordinates) for l in node.coordinates])) == 1:
                self.handle_error('Line {}: Matrix initialization with vectors of different sizes'.format(node.line))
                return INVALID
            return MATRIX
        self.handle_error('Line {}: Vector initialization with illegal type: {}'.format(node.line, coor_type))
        return INVALID

    def visit_MatrixInit(self, node, table):
        dim1_type = self.visit(node.dim_1, table)
        dim2_type = dim1_type
        if node.dim_2:
            dim2_type = self.visit(node.dim_2, table)
        if dim1_type != INTEGER:
            print('Line {}: Matrix initialization with illegal dims type: {}'.format(node.line, dim1_type))
            return INVALID
        if dim2_type != INTEGER:
            print('Line {}: Matrix initialization with illegal dims type: {}'.format(node.line, dim2_type))
            return INVALID
        return MATRIX

    def visit_Eye(self, node, table):
        return self.visit_MatrixInit(node, table)

    def visit_Zeros(self, node, table):
        return self.visit_MatrixInit(node, table)

    def visit_Ones(self, node, table):
        return self.visit_MatrixInit(node, table)

    def visit_String(self, node, table):
        return STRING



