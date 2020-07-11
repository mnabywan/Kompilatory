from __future__ import print_function
import ast_new as ast

def addToClass(cls):

    def decorator(func):
        setattr(cls,func.__name__,func)
        return func
    return decorator

class TreePrinter:

    @classmethod
    def printIndented(cls, string, level):
        print ("|  " * level + string)

    @addToClass(ast.Node)
    def printTree(self, indent=0):
        raise Exception("printTree not defined in class " + self.__class__.__name__)

    @addToClass(ast.Program)
    def printTree(self, indent=0):
        self.instructions.printTree(indent)

    @addToClass(ast.Instructions)
    def printTree(self, indent=0):
        for instruction in self.instructions:
            instruction.printTree(indent)

    @addToClass(ast.IntNum)
    def printTree(self, indent=0):
        TreePrinter.printIndented(str(self.value), indent)

    @addToClass(ast.FloatNum)
    def printTree(self, indent=0):
        TreePrinter.printIndented(str(self.value), indent)

    @addToClass(ast.Variable)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.name, indent)

    @addToClass(ast.Ref)
    def printTree(self, indent=0):
        TreePrinter.printIndented("REF", indent)
        self.variable.printTree(indent + 1)
        for index in self.indexes:
            index.printTree(indent + 1)

    @addToClass(ast.BinExpr)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.op, indent)
        self.left.printTree(indent + 1)
        self.right.printTree(indent + 1)

    @addToClass(ast.Assignment)
    def printTree(self, indent=0):
        TreePrinter.printIndented("=", indent)
        self.left.printTree(indent + 1)
        self.right.printTree(indent + 1)

    @addToClass(ast.AssignmentAndExpr)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.op, indent)
        self.left.printTree(indent + 1)
        self.right.printTree(indent + 1)

    @addToClass(ast.RelExpr)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.op, indent)
        self.left.printTree(indent + 1)
        self.right.printTree(indent + 1)

    @addToClass(ast.UnaryExpr)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.op, indent)
        self.arg.printTree(indent + 1)

    @addToClass(ast.For)
    def printTree(self, indent=0):
        TreePrinter.printIndented("FOR", indent)
        self.variable.printTree(indent + 1)
        self.range_.printTree(indent + 1)
        self.instruction.printTree(indent + 1)

    @addToClass(ast.While)
    def printTree(self, indent=0):
        TreePrinter.printIndented("WHILE", indent)
        self.condition.printTree(indent + 1)
        self.instruction.printTree(indent + 1)

    @addToClass(ast.If)
    def printTree(self, indent=0):
        TreePrinter.printIndented("IF", indent)
        self.condition.printTree(indent + 1)
        TreePrinter.printIndented("THEN", indent)
        self.instruction.printTree(indent + 1)
        if self.else_instruction:
            TreePrinter.printIndented("ELSE", indent)
            self.else_instruction.printTree(indent + 1)

    @addToClass(ast.Range)
    def printTree(self, indent=0):
        TreePrinter.printIndented("RANGE", indent)
        self.start.printTree(indent + 1)
        self.end.printTree(indent + 1)

    @addToClass(ast.Return)
    def printTree(self, indent=0):
        TreePrinter.printIndented("RETURN", indent)
        if self.value:
            self.value.printTree(indent + 1)

    @addToClass(ast.Print)
    def printTree(self, indent=0):
        TreePrinter.printIndented("PRINT", indent)
        for expression in self.expressions:
            expression.printTree(indent + 1)

    @addToClass(ast.Continue)
    def printTree(self, indent=0):
        TreePrinter.printIndented("CONTINUE", indent)

    @addToClass(ast.Break)
    def printTree(self, indent=0):
        TreePrinter.printIndented("BREAK", indent)

    @addToClass(ast.Vector)
    def printTree(self, indent=0):
        TreePrinter.printIndented("VECTOR", indent)
        for coordinate in self.coordinates:
            coordinate.printTree(indent + 1)

    @addToClass(ast.Eye)
    def printTree(self, indent=0):
        TreePrinter.printIndented("EYE", indent)
        self.dim_1.printTree(indent + 1)
        if self.dim_2:
            self.dim_2.printTree(indent + 1)

    @addToClass(ast.Zeros)
    def printTree(self, indent=0):
        TreePrinter.printIndented("ZEROS", indent)
        self.dim_1.printTree(indent + 1)
        if self.dim_2:
            self.dim_2.printTree(indent + 1)

    @addToClass(ast.Ones)
    def printTree(self, indent=0):
        TreePrinter.printIndented("ONES", indent)
        self.dim_1.printTree(indent + 1)
        if self.dim_2:
            self.dim_2.printTree(indent + 1)

    @addToClass(ast.String)
    def printTree(self, indent=0):
        TreePrinter.printIndented(self.value, indent)

    @addToClass(ast.Error)
    def printTree(self, indent=0):
        pass
        # fill in the body
