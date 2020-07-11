
import sys
import ply.yacc as yacc
# import Mparser
import parser_new
import scanner
from TreePrinter_new import TreePrinter
from TypeChecker_new import TypeChecker
from Interpreter import  Interpreter

if __name__ == '__main__':

    try:
        filename = sys.argv[1] if len(sys.argv) > 1 else "input/opers2.m"
        file = open(filename, "r")
    except IOError:
        print("Cannot open {0} file".format(filename))
        sys.exit(0)

    text = file.read()

    parser = parser_new.parser
    lexer = scanner.lexer
    ast = parser.parse(text, lexer=lexer)
    # ast.printTree()
    # Below code shows how to use visitor
    typeChecker = TypeChecker()

    typeChecker.visit(ast)   # or alternatively ast.accept(typeChecker)
    ast.accept(Interpreter())