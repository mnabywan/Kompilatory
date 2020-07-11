import scanner
import ply.yacc as yacc
import AST

tokens = scanner.tokens


precedence = (
    ("nonassoc", 'IFX'),
    ("nonassoc", 'ELSE'),
    ("nonassoc", '>', 'GE', '<', 'LE', 'EQ', 'NE', ),
    ("nonassoc", 'ADDASSIGN', 'SUBASSIGN', 'DIVASSIGN', 'MULASSIGN'),
    ("right", '='),
    ("left", '+', '-'),
    ("left", '*', '/'),
    ("left", 'DOTADD', 'DOTSUB'),
    ("left", 'DOTMUL', 'DOTDIV'),
    ("left", '\''),
)


def p_error(p):
    if p:
        print("Syntax error at line {0}, column {1} : LexToken({2}, '{3}')".format(p.lineno,
                                                                                   scanner.find_column(p),
                                                                                  p.type, p.value))
    else:
        print("Unexpected end of input")




def p_program(p):
    """program : statements"""
    p[0] = AST.Program(p[1])
    #print(p[0])


def p_statements(p):
    """statements : statements statement
                    | statement"""
    if len(p) > 2:
        p[1].statements.append(p[2])
        p[0] = p[1]
    else:
        p[0] = AST.Statements(p[1])


def p_statement(p):
    """statement : assignment
            | if_else_statement
            | while_statement
            | for_statement
            | print_statement
            | break_statement
            | continue_statement
            | return_statement
            | curly_brackets_statement
            """
    p[0] = p[1]


def p_assignment(p):
    """assignment : variable assignment_op arithmetic_expression ';'
                    | ref_var assignment_op arithmetic_expression ';' """
    p[0] = AST.Assignment(p[1], p[2], p[3])


def p_assignment_op(p):
    """assignment_op : '='
                    | ADDASSIGN
                    | SUBASSIGN
                    | MULASSIGN
                    | DIVASSIGN
                    """
    p[0] = AST.Assignment_Op(p[1])


def p_if_else_statement(p):
    """if_else_statement : IF '(' relational_expression ')'  statement  %prec IFX
                        | IF '(' relational_expression ')' statement  ELSE  statement
                        | IF '(' error ')' statement %prec IFX
                        | IF '(' error ')' statement ELSE statement """
    if len(p) > 6:
        p[0] = AST.If_statement(p[3], p[5], p[7])
    else:
        p[0] = AST.If_statement(p[3], p[5])


def p_while_statement(p):
    """while_statement : WHILE '(' relational_expression ')' statement
                        | WHILE '(' error ')' statement"""

    p[0] = AST.While(p[3], p[5])


def p_for_statement(p):
    """for_statement : FOR variable '='  range_expression statement """
    p[0] = AST.For(p[2], p[4], p[5])


def p_range_expression(p):
    """range_expression : arithmetic_expression ':' arithmetic_expression """
    p[0] = AST.Range(p[1], p[3])


def p_print_statement(p):
    """print_statement : PRINT expr_list ';'
                        | PRINT error ';' """

    p[0] = AST.Print(p[2])


def p_break_statement(p):
    """break_statement : BREAK ';'"""
    p[0] = AST.Break()


def p_return_statement(p):
    """return_statement : RETURN arithmetic_expression ';'"""
    p[0] = AST.Return(p[2])


def p_continue_statement(p):
    """continue_statement : CONTINUE ';' """
    p[0] = AST.Continue()


def p_curly_brackets_statement(p):
    """ curly_brackets_statement : '{' statements '}' """
    p[0] = AST.CurlyBracketsStatement(p[2])


def p_const(p):
    """const : STRING
            | FLOATNUM
            | INTNUM """
    p[0] = AST.Const(p[1])


def p_matrix(p):
    """ matrix : variable
                | minus_matrix
                | matrix_transposed """
    p[0] = AST.Matrix(p[1])


def p_minus_matrix(p):
    """ minus_matrix : '-' matrix"""
    p[0] = AST.MinusMatrix(p[2])


def p_matrix_transposed(p):
    """ matrix_transposed : matrix "'" """
    p[0] = AST.MatrixTransposed(p[1])


def p_variable(p):
    """variable : ID """
    p[0] = AST.Variable(p[1])


def p_ref_var(p):
    """ref_var : variable '[' expr_list ']' """
    p[0] = AST.RefVar(p[1], p[3])


def p_expr_list(p):
    """expr_list : expr_list ',' arithmetic_expression
                     | arithmetic_expression  """

    if len(p) == 4:
        p[1].expressions.append(p[3])
        p[0] = p[1]
    else:
        p[0] = AST.Expressions(p[1])


def p_relational_expression(p):
    """relational_expression : arithmetic_expression '>' arithmetic_expression
                 | arithmetic_expression GE arithmetic_expression
                 | arithmetic_expression '<' arithmetic_expression
                 | arithmetic_expression LE arithmetic_expression
                 | arithmetic_expression EQ arithmetic_expression
                 | arithmetic_expression NE arithmetic_expression
                 | '(' relational_expression ')'
                 | arithmetic_expression"""

    if len(p) == 2:
        p[0] = p[1]
    elif p[1] == '(':
        p[0] = p[2]
    else:
        p[0] = AST.RelExpr(p[1], p[2], p[3])


def p_arithmetic_expression(p):
    """arithmetic_expression : const
                        | vector
                        | variable
                        | matrix
                        | minus_matrix
                        | matrix_function
                        | matrix_operation
                        | arithmetic_expression '+' arithmetic_expression
                       | arithmetic_expression '-' arithmetic_expression
                       | arithmetic_expression '*' arithmetic_expression
                       | arithmetic_expression '/' arithmetic_expression
                       | '(' arithmetic_expression ')'
                       """
    if p[1] == '(':
        p[0] = p[2]
    if len(p) > 2 and p[2] in ['+', '-', '/', '*']:
        p[0] = AST.BinExpr(p[1], p[2], p[3])
    elif len(p) == 2:
        p[0] = p[1]


def p_vector_more_dimensions(p):
    """ vector : '[' vector_row ']' """
    p[0] = AST.Vector(p[2])


def p_vector_row(p):
    """ vector_row : vector_row ';' expr_list
                    | expr_list """

    if len(p) == 4:
        p[1].vector_row1.append(p[3])
        p[0] = p[1]
    else:
        p[0] = AST.VectorRow(p[1])



def p_matrix_function(p):
    """ matrix_function : ZEROS '(' const ',' const ')'
                        | ONES '(' const ',' const ')'
                        | EYE '(' const ',' const ')'
                        | ZEROS '(' const ')'
                        | ONES '(' const ')'
                        | EYE '(' const ')'
                        """
    if len(p) == 5:
        p[0] = AST.MatrixFunction(p[1], p[3])
    elif len(p) == 7:
        p[0] = AST.MatrixFunction(p[1], p[3], p[5])

def p_matrix_operation(p):
    """ matrix_operation : matrix DOTADD matrix
                           | matrix DOTSUB matrix
                           | matrix DOTMUL matrix
                           | matrix DOTDIV matrix """
    p[0] = AST.MatrixOperation(p[1], p[2], p[3])


parser = yacc.yacc()
