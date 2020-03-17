import ply.lex as lex
import sys

literals = ['+', '-', '*', '/', '(', ')', '=', '<', '>', '[', ']', '{', '}',  ',', ';', ':', '\'']


tokens = ( 'FLOATNUM', 'INTNUM',
           'LE', 'GE', 'NE', 'EQ',
            'DOTADD', 'DOTSUB', 'DOTMUL', 'DOTDIV',
            'ADDASSIGN', 'SUBASSIGN', 'MULASSIGN', 'DIVASSIGN',
           'COMMENT', 'STRING', 'ID',
          )

t_LE = r'\<='
t_GE = r'\>='
t_NE = r'\!='
t_EQ = r'=='

#matrix operators
t_DOTADD = r'\.\+'
t_DOTSUB = r'\.\-'
t_DOTMUL = r'\.\*'
t_DOTDIV = r'\.\/'

#operations with assign
t_ADDASSIGN = r'\+='
t_SUBASSIGN = r'\-='
t_MULASSIGN = r'\*='
t_DIVASSIGN = r'\/='

reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'for': 'FOR',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'return': 'RETURN',
    'eye': 'EYE',
    'zeros': 'ZEROS',
    'ones': 'ONES',
    'print': 'PRINT'
}

tokens = list(tokens) + list(reserved.values())


def t_FLOATNUM(t):
    r'((\d+\.\d*)|(\.\d+))(E|e)[+|-]?\d+|((\d+\.\d*)|(\.\d+))|(\d+)(E|e)[+|-]?(\d+)'
    t.value = float(t.value)
    return t


def t_INTNUM(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t


def t_COMMENT(t):
    r'\#.*'
    pass


def t_STRING(t):
    r'\".+\"'
    t.value = str(t.value)
    return t


t_ignore = '  \t'


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_error(t):
    print("Illegal character '{}' in line {}".format(t.value[0], t.lineno))
    t.lexer.skip(1)


def find_column(input, token):
    line_start = input.rfind('\n', 0, token.lexpos) + 1
    return (token.lexpos - line_start) + 1

#print(tokens)

lexer = lex.lex()
