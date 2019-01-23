import numpy as np
from copy import copy, deepcopy

def column(matrix, i):
    return [row[i] for row in matrix]

def block(matrix, i, j):
    return [row[i:(i+3)] for row in matrix[j:(j+3)]]

def possibilities(i,j,board):
    # to generate possibilities of each entry of the board
    begin = [1,2,3,4,5,6,7,8,9]
    for col in range(0,9):
        if board[i][col] in begin:
            begin.remove(board[i][col])
    for row in range(0,9):
        if board[row][j] in begin:
            begin.remove(board[row][j])
    row_index = int(i/3)*3
    col_index = int(j/3)*3
    for row in range(row_index, row_index+3):
        for col in range(col_index, col_index+3):
            if board[row][col] in begin:
                begin.remove(board[row][col])
    return begin

def generate_poss(board):
    out = [[[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], []]
           ]
    for i in range(0,9):
        for j in range(0,9):
            if board[i][j] == 0:
                out[i][j] = possibilities(i,j,board)
    return out

def find_hidden_single(poss, board):
    sol = []
    for i in range(0,9):
        # check row
        rows = sum(poss[i],[])
        cols = sum(column(poss, i),[])
        x = int(i/3)
        y = i//3
        blocks = sum(block(poss,x,y),[])
        for k in range(1,10):
            if rows.count(k) == 1:
                for j in range(0,9):
                    if k in poss[i][j] and [i,j,k] not in sol:
                        # this is the hidden single
                        sol.append([i,j,k])
            if cols.count(k) == 1:
                for j in range(0,9):
                    if k in poss[j][i] and [j,i,k] not in sol:
                        # this is the hidden single
                        sol.append([j,i,k])
            if blocks.count(k) == 1:
                for ix in range(x*3, x*3+3):
                    for iy in range(y*3, y*3+3):
                        if k in poss[ix][iy] and [ix,iy,k] not in sol:
                        # this is the hidden single
                            sol.append([ix,iy,k])
    return sol

def update_board(board, sol):
    for key in sol:
        board[key[0]][key[1]] = key[2]
    return board

def get_min_poss(poss):
    min_poss = [9,9,9]
    for i in range(0,9):
        for j in range(0,9):
            if len(poss[i][j]) < min_poss[2] and len(poss[i][j]) != 0:
                min_poss = [i,j,len(poss[i][j])]
    return min_poss

def check_complete(board):
    success = True
    for row in board:
        if 0 in row:
            success = False
    return success

def check_wrong_path(board):
    wrong_path = False
    poss = generate_poss(board)
    for i in range(0,9):
        for j in range(0,9):
            if board[i][j] == 0 and poss[i][j] == []:
                wrong_path = True
    return wrong_path

def solve(board):
    global result
    hidden_single = True
    while hidden_single:
        poss = generate_poss(board)
        sol = find_hidden_single(poss, board)
        if sol == []:
            #can't find hidden single any more
            hidden_single = False
        else:
            board = update_board(board, sol)
    if check_complete(board):
        result = board
    elif check_wrong_path(board):
        #do nothing
        pass
    else:
        poss = generate_poss(board)
        guess = get_min_poss(poss)
        for i in range(0, guess[2]):
            #print(str(i) + ' ' + str(guess[2]))
            temp = deepcopy(board)
            temp[guess[0]][guess[1]] = poss[guess[0]][guess[1]][i]
            solve(temp)

board = [[0,0,0,0,3,0,0,0,2],
         [0,0,0,9,0,0,8,3,0],
         [1,0,0,7,0,0,5,0,0],
         [0,8,0,0,0,4,0,0,0],
         [0,0,0,0,5,0,0,0,0],
         [4,7,0,0,0,0,3,0,6],
         [0,0,0,0,6,0,4,1,5],
         [0,0,9,5,0,1,0,6,0],
         [0,0,0,0,0,0,0,0,0]]

result = []
solve(board)
for i in result:
    for j in i:
        print(j, end = ' ')
    print()

