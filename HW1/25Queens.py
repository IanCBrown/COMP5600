import numpy as np

# Set up board 
master_board = np.zeros((25, 25), int)
np.fill_diagonal(master_board, 1)


def vertical_sums(board):
    return [sum(x) for x in zip(*board)]

def horizontal_sums(board):
    return [sum(row) for row in board]

def diagonal_sums(board):
    bottom = [sum(board.diagonal(i)) for i in range(len(board))]
    top = [sum(board.diagonal(i)) for i in range(len(board), 0, -1)]

    return bottom + top

def safe_horizontal(board):
    safe = 1
    ret = True 
    for s in horizontal_sums(board):
        if s > safe:
            ret = False
    return ret

def is_curr_horizontal_safe(row, board):
    return sum(board[row]) < 1

def is_curr_diagonal_safe(i, board):
    return (sum(board.diagonal(i)) < 1)

def safe_diagonal(board):
    ret = True 
    for num in diagonal_sums(board):
        if num > 1:
            ret = False
    return ret

def is_board_safe(board):
    return safe_diagonal(board) and safe_horizontal(board) 

def get_neighbors(col, row, board):
    return board[:,col]

def number_of_conflicting_queens(board):
    # score function
    # number of queens in check
    count = 0 
    for d in diagonal_sums(board):
        if d > 1:
            count += d - 1
    for h in horizontal_sums(board):
        if h > 1:
            count += h - 1
    return count

def solve(board):
    max_iterations = 250
    curr_score = 0
    best_state = None
    best_score = 25

    # start state is a local optima, randomize from start state to get out 
    for x in range(max_iterations):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 1 and not is_curr_horizontal_safe(i, board):
                    move_queen(i,j,np.random.randint(0, 25),j, board)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 1 and not is_curr_diagonal_safe(i,board) and not is_curr_horizontal_safe(i,board):
                    copy = board
                    for k in range(len(board)):      
                        if is_curr_diagonal_safe(k, board) and is_curr_horizontal_safe(k,board):
                            # if there is a conflict, move the queen in it's neighborhood (column)
                            # until it no longer causes conflict 
                            move_queen(i,j,k,j, copy)
                            break
                    if number_of_conflicting_queens(copy) < number_of_conflicting_queens(board):
                        board = copy
        
        curr_score = number_of_conflicting_queens(board)
        if curr_score < best_score:
            best_score = curr_score
            best_state = board
    
    return best_state
  

def move_queen(old_row, old_col, new_row, new_col, board):
    board[old_row][old_col] = 0
    board[new_row][new_col] = 1




print(master_board)
print(number_of_conflicting_queens(master_board))
print(solve(master_board))
print(number_of_conflicting_queens(master_board))