import matplotlib.pyplot as pyplot
import numpy as np

# Set board 
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
    safe = 1    
    for num in diagonal_sums(board):
        if num > 1:
            ret = False
    return ret

def is_board_safe(board):
    return safe_diagonal(board) and safe_horizontal(board) 

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
    max_iterations = 100
    curr_score = 1000000
    best_state = None
    best_score = curr_score
    for x in range(max_iterations):
        for i in range(len(board)):
            for j in range(len(board[0])):
                new_board_state = board
                if board[i][j] == 1:
                    # if i + 1 <= len(board) - 1 and is_curr_horizontal_safe(i + 1, new_board_state) and is_curr_diagonal_safe(i + 1, new_board_state):
                    #     move_queen(i,j,i + 1,j, new_board_state)
                    # elif i - 1 >= 0 and is_curr_horizontal_safe(i - 1, new_board_state) and is_curr_diagonal_safe(i + 1, new_board_state):
                    #     move_queen(i,j,i - 1,j, new_board_state)
                    move_queen(i,j,np.random.randint(0, 25),j, board)
                if number_of_conflicting_queens(new_board_state) < curr_score:
                    curr_score = number_of_conflicting_queens(new_board_state)
                    board = new_board_state
        if curr_score < best_score:
            best_score = curr_score
            best_state = board
    return  best_state


            

def move_queen(old_row, old_col, new_row, new_col, board):
    board[old_row][old_col] = 0
    board[new_row][new_col] = 1




# print(number_of_conflicting_queens(master_board))
# solve(master_board)
# print(master_board)
# print(number_of_conflicting_queens(master_board))
# print(is_board_safe(master_board))

print(master_board)
print(number_of_conflicting_queens(master_board))
print(solve(master_board))
print(number_of_conflicting_queens(master_board))