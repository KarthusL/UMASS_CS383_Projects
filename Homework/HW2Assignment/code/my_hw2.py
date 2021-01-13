from sudoku import Sudoku
from collections import OrderedDict
import time


class CSP_Solver(object):
    """
    This class is used to solve the CSP with backtracking.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        num_guesses = [0]
        return self.solve_helper(num_guesses)




    def solve_helper(self, num):
        csp_H = CSP_Helper(self.sudoku)
        if self.sudoku.complete():
            return self.sudoku.board, num[0]
        empty_node = csp_H.get_state(self.sudoku.board)
        domain = csp_H.get_domain(self.sudoku.board, empty_node)
        for value in domain:
            num[0] += 1
            if value in domain:
                self.sudoku.board[empty_node[1]][empty_node[0]] = value
                if self.solve_helper(num) is not None:
                    return self.solve_helper(num)
                self.sudoku.board[empty_node[1]][empty_node[0]] = 0
        return None

class CSP_Solver_MRV(object):
    """
    This class is used to solve the CSP with backtracking and the MRV
    heuristic.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver_mrv = CSP_Solver_MRV('puz-001.txt')
    ### solved_board, num_guesses = csp_solver_mrv.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        num_guesses = [0]
        return self.solve_helper(num_guesses)

    def solve_helper(self, num):
        csp_H = CSP_Helper(self.sudoku)
        if self.sudoku.complete():
            return self.sudoku.board, num[0]
        empty_node = csp_H.get_sorted_state(self.sudoku.board)
        domain = csp_H.get_domain(self.sudoku.board, empty_node)
        for value in domain:
            num[0] += 1
            if value in domain:
                self.sudoku.board[empty_node[1]][empty_node[0]] = value
                if self.solve_helper(num) is not None:
                    return self.solve_helper(num)
                self.sudoku.board[empty_node[1]][empty_node[0]] = 0
        return None


class CSP_Helper(Sudoku):


    def __init__(self, puzzle):
        self.puzzle = puzzle

    '''return the empty node in the board, e.g.(4,1)'''
    def get_state(self, puzzle):
        flattened = [val for num in puzzle for val in num]
        for number in flattened:
            if number == 0:
                position = flattened.index(number)
                return [position % 9, position // 9]
                break

    '''returns the domain of the current state that is being checked, e.g.[4, 6]'''
    def get_domain(self, puzzle, coordinator):
        x,y = coordinator
        square = []
        for col in range((x // 3) * 3, (x // 3) * 3 + 3):
            for row in range((y // 3) * 3, (y // 3) * 3 + 3):
                square.append(puzzle[row][col])
        flattened = list(set(puzzle[y] + [ele[x] for ele in puzzle] + square).symmetric_difference(set(range(0, 10))))
        return flattened


    def complete(self, puzzle):
        """
        Tests whether all the tiles in the board are filled in.
        Returns true if the board is filled. False, otherwise.
        """
        return all([all(row) for row in puzzle])

    '''returns all the empty nodes in the board'''
    def get_all_state(self, puzzle):
        all_state = []
        flattened = [val for num in puzzle for val in num]
        for number in flattened:
            if number == 0:
                position = flattened.index(number)
                all_state.append([position % 9, position // 9])
                flattened[position] = "*"
        return all_state

    '''only returns the most constrained empty node in the board accordingly'''
    def get_sorted_state(self, puzzle):
        dict = {}
        final_state = []
        x = CSP_Helper.get_all_state(self, puzzle)
        for coordinate in x:
            dict.update({tuple(coordinate): CSP_Helper.get_domain(self, puzzle, coordinate)})
        ordered_dict = sorted(dict.items(), key = lambda x: len(x[1]))
        for x in ordered_dict:
            final_state = x[0]
            break
        return final_state



if __name__ == '__main__':
    csp_solver = CSP_Solver('puz-001.txt')
    start_time = time.time()
    solved_board, num_guesses = csp_solver.solve()
    print("CSP time", time.time() - start_time)
    for i in solved_board:
        print(i)
    print(num_guesses)
    csp_solver_mrv = CSP_Solver_MRV('puz-001.txt')
    start_time = time.time()
    solved_board, num_guesses = csp_solver_mrv.solve()
    print("MRV time", time.time() - start_time)
    print(num_guesses)
    for i in solved_board:
        print(i)
