from eight_puzzle import Puzzle
import time, heapq


##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
class Node:
    """
    A class representing a node.
    - 'state' holds the state of the node.
    - 'parent' points to the node's parent.
    - 'action' is the action taken by the parent to produce this node.
    - 'path_cost' is the cost of the path from the root to this node.
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def gen_child(self, problem, action):
        """
        Returns the child node resulting from applying 'action' to this node.
        """
        return Node(state=problem.transitions(self.state, action),
                    parent=self,
                    action=action,
                    path_cost=self.path_cost + problem.step_cost(self.state, action))

    @property
    def state_hashed(self):
        """
        Produces a hashed representation of the node's state for easy
        lookup in a python 'set'.
        """
        return hash(str(self.state))

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def retrieve_solution(node, num_explored, num_generated):
    """
    Returns the list of actions and the list of states on the
    path to the given goal_state node. Also returns the number
    of nodes explored and generated.
    """
    actions = []
    states = []
    while node.parent is not None:
        actions += [node.action]
        states += [node.state]
        node = node.parent
    states += [node.state]
    return actions[::-1], states[::-1], num_explored, num_generated

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def print_solution(solution):
    """
    Prints out the path from the initial state to the goal given
    a tuple of (actions,states) corresponding to the solution.
    """
    actions, states, num_explored, num_generated = solution
    print('Start')
    for step in range(len(actions)):
        print(puzzle.board_str(states[step]))
        print()
        print(actions[step])
        print()
    print('Goal')
    print(puzzle.board_str(states[-1]))
    print()
    print('Number of steps: {:d}'.format(len(actions)))
    print('Nodes explored: {:d}'.format(num_explored))
    print('Nodes generated: {:d}'.format(num_generated))


################################################################
### Skeleton code for your Astar implementation. Fill in here.
################################################################
class Astar:

    """
    A* search.
    - 'problem' is a Puzzle instance.
    """

    def __init__(self, problem):
        self.problem = problem

    def solve(self):
        """
        Perform A* search and return a solution using `retrieve_solution'
        (if a solution exists).
        IMPORTANT: Use node generation time (i.e., time.time()) to split
        ties among nodes with equal f(n).
        """
        ################################################################
        ### Your code here.
        ################################################################
        num_explored = 0; num_generated = 0
        Node.state = self.problem
        Node.path_cost = 0
        frontier = []
        node = Node(self.problem.init_state, None, None, 0)
        heapq.heappush(frontier, (self.f(node), time.perf_counter(), node))
        explored = set()
        while frontier:
            if frontier.__sizeof__() == 0:
                return False
            num_explored += 1
            temp_node = heapq.heappop(frontier)[2]

            if self.problem.is_goal(temp_node.state):
                return retrieve_solution(temp_node, num_explored, num_generated)

            explored.add(temp_node)
            for action in list(Puzzle.actions(self.problem, temp_node.state)):
                child = temp_node.gen_child(self.problem, action)
                if child not in explored or child.state not in frontier:
                    num_generated += 1
                    heapq.heappush(frontier, (self.f(child), time.perf_counter(), child))
                elif child in frontier:
                    print("-------------replace--------------")
                    deleted = frontier[child]
                    del frontier[deleted]
                    frontier.append(child)

    def f(self,node):
        '''
        Returns a lower bound estimate on the cost from root through node
        to the goal.
        '''
        return node.path_cost + self.h(node)

    def h(self,node):
        '''
        Returns a lower bound estimate on the cost from node to the goal
        using the Manhattan distance heuristic.
        '''
        ################################################################
        ### Your code here.
        ################################################################
        standard_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        temp_state = node.state
        manhattan = 0
        for var in range(1,9):
            temp_x = var % 3
            temp_y = var // 3
            node_x = node.state.index(var) % 3
            node_y = node.state.index(var) // 3
            manhattan += (abs(temp_x - node_x) + abs(temp_y - node_y))

        return manhattan


    def branching_factor(self, board, trials=100):
        '''
        Returns an average upper bound for the effective branching factor.
        '''
        b_hi = 0  # average upper bound for branching factor
        for t in range(trials):
            puzzle = Puzzle(board).shuffle()
            solver = Astar(puzzle)
            actions, states, num_explored, num_generated = solver.solve()
            b_hi += num_generated**(1/actions.__sizeof__())
            ############################################################
            ### Compute upper bound for branching factor and update b_hi
            ### Your code here.
            ############################################################
        return b_hi/trials


if __name__ == '__main__':
    # Simple puzzle test
    print("simple")
    board = [[3,1,2],
             [4,0,5],
             [6,7,8]]

    '''print(help(Puzzle))'''


    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    print_solution(solution)
    print("simple finished")


    print("hard")
    # Harder puzzle test
    board = [[7,2,4],
             [5,0,6],
             [8,3,1]]

    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    print_solution(solution)
    print(len(solution[0]))
    print("hard finished")
    # branching factor test
    b_hi = solver.branching_factor(board, trials=100)
    print('Upper bound on effective branching factor: {:.2f}'.format(b_hi))
