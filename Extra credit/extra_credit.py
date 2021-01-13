
class Search():

    def __init__(self, graph):
        self.graph = graph

    def bfs(self, graph, start):
        visited, queue = set(), [start]
        route = []
        cost = 0
        while queue:
            vertex = queue.pop(0)
            route.append(vertex)
            if vertex == 'G2' or vertex == 'G1':
                return route, cost
            if vertex not in visited:
                visited.add(vertex)
                temp = graph[vertex]
                print(temp)
                queue.extend(temp.difference(visited))
        return

    def dfs(self, graph, start):
        visited, stack = set(), [start]
        route = []
        cost = 0
        index = 0
        while stack:
            vertex = stack.pop()
            if vertex == 'G2' or vertex == 'G1':
                route.append(vertex)
                for elem in route:
                    index = route.index(elem)
                    keys = graph[elem][0].keys()
                    if route[index] in (graph[elem][0].keys()):
                        cost += graph[elem][0][route[route.index(elem)]]
                return route, cost
            route.append(vertex)
            if vertex not in visited :
                visited.add(vertex)
                temp = graph[vertex][0].keys()
                print("temp", temp)
                stack.extend(set(temp).difference(visited))
        return


if __name__ == '__main__':

    graph = {'S': ({'A': 3, 'B': 7}, 10),
             'A': ({'C': 1, 'D': 6}, 5),
             'B': ({'E': 1, 'G2': 9}, 5),
             'C': ({'S': 2, 'D': 4}, 3),
             'D': ({'G1': 6}, 2),
             'E': ({'G2': 5}, 4),
             'G1': ({'C': 2}, 0),
             'G2': ({'B': 8}, 0)}
    search = Search(graph)
    #route1 = search.bfs(graph, ('S'))
    route2 = search.dfs(graph, ('S'))
    #print("BFS", route1)
    print("DFS", route2)