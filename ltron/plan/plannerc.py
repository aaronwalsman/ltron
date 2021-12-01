import time
from bisect import insort

from ltron.exceptions import LtronException

class PlanningException(LtronException):
    pass

class PathNotFoundError(PlanningException):
    pass

class RoadMap:
    def __init__(self, env):
        self.env = env
        self.nodes = set()
        self.edges = {}
        
    def plan(self, start, goal, max_cost=float('inf'), timeout=float('inf')):
        t_start = time.time()
        self.nodes.add(start)
        self.nodes.add(goal)
        while True:
            t_loop = time.time()
            if t_loop - t_start >= timeout:
                raise PathNotFoundError
            
            try:
                path, cost = self.graph_search(
                    start, goal, max_cost=max_cost)
                expanded_path = []
                expanded_cost = 0
                for a, b in zip(path[:-1], path[1:]):
                    try:
                        expanded_nodes, edge_cost = self.check_edge(a, b)
                        expanded_path.extend(expanded_nodes)
                        expanded_cost += edge_cost
                        if expanded_cost > max_cost:
                            raise PathNotFoundError
                    
                    except InvalidEdgeError:
                        raise PathNotFoundError
                
                return fine_path, cost
            
            except PathNotFoundError:
                new_nodes = self.expand_graph()
                if not new_nodes:
                    raise PathNotFoundError
    
    def graph_search(self, start, goal, max_cost):
        # A* search through the current graph
        precursors = {}
        frontier = [(0, None, start)]
        while frontier:
            expand_cost, precursor_node, expand_node = frontier.pop()
            if expand_node in precursors:
                continue
            precursors[expand_node] = precursor_node
            
            if expand_node == goal:
                path = []
                path_node = expand_node
                while path_node != start:
                    path.append(path_node)
                    path_node = precursors[path_node]
                path.append(start)
                return reversed(path), expand_cost
            
            neighbor_nodes = self.edges[expand_node]
            for neighbor_node in neighbor_nodes:
                neighbor_cost = (
                    expand_cost +
                    self.edge_cost(expand_node, neighbor_node) +
                    self.heuristic_cost(neighbor_node, goal)
                )
                if neighbor_cost < max_cost:
                    frontier.insort((neighbor_cost, expand_node, neighbor_node))
        
        raise PathNotFoundError
    
    def edge_cost(self, previous, current):
        raise NotImplementedError
    
    def heuristic_cost(self, current, goal):
        raise NotImplementedError
    
    def expand_graph(self):
        raise NotImplementedError
    
    def get_neighbors(self):
        raise NotImplementedError
    
    def check_edge(self, edge):
        raise NotImplementedError
