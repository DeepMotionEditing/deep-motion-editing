import numpy as np
import heapq

class AStar:
    
    def __init__(self,
        neighbor_func, 
        dist_func='euclidian', 
        heuristic_func='euclidian',
        bias=0.0, silent=True):
        
        self.neighbor_func = neighbor_func
        self.heuristic_func = heuristic_func
        self.dist_func = dist_func
        
        if heuristic_func == 'euclidian':
            self.heuristic_func = lambda x, y: np.sqrt(np.sum((x-y)**2))
        
        if dist_func == 'euclidian':
            self.dist_func = lambda x, y: np.sqrt(np.sum((x-y)**2))
        
        self.bias = bias
        self.silent = silent
    
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path
    
    def __call__(self, current, goal):
        
        neighbor_func = self.neighbor_func
        heuristic_func = self.heuristic_func
        dist_func = self.dist_func
        bias = self.bias
        silent = self.silent

        closedset = set([])
        openset = set([current])
        openheap = [(0, current)]
        came_from = {}
        g_score = {current:0}
        
        i = 0
        while len(openset):
            
            current = heapq.heappop(openheap)[1]
            if current == goal:
                self.closedset = closedset
                return self.reconstruct_path(came_from, goal)
            
            if not silent and i % 100000 == 0:
                print('[AStar] current: ', current)
            
            openset.remove(current)
            closedset.add(current)
            
            for neighbor in neighbor_func(current):
                
                if neighbor in closedset: continue
                
                tentative_g_score = g_score[current] + dist_func(current, neighbor)
     
                if neighbor not in openset or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    if neighbor not in openset:
                        score = tentative_g_score + (1 + bias) * heuristic_func(neighbor, goal)
                        openset.add(neighbor)
                        heapq.heappush(openheap, (score, neighbor))
            
            i += 1
            
        raise Exception('Goal State Not Found')
