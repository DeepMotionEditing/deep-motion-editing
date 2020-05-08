import numpy as np
import scipy.spatial as sp_spatial

from AStar import AStar

class AStarTW:
    
    def __init__(self, X, Y, height=100.0, metric='euclidean', p=2, smoothing=0.5, maxdrift=100, silent=True):
        
        if not silent:
            print("[DTW] Calculating Distance Matrix", len(X), len(Y))
                
        cost = sp_spatial.distance.cdist(X, Y, metric=metric, p=p)
        
        im, jm = cost.shape[0], cost.shape[1]

        def neighbor_func(n):
            ns = []
            r = n[0] != im-1 and ((n[0]/im) - (n[1]/jm))*((im+jm)/2) < maxdrift
            d = n[1] != jm-1 and ((n[1]/jm) - (n[0]/im))*((im+jm)/2) < maxdrift
            if r: ns.append((n[0]+1, n[1]))
            if d: ns.append((n[0], n[1]+1))
            if r and d: ns.append((n[0]+1, n[1]+1))
            return ns
        
        def dist_func(n, m):
            return np.sqrt(np.sum((np.array(n)-np.array(m))**2.0)) + height * cost[m[0],m[1]]
        
        if not silent:
            print("[DTW] Performing Path Search", len(X))
        
        astar = AStar(neighbor_func, dist_func, dist_func, bias=0.0, silent=silent)
        
        path = np.array(astar((0,0), (im-1, jm-1))).astype(np.float)
        path[0] = ((1-smoothing) * path[0] + (smoothing) * path[1])
        path[1:-1] = (
           0.5 * (1-smoothing) * path[:-2] + 
           (smoothing) * path[1:-1] + 
           0.5 * (1-smoothing) * path[2:])
        path[-1] = ((1-smoothing) * path[-1] + (smoothing) * path[-2])
        
        for i in range(1, len(path)):
            if path[i,1] <= path[i-1,1]:
                path[i,1] = path[i-1,1] + 1e-5
        
        self.path = path
        
        #import matplotlib.pyplot as plt
        #plt.imshow(cost[::4,::4])
        #plt.plot(self.path[::4,1]/4, self.path[::4,0]/4)
        #closed_nodes = np.array(list(astar.closedset))
        #plt.plot(closed_nodes[:,0], closed_nodes[:,1], '.', alpha=0.5)
        #plt.show() 
    
    def __call__(self, Xp):
        return np.interp(Xp, self.path[:,1], self.path[:,0])
        

class DTW:
    
    def __init__(self, X, Y, metric='euclidean', p=2, type='linear', bias=0.5):

        """ Create Cost Matrix """
    
        cost = sp_spatial.distance.cdist(X, Y, metric=metric, p=p)
        cost[0] = np.cumsum(cost[0])
        cost[:,0] = np.cumsum(cost[:,0])
        
        for j in range(1, cost.shape[1]):
            for i in range(1, cost.shape[0]):
                cost[i,j] += min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1])
        
        m, n = cost.shape[0]-1, cost.shape[1]-1
        
        """ Find Path """
        
        path = []
        
        while (m,n) != (0,0):
            path.append((m,n))
            m,n = min((m-1, n), (m, n-1), (m-1, n-1), key = lambda x: cost[x[0], x[1]])
        
        path.append((0,0))
        path = np.flipud(path)
        
        """ Save Variables """
        
        self.path = path
        self.path_y = np.unique(path[:,0], return_index=True)[1]
        self.path_x = path[self.path_y,0]
        
        self.lenx = len(X)
        self.leny = len(Y)
        self.bias = bias
        self.type = type
        
    def __call__(self, Xp):
        
        if self.type == 'discrete':
            raise Exception('TODO: Implement')
        
        if self.type == 'linear':
            p0 = self.leny * (Xp.astype(np.float) / self.lenx)
            p1 = np.interp(Xp, self.path_x, self.path_y)
            return p0 * (1-self.bias) + p1 * (self.bias)
            