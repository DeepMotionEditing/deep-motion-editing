import numpy as np
import scipy.sparse as sparse
import Animation as Animation


""" Maya Functions """

def load_from_maya(root):
    """
    Load joint parents and names from maya
    
    Parameters
    ----------
    
    root : PyNode
        Root Maya Node
        
    Returns
    -------
    
    (names, parents) : ([str], (J) ndarray)
        List of joint names and array
        of indices representing the parent
        joint for each joint J.
        
        Joint index -1 is used to represent
        that there is no parent joint
    """
    
    import pymel.core as pm
    
    names = []
    parents = []

    def unload_joint(j, parents, par):
        
        id = len(names)
        names.append(j)
        parents.append(par)
        
        children = [c for c in j.getChildren() if
                isinstance(c, pm.nt.Transform) and
            not isinstance(c, pm.nt.Constraint) and
            not any(pm.listRelatives(c, s=True)) and 
            (any(pm.listRelatives(c, ad=True, ap=False, type='joint')) or isinstance(c, pm.nt.Joint))]
        
        map(lambda c: unload_joint(c, parents, id), children)

    unload_joint(root, parents, -1)
    
    return (names, parents)

    
""" Family Functions """

def joints(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    joints : (J) ndarray
        Array of joint indices
    """
    return np.arange(len(parents), dtype=int)

def joints_list(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    joints : [ndarray]
        List of arrays of joint idices for
        each joint
    """
    return list(joints(parents)[:,np.newaxis])
    
def parents_list(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    parents : [ndarray]
        List of arrays of joint idices for
        the parents of each joint
    """
    return list(parents[:,np.newaxis])
    
    
def children_list(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    children : [ndarray]
        List of arrays of joint indices for
        the children of each joint
    """
    
    def joint_children(i):
        return [j for j, p in enumerate(parents) if p == i]
        
    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))
     
     
def descendants_list(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    descendants : [ndarray]
        List of arrays of joint idices for
        the descendants of each joint
    """
    
    children = children_list(parents)
    
    def joint_descendants(i):
        return sum([joint_descendants(j) for j in children[i]], list(children[i]))
    
    return list(map(lambda j: np.array(joint_descendants(j)), joints(parents)))
    
    
def ancestors_list(parents):
    """
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
    
    Returns
    -------
    
    ancestors : [ndarray]
        List of arrays of joint idices for
        the ancestors of each joint
    """
    
    decendants = descendants_list(parents)
    
    def joint_ancestors(i):
        return [j for j in joints(parents) if i in decendants[j]]
        
    return list(map(lambda j: np.array(joint_ancestors(j)), joints(parents)))
    
    
""" Mask Functions """

def mask(parents, filter):
    """
    Constructs a Mask for a give filter
    
    A mask is a (J, J) ndarray truth table for a given
    condition over J joints. For example there
    may be a mask specifying if a joint N is a
    child of another joint M.

    This could be constructed into a mask using
    `m = mask(parents, children_list)` and the condition
    of childhood tested using `m[N, M]`.
    
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
        
    filter : (J) ndarray -> [ndarray]
        function that outputs a list of arrays
        of joint indices for some condition

    Returns
    -------
    
    mask : (N, N) ndarray
        boolean truth table of given condition
    """
    m = np.zeros((len(parents), len(parents))).astype(bool)
    jnts = joints(parents)
    fltr = filter(parents)
    for i,f in enumerate(fltr): m[i,:] = np.any(jnts[:,np.newaxis] == f[np.newaxis,:], axis=1)
    return m

def joints_mask(parents): return np.eye(len(parents)).astype(bool)
def children_mask(parents): return mask(parents, children_list)
def parents_mask(parents): return mask(parents, parents_list)
def descendants_mask(parents): return mask(parents, descendants_list)
def ancestors_mask(parents): return mask(parents, ancestors_list)
    
""" Search Functions """
    
def joint_chain_ascend(parents, start, end):
    chain = []
    while start != end:
        chain.append(start)
        start = parents[start]
    chain.append(end)
    return np.array(chain, dtype=int)
    
    
""" Constraints """
    
def constraints(anim, **kwargs):
    """
    Constraint list for Animation
    
    This constraint list can be used in the
    VerletParticle solver to constrain
    a animation global joint positions.
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
        
    masses : (F, J) ndarray
        Optional list of masses
        for joints J across frames F
        defaults to weighting by
        vertical height
    
    Returns
    -------
    
    constraints : [(int, int, (F, J) ndarray, (F, J) ndarray, (F, J) ndarray)]
        A list of constraints in the format:
        (Joint1, Joint2, Masses1, Masses2, Lengths)
    
    """
    
    masses = kwargs.pop('masses', None)
    
    children = children_list(anim.parents)
    constraints = []

    points_offsets = Animation.offsets_global(anim)
    points = Animation.positions_global(anim)
    
    if masses is None:
        masses = 1.0 / (0.1 + np.absolute(points_offsets[:,1]))
        masses = masses[np.newaxis].repeat(len(anim), axis=0)
    
    for j in xrange(anim.shape[1]):
                
        """ Add constraints between all joints and their children """
        for c0 in children[j]:
            
            dists = np.sum((points[:, c0] - points[:, j])**2.0, axis=1)**0.5
            constraints.append((c0, j, masses[:,c0], masses[:,j], dists))
            
            """ Add constraints between all children of joint """
            for c1 in children[j]:
                if c0 == c1: continue

                dists = np.sum((points[:, c0] - points[:, c1])**2.0, axis=1)**0.5
                constraints.append((c0, c1, masses[:,c0], masses[:,c1], dists))  
    
    return constraints
    
""" Graph Functions """

def graph(anim):
    """
    Generates a weighted adjacency matrix
    using local joint distances along
    the skeletal structure.
    
    Joints which are not connected
    are assigned the weight `0`.
    
    Joints which actually have zero distance
    between them, but are still connected, are
    perturbed by some minimal amount.
    
    The output of this routine can be used
    with the `scipy.sparse.csgraph`
    routines for graph analysis.
    
    Parameters
    ----------
    
    anim : Animation
        input animation
        
    Returns
    -------
    
    graph : (N, N) ndarray
        weight adjacency matrix using
        local distances along the
        skeletal structure from joint
        N to joint M. If joints are not
        directly connected are assigned
        the weight `0`.
    """
    
    graph = np.zeros(anim.shape[1], anim.shape[1])
    lengths = np.sum(anim.offsets**2.0, axis=1)**0.5 + 0.001
    
    for i,p in enumerate(anim.parents):
        if p == -1: continue
        graph[i,p] = lengths[p]
        graph[p,i] = lengths[p]
    
    return graph

    
def distances(anim):
    """
    Generates a distance matrix for
    pairwise joint distances along
    the skeletal structure
    
    Parameters
    ----------
    
    anim : Animation
        input animation
        
    Returns
    -------
    
    distances : (N, N) ndarray
        array of pairwise distances
        along skeletal structure
        from some joint N to some
        joint M
    """
    
    distances = np.zeros((anim.shape[1], anim.shape[1]))
    generated = distances.copy().astype(bool)
    
    joint_lengths = np.sum(anim.offsets**2.0, axis=1)**0.5
    joint_children = children_list(anim)
    joint_parents  = parents_list(anim)
    
    def find_distance(distances, generated, prev, i, j):
        
        """ If root, identity, or already generated, return """ 
        if j == -1: return (0.0, True)
        if j ==  i: return (0.0, True)
        if generated[i,j]: return (distances[i,j], True)
        
        """ Find best distances along parents and children """
        par_dists = [(joint_lengths[j], find_distance(distances, generated, j, i, p)) for p in joint_parents[j]  if p != prev]
        out_dists = [(joint_lengths[c], find_distance(distances, generated, j, i, c)) for c in joint_children[j] if c != prev]
        
        """ Check valid distance and not dead end """
        par_dists = [a + d for (a, (d, f)) in par_dists if f]
        out_dists = [a + d for (a, (d, f)) in out_dists if f]
        
        """ All dead ends """
        if (out_dists + par_dists) == []: return (0.0, False)
        
        """ Get minimum path """
        dist = min(out_dists + par_dists)
        distances[i,j] = dist; distances[j,i] = dist
        generated[i,j] = True; generated[j,i] = True
    
    for i in xrange(anim.shape[1]):
        for j in xrange(anim.shape[1]):
            find_distance(distances, generated, -1, i, j)
        
    return distances
    
def edges(parents):
    """
    Animation structure edges
    
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
        
    Returns
    -------
    
    edges : (M, 2) ndarray
        array of pairs where each
        pair contains two indices of a joints
        which corrisponds to an edge in the
        joint structure going from parent to child.
    """
    
    return np.array(list(zip(parents, joints(parents)))[1:])
    
    
def incidence(parents):
    """
    Incidence Matrix
    
    Parameters
    ----------
    
    parents : (J) ndarray
        parents array
        
    Returns
    -------
    
    incidence : (N, M) ndarray
        
        Matrix of N joint positions by
        M edges which each entry is either
        1 or -1 and multiplication by the
        joint positions returns the an
        array of vectors along each edge
        of the structure
    """
    
    es = edges(parents)
    
    inc = np.zeros((len(parents)-1, len(parents))).astype(np.int)
    for i, e in enumerate(es):
        inc[i,e[0]] =  1
        inc[i,e[1]] = -1
    
    return inc.T
    