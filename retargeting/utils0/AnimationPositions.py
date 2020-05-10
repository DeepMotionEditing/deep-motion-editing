import numpy as np
from scipy.spatial import distance

from Quaternions import Quaternions
import Animation
import AnimationStructure
    
def constrain(positions, constraints):
    """
    Constrain animation positions given
    a number of VerletParticles constrains
    
    Parameters
    ----------
    
    positions : (F, J, 3) ndarray
        array of joint positions for
        F frames and J joints
    
    constraints : [(int, int, float, float, float)]
        A list of constraints in the format:
        (Joint1, Joint2, Masses1, Masses2, Lengths)
        
    Returns
    -------
    
    positions : (F, J, 3) ndarray
        joint positions for F
        frames and J joints constrained
        using the supplied constraints
    """
    
    from VerletParticles import VerletParticles
    
    particles = VerletParticles(positions, gravity=0.0, timestep=0.0)
    for i, j, w0, w1, l in constraints:
        particles.add_length_constraint(i, j, w0, w1, l)

    return particles.constrain()

def extremities(positions, count, **kwargs):
    """
    List of most extreme frame indices
    
    Parameters
    ----------
    
    positions : (F, J, 3) ndarray
        array of joint positions for
        F frames and J joints
    
    count : int
        Number of indices to return,
        does not include first and last
        frame which are always included
    
    static : bool
        Find extremities where root
        translation has been removed
    
    Returns
    -------
    
    indices : (C) ndarray
        Returns C frame indices of the
        most extreme frames including
        the first and last frames.
        
        Therefore if `count` it specified
        as `4` will return and array of
        `6` indices.
    """
    
    if kwargs.pop('static', False):
        positions = positions - positions[:,0][:,np.newaxis,:]
        
    positions = positions.reshape((len(positions), -1))
    
    distance_matrix = distance.squareform(distance.pdist(positions))
    
    keys = [0]
    for _ in range(count-1):
        keys.append(int(np.argmax(np.min(distance_matrix[keys], axis=0))))
    return np.array(keys)
    
    
def load_to_maya(positions, names=None, parents=None, color=None, radius=0.1, thickness=5.0):

    import pymel.core as pm
    import maya.mel as mel
    
    if names is None:
        names = ['joint_%i' % i for i in xrange(positions.shape[1])]
    
    if color is None:
        color = (0.5, 0.5, 0.5)
    
    mpoints = []
    frames = range(1, len(positions)+1)
    for i, name in enumerate(names):
    
        #try:
        #    point = pm.PyNode(name)
        #except pm.MayaNodeError:
        #    point = pm.sphere(p=(0,0,0), n=name, radius=radius)[0]
        point = pm.sphere(p=(0,0,0), n=name, radius=radius)[0]
    
        jpositions = positions[:,i]

        for j,attr,attr_name in zip(xrange(3),
                                    [point.tx, point.ty, point.tz],
                                    ["_translateX", "_translateY", "_translateZ"]):
            conn = attr.listConnections()
            if len(conn) == 0:
                curve = pm.nodetypes.AnimCurveTU(n=name + attr_name)
                pm.connectAttr(curve.output, attr)
            else:
                curve = conn[0]
            curve.addKeys(frames, jpositions[:,j])

        mpoints.append(point)
         
    if parents != None:

        for i, p in enumerate(parents):
            if p == -1: continue
            pointname = names[i]
            parntname = names[p]
            conn = pm.PyNode(pointname).t.listConnections()
            if len(conn) != 0: continue
            
            curve = pm.curve(p=[[0,0,0],[0,1,0]], d=1, n=names[i]+'_curve')
            pm.connectAttr(pointname+'.t', names[i]+'_curve.cv[0]')
            pm.connectAttr(parntname+'.t', names[i]+'_curve.cv[1]')
            pm.select(curve)
            pm.runtime.AttachBrushToCurves()
            stroke = pm.selected()[0]
            brush = pm.listConnections(stroke.getChildren()[0]+'.brush')[0]
            pm.setAttr(brush+'.color1', color)
            pm.setAttr(brush+'.globalScale', thickness)
            pm.setAttr(brush+'.endCaps', 1)
            pm.setAttr(brush+'.tubeSections', 20)
            mel.eval('doPaintEffectsToPoly(1,0,0,1,100000);')
            mpoints += [stroke, curve]
            
    return pm.group(mpoints, n='AnimationPositions'), mpoints

        
def load_from_maya(root, start, end):
    
    import pymel.core as pm
        
    def rig_joints_list(s, js):
        for c in s.getChildren():
            if 'Geo' in c.name(): continue
            if isinstance(c, pm.nodetypes.Joint):     js = rig_joints_list(c, js); continue
            if isinstance(c, pm.nodetypes.Transform): js = rig_joints_list(c, js); continue
        return [s] + js
        
    joints = rig_joints_list(root, [])
    
    names = map(lambda j: j.name(), joints)
    positions = np.empty((end - start, len(names), 3))
    
    original_time = pm.currentTime(q=True)
    pm.currentTime(start)
    
    for i in range(start, end):
        
        pm.currentTime(i)
        for j in joints: positions[i-start, names.index(j.name())] = j.getTranslation(space='world')
    
    pm.currentTime(original_time)
    
    return positions, names

    
def loop(positions, forward='z'):
    
    fid = 'xyz'.index(forward)
    
    data = positions.copy()
    trajectory = data[:,0:1,fid].copy()
    
    data[:,:,fid] -= trajectory
    diff = data[0] - data[-1] 
    data += np.linspace(
        0, 1, len(data))[:,np.newaxis,np.newaxis] * diff[np.newaxis]
    data[:,:,fid] += trajectory
    
    return data

    
def extend(positions, length, forward='z'):
    
    fid = 'xyz'.index(forward)
    
    data = positions.copy()
    
    while len(data) < length:
        
        next = positions[1:].copy()
        next[:,:,fid] += data[-1,0,fid]
        data = np.concatenate([data, next], axis=0)
    
    return data[:length]
    
    
def redirect(positions, joint0, joint1, forward='z'):
    
    forwarddir = {
        'x': np.array([[[1,0,0]]]),
        'y': np.array([[[0,1,0]]]),
        'z': np.array([[[0,0,1]]]),
    }[forward]
    
    direction = (positions[:,joint0] - positions[:,joint1]).mean(axis=0)[np.newaxis,np.newaxis]
    direction = direction / np.sqrt(np.sum(direction**2))
    
    rotation = Quaternions.between(direction, forwarddir).constrained_y()
    
    return rotation * positions
    
