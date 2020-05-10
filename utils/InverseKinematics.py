import numpy as np
import scipy.linalg as linalg

import Animation
import AnimationStructure

from Quaternions_old import Quaternions

class BasicInverseKinematics:
    """
    Basic Inverse Kinematics Solver
    
    This is an extremely simple full body IK
    solver.
    
    It works given the following conditions:
    
        * All joint targets must be specified
        * All joint targets must be in reach
        * All joint targets must not differ 
          extremely from the starting pose
        * No bone length constraints can be violated
        * The root translation and rotation are 
          set to good initial values
    
    It works under the observation that if the
    _directions_ the joints are pointing toward
    match the _directions_ of the vectors between
    the target joints then the pose should match
    that of the target pose.
    
    Therefore it iterates over joints rotating
    each joint such that the vectors between it
    and it's children match that of the target
    positions.
    
    Parameters
    ----------
    
    animation : Animation
        animation input
        
    positions : (F, J, 3) ndarray
        target positions for each frame F
        and each joint J
    
    iterations : int
        Optional number of iterations.
        If the above conditions are met
        1 iteration should be enough,
        therefore the default is 1
        
    silent : bool
        Optional if to suppress output
        defaults to False
    """
    
    def __init__(self, animation, positions, iterations=1, silent=True):
    
        self.animation = animation
        self.positions = positions
        self.iterations = iterations
        self.silent = silent
        
    def __call__(self):
        
        children = AnimationStructure.children_list(self.animation.parents)
        
        for i in range(self.iterations):
        
            for j in AnimationStructure.joints(self.animation.parents):
                
                c = np.array(children[j])
                if len(c) == 0: continue
                
                anim_transforms = Animation.transforms_global(self.animation)
                anim_positions = anim_transforms[:,:,:3,3]
                anim_rotations = Quaternions.from_transforms(anim_transforms)
                
                jdirs = anim_positions[:,c] - anim_positions[:,np.newaxis,j]
                ddirs = self.positions[:,c] - anim_positions[:,np.newaxis,j]

                jsums = np.sqrt(np.sum(jdirs**2.0, axis=-1)) + 1e-10
                dsums = np.sqrt(np.sum(ddirs**2.0, axis=-1)) + 1e-10
                
                jdirs = jdirs / jsums[:,:,np.newaxis]
                ddirs = ddirs / dsums[:,:,np.newaxis]
                
                angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
                axises = np.cross(jdirs, ddirs)
                axises = -anim_rotations[:,j,np.newaxis] * axises
                
                rotations = Quaternions.from_angle_axis(angles, axises)
                
                if rotations.shape[1] == 1:
                    averages = rotations[:,0]
                else:
                    averages = Quaternions.exp(rotations.log().mean(axis=-2))                
                
                self.animation.rotations[:,j] = self.animation.rotations[:,j] * averages
            
            if not self.silent:
                anim_positions = Animation.positions_global(self.animation)
                error = np.mean(np.sum((anim_positions - self.positions)**2.0, axis=-1)**0.5, axis=-1)
                print('[BasicInverseKinematics] Iteration %i Error: %f' % (i+1, error))
        
        return self.animation


class JacobianInverseKinematics:
    """
    Jacobian Based Full Body IK Solver
    
    This is a full body IK solver which
    uses the dampened least squares inverse
    jacobian method.
    
    It should remain fairly stable and effective
    even for joint positions which are out of
    reach and it can also take any number of targets
    to treat as end effectors.
    
    Parameters
    ----------
    
    animation : Animation
        animation to solve inverse problem on

    targets : {int : (F, 3) ndarray}
        Dictionary of target positions for each
        frame F, mapping joint index to 
        a target position
    
    references : (F, 3)
        Optional list of J joint position
        references for which the result
        should bias toward
        
    iterations : int
        Optional number of iterations to
        compute. More iterations results in
        better accuracy but takes longer to 
        compute. Default is 10.
        
    recalculate : bool
        Optional if to recalcuate jacobian
        each iteration. Gives better accuracy
        but slower to compute. Defaults to True
        
    damping : float
        Optional damping constant. Higher
        damping increases stability but
        requires more iterations to converge.
        Defaults to 5.0
        
    secondary : float
        Force, or bias toward secondary target.
        Defaults to 0.25
        
    silent : bool
        Optional if to suppress output
        defaults to False
    """
    
    def __init__(self, animation, targets,
        references=None, iterations=10,
        recalculate=True, damping=2.0,
        secondary=0.25, translate=False,
        silent=False, weights=None,
        weights_translate=None):
        
        self.animation = animation
        self.targets = targets
        self.references = references
        
        self.iterations  = iterations
        self.recalculate = recalculate
        self.damping   = damping
        self.secondary   = secondary
        self.translate   = translate
        self.silent      = silent
        self.weights     = weights
        self.weights_translate = weights_translate
        
    def cross(self, a, b):
        o = np.empty(b.shape)
        o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
        o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
        o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
        return o
        
    def jacobian(self, x, fp, fr, ts, dsc, tdsc):
        
        """ Find parent rotations """
        prs = fr[:,self.animation.parents]
        prs[:,0] = Quaternions.id((1))
        
        """ Find global positions of target joints """
        tps = fp[:,np.array(list(ts.keys()))]
        
        """ Get partial rotations """
        qys = Quaternions.from_angle_axis(x[:,1:prs.shape[1]*3:3], np.array([[[0,1,0]]]))
        qzs = Quaternions.from_angle_axis(x[:,2:prs.shape[1]*3:3], np.array([[[0,0,1]]]))
        
        """ Find axis of rotations """
        es = np.empty((len(x),fr.shape[1]*3, 3))
        es[:,0::3] = ((prs * qzs) * qys) * np.array([[[1,0,0]]])
        es[:,1::3] = ((prs * qzs) * np.array([[[0,1,0]]]))
        es[:,2::3] = ((prs * np.array([[[0,0,1]]])))
        
        """ Construct Jacobian """
        j = fp.repeat(3, axis=1)
        j = dsc[np.newaxis,:,:,np.newaxis] * (tps[:,np.newaxis,:] - j[:,:,np.newaxis])
        j = self.cross(es[:,:,np.newaxis,:], j)
        j = np.swapaxes(j.reshape((len(x), fr.shape[1]*3, len(ts)*3)), 1, 2)
                
        if self.translate:
            
            es = np.empty((len(x),fr.shape[1]*3, 3))
            es[:,0::3] = prs * np.array([[[1,0,0]]])
            es[:,1::3] = prs * np.array([[[0,1,0]]])
            es[:,2::3] = prs * np.array([[[0,0,1]]])
            
            jt = tdsc[np.newaxis,:,:,np.newaxis] * es[:,:,np.newaxis,:].repeat(tps.shape[1], axis=2)
            jt = np.swapaxes(jt.reshape((len(x), fr.shape[1]*3, len(ts)*3)), 1, 2)
            
            j = np.concatenate([j, jt], axis=-1)
        
        return j
        
    #@profile(immediate=True)
    def __call__(self, descendants=None, gamma=1.0):
        
        self.descendants = descendants
        
        """ Calculate Masses """
        if self.weights is None:
            self.weights = np.ones(self.animation.shape[1])
            
        if self.weights_translate is None:
            self.weights_translate = np.ones(self.animation.shape[1])
        
        """ Calculate Descendants """
        if self.descendants is None:
            self.descendants = AnimationStructure.descendants_mask(self.animation.parents)
        
        self.tdescendants = np.eye(self.animation.shape[1]) + self.descendants
        
        self.first_descendants = self.descendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)
        self.first_tdescendants = self.tdescendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)
        
        """ Calculate End Effectors """
        self.endeff = np.array(list(self.targets.values()))
        self.endeff = np.swapaxes(self.endeff, 0, 1) 
        
        if not self.references is None:
            self.second_descendants = self.descendants.repeat(3, axis=0).astype(int)
            self.second_tdescendants = self.tdescendants.repeat(3, axis=0).astype(int)
            self.second_targets = dict([(i, self.references[:,i]) for i in xrange(self.references.shape[1])])
        
        nf = len(self.animation)
        nj = self.animation.shape[1]
        
        if not self.silent:
            gp = Animation.positions_global(self.animation)
            gp = gp[:,np.array(list(self.targets.keys()))]            
            error = np.mean(np.sqrt(np.sum((self.endeff - gp)**2.0, axis=2)))
            print('[JacobianInverseKinematics] Start | Error: %f' % error)
        
        for i in range(self.iterations):

            """ Get Global Rotations & Positions """
            gt = Animation.transforms_global(self.animation)
            gp = gt[:,:,:,3]
            gp = gp[:,:,:3] / gp[:,:,3,np.newaxis]
            gr = Quaternions.from_transforms(gt)
            
            x = self.animation.rotations.euler().reshape(nf, -1)
            w = self.weights.repeat(3)
            
            if self.translate:
                x = np.hstack([x, self.animation.positions.reshape(nf, -1)])
                w = np.hstack([w, self.weights_translate.repeat(3)])
            
            """ Generate Jacobian """
            if self.recalculate or i == 0:
                j = self.jacobian(x, gp, gr, self.targets, self.first_descendants, self.first_tdescendants)
            
            """ Update Variables """            
            l = self.damping * (1.0 / (w + 0.001))
            d = (l*l) * np.eye(x.shape[1])
            e = gamma * (self.endeff.reshape(nf,-1) - gp[:,np.array(list(self.targets.keys()))].reshape(nf, -1))
            
            x += np.array(list(map(lambda jf, ef:
                linalg.lu_solve(linalg.lu_factor(jf.T.dot(jf) + d), jf.T.dot(ef)), j, e)))
            
            """ Generate Secondary Jacobian """
            if self.references is not None:
                
                ns = np.array(list(map(lambda jf:
                    np.eye(x.shape[1]) - linalg.solve(jf.T.dot(jf) + d, jf.T.dot(jf)), j)))
                    
                if self.recalculate or i == 0:
                    j2 = self.jacobian(x, gp, gr, self.second_targets, self.second_descendants, self.second_tdescendants)
                        
                e2 = self.secondary * (self.references.reshape(nf, -1) - gp.reshape(nf, -1))
                
                x += np.array(list(map(lambda nsf, j2f, e2f:
                    nsf.dot(linalg.lu_solve(linalg.lu_factor(j2f.T.dot(j2f) + d), j2f.T.dot(e2f))), ns, j2, e2)))

            """ Set Back Rotations / Translations """
            self.animation.rotations = Quaternions.from_euler(
                x[:,:nj*3].reshape((nf, nj, 3)), order='xyz', world=True)
                
            if self.translate:
                self.animation.positions = x[:,nj*3:].reshape((nf,nj, 3))
                
            """ Generate Error """
            
            if not self.silent:
                gp = Animation.positions_global(self.animation)
                gp = gp[:,np.array(list(self.targets.keys()))]
                error = np.mean(np.sum((self.endeff - gp)**2.0, axis=2)**0.5)
                print('[JacobianInverseKinematics] Iteration %i | Error: %f' % (i+1, error))
            

class BasicJacobianIK:
    """
    Same interface as BasicInverseKinematics
    but uses the Jacobian IK Solver Instead
    """
    
    def __init__(self, animation, positions, iterations=10, silent=True, **kw):
       
        targets = dict([(i, positions[:,i]) for i in range(positions.shape[1])])
        self.ik = JacobianInverseKinematics(animation, targets, iterations=iterations, silent=silent, **kw)
       
    def __call__(self, **kw):
        return self.ik(**kw)
        

class ICP:
    
    
    def __init__(self, 
        anim, rest, weights, mesh, goal,
        find_closest=True, damping=10,
        iterations=10, silent=True,
        translate=True, recalculate=True,
        weights_translate=None):
        
        self.animation = anim
        self.rest = rest
        self.vweights = weights
        self.mesh = mesh
        self.goal = goal
        self.find_closest = find_closest
        self.iterations = iterations
        self.silent = silent
        self.translate = translate
        self.damping   = damping
        self.weights = None
        self.weights_translate = weights_translate
        self.recalculate = recalculate
        
    def cross(self, a, b):
        o = np.empty(b.shape)
        o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
        o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
        o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
        return o
        
    def jacobian(self, x, fp, fr, goal, weights, des_r, des_t):
        
        """ Find parent rotations """
        prs = fr[:,self.animation.parents]
        prs[:,0] = Quaternions.id((1))
        
        """ Get partial rotations """
        qys = Quaternions.from_angle_axis(x[:,1:prs.shape[1]*3:3], np.array([[[0,1,0]]]))
        qzs = Quaternions.from_angle_axis(x[:,2:prs.shape[1]*3:3], np.array([[[0,0,1]]]))
        
        """ Find axis of rotations """
        es = np.empty((len(x),fr.shape[1]*3, 3))
        es[:,0::3] = ((prs * qzs) * qys) * np.array([[[1,0,0]]])
        es[:,1::3] = ((prs * qzs) * np.array([[[0,1,0]]]))
        es[:,2::3] = ((prs * np.array([[[0,0,1]]])))
        
        """ Construct Jacobian """
        j = fp.repeat(3, axis=1)
        j = des_r[np.newaxis,:,:,:,np.newaxis] * (goal[:,np.newaxis,:,np.newaxis] - j[:,:,np.newaxis,np.newaxis])
        j = np.sum(j * weights[np.newaxis,np.newaxis,:,:,np.newaxis], 3)
        j = self.cross(es[:,:,np.newaxis,:], j)
        j = np.swapaxes(j.reshape((len(x), fr.shape[1]*3, goal.shape[1]*3)), 1, 2)
        
        if self.translate:
            
            es = np.empty((len(x),fr.shape[1]*3, 3))
            es[:,0::3] = prs * np.array([[[1,0,0]]])
            es[:,1::3] = prs * np.array([[[0,1,0]]])
            es[:,2::3] = prs * np.array([[[0,0,1]]])
            
            jt = des_t[np.newaxis,:,:,:,np.newaxis] * es[:,:,np.newaxis,np.newaxis,:].repeat(goal.shape[1], axis=2)
            jt = np.sum(jt * weights[np.newaxis,np.newaxis,:,:,np.newaxis], 3)
            jt = np.swapaxes(jt.reshape((len(x), fr.shape[1]*3, goal.shape[1]*3)), 1, 2)
            
            j = np.concatenate([j, jt], axis=-1)
        
        return j
        
    #@profile(immediate=True)
    def __call__(self, descendants=None, maxjoints=4, gamma=1.0, transpose=False):
        
        """ Calculate Masses """
        if self.weights is None:
            self.weights = np.ones(self.animation.shape[1])
            
        if self.weights_translate is None:
            self.weights_translate = np.ones(self.animation.shape[1]) 
        
        nf = len(self.animation)
        nj = self.animation.shape[1]
        nv = self.goal.shape[1]
        
        weightids = np.argsort(-self.vweights, axis=1)[:,:maxjoints]
        weightvls = np.array(list(map(lambda w, i: w[i], self.vweights, weightids)))
        weightvls = weightvls / weightvls.sum(axis=1)[...,np.newaxis]
        
        if descendants is None:
            self.descendants = AnimationStructure.descendants_mask(self.animation.parents)
        else:
            self.descendants = descendants
        
        des_r = np.eye(nj) + self.descendants
        des_r = des_r[:,weightids].repeat(3, axis=0)

        des_t = np.eye(nj) + self.descendants
        des_t = des_t[:,weightids].repeat(3, axis=0)
        
        if not self.silent:
            curr = Animation.skin(self.animation, self.rest, self.vweights, self.mesh, maxjoints=maxjoints)
            error = np.mean(np.sqrt(np.sum((curr - self.goal)**2.0, axis=-1)))
            print('[ICP] Start | Error: %f' % error)
        
        for i in range(self.iterations):
            
            """ Get Global Rotations & Positions """
            gt = Animation.transforms_global(self.animation)
            gp = gt[:,:,:,3]
            gp = gp[:,:,:3] / gp[:,:,3,np.newaxis]
            gr = Quaternions.from_transforms(gt)
            
            x = self.animation.rotations.euler().reshape(nf, -1)
            w = self.weights.repeat(3)
            
            if self.translate:
                x = np.hstack([x, self.animation.positions.reshape(nf, -1)])
                w = np.hstack([w, self.weights_translate.repeat(3)])
            
            """ Get Current State """
            curr = Animation.skin(self.animation, self.rest, self.vweights, self.mesh, maxjoints=maxjoints)

            """ Find Cloest Points """
            if self.find_closest:
                mapping = np.argmin(
                    (curr[:,:,np.newaxis] - 
                     self.goal[:,np.newaxis,:])**2.0, axis=2)
                e = gamma * (np.array(list(map(lambda g, m: g[m], self.goal, mapping))) - curr).reshape(nf, -1)
            else:
                e = gamma * (self.goal - curr).reshape(nf, -1)
            
            """ Generate Jacobian """
            if self.recalculate or i == 0:
                j = self.jacobian(x, gp, gr, self.goal, weightvls, des_r, des_t)
            
            """ Update Variables """            
            l = self.damping * (1.0 / (w + 1e-10))
            d = (l*l) * np.eye(x.shape[1])
            
            if transpose:
                x += np.array(list(map(lambda jf, ef: jf.T.dot(ef), j, e)))
            else:
                x += np.array(list(map(lambda jf, ef:
                    linalg.lu_solve(linalg.lu_factor(jf.T.dot(jf) + d), jf.T.dot(ef)), j, e)))
            
            """ Set Back Rotations / Translations """
            self.animation.rotations = Quaternions.from_euler(
                x[:,:nj*3].reshape((nf, nj, 3)), order='xyz', world=True)
                
            if self.translate:
                self.animation.positions = x[:,nj*3:].reshape((nf, nj, 3))
        
            if not self.silent:
                curr = Animation.skin(self.animation, self.rest, self.vweights, self.mesh)
                error = np.mean(np.sqrt(np.sum((curr - self.goal)**2.0, axis=-1)))
                print('[ICP] Iteration %i | Error: %f' % (i+1, error))
                
