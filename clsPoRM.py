import random
import numpy as np

class PoRM(object):
    
    def __init__(self, matlist, pvec):
        self.matlist = matlist
        self.pvec = pvec
        self.matlistsize = len(matlist)
        psize = len(pvec)
        if (self.matlistsize != psize):
            raise RuntimeError('Number of matrices do not match with probability vector.')
        if not np.isclose(sum(pvec),1):
            self.pvec = np.dot(pvec, 1/sum(pvec))
            print('Probability vector is normalised to {}.'.format(self.pvec))
        if not self.matlistsize:
            raise RuntimeError('No matrix in the matrix list.')
        r, c = matlist[0].shape
        if (r != c):
            raise RuntimeError('Matrix is not square.')
        self.matdim = r
        for mat in matlist:
            if ((r, c) != mat.shape):
                raise RuntimeError('Matrix sizes do not match in the list.')
                
    def description(self, info=True, invertibility=True, positivity=True):
        if info:
            if (self.matlistsize <= 5):
                for i in range(self.matlistsize):
                    print('With probability {} to choose matrix {}'.format(self.pvec[i], self.matlist[i]))
            else:
                for i in range(0, 3):
                    print('With probability {} to choose matrix {}'.format(self.pvec[i], self.matlist[i]))
                print('...\n')
                for i in range(-2,0):
                    print('With probability {} to choose matrix {}'.format(self.pvec[i], self.matlist[i]))       
        if invertibility:
            noninvertible = 0
            for mat in matlist:
                if (np.linalg.det(mat) == 0):
                    noninvertible += 1
            if not noninvertible: print('All matrices are invertible.')
            else: print('There are {} non-invertible matrices. Pollicott\'s algorithm is not applicable.'.format(noninvertible))
        if positivity:
            nonpositive = 0
            for mat in matlist:
                if (mat <= 0).any():
                    nonpositive += 1
            if not nonpositive: print('All matrices are positive.')
            else: print('There are {} non-positive matrices. Pollicott\'s algorithm is not applicable.'.format(nonpositive))
    
    def monte_carlo_stochastic(self, init_vec = None, niter = 1000):
        if not init_vec:
            init_vec = np.random.randn(self.matdim)
        if (len(init_vec) != self.matdim):
            raise RuntimeError(
                'Size of initial vector ({}) does not match the dimension of matrix ({}).'.format(
                    len(init_vec), self.matdim))
        if (init_vec == [0]*self.matdim):
            raise RuntimeError('Initial vector is not allowed to set to be a zero vector.')
        v, _ = normalize_vec(init_vec)
        LE = 0
        for i in range(1, niter):
            randomID = np.random.choice(np.arange(0, self.matlistsize), p=self.pvec)
            v = np.dot(self.matlist[randomID], v)
            v, norm = normalize_vec(v)
            LE += (np.log(norm) - LE)/i
        # print('stochastic monte-carlo after {} iterations: {}'.format(niter, LE))
        return LE
    
    def monte_carlo_enum(self, init_vec = None, depth = 10):
        if not init_vec:
            init_vec = np.random.randn(self.matdim)
        if (len(init_vec) != self.matdim):
            raise RuntimeError(
                'Size of initial vector ({}) does not match the dimension of matrix ({}).'.format(
                    len(init_vec), self.matdim))
        if (init_vec == [0]*self.matdim):
            raise RuntimeError('Initial vector is not allowed to set to be a zero vector.')
        v, _ = normalize_vec(init_vec)
        LE = 0
        for i in range(self.matlistsize):
            LE = self.backtrack(0, depth, i, 0, v, LE)
        return LE       
        
    def backtrack(self, currdepth, depth, currmatID, currlogprob, currv, LE):
        nextlogprob = currlogprob + np.log(self.pvec[currmatID])
        nextv = np.dot(self.matlist[currmatID], currv)
        if (currdepth == depth):
            _, norm = normalize_vec(nextv)
            LE += (np.log(norm)/depth)*np.exp(nextlogprob)
            return LE
        for i in range(self.matlistsize):
            LE = self.backtrack(currdepth+1, depth, i, nextlogprob, nextv, LE)
        return LE
        

    
def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    return (vec/norm, norm)