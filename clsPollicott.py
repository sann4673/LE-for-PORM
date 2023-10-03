import math
import numpy as np
from clsPoRM import PoRM

class Pollicott(object):
    
    def __init__(self, porm : 'PoRM', n : int):
        self.matlist = porm.matlist
        self.matlistsize = len(self.matlist)
        self.pvec = porm.pvec
        self.iter = n
        self.savedproducts = dict()
        self.savedprob = dict()
        for i in range(self.matlistsize):
            self.savedproducts[str(i)] = self.matlist[i]
            self.savedprob[str(i)] = self.pvec[i]
        self.savedtraces = dict()
        self.savedcoeff = dict()
        self.approximations = np.zeros(n+1, dtype = np.longdouble)
        self.LEapprox(n)
            
        
    def matProd(self, s : str): # given string s, return the product as np.array (used DP)
        if s in self.savedproducts:
            return self.savedproducts[s]
        hlen = len(s)// 2
        prod = np.matmul(self.matProd(s[:-hlen]), self.matProd(s[-hlen:]))
        if (len(s) <= (self.iter+1) // 2):
            self.savedproducts[s] = prod
        return prod
    
    def probProd(self, s : str): # given string s, return the corresponding probability
        if s in self.savedprob:
            return self.savedprob[s]
        hlen = len(s)//2
        prob = self.probProd(s[:-hlen]) * self.probProd(s[-hlen:])
        if (len(s) <= (self.iter+1) // 2):
            self.savedprob[s] = prob
        return prob
    
        
    def traceHelper(self, s : str):
        # given string s, return the probability product, denominator, and numerator in trace
        A = self.matProd(s)
        Aevals, _ = np.linalg.eig(A)
        Atopeval = Aevals[0]
        denom = np.prod(1 - Aevals[1:]/Atopeval)
        strprob = self.probProd(s)
        numer = strprob * np.log(Atopeval)    
        return (strprob, denom, numer)
        
        
    def traceL(self, m : int):
        # calculate tr(L^m) when t = 0 as well as the derivative 
        # of tr(L^m) wrt t at t = 0
        if m in self.savedtraces:
            return self.savedtraces[m]
        tr, dtr = self.backtracking('', m, 0, 0)
        self.savedtraces[m] = (tr, dtr)
        return (tr, dtr)
        
        
    def backtracking(self, currstr:str, m:int, tr, dtr):
        # use backtracking to generate all possible strings with length m to help compute traceL
        if (len(currstr) == m):
            strprob, denom, numer = self.traceHelper(currstr)
            tr += strprob / denom
            dtr += numer / denom
            return (tr, dtr)
        for i in range(self.matlistsize):
            nextstr = currstr + str(i)
            tr, dtr = self.backtracking(nextstr, m, tr, dtr)
        return (tr, dtr)
    
    def coefficientHelper(self, sol):
        # given a solution of n_1 + ... + n_k = m (k undetermined) return the list of tr(L^n_i)/n_i and the list
        # of dtrL(n_i)/tr(L^n_i)
        trlist = np.zeros(len(sol), dtype = np.longdouble)
        qtrlist = np.zeros(len(sol), dtype = np.longdouble)
        for i in range(len(sol)):
            trinfo, dtrinfo = self.traceL(sol[i])
            trlist[i] = trinfo / sol[i]
            qtrlist[i] = dtrinfo / trinfo
        return (trlist, qtrlist)
    
    def partition(self, m):
        # given m, return all solutions of n_1 + ... + n_k = m (k undetermined)
        sollist = [[m]]
        for i in range(1, m):
            for p in self.partition(m - i):
                sol = [i] + p
                sollist.append(sol)
        return sollist

    def coeff(self, m):
        # return the list of (a_m^j)_j at t = 0
        if m in self.savedcoeff:
            return self.savedcoeff[m]
        ares = np.longdouble(0)
        cres = np.longdouble(0)
        for sol in self.partition(m):
            trlist, qtrlist = self.coefficientHelper(sol)
            k = len(sol)
            aterm = (-1)**k * np.prod(trlist)/math.factorial(k)
            ares += aterm
            cres += aterm * np.sum(qtrlist)
        bres = m * ares
        self.savedcoeff[m] = (ares, bres, cres)
        return (ares, bres, cres)
    
    def LEapprox(self, n):
        # return the n-th approximation of LE
        bsum = np.longdouble(0)
        csum = np.longdouble(0)
        for i in range(1, n+1):
            _, bres, cres = self.coeff(i)
            csum += cres
            bsum += bres
            self.approximations[i] = csum / bsum
            print('The ({})-approximation is {}'.format(i, self.approximations[i]))
        return self.approximations
        
        
        