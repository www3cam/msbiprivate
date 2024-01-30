import autograd.numpy as np
from autograd import jacobian, grad
from autograd.extend import defvjp, primitive
np.set_printoptions(suppress=True,precision=4)
from scipy.stats import norm

from scipy.stats import multivariate_normal

#import numdifftools as nd

from scipy.optimize import brentq
import time
import multiprocessing as mp
import random

import pykalman
import csv
from autograd.numpy.random import default_rng

def solve_and_sim_reiter(vec11):
    #adapted from https://alisdairmckay.com/Notes/HetAgents/ code: https://alisdairmckay.com/files/HetAgentsPython3.zip
    #to solve a HANK model via Reiter's method
    #solve
    # Parameters and setup
    iterations = 200
    try:#in case the model doesn't converge
        vecin = vec11.numpy().reshape(-1)
        beta = 0.97
        gamma = 2.0
        rho = vecin[0]#0.95
        rho_xi = vecin[1]#0.80
        zeta = vecin[2]#0.8
        psi = vecin[3]#0.1
        mu = vecin[4]#mu_epsilon / (mu_epsilon - 1)#1.2 
        mu_epsilon = mu/(mu-1)
        theta = vecin[5]#0.75
        omega = vecin[6]#1.5
        ubar = vecin[7]#0.06
        delta = vecin[8]#0.15 # job separation prob
        Mbar = (1-ubar)*delta/(ubar+delta*(1-ubar))
        wbar = 1/mu - delta * psi * Mbar
        B = vecin[9]#0.6
        ben = wbar * 0.5
        History = []
            
        amin = 0.0  # borrowing constraint
        #exogenous transition matrix
        #has form rows = future state, columns = current  state
        N = 2
        nX = 812
    
        #grid for savings
        Amin, Amax, Asize = amin, 200, 201
        A = np.linspace(Amin**(0.25), Amax**(0.25), Asize)**4
        
        tiledGrid = np.tile(A,(2,1))
        
        #initialize
        class Prices:
          def __init__(self,R,M=Mbar,Z=1.0,u=ubar,ulag=ubar):
              self.R = R
              self.M = M
              self.w = wbar * (M/Mbar)**zeta
              Y = Z * (1-u)
              H = (1-u) - (1-delta)*(1-ulag)
              d = (Y-psi*M*H)/(1-u) - self.w
              tau = ((self.R-1)*B + ben * u)/(self.w+d)/(1-u)
              self.employedIncome = (1-tau)*(self.w+d)
              self.earnings = np.array([ben,self.employedIncome])
              self.employTrans = np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]])
     
          def tiledEndow(self):
              return np.tile(self.earnings[np.newaxis].T,(1,Asize))
           
        G = 10+tiledGrid
        #R = 1.01
        #Pr0 = Prices(R)
        def interp(x,y,x1):
            N = len(x)
            i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
            xl = x[i-1]
            xr = x[i]
            yl = y[i-1]
            yr = y[i]
            y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
            above = x1 > x[-1]
            below = x1 < x[0]
            y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
            y1 = np.where(below,y[0],y1)
        
            return y1, i
        
        def get_c(G,Pr,CurrentAssets = tiledGrid):
            return np.vstack( [Pr.R * CurrentAssets[i] + Pr.earnings[i] - interp(G[i],A,CurrentAssets[i])[0] for i in range(N)] )
     
        def uPrime(c):
            return c**(-gamma)
        
        def uPrimeInv(up):
            return up**(-1.0/gamma)
        
        def eulerBack(G,Pr,Pr_P):
        # The argument is the savings policy rule in the next period
        # it is parameterized as the a values associated with savings grid for a'
        
            # compute next period's consumption conditional on next period's income
            cp = get_c(G,Pr_P)
            upcp = uPrime(cp)
        
            #compute E(u'(cp))
            # In principle we could do it like this: Eupcp = np.dot(exogTrans.T , uPrime(cp) )
            # But automatic differentiation doesnt work with matrix-matrix multiplication
            # because it isn't obvious how to take the gradient of a function that produces a matrix
            # so we loop over the values instead
            Eupcp = []
            for ip in range(N):
                Eupcp_i = 0.
                for jp in range(N):
                    Eupcp_i += Pr_P.employTrans[jp,ip] * upcp[jp]
                Eupcp.append(Eupcp_i)
            Eupcp = np.vstack(Eupcp)
        
            #use  upc = R *  beta*Eupcp to solve for upc
            upc = beta*Pr_P.R*Eupcp
        
            #invert uprime to solve for c
            c = uPrimeInv(upc)
        
        
            #use budget constraint to find previous assets
            # (a' + c - y)/R = a
            a = (tiledGrid + c - Pr.tiledEndow())/ Pr.R
        
            return a, c
        
        def SolveEGM(G,Pr):
            #loop until convergence
            #print("solving for policy rules")
            tol = 1e-15
            test = True
            for it in range(10000):
                a = eulerBack(G,Pr,Pr)[0]
        
                if it % 50 == 0:
                    test = np.abs(a-G)/(np.abs(a)+np.abs(G)+tol)
                    #print("it = {0}, test = {1}".format(it,test.max()))
                    if np.all(test  < tol):
                        break
        
                G = a
        
            return G
        
        def Agg_Assets(D):
            return (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
        
        def MakeTransMat_Savings(G):
           # Rows of T correspond to where we are going, cols correspond to where we are coming from
           T = np.zeros((N*Asize,N*Asize))
           for j in range(N):
               x, i = interp(G[j],A,A)
               p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
               p = np.minimum(np.maximum(p,0.0),1.0)
               sj = j*Asize
               T[sj + i,sj+np.arange(Asize)]= p
               T[sj + i - 1,sj+np.arange(Asize)] = (1.0-p)
       
           assert np.allclose(T.sum(axis=0), np.ones(N*Asize))
           return T
       
        
        def MakeTransMat_Emp(M):
            return np.kron(np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]]), np.eye(Asize))
        
        
        def MakeTransMat(G, M):
            return MakeTransMat_Savings(G) @ MakeTransMat_Emp(M)
        
        
        def GetStationaryDist(T):
            eval,evec = np.linalg.eig(T)
            i = np.argmin(np.abs(eval-1.0))
            D = np.array(evec[:,i]).flatten()
            assert np.max(np.abs(np.imag(D))) < 1e-6
            D = np.real(D)  # just recasts as float
            return D/D.sum()
        
        def Check_Assets(R):
            Pr = Prices(R)
            G[:] = SolveEGM(G,Pr)
            Assets_implied = Agg_Assets(GetStationaryDist(MakeTransMat(G,Pr.M)))
            History.append([R,Assets_implied])
            #print("Checking R = {0}, assets implied = {1}".format(R,Assets_implied))
            return Assets_implied - B
        
        Rstar = brentq(Check_Assets,.8,1.2,xtol = 1e-14)
        
       
            
        Pr = Prices(Rstar)
    
        G[:] = SolveEGM(G,Pr)
        
        D = GetStationaryDist(MakeTransMat(G,Pr.M))
        pB  = (1-ubar)  /(1-theta/Rstar)
        Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
        
        X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
        epsilon_SS = np.zeros(2)
        
        
        #from HANK_EGM import SolveEGM, Pr0, Mbar, ubar, G, A, Asize, N, B, Prices, interp, delta
        
        # note, D tracks end of period states
        # For example, aggregate consumption in t should be computed with D_L shuffled
        # with MakeTransMat_Emp(Pr): AggC(G, Pr, MakeTransMat_Emp(Pr) @ D_L )
        
        
        # rewrite eulerBack to deliver residual function
        def eulerResidual(G,G_P,Pr,Pr_P):
        # The arguments are the savings policy rules in the current and next period
        # parameterized as the a values associated with savings grid for a'
        
            a, c = eulerBack(G_P.reshape(N,Asize),Pr,Pr_P)
            c2 = get_c(G.reshape(N,Asize),Pr,CurrentAssets = a)
            return (c/c2 - 1).reshape(-1)
        
        
        # get residuals of distribution of wealth equations
        @primitive
        def wealthResidual(G,D_L,D,M):
            return (D - MakeTransMat(G.reshape(N,Asize),M) @ D_L)[1:]  # drop one redundant equation (prob distribution sums to 1)
        
        
        
        def Deriv_MakeTransMat_Savings(G):
            # Create a 3-D array TD
            # TD[i,j,k] is the derivative of the transition probability T[i,j] with respect to G[k] (where G has been flattened)
            G = G.reshape(N,Asize)
            TD = np.zeros((N*Asize,N*Asize,G.size))
            for j in range(N):
                x, i = interp(G[j],A,A)
                p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
        
                dpdGLeft = (p-1) / (G[j,i] - G[j,i-1])
                dpdGRight = - p / (G[j,i] - G[j,i-1])
        
                dpdGLeft = np.where( (p > 0) & (p<1) , dpdGLeft, 0.0 )
                dpdGRight = np.where( (p > 0) & (p<1) , dpdGRight, 0.0 )
        
                sj = j*Asize
        
                TD[sj + i,sj+np.arange(Asize), sj + i ] += dpdGRight
                TD[sj + i,sj+np.arange(Asize), sj + i-1] += dpdGLeft
        
                TD[sj + i - 1,sj+np.arange(Asize), sj + i ] += -dpdGRight
                TD[sj + i - 1,sj+np.arange(Asize), sj + i-1] += -dpdGLeft
        
            assert np.allclose(TD.sum(axis=0), np.zeros(N*Asize))
        
            return TD
        
        
        
        def Deriv_wealthResidual_G(ans,G,D_L,D,M):
            J = -(Deriv_MakeTransMat_Savings(G.reshape(N,Asize)) * (MakeTransMat_Emp(M) @ D_L).reshape(1,N*Asize,1)).sum(axis = 1)
            J = J[1:] # drop first equation
            return lambda g : g  @ J
        
        def Deriv_wealthResidual_D(ans,G,D_L,D,M):
            J = np.eye(len(D))
            J = J[1:] # drop first equation
            return lambda g : g  @ J
        
        def Deriv_wealthResidual_D_L(ans,G,D_L,D,M):
            J = -MakeTransMat(G.reshape(N,Asize),M)
            J = J[1:] # drop first equation
            return lambda g : g  @ J
        
        def Deriv_wealthResidual_M(ans,G,D_L,D,M):
            TS = MakeTransMat_Savings(G.reshape(N,Asize))
            DTE = np.array([[-1,-delta],[1,delta]])
            J = []
            for e in range(N):
                ofs = np.array((e,e+1)) * Asize
                Je = 0.0
                for elag in range(N):
                    ofslag = elag * Asize + np.arange(Asize)
                    Je +=  DTE[e,elag] *  D_L[ofslag]
        
                J.append(TS[ofs[0]:ofs[1],ofs[0]:ofs[1]] @ Je )
        
            J = np.hstack(J)
            J = -J[1:] # drop first equation
            return lambda g : g  @ J
        
        
        
        defvjp(wealthResidual,Deriv_wealthResidual_G,Deriv_wealthResidual_D_L,Deriv_wealthResidual_D,Deriv_wealthResidual_M)
        
        
        
        # def Agg_C(D_L,Pr,u):
        #     savings = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
        #     assets = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
        #     return Pr.RLag * assets + (1-u)*Pr.employedIncome + u * ben - savings
        
        def AggResidual(Pr,D, u, u_L, R,i_L,i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z, xi_L, xi,epsilon):
            #               1   2  3   4   5   6  7 8
            # Equations for u , R, M, pi, pA, pB, Z xi
            Y = Z * (1-u)
            H = 1-u - (1-delta)*(1-u_L)
            marg_cost = (wbar * (M/Mbar)**zeta + psi *M  - (1-delta)*psi *M_P)/ Z
            # C = Agg_C(D_L,Pr,u)
            return np.hstack((Agg_Assets(D) - B,  # 1  Bond clearing
                              1+i - Rstar * pi**omega * xi,         # 2  mon pol rule
                              R - (1+i_L)/pi,
                              M - (1-u-(1-delta)*(1-u_L))/(u_L + delta*(1-u_L)),  # 3 labor market dynamics
                              pi - theta**(1./(1-mu_epsilon))*(1-(1-theta)*(pA/pB)**(1-mu_epsilon))**(1./(mu_epsilon-1)), # 4 inflation
                              -pA + mu * Y * marg_cost + theta * pi_P**mu_epsilon * pA_P / R, # 5 aux inflation equ 1
                              -pB + Y  + theta * pi_P**(mu_epsilon-1) * pB_P / R, # 6 aux inflation equ 2
                              np.log(Z) - rho*np.log(Z_L)-epsilon[0] ,   # 7 TFP evolution
                              np.log(xi) - rho_xi*np.log(xi_L)-epsilon[1])) # monetary shock evolution
        
        
        def F(X_L,X,X_P,epsilon):
            # Bundle the equations of the model
        
            # Step 1: unpack
            m = N*Asize
            G_L,D_L,Agg_L = X_L[:m], X_L[m:(2*m-1)], X_L[2*m-1:]
            G  ,D  ,Agg   = X[:m]  , X[m:(2*m-1)],   X[2*m-1:]
            G_P,D_P,Agg_P = X_P[:m], X_P[m:(2*m-1)], X_P[2*m-1:]
        
            u_L, R_L, i_L, M_L, pi_L, pA_L, pB_L, Z_L, xi_L = Agg_L
            u, R, i, M, pi, pA, pB, Z, xi = Agg
            u_P, R_P, i_P, M_P, pi_P, pA_P, pB_P, Z_P, xi_P = Agg_P
        
            D_L = np.hstack((1-D_L.sum(), D_L))
            D = np.hstack((1-D.sum(), D))
            D_P = np.hstack((1-D_P.sum(), D_P))
        
            # Step 2: prices
            Pr = Prices(R,M,Z,u,u_L)
            Pr_P = Prices(R_P,M_P,Z_P,u_P,u)
        
            # Step 3: bundle equations
            return np.hstack( (eulerResidual(G,G_P,Pr,Pr_P), wealthResidual(G,D_L,D,Pr.M), AggResidual(Pr,D, u, u_L, R, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z,xi_L,xi,epsilon) ) )
    
        
        #prepare some useful arrays

        # try:
        pB  = (1-ubar)  /(1-theta/Rstar)
        Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
        
        X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
        epsilon_SS = np.zeros(2)
        # Linearizexcept:
        #     xvec = np.empty((1,5*iterations))
        #     xvec.fill(np.nan)e
        AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
        BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
        CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
        EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)
        
        
        
        P, Q = solve(AMat,BMat,CMat,EMat)
        
        
        #simulate
    
        xvec = np.zeros((iterations + 1,nX))
        
        noise = .01*np.random.randn(iterations,2)
        for idx in range(iterations):
            xvec[1 + idx,:] = P@xvec[idx,:] + Q@noise[idx,:]
        xvec = xvec[1:,[-6,-4]].reshape((1,2*iterations))
    
    except:
        xvec = np.empty((1,2*iterations))
        xvec.fill(np.nan)
    
    return torch.tensor(xvec, dtype = torch.float32)

rv = norm()

#vec should be numpy array
def likelihood_reiter(vec, xvalue, ranvar):
    #solve
    # Parameters and setup
    vecin = vec.reshape(-1)
    beta = 0.97
    gamma = 2.0
    rho = vecin[0]#0.95
    rho_xi = vecin[1]#0.80
    zeta = vecin[2]#0.8
    psi = vecin[3]#0.1
    mu = vecin[4]#mu_epsilon / (mu_epsilon - 1)#1.2 
    mu_epsilon = mu/(mu-1)
    theta = vecin[5]#0.75
    omega = vecin[6]#1.5
    ubar = vecin[7]#0.06
    delta = vecin[8]#0.15 # job separation prob
    Mbar = (1-ubar)*delta/(ubar+delta*(1-ubar))
    wbar = 1/mu - delta * psi * Mbar
    B = vecin[9]#0.6
    ben = wbar * 0.5
    History = []
        
    amin = 0.0  # borrowing constraint
    #exogenous transition matrix
    #has form rows = future state, columns = current  state
    N = 2
    nX = 2

    #grid for savings
    Amin, Amax, Asize = amin, 200, 201
    A = np.linspace(Amin**(0.25), Amax**(0.25), Asize)**4
    
    tiledGrid = np.tile(A,(2,1))
    
       # #initialize
    class Prices:
      def __init__(self,R,M=Mbar,Z=1.0,u=ubar,ulag=ubar):
          self.R = R
          self.M = M
          self.w = wbar * (M/Mbar)**zeta
          Y = Z * (1-u)
          H = (1-u) - (1-delta)*(1-ulag)
          d = (Y-psi*M*H)/(1-u) - self.w
          tau = ((self.R-1)*B + ben * u)/(self.w+d)/(1-u)
          self.employedIncome = (1-tau)*(self.w+d)
          self.earnings = np.array([ben,self.employedIncome])
          self.employTrans = np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]])
 
      def tiledEndow(self):
          return np.tile(self.earnings[np.newaxis].T,(1,Asize))
       
    G = 10+tiledGrid
    #R = 1.01
    #Pr0 = Prices(R)
    def interp(x,y,x1):
        N = len(x)
        i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
        xl = x[i-1]
        xr = x[i]
        yl = y[i-1]
        yr = y[i]
        y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
        above = x1 > x[-1]
        below = x1 < x[0]
        y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
        y1 = np.where(below,y[0],y1)
    
        return y1, i
    
    def get_c(G,Pr,CurrentAssets = tiledGrid):
        return np.vstack( [Pr.R * CurrentAssets[i] + Pr.earnings[i] - interp(G[i],A,CurrentAssets[i])[0] for i in range(N)] )
 
    def uPrime(c):
        return c**(-gamma)
    
    def uPrimeInv(up):
        return up**(-1.0/gamma)
    
    def eulerBack(G,Pr,Pr_P):
    # The argument is the savings policy rule in the next period
    # it is parameterized as the a values associated with savings grid for a'
    
        # compute next period's consumption conditional on next period's income
        cp = get_c(G,Pr_P)
        upcp = uPrime(cp)
    
        #compute E(u'(cp))
        # In principle we could do it like this: Eupcp = np.dot(exogTrans.T , uPrime(cp) )
        # But automatic differentiation doesnt work with matrix-matrix multiplication
        # because it isn't obvious how to take the gradient of a function that produces a matrix
        # so we loop over the values instead
        Eupcp = []
        for ip in range(N):
            Eupcp_i = 0.
            for jp in range(N):
                Eupcp_i += Pr_P.employTrans[jp,ip] * upcp[jp]
            Eupcp.append(Eupcp_i)
        Eupcp = np.vstack(Eupcp)
    
        #use  upc = R *  beta*Eupcp to solve for upc
        upc = beta*Pr_P.R*Eupcp
    
        #invert uprime to solve for c
        c = uPrimeInv(upc)
    
    
        #use budget constraint to find previous assets
        # (a' + c - y)/R = a
        a = (tiledGrid + c - Pr.tiledEndow())/ Pr.R
    
        return a, c
    
    def SolveEGM(G,Pr):
        #loop until convergence
        #print("solving for policy rules")
        tol = 1e-15
        test = True
        for it in range(10000):
            a = eulerBack(G,Pr,Pr)[0]
    
            if it % 50 == 0:
                test = np.abs(a-G)/(np.abs(a)+np.abs(G)+tol)
                #print("it = {0}, test = {1}".format(it,test.max()))
                if np.all(test  < tol):
                    break
    
            G = a
    
        return G
    
    def Agg_Assets(D):
        return (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
    
    def MakeTransMat_Savings(G):
       # Rows of T correspond to where we are going, cols correspond to where we are coming from
       T = np.zeros((N*Asize,N*Asize))
       for j in range(N):
           x, i = interp(G[j],A,A)
           p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
           p = np.minimum(np.maximum(p,0.0),1.0)
           sj = j*Asize
           T[sj + i,sj+np.arange(Asize)]= p
           T[sj + i - 1,sj+np.arange(Asize)] = (1.0-p)
   
       #assert np.allclose(T.sum(axis=0), np.ones(N*Asize))
       return T
   
    
    def MakeTransMat_Emp(M):
        return np.kron(np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]]), np.eye(Asize))
    
    
    def MakeTransMat(G, M):
        return MakeTransMat_Savings(G) @ MakeTransMat_Emp(M)
    
    
    def GetStationaryDist(T):
        eval,evec = np.linalg.eig(T)
        i = np.argmin(np.abs(eval-1.0))
        D = np.array(evec[:,i]).flatten()
        assert np.max(np.abs(np.imag(D))) < 1e-6
        D = np.real(D)  # just recasts as float
        return D/D.sum()
    
    def Check_Assets(R):
        Pr = Prices(R)
        G[:] = SolveEGM(G,Pr)
        Assets_implied = Agg_Assets(GetStationaryDist(MakeTransMat(G,Pr.M)))
        History.append([R,Assets_implied])
        #print("Checking R = {0}, assets implied = {1}".format(R,Assets_implied))
        return Assets_implied - B
    
    Rstar = brentq(Check_Assets,.8,1.2,xtol = 1e-14)
    
   
        
    Pr = Prices(Rstar)

    G[:] = SolveEGM(G,Pr)
    
    

    
    D = GetStationaryDist(MakeTransMat(G,Pr.M))
    pB  = (1-ubar)  /(1-theta/Rstar)
    Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
    
    X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
    epsilon_SS = np.zeros(2)
    
    
    #from HANK_EGM import SolveEGM, Pr0, Mbar, ubar, G, A, Asize, N, B, Prices, interp, delta
    
    # note, D tracks end of period states
    # For example, aggregate consumption in t should be computed with D_L shuffled
    # with MakeTransMat_Emp(Pr): AggC(G, Pr, MakeTransMat_Emp(Pr) @ D_L )
    
    
    # rewrite eulerBack to deliver residual function
    def eulerResidual(G,G_P,Pr,Pr_P):
    # The arguments are the savings policy rules in the current and next period
    # parameterized as the a values associated with savings grid for a'
    
        a, c = eulerBack(G_P.reshape(N,Asize),Pr,Pr_P)
        c2 = get_c(G.reshape(N,Asize),Pr,CurrentAssets = a)
        return (c/c2 - 1).reshape(-1)
    
    
    # get residuals of distribution of wealth equations
    @primitive
    def wealthResidual(G,D_L,D,M):
        return (D - MakeTransMat(G.reshape(N,Asize),M) @ D_L)[1:]  # drop one redundant equation (prob distribution sums to 1)
    
    
    
    def Deriv_MakeTransMat_Savings(G):
        # Create a 3-D array TD
        # TD[i,j,k] is the derivative of the transition probability T[i,j] with respect to G[k] (where G has been flattened)
        G = G.reshape(N,Asize)
        TD = np.zeros((N*Asize,N*Asize,G.size))
        for j in range(N):
            x, i = interp(G[j],A,A)
            p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
    
            dpdGLeft = (p-1) / (G[j,i] - G[j,i-1])
            dpdGRight = - p / (G[j,i] - G[j,i-1])
    
            dpdGLeft = np.where( (p > 0) & (p<1) , dpdGLeft, 0.0 )
            dpdGRight = np.where( (p > 0) & (p<1) , dpdGRight, 0.0 )
    
            sj = j*Asize
    
            TD[sj + i,sj+np.arange(Asize), sj + i ] += dpdGRight
            TD[sj + i,sj+np.arange(Asize), sj + i-1] += dpdGLeft
    
            TD[sj + i - 1,sj+np.arange(Asize), sj + i ] += -dpdGRight
            TD[sj + i - 1,sj+np.arange(Asize), sj + i-1] += -dpdGLeft
    
        assert np.allclose(TD.sum(axis=0), np.zeros(N*Asize))
    
        return TD
    
    
    
    def Deriv_wealthResidual_G(ans,G,D_L,D,M):
        J = -(Deriv_MakeTransMat_Savings(G.reshape(N,Asize)) * (MakeTransMat_Emp(M) @ D_L).reshape(1,N*Asize,1)).sum(axis = 1)
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_D(ans,G,D_L,D,M):
        J = np.eye(len(D))
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_D_L(ans,G,D_L,D,M):
        J = -MakeTransMat(G.reshape(N,Asize),M)
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_M(ans,G,D_L,D,M):
        TS = MakeTransMat_Savings(G.reshape(N,Asize))
        DTE = np.array([[-1,-delta],[1,delta]])
        J = []
        for e in range(N):
            ofs = np.array((e,e+1)) * Asize
            Je = 0.0
            for elag in range(N):
                ofslag = elag * Asize + np.arange(Asize)
                Je +=  DTE[e,elag] *  D_L[ofslag]
    
            J.append(TS[ofs[0]:ofs[1],ofs[0]:ofs[1]] @ Je )
    
        J = np.hstack(J)
        J = -J[1:] # drop first equation
        return lambda g : g  @ J
    
    
    
    defvjp(wealthResidual,Deriv_wealthResidual_G,Deriv_wealthResidual_D_L,Deriv_wealthResidual_D,Deriv_wealthResidual_M)
    
    
    
    # def Agg_C(D_L,Pr,u):
    #     savings = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
    #     assets = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
    #     return Pr.RLag * assets + (1-u)*Pr.employedIncome + u * ben - savings
    
    def AggResidual(Pr,D, u, u_L, R,i_L,i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z, xi_L, xi,epsilon):
        #               1   2  3   4   5   6  7 8
        # Equations for u , R, M, pi, pA, pB, Z xi
        Y = Z * (1-u)
        H = 1-u - (1-delta)*(1-u_L)
        marg_cost = (wbar * (M/Mbar)**zeta + psi *M  - (1-delta)*psi *M_P)/ Z
        # C = Agg_C(D_L,Pr,u)
        return np.hstack((Agg_Assets(D) - B,  # 1  Bond clearing
                          1+i - Rstar * pi**omega * xi,         # 2  mon pol rule
                          R - (1+i_L)/pi,
                          M - (1-u-(1-delta)*(1-u_L))/(u_L + delta*(1-u_L)),  # 3 labor market dynamics
                          pi - theta**(1./(1-mu_epsilon))*(1-(1-theta)*(pA/pB)**(1-mu_epsilon))**(1./(mu_epsilon-1)), # 4 inflation
                          -pA + mu * Y * marg_cost + theta * pi_P**mu_epsilon * pA_P / R, # 5 aux inflation equ 1
                          -pB + Y  + theta * pi_P**(mu_epsilon-1) * pB_P / R, # 6 aux inflation equ 2
                          np.log(Z) - rho*np.log(Z_L)-epsilon[0] ,   # 7 TFP evolution
                          np.log(xi) - rho_xi*np.log(xi_L)-epsilon[1])) # monetary shock evolution
    
    
    def F(X_L,X,X_P,epsilon):
        # Bundle the equations of the model
    
        # Step 1: unpack
        m = N*Asize
        G_L,D_L,Agg_L = X_L[:m], X_L[m:(2*m-1)], X_L[2*m-1:]
        G  ,D  ,Agg   = X[:m]  , X[m:(2*m-1)],   X[2*m-1:]
        G_P,D_P,Agg_P = X_P[:m], X_P[m:(2*m-1)], X_P[2*m-1:]
    
        u_L, R_L, i_L, M_L, pi_L, pA_L, pB_L, Z_L, xi_L = Agg_L
        u, R, i, M, pi, pA, pB, Z, xi = Agg
        u_P, R_P, i_P, M_P, pi_P, pA_P, pB_P, Z_P, xi_P = Agg_P
    
        D_L = np.hstack((1-D_L.sum(), D_L))
        D = np.hstack((1-D.sum(), D))
        D_P = np.hstack((1-D_P.sum(), D_P))
    
        # Step 2: prices
        Pr = Prices(R,M,Z,u,u_L)
        Pr_P = Prices(R_P,M_P,Z_P,u_P,u)
    
        # Step 3: bundle equations
        return np.hstack( (eulerResidual(G,G_P,Pr,Pr_P), wealthResidual(G,D_L,D,Pr.M), AggResidual(Pr,D, u, u_L, R, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z,xi_L,xi,epsilon) ) )

    
    #prepare some useful arrays
    iterations = 61
    # try:
    pB  = (1-ubar)  /(1-theta/Rstar)
    Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
    
    X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
    epsilon_SS = np.zeros(2)
    # Linearizexcept:
    #     xvec = np.empty((1,5*iterations))
    #     xvec.fill(np.nan)e
    AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
    BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
    CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
    EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)
    
    
    #print(AMat)
    #print(BMat)
    #print(CMat)
    #print(EMat)
    P, Q = solve(AMat,BMat,CMat,EMat)

    Q1 = Q.reshape(-1,2)
    
    errorcov = Q1@np.transpose(Q1)
    
    
    #simulate

    zscorevec = np.zeros((iterations + 1,2))
    
    #Qinv = np.linalg.pinv(Q)
    
    zscorevec[0,:] = multivariate_normal.pdf(xvalue[0,:], np.zeros(812), errorcov)
    for idx in range(1,iterations):
        zscorevec[idx,:] = multivariate_normal.pdf(xvalue[idx,:]  - P@xvalue[idx-1,:], np.zeros(812), errorcov)
    llike = np.sum(np.log(zscorevec))
    
    return -llike

def likelihood_reiter_po(vec, xvalue):
    #solve
    # Parameters and setup
    vecin = vec.reshape(-1)
    beta = 0.97
    gamma = 2.0
    rho = vecin[0]#0.95
    rho_xi = vecin[1]#0.80
    zeta = vecin[2]#0.8
    psi = vecin[3]#0.1
    mu = vecin[4]#mu_epsilon / (mu_epsilon - 1)#1.2 
    mu_epsilon = mu/(mu-1)
    theta = vecin[5]#0.75
    omega = vecin[6]#1.5
    ubar = vecin[7]#0.06
    delta = vecin[8]#0.15 # job separation prob
    Mbar = (1-ubar)*delta/(ubar+delta*(1-ubar))
    wbar = 1/mu - delta * psi * Mbar
    B = vecin[9]#0.6
    ben = wbar * 0.5
    History = []
        
    amin = 0.0  # borrowing constraint
    #exogenous transition matrix
    #has form rows = future state, columns = current  state
    N = 2
    nX = 812

    #grid for savings
    Amin, Amax, Asize = amin, 200, 25#201
    A = np.linspace(Amin**(0.25), Amax**(0.25), Asize)**4
    
    tiledGrid = np.tile(A,(2,1))
    
       # #initialize
    class Prices:
      def __init__(self,R,M=Mbar,Z=1.0,u=ubar,ulag=ubar):
          self.R = R
          self.M = M
          self.w = wbar * (M/Mbar)**zeta
          Y = Z * (1-u)
          H = (1-u) - (1-delta)*(1-ulag)
          d = (Y-psi*M*H)/(1-u) - self.w
          tau = ((self.R-1)*B + ben * u)/(self.w+d)/(1-u)
          self.employedIncome = (1-tau)*(self.w+d)
          self.earnings = np.array([ben,self.employedIncome])
          self.employTrans = np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]])
 
      def tiledEndow(self):
          return np.tile(self.earnings[np.newaxis].T,(1,Asize))
       
    G = 10+tiledGrid
    #R = 1.01
    #Pr0 = Prices(R)
    def interp(x,y,x1):
        N = len(x)
        i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
        xl = x[i-1]
        xr = x[i]
        yl = y[i-1]
        yr = y[i]
        y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
        above = x1 > x[-1]
        below = x1 < x[0]
        y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
        y1 = np.where(below,y[0],y1)
    
        return y1, i
    
    def get_c(G,Pr,CurrentAssets = tiledGrid):
        return np.vstack( [Pr.R * CurrentAssets[i] + Pr.earnings[i] - interp(G[i],A,CurrentAssets[i])[0] for i in range(N)] )
 
    def uPrime(c):
        return c**(-gamma)
    
    def uPrimeInv(up):
        return up**(-1.0/gamma)
    
    def eulerBack(G,Pr,Pr_P):
    # The argument is the savings policy rule in the next period
    # it is parameterized as the a values associated with savings grid for a'
    
        # compute next period's consumption conditional on next period's income
        cp = get_c(G,Pr_P)
        upcp = uPrime(cp)
    
        #compute E(u'(cp))
        # In principle we could do it like this: Eupcp = np.dot(exogTrans.T , uPrime(cp) )
        # But automatic differentiation doesnt work with matrix-matrix multiplication
        # because it isn't obvious how to take the gradient of a function that produces a matrix
        # so we loop over the values instead
        Eupcp = []
        for ip in range(N):
            Eupcp_i = 0.
            for jp in range(N):
                Eupcp_i += Pr_P.employTrans[jp,ip] * upcp[jp]
            Eupcp.append(Eupcp_i)
        Eupcp = np.vstack(Eupcp)
    
        #use  upc = R *  beta*Eupcp to solve for upc
        upc = beta*Pr_P.R*Eupcp
    
        #invert uprime to solve for c
        c = uPrimeInv(upc)
    
    
        #use budget constraint to find previous assets
        # (a' + c - y)/R = a
        a = (tiledGrid + c - Pr.tiledEndow())/ Pr.R
    
        return a, c
    
    def SolveEGM(G,Pr):
        #loop until convergence
        #print("solving for policy rules")
        tol = 1e-15
        test = True
        for it in range(10000):
            a = eulerBack(G,Pr,Pr)[0]
    
            if it % 50 == 0:
                test = np.abs(a-G)/(np.abs(a)+np.abs(G)+tol)
                #print("it = {0}, test = {1}".format(it,test.max()))
                if np.all(test  < tol):
                    break
    
            G = a
    
        return G
    
    def Agg_Assets(D):
        return (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
    
    def MakeTransMat_Savings(G):
       # Rows of T correspond to where we are going, cols correspond to where we are coming from
       T = np.zeros((N*Asize,N*Asize))
       for j in range(N):
           x, i = interp(G[j],A,A)
           p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
           p = np.minimum(np.maximum(p,0.0),1.0)
           sj = j*Asize
           T[sj + i,sj+np.arange(Asize)]= p
           T[sj + i - 1,sj+np.arange(Asize)] = (1.0-p)
   
       #assert np.allclose(T.sum(axis=0), np.ones(N*Asize))
       return T
   
    
    def MakeTransMat_Emp(M):
        return np.kron(np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]]), np.eye(Asize))
    
    
    def MakeTransMat(G, M):
        return MakeTransMat_Savings(G) @ MakeTransMat_Emp(M)
    
    
    def GetStationaryDist(T):
        eval,evec = np.linalg.eig(T)
        i = np.argmin(np.abs(eval-1.0))
        D = np.array(evec[:,i]).flatten()
        assert np.max(np.abs(np.imag(D))) < 1e-6
        D = np.real(D)  # just recasts as float
        return D/D.sum()
    
    def Check_Assets(R):
        Pr = Prices(R)
        G[:] = SolveEGM(G,Pr)
        Assets_implied = Agg_Assets(GetStationaryDist(MakeTransMat(G,Pr.M)))
        History.append([R,Assets_implied])
        #print("Checking R = {0}, assets implied = {1}".format(R,Assets_implied))
        return Assets_implied - B
    
    Rstar = brentq(Check_Assets,.8,1.2,xtol = 1e-14)
    
   
        
    Pr = Prices(Rstar)

    G[:] = SolveEGM(G,Pr)
    
    

    
    D = GetStationaryDist(MakeTransMat(G,Pr.M))
    pB  = (1-ubar)  /(1-theta/Rstar)
    Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
    
    X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
    epsilon_SS = np.zeros(2)
    
    
    #from HANK_EGM import SolveEGM, Pr0, Mbar, ubar, G, A, Asize, N, B, Prices, interp, delta
    
    # note, D tracks end of period states
    # For example, aggregate consumption in t should be computed with D_L shuffled
    # with MakeTransMat_Emp(Pr): AggC(G, Pr, MakeTransMat_Emp(Pr) @ D_L )
    
    
    # rewrite eulerBack to deliver residual function
    def eulerResidual(G,G_P,Pr,Pr_P):
    # The arguments are the savings policy rules in the current and next period
    # parameterized as the a values associated with savings grid for a'
    
        a, c = eulerBack(G_P.reshape(N,Asize),Pr,Pr_P)
        c2 = get_c(G.reshape(N,Asize),Pr,CurrentAssets = a)
        return (c/c2 - 1).reshape(-1)
    
    
    # get residuals of distribution of wealth equations
    @primitive
    def wealthResidual(G,D_L,D,M):
        return (D - MakeTransMat(G.reshape(N,Asize),M) @ D_L)[1:]  # drop one redundant equation (prob distribution sums to 1)
    
    
    
    def Deriv_MakeTransMat_Savings(G):
        # Create a 3-D array TD
        # TD[i,j,k] is the derivative of the transition probability T[i,j] with respect to G[k] (where G has been flattened)
        G = G.reshape(N,Asize)
        TD = np.zeros((N*Asize,N*Asize,G.size))
        for j in range(N):
            x, i = interp(G[j],A,A)
            p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
    
            dpdGLeft = (p-1) / (G[j,i] - G[j,i-1])
            dpdGRight = - p / (G[j,i] - G[j,i-1])
    
            dpdGLeft = np.where( (p > 0) & (p<1) , dpdGLeft, 0.0 )
            dpdGRight = np.where( (p > 0) & (p<1) , dpdGRight, 0.0 )
    
            sj = j*Asize
    
            TD[sj + i,sj+np.arange(Asize), sj + i ] += dpdGRight
            TD[sj + i,sj+np.arange(Asize), sj + i-1] += dpdGLeft
    
            TD[sj + i - 1,sj+np.arange(Asize), sj + i ] += -dpdGRight
            TD[sj + i - 1,sj+np.arange(Asize), sj + i-1] += -dpdGLeft
    
        assert np.allclose(TD.sum(axis=0), np.zeros(N*Asize))
    
        return TD
    
    
    
    def Deriv_wealthResidual_G(ans,G,D_L,D,M):
        J = -(Deriv_MakeTransMat_Savings(G.reshape(N,Asize)) * (MakeTransMat_Emp(M) @ D_L).reshape(1,N*Asize,1)).sum(axis = 1)
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_D(ans,G,D_L,D,M):
        J = np.eye(len(D))
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_D_L(ans,G,D_L,D,M):
        J = -MakeTransMat(G.reshape(N,Asize),M)
        J = J[1:] # drop first equation
        return lambda g : g  @ J
    
    def Deriv_wealthResidual_M(ans,G,D_L,D,M):
        TS = MakeTransMat_Savings(G.reshape(N,Asize))
        DTE = np.array([[-1,-delta],[1,delta]])
        J = []
        for e in range(N):
            ofs = np.array((e,e+1)) * Asize
            Je = 0.0
            for elag in range(N):
                ofslag = elag * Asize + np.arange(Asize)
                Je +=  DTE[e,elag] *  D_L[ofslag]
    
            J.append(TS[ofs[0]:ofs[1],ofs[0]:ofs[1]] @ Je )
    
        J = np.hstack(J)
        J = -J[1:] # drop first equation
        return lambda g : g  @ J
    
    
    
    defvjp(wealthResidual,Deriv_wealthResidual_G,Deriv_wealthResidual_D_L,Deriv_wealthResidual_D,Deriv_wealthResidual_M)
    
    
    def AggResidual(Pr,D, u, u_L, R,i_L,i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z, xi_L, xi,epsilon):
        #               1   2  3   4   5   6  7 8
        # Equations for u , R, M, pi, pA, pB, Z xi
        Y = Z * (1-u)
        H = 1-u - (1-delta)*(1-u_L)
        marg_cost = (wbar * (M/Mbar)**zeta + psi *M  - (1-delta)*psi *M_P)/ Z
        # C = Agg_C(D_L,Pr,u)
        return np.hstack((Agg_Assets(D) - B,  # 1  Bond clearing
                          1+i - Rstar * pi**omega * xi,         # 2  mon pol rule
                          R - (1+i_L)/pi,
                          M - (1-u-(1-delta)*(1-u_L))/(u_L + delta*(1-u_L)),  # 3 labor market dynamics
                          pi - theta**(1./(1-mu_epsilon))*(1-(1-theta)*(pA/pB)**(1-mu_epsilon))**(1./(mu_epsilon-1)), # 4 inflation
                          -pA + mu * Y * marg_cost + theta * pi_P**mu_epsilon * pA_P / R, # 5 aux inflation equ 1
                          -pB + Y  + theta * pi_P**(mu_epsilon-1) * pB_P / R, # 6 aux inflation equ 2
                          np.log(Z) - rho*np.log(Z_L)-epsilon[0] ,   # 7 TFP evolution
                          np.log(xi) - rho_xi*np.log(xi_L)-epsilon[1])) # monetary shock evolution
    
    
    def F(X_L,X,X_P,epsilon):
        # Bundle the equations of the model
    
        # Step 1: unpack
        m = N*Asize
        G_L,D_L,Agg_L = X_L[:m], X_L[m:(2*m-1)], X_L[2*m-1:]
        G  ,D  ,Agg   = X[:m]  , X[m:(2*m-1)],   X[2*m-1:]
        G_P,D_P,Agg_P = X_P[:m], X_P[m:(2*m-1)], X_P[2*m-1:]
    
        u_L, R_L, i_L, M_L, pi_L, pA_L, pB_L, Z_L, xi_L = Agg_L
        u, R, i, M, pi, pA, pB, Z, xi = Agg
        u_P, R_P, i_P, M_P, pi_P, pA_P, pB_P, Z_P, xi_P = Agg_P
    
        D_L = np.hstack((1-D_L.sum(), D_L))
        D = np.hstack((1-D.sum(), D))
        D_P = np.hstack((1-D_P.sum(), D_P))
    
        # Step 2: prices
        Pr = Prices(R,M,Z,u,u_L)
        Pr_P = Prices(R_P,M_P,Z_P,u_P,u)
    
        # Step 3: bundle equations
        return np.hstack( (eulerResidual(G,G_P,Pr,Pr_P), wealthResidual(G,D_L,D,Pr.M), AggResidual(Pr,D, u, u_L, R, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z,xi_L,xi,epsilon) ) )

    
    #prepare some useful arrays
    iterations = 61
    # try:
    pB  = (1-ubar)  /(1-theta/Rstar)
    Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))
    
    X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
    epsilon_SS = np.zeros(2)
    # Linearizexcept:
    #     xvec = np.empty((1,5*iterations))
    #     xvec.fill(np.nan)e
    AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
    BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
    CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
    EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)
    
    
    #print(AMat)
    #print(BMat)
    #print(CMat)
    #print(EMat)
    P, Q = solve(AMat,BMat,CMat,EMat)

    Q1 = Q.reshape(-1,2)
    
    errorcov = Q1@np.transpose(Q1)
    
    transit = P
    obsmat = np.zeros((2,812))
    obsmat[0,-8] = 1
    obsmat[0,-7] = 1
     
    xvalue = xvalue.reshape(-1,2)
    
    kalfilter = pykalman.KalmanFilter(transition_matrices = transit, 
                                      observation_matrices = obsmat, 
                                      transition_covariance=errorcov)
    
    print(xvalue.shape)
    kf = kalfilter.em(xvalue, n_iter=5)
    
    (filtered_state_means, filtered_state_covariances) = kf.filter(xvalue)
    

    
    observed_means = filtered_state_means[:,[-8,-7]]
    observed_cov = filtered_state_covariances[:,[-8,-7],:]
    observed_cov = observed_cov[:,:,[-8,-7]]
    
    #simulate

    zscorevec = np.zeros((iterations,1))
    
    #Qinv = np.linalg.pinv(Q)
    
    for idx in range(0,iterations):
        #w,_ = np.linalg.eig(observed_cov[idx,:,:])
        zscorevec[idx,:] = multivariate_normal.pdf(xvalue[idx,:], observed_means[idx,:], observed_cov[idx,:,:], allow_singular=True)
    llike = np.sum(np.log(zscorevec))
    
    return -llike

#samples = solve_and_sim_reiter(torch.tensor([[.95,.80,.8,.1,1.2, .75,1.5,.06,.15,.6]], dtype = torch.float32))
 


#print(xdata.shape)
#print(likelihood_reiter(torch.tensor([[.95,.80,.8,.1,1.2, .75,1.5,.06,.15,.6]], dtype = torch.float32), xdata, rv))


#xdata = np.loadtxt('reiter_RBC_po.csv', delimiter = ',').reshape((200,2))
 
def likewrapper(param):
    return likelihood_reiter(param, xdata, rv)

#test1 = likewrapper(np.array([.95,.80,.8,.1,1.2, .75,1.5,.06,.15,.6]))

#grad_like = nd.Gradient(likewrapper, method='forward', order = 1, step=1e-8)
#print(grad_like(np.array([.85,.80,.8,.1,1.2, .75,1.5,.06,.15,.5])))
#arraystart = np.array([.9227,.883,.8009,.1124,1.1727, .7117,1.46431,.684152,.0478929,.571386])
arraystart = np.array([8.951539363058206300e-01, 7.754953573005082257e-01, 1.151456649946240307e-01, 3.009347272592394762e-01, 1.224513259144915045e+00,
                      4.781080604823965130e-01, 1.212845496580873483e+00, 7.723220570197521229e-02, 5.598079884478802948e-02, 5.029026760689112585e-01])#random sample from prior
prior_min = np.array([0.0001,0.0001,0.0001,0.0001,1.0201,0.0001,1.0001,0.0101,0.0101,0.4001])
prior_max =  np.array([1.,1.,3.,.3,1.4,1.,2.,.2,.3,1.])-.0001
array_like = np.zeros((500))

class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights
        
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta

count = 0

def kernelfunct(logpdf, initial_position, iterations, seed1):
    position_old = initial_position
    log_prob = logpdf(initial_position)
    count = 0
    vector = [[] for each in range(iterations)]
    # np.random.seed(seed=seed1)
    # random.seed(seed = 5*seed1-4)
    rng = default_rng(seed1)
    while True:
        # try:
            move_proposals = rng.multivariate_normal(0.*prior_max,np.square(np.diag((prior_max - prior_min)/500)))
            proposal1 = position_old + move_proposals
            # print()
            #print(move_proposals)
            #in1s = input("press enter")
            while np.any(proposal1 > prior_max) or np.any(proposal1 < prior_min):
                move_proposals = rng.multivariate_normal(0.*prior_max,np.square(np.diag((prior_max - prior_min)/500)))
                proposal1 = position_old + move_proposals
            proposal_log_prob = logpdf(proposal1)
            
            log_uniform = np.log(np.random.rand())#initial_position.shape[0], initial_position.shape[1]
            do_accept = log_uniform < -proposal_log_prob + log_prob
            
            position = np.where(do_accept, proposal1, position_old)
            log_prob = np.where(do_accept, proposal_log_prob, log_prob)
            vector[count] = position_old
            position_old = position
            count = count + 1
            if count >= iterations:
                return vector


        
def iter_mp(queue, likewrapper1, arraystart1, iters1, seed2):
    queue.put(kernelfunct(likewrapper1, arraystart1, iters1, seed2))
