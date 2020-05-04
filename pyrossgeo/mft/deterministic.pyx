import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, tanh
from cython.parallel import prange
cdef double PI = 3.14159265359

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M, Nd, wL, tL
        readonly double beta, gI, rW, rT
        readonly np.ndarray rp0, Ni, drpdt, lld, CM, CC, Wo, Tr
    
    def __init__(self, parameters, Nd, M, Ni, Wo, Tr):
        self.beta  = parameters.get('beta')                     # infection rate 
        self.gI    = parameters.get('gI')                      # recovery rate of Ia
        self.rW    = parameters.get('rW')                      # recovery rate of Ia
        self.rT    = parameters.get('rT')                      # recovery rate of Ia

        self.N     = np.sum(Ni)
        self.Nd    = Nd
        self.M     = M
        self.Ni    = np.zeros( self.Nd, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 2*self.Nd, dtype=DTYPE)           # right hand side

        self.Wo   = np.zeros(Wo.shape, dtype=DTYPE)
        self.Tr   = np.zeros(Tr.shape, dtype=DTYPE)
        self.Wo   = Wo
        self.Tr   = Tr

        self.wL   = Wo[0,:].size
        self.tL   = Tr[0,:].size
    
    
    cdef rhs(self, rp, tt):
        cdef: 
            int N=self.N, Nd=self.Nd, i, j
            double beta=self.beta, gI=self.gI, aa, bb
            double [:] S    = rp[0  :Nd]        
            double [:] I    = rp[Nd :2*Nd]       
            double [:] Ni   = self.Ni       
            double [:] X    = self.drpdt        

        for i in prange(Nd, nogil=True):
            bb=0
            aa = beta*I[i]*S[i]/Ni[i]
            X[i]    = -aa
            X[i+Nd] = aa - gI*I[i]
        return

       
    cdef rhs1(self, rp, tt):
        cdef: 
            int N=self.N, Nd=self.Nd, i, j
            double beta=self.beta, gI=self.gI, aa, bb
            double [:] S    = rp[0  :Nd]        
            double [:] I    = rp[Nd :2*Nd]       
            double [:] Ni   = self.Ni       
            double [:] X    = self.drpdt        

        for i in prange(Nd, nogil=True):
            bb=0
            aa = beta*I[i]*S[i]/Ni[i]
            X[i]    = -aa
            X[i+Nd] = aa - gI*I[i]
        return

         
    cdef rhs2(self, rp, tt):
        '''contact in travel'''
        cdef: 
            int N=self.N, Nd=self.Nd, i, j, ii, tL=self.tL
            double beta=self.beta, aa, bb, rT=self.rT
            double gI=self.gI*self.rT
            double [:] S    = rp[0  :Nd]        
            double [:] I    = rp[Nd :2*Nd]       
            double [:] Ni   = self.Ni       
            double [:] X    = self.drpdt        
            double [:,:] Tr = self.Tr        

        for i in range(Nd):#, nogil=True):
            bb=0
            for j in range(tL):
                ii=int(Tr[i,j])
                bb += beta*I[ii]*S[ii]/Ni[ii]
            bb=bb*rT
            print(ii, bb)
            X[i]    = -bb
            X[i+Nd] = bb - gI*I[i]
        return

         
    cdef rhs3(self, rp, tt):
        '''contact at work'''
        cdef: 
            int N=self.N, Nd=self.Nd, i, j, ii, wL=self.wL
            double beta=self.beta, gI=self.gI*self.rW, aa, bb, rW=self.rW
            double [:] S    = rp[0  :Nd]        
            double [:] I    = rp[Nd :2*Nd]       
            double [:] Ni   = self.Ni       
            double [:] X    = self.drpdt        
            double [:,:] Wo = self.Wo        

        for i in range(Nd):#, nogil=True):
            bb=0
            for j in range(wL):
                ii=int(Wo[i,j])
                bb += beta*I[ii]*S[ii]/Ni[ii]
            bb=bb*rW
            X[i]    = -bb
            X[i+Nd] = bb - gI*I[i]
        return

         
    def simulate(self, S0, I0, contactMatrix, Tf, Nf, integrator='odeint', nodeInteraction='False'):
        from scipy.integrate import odeint
        self.CM = contactMatrix(1)
        
        if nodeInteraction=='False':
            def rhs0(rp, t):
                self.rhs1(rp, t)
                return self.drpdt
        else:
            def rhs0(rp, t):
                tI=t%24
                tt=tI*tI
                 
                if 6<=tI<10:
                    self.rhs1(rp, t)
                    dy = self.drpdt
                    self.rhs2(rp, t)
                    dy = dy - self.rT*(36-tt)*(100-tt)*self.drpdt
                elif (10<=tI<17):
                    self.rhs1(rp, t)
                    dy = self.drpdt
                    self.rhs3(rp, t)
                    dy = dy - self.rW*(100-tt)*(289-tt)*self.drpdt
                else:
                    self.rhs1(rp, t)
                    dy = self.drpdt
                return dy
            
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator. 
        u = odeint(rhs0, np.concatenate((S0, I0)), time_points, mxstep=5000000)
        
        data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'beta':self.beta,'gI':self.gI}
        return data
       



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI5R:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    The infected class has 5 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Im: Mortality

    S  ---> E
    E  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, R
    Ih ---> Ic, R
    Ic ---> Im, R
    """
    cdef:
        readonly int N, M, Nd, wL, tL
        readonly double alpha, beta, gE, gIa, gIs, gIh, gIc, fsa, fh, rW, rT
        readonly np.ndarray rp0, Ni, drpdt, CM, FM, CC, Wo, Tr, hh, cc, mm

    def __init__(self, parameters, Nd, M, Ni, Wo, Tr):
        self.alpha = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta  = parameters.get('beta')                     # infection rate
        self.gE    = parameters.get('gE')                       # recovery rate of E class
        self.gIa   = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs   = parameters.get('gIs')                      # recovery rate of Is
        self.gIh   = parameters.get('gIh')                      # recovery rate of Is
        self.gIc   = parameters.get('gIc')                      # recovery rate of Ih
        self.fsa   = parameters.get('fsa')                      # the self-isolation parameter of symptomatics

        self.rW    = parameters.get('rW')                      # recovery rate of Ia
        self.rT    = parameters.get('rT')                      # recovery rate of Ia
        self.Nd    = Nd


        hh         = parameters.get('hh')                       # hospital
        cc         = parameters.get('cc')                       # ICU
        mm         = parameters.get('mm')                       # mortality

        self.Wo   = np.zeros(Wo.shape, dtype=DTYPE)
        self.Tr   = np.zeros(Tr.shape, dtype=DTYPE)
        self.Wo   = Wo
        self.Tr   = Tr

        self.wL   = Wo[0,:].size
        self.tL   = Tr[0,:].size
        self.N     = np.sum(Ni)
        self.M     = M
        self.Ni    = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni    = Ni

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)   # contact matrix C
        self.drpdt = np.zeros( 8*self.M*Nd, dtype=DTYPE)           # right hand side

        self.hh    = np.zeros( self.M, dtype = DTYPE)
        if np.size(hh)==1:
            self.hh = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh= hh
        else:
            print('hh can be a number or an array of size M')

        self.cc    = np.zeros( self.M, dtype = DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        self.mm    = np.zeros( self.M, dtype = DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')

    
    cdef rhs1(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j, n, n1, Nd=self.Nd, M1=self.M*self.Nd, ii
            double alpha=self.alpha, beta=self.beta, aa, bb
            double fsa=self.fsa, fh=self.fh, alphab=1-self.alpha, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1=self.gE*self.alpha, ce2=self.gE*(1-self.alpha)
            double [:] S    = rp[0   :  M1]
            double [:] E    = rp[M1  :2*M1]
            double [:] Ia   = rp[2*M1:3*M1]
            double [:] Is   = rp[3*M1:4*M1]
            double [:] Ih   = rp[4*M1:5*M1]
            double [:] Ic   = rp[5*M1:6*M1]
            double [:] Im   = rp[6*M1:7*M1]
            double [:] Ni   = rp[7*M1:8*M1]
            double [:,:] CM = self.CM
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] X    = self.drpdt

        for n in range(Nd):
            n1=n*M
            for i in range(M):
                bb=0
                for j in range(M):
                     bb += beta*CM[i,j]*(Ia[n1+j]+fsa*Is[n1+j])/Ni[n1+j]
                aa = bb*S[i]
                ii = i+M*n
                X[ii]     = -aa 
                X[ii+M1]   = aa  - gE*E[i+n1]                     
                X[ii+2*M1] = ce1*E[i+n1] - gIa*Ia[i+n1]              
                X[ii+3*M1] = ce2*E[i+n1] - gIs*Is[i+n1]              
                X[ii+4*M1] = gIs*hh[i]*Is[i+n1] - gIh*Ih[i+n1]       
                X[ii+5*M1] = gIh*cc[i]*Ih[i+n1] - gIc*Ic[i+n1]       
                X[ii+6*M1] =  gIc*mm[i]*Ic[i+n1]                   
                X[ii+7*M1] = -gIc*mm[i]*Im[i+n1]           
        return


    cdef rhs2(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j, m1, n1, n2, Nd=self.Nd, M1=self.M*self.Nd, ii, tL=self.tL
            double alpha=self.alpha, beta=self.beta, aa, bb, rT=self.rT
            double fsa=self.fsa, fh=self.fh, alphab=1-self.alpha, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1=self.gE*self.alpha, ce2=self.gE*(1-self.alpha)
            double [:] S    = rp[0   :  M1]
            double [:] E    = rp[M1  :2*M1]
            double [:] Ia   = rp[2*M1:3*M1]
            double [:] Is   = rp[3*M1:4*M1]
            double [:] Ih   = rp[4*M1:5*M1]
            double [:] Ic   = rp[5*M1:6*M1]
            double [:] Im   = rp[6*M1:7*M1]
            double [:] Ni   = rp[7*M1:8*M1]
            double [:,:] CM = self.CM
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] X    = self.drpdt
            double [:,:] Tr = self.Tr        

        for n in range(Nd):
            n1=n*M
            for i in range(M):
                for m1 in range(tL):
                    bb=0
                    n2=int(Tr[n, m1])*M
                    for j in range(M):
                         bb += beta*CM[i,j]*(Ia[n2+j]+fsa*Is[n2+j])/Ni[n2+j]
                aa = rT*bb*S[i]
                ii = i+M*n
                X[ii]     = -aa 
                X[ii+M1]   = aa  - gE*E[i+n1]                     
                X[ii+2*M1] = ce1*E[i+n1] - gIa*Ia[i+n1]              
                X[ii+3*M1] = ce2*E[i+n1] - gIs*Is[i+n1]              
                X[ii+4*M1] = gIs*hh[i]*Is[i+n1] - gIh*Ih[i+n1]       
                X[ii+5*M1] = gIh*cc[i]*Ih[i+n1] - gIc*Ic[i+n1]       
                X[ii+6*M1] =  gIc*mm[i]*Ic[i+n1]                   
                X[ii+7*M1] = -gIc*mm[i]*Im[i+n1]           
        return


    cdef rhs3(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, i, j, n, n1, m1,n2, Nd=self.Nd, M1=self.M*self.Nd, ii, wL=self.wL
            double alpha=self.alpha, beta=self.beta, aa, bb, rW=self.rW
            double fsa=self.fsa, fh=self.fh, alphab=1-self.alpha, gE=self.gE
            double gIs=self.gIs, gIa=self.gIa, gIh=self.gIh, gIc=self.gIh
            double ce1=self.gE*self.alpha, ce2=self.gE*(1-self.alpha)
            double [:] S    = rp[0   :  M1]
            double [:] E    = rp[M1  :2*M1]
            double [:] Ia   = rp[2*M1:3*M1]
            double [:] Is   = rp[3*M1:4*M1]
            double [:] Ih   = rp[4*M1:5*M1]
            double [:] Ic   = rp[5*M1:6*M1]
            double [:] Im   = rp[6*M1:7*M1]
            double [:] Ni   = rp[7*M1:8*M1]
            double [:] hh   = self.hh
            double [:] cc   = self.cc
            double [:] mm   = self.mm
            double [:] X    = self.drpdt
            double [:,:] CM = self.CM
            double [:,:] Wo = self.Wo        

        for n in range(Nd):
            n1=n*M
            for i in range(M):
                for m1 in range(wL):
                    bb=0
                    n2=int(Wo[n, m1])*M
                    for j in range(M):
                         bb += beta*CM[i,j]*(Ia[n2+j]+fsa*Is[n2+j])/Ni[n2+j]
                aa = rW*bb*S[i]
#                print(n, aa, self.Wo[n,:])
                ii = i+M*n
                X[ii]     = -aa 
                X[ii+M1]   = aa  - gE*E[i+n1]                     
                X[ii+2*M1] = ce1*E[i+n1] - gIa*Ia[i+n1]              
                X[ii+3*M1] = ce2*E[i+n1] - gIs*Is[i+n1]              
                X[ii+4*M1] = gIs*hh[i]*Is[i+n1] - gIh*Ih[i+n1]       
                X[ii+5*M1] = gIh*cc[i]*Ih[i+n1] - gIc*Ic[i+n1]       
                X[ii+6*M1] =  gIc*mm[i]*Ic[i+n1]                   
                X[ii+7*M1] = -gIc*mm[i]*Im[i+n1]           
        return


    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0, contactMatrix, Tf, Nf, integrator='odeint', nodeInteraction='False'):
        self.CM = contactMatrix(1)
        
        # contruct the smoothening funtion for control/
        x = np.linspace(0, 26, 26000); xW=3.25
        sT  = 0.5*(1+np.tanh(xW*(x-6)))  - 0.5*(1+np.tanh(xW*(x-8)))
        sT += 0.5*(1+np.tanh(xW*(x-10))) - 0.5*(1+np.tanh(xW*(x-16)))
        sT += 0.5*(1+np.tanh(xW*(x-18))) - 0.5*(1+np.tanh(xW*(x-20)))
        
        if nodeInteraction=='False':
            def rhs0(rp, t):
                self.rhs1(rp, t)
                return self.drpdt
        else:
            def rhs0(rp, t):
                tI=t%24
                if 5<=tI<9:
                    self.rhs1(rp, t)
                    dy = self.drpdt
                    self.rhs2(rp, t)
                    dy = dy + sT[int(tI*1000)]*self.drpdt
                elif (9<=tI<17):
                    self.rhs1(rp, t)
                    dy = self.drpdt
                    self.rhs3(rp, t)
                    dy = dy + sT[int(tI*1000)]*self.drpdt
                elif 17<=tI<21:
                    self.rhs1(rp, t)
                    dy = self.drpdt
                    self.rhs2(rp, t)
                    dy = dy + sT[int(tI*1000)]*self.drpdt
                else:
                    self.rhs1(rp, t)
                    dy = self.drpdt
                return dy
            
        import odespy
        time_points=np.linspace(0, Tf, Nf);  ## intervals at which output is returned by integrator.
        solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
        #solver = odespy.RKF45(rhs0)
        #solver = odespy.RK4(rhs0)
        solver.set_initial_condition(np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, self.Ni)))
        u, time_points = solver.solve(time_points)
        
        data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'beta':self.beta}
        return data


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class simpler_network_SIR_full:
    """
    Susceptible, Infected, Recovered (simpler_network_SIR_full)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int N, M, Nnode
        readonly double alpha, beta, gIa, gIs, fsa, rW, rT, cutoff
        readonly np.ndarray rp0, Ni, Njr, Njw, Ntrans, drpdt, CM, WRM, PWRr, PWRw, FM, CC, FF
        readonly np.ndarray iNjr, iNjw, iNtrans, indexJ, indexI, aveS, aveI #, CSW, CIW
        
    def __init__(self, parameters, M, Nnode, Ni, Njr, Njw, Ntrans):
        self.alpha  = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta   = parameters.get('beta')                     # infection rate
        self.gIa    = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs    = parameters.get('gIs')                      # recovery rate of Is
        self.fsa    = parameters.get('fsa')                      # the self-isolation parameter
        self.rW     = parameters.get('rW')                       # fitting parameter at work
        self.rT     = parameters.get('rT')                       # fitting parameter in trans
        self.cutoff = parameters.get('cutoff')                   # cutoff value of census data
        
        self.N      = np.sum(Ni)
        self.M      = M
        self.Nnode  = Nnode
        self.Ni     = np.zeros( self.M, dtype=DTYPE)             # # people in each age-group
        self.Ni     = Ni
        self.Njr    = np.zeros( self.Nnode, dtype=DTYPE)         # # people living in node i
        self.Njr    = Njr
        self.Njw    = np.zeros( self.Nnode, dtype=DTYPE)         # # people working at node i
        self.Njw    = Njw
        self.Ntrans = np.zeros( (self.Nnode, self.Nnode), dtype=DTYPE)# people comuteing i->j
        self.Ntrans = Ntrans
        self.iNjr    = np.zeros( self.Nnode, dtype=DTYPE)        # inv people living in node i
        self.iNjw    = np.zeros( self.Nnode, dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.Nnode, self.Nnode), dtype=DTYPE)# inv people comuteing i->j

        self.CM    = np.zeros( (self.M, self.M), dtype=DTYPE)        # contact matrix C
        self.WRM   = np.zeros( (self.Nnode, self.Nnode), dtype=DTYPE)# census data matrix WR
        self.PWRr  = np.zeros( (self.Nnode, self.Nnode), dtype=DTYPE)# probability of WRM at w
        self.PWRw  = np.zeros( (self.Nnode, self.Nnode), dtype=DTYPE)# probability of WRM at w
        self.aveS  = np.zeros( self.Nnode, dtype=DTYPE) # average S at i node
        self.aveI  = np.zeros( self.Nnode, dtype=DTYPE) # average I at i node
        #self.CSW   = np.zeros( self.Nnode, dtype=DTYPE) # correalation S and PWRw at i node
        #self.CIW   = np.zeros( self.Nnode, dtype=DTYPE) # correalation I and PWRw at i node
        self.FF    = np.zeros( self.Nnode, dtype=DTYPE)
        self.FM    = np.zeros( self.M, dtype = DTYPE)                # seed function F
        self.drpdt = np.zeros( 3*self.Nnode, dtype=DTYPE)            # right hand side
        self.indexJ= np.zeros( (self.Nnode, self.Nnode + 1), dtype=int)
        self.indexI= np.zeros( (self.Nnode, self.Nnode + 1), dtype=int)

    cdef rhs(self, rp, tt):
        cdef:
            int N=self.N, M=self.M, Nnode=self.Nnode, i, j
            long ii, jj
            double alpha=self.alpha, beta=self.beta, gIa=self.gIa, aa, bb, cc, t_p_24
            double fsa=self.fsa, alphab=1-self.alpha, gIs=self.gIs
            double aveSi, aveIi, aveIS, aveITs
            double rW=self.rW, rT=self.rT
            #double [:,:] S      = rp[0:M, 0      :Nnode  ]
            #double [:,:] Ia     = rp[0:M, Nnode  :2*Nnode]
            #double [:,:] Is     = rp[0:M, 2*Nnode:3*Nnode]
            double [:] S        = rp[0      :Nnode  ]
            double [:] Ia       = rp[Nnode  :2*Nnode]
            double [:] Is       = rp[2*Nnode:3*Nnode]
            double [:] Ni       = self.Ni
            double [:]   Njr    = self.Njr
            double [:]   Njw    = self.Njw
            double [:,:] Ntrans = self.Ntrans
            double [:]   iNjr    = self.iNjr
            double [:]   iNjw    = self.iNjw
            double [:,:] iNtrans = self.iNtrans
            #double [:,:] CM     = self.CM
            double [:,:] WRM    = self.WRM
            double [:,:] PWRr   = self.PWRr
            double [:,:] PWRw   = self.PWRw
            double [:]   aveS   = self.aveS
            double [:]   aveI   = self.aveI
            #double [:]   CSW    = self.CSW
            #double [:]   CIW    = self.CIW
            double [:]   FF     = self.FF
            double [:]   FM     = self.FM
            double [:]   X      = self.drpdt
            long [:,:]   indexJ = self.indexJ
            long [:,:]   indexI = self.indexI
            #long [:]     EE     = self.EE

        t_p_24 = tt%24
        if t_p_24 < 8.0 and t_p_24 > 18.0: #HOME
        #if True: #HOME
            for i in range(Nnode):
                if S[i] > 0.0:
                    aa = beta*(Ia[i]+fsa*Is[i])*S[i]*iNjr[i]
                    X[i]         = -aa #- FM[0]
                    X[i+Nnode]   = alpha *aa - gIa*Ia[i]# + alpha * FM[0]
                    X[i+2*Nnode] = alphab*aa - gIs*Is[i]# + alphab* FM[0]
                else:
                    X[i+Nnode]   = -gIa*Ia[i]# + alpha * FM[0]
                    X[i+2*Nnode] = -gIs*Is[i]# + alphab* FM[0]
        elif t_p_24  > 9.0 and t_p_24  < 17.0: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            for i in range(Nnode):
                #EE = indexJ[i]
                aa      = 0.0
                bb      = 0.0
                aveS[i] = 0.0
                aveI[i] = 0.0
                #CSW[i] = 0.0
                #CIW[i] = 0.0
                for j in range(1, indexJ[i][0] + 1):
                #for j in range(Nnode):
                    jj = indexJ[i][j]
                    #jj = j
                    aveS[i] += PWRr[i,jj]*S[jj]
                    aveI[i] += PWRr[i,jj]*(Ia[jj]+fsa*Is[jj])
                    #CSW[i] += PWRw[i,jj]*PWRr[i,jj]*S[jj]
                    #CIW[i] += PWRw[i,jj]*PWRr[i,jj]*(Ia[jj]+fsa*Is[jj])
                    #bb += PWRw[i,jj]*PWRw[i,jj]
                #aveS[i] = round(aveS[i])#XX
                #aveI[i] = round(aveI[i])#XX
                FF[i] = rW*beta*aveS[i]*aveI[i]*iNjw[i]

                X[i]         = 0.0
                X[i+Nnode]   = 0.0
                X[i+2*Nnode] = 0.0

            for i in range(Nnode):
                #if Ia[i]+fsa*Is[i] > 0.0:
                X[i+Nnode]   = -gIa*Ia[i]# + alpha * FM[0]
                X[i+2*Nnode] = -gIs*Is[i]# + alphab* FM[0]
                #bb = rW*beta*(Ia[i]+fsa*Is[i])
                #cc += PWRr[i,0]*aveS[0]*iNjw[0]
                for j in range(1, indexI[i][0] + 1):
                #for j in range(Nnode):
                    ii     = indexI[i][j]
                    #ii = j
                    #aa = rW*beta*PWRw[ii,i]*aveS[ii]*aveI[ii]*iNjw[ii]
                    aa = PWRw[ii,i]*FF[ii]
                    X[i]         += -aa# - FM[0]
                    X[i+Nnode]   += alpha *aa# + alpha * FM[0]
                    X[i+2*Nnode] += alphab*aa# + alphab* FM[0]

        else: #TRANS
            #print("TIME_in_TRANS", tt)
            for i in range(Nnode):
                aa     = 0.0
                bb     = 0.0
                aveS[i] = 0.0
                aveIS  = 0.0
                aveITs = 0.0
                if Njw[i] != 0.0:
                    for j in range(1, indexJ[i][0] + 1):
                        jj = indexJ[i][j]
                        cc = Ia[jj]+fsa*Is[jj]
                        bb = PWRr[i,jj]*cc*iNtrans[i,jj]
                        aveS[i]  += PWRr[i,jj]*S[jj]
                        ##aveI   += PWRw[i,jj]*cc
                        
                        aveIS  += bb*PWRr[i,jj]*S[jj]
                        aveITs += bb*(Ntrans[i,jj] - WRM[i,jj])*iNjw[i]#should be multiply aveS
                        
                        #bb += PWRw[i,jj]*(cc*iNtrans[i,jj])*(PWRw[i,jj]*S[jj] + (Ntrans[i,jj] - WRM[i,jj])*aveS*iNjw[i])
                    bb = aveIS + aveITs*aveS[i]
                aa = rT*beta*bb
                FF[i] = aa

                X[i]         = 0.0
                X[i+Nnode]   = 0.0
                X[i+2*Nnode] = 0.0
                
            for i in range(Nnode):
                X[i+Nnode]   = -gIa*Ia[i]# + alpha * FM[0]
                X[i+2*Nnode] = -gIs*Is[i]# + alphab* FM[0]
                for j in range(1, indexI[i][0] + 1):
                    #for j in range(Nnode):
                    ii     = indexI[i][j]
                    #if S[jj] > 0.0:
                    aa = PWRw[ii,i]*FF[ii]
                    X[i]         += -aa# - FM[0]
                    X[i+Nnode]   += alpha *aa# + alpha * FM[0]
                    X[i+2*Nnode] += alphab*aa# + alphab* FM[0]
        return


    def simulate(self, S0, Ia0, Is0, contactMatrix, workResidenceMatrix, Tf, Nf, Ti=0, integrator='odeint', seedRate=None):
        from scipy.integrate import solve_ivp
        from scipy.integrate import odeint

        print(self.rW, self.rT, self.cutoff)
        self.CM  = contactMatrix(0.0)
        self.WRM = workResidenceMatrix(0.0)
        for i in range(self.Nnode):
            self.iNjr[i] = 1.0/self.Njr[i]
                
            if self.Njw[i] != 0:
                self.iNjw[i] = 1.0/self.Njw[i]
            else:
                self.iNjw[i] = 0.0
                
            index_j = 0
            index_i = 0
            for j in range(self.Nnode):
                self.PWRr[i][j] = self.WRM[i][j]/self.Njr[j]
                
                if self.Njw[i] != 0:
                    self.PWRw[i][j] = self.WRM[i][j]/self.Njw[i]
                else:
                    self.PWRw[i][j] = 0.0
                    
                if self.Ntrans[i][j] != 0:
                    self.iNtrans[i][j] = 1.0/self.Ntrans[i][j]
                else:
                    self.iNtrans[i][j] = 0.0
                    
                if self.WRM[i][j] > self.cutoff:
                    self.indexJ[i][index_j + 1] = j
                    index_j += 1
                elif self.WRM[i][j] != 0:
                    print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(self.WRM[i,j]) + '\n')

                if self.WRM[j][i] > self.cutoff:
                    self.indexI[i][index_i + 1] = j
                    index_i += 1
                elif self.WRM[j][i] != 0:
                    print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(self.WRM[j,i]) + '\n')
                    
            self.indexJ[i][0] = index_j
            self.indexI[i][0] = index_i
        
        def rhs0(t, rp):
            #self.CM  = contactMatrix(t)
            #self.WRM = workResidenceMatrix(t)
            if None != seedRate :
                self.FM = seedRate(t)
            else :
                self.FM = np.zeros( self.M, dtype = DTYPE)
            print("Time", t)
            self.rhs(rp, t)
            return self.drpdt

        if integrator=='odeint':
            #time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            #u = odeint(rhs0, np.concatenate((S0, Ia0, Is0)), time_points, mxstep=100000)
            time_points=np.linspace(Ti, Tf);  ## intervals at which output is returned by integrator.
            u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, Ia0, Is0)), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, max_step=6.0)
            #u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, Ia0, Is0)), method='RK45')
            #u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, Ia0, Is0)), method='BDF')
            
            data={'X':u.y, 't':u.t, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        else:
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
            #solver = odespy.RKF45(rhs0)
            #solver = odespy.RK4(rhs0)
            solver.set_initial_condition(np.concatenate((S0, Ia0, Is0)))
            u, time_points = solver.solve(time_points)

            data={'X':u, 't':time_points, 'N':self.N, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        return data
