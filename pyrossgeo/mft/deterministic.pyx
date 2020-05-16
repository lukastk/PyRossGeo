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
cdef class SIR_1storder:
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
cdef class SEI5R_1storder:
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
cdef class SIR:
    """
    Susceptible, Infected, Recovered (SIR)
    Ia: asymptomatic
    Is: symptomatic
    """
    cdef:
        readonly int highSpeed
        readonly int M, Nnode, max_route_num, t_divid_100, t_old
        readonly double alpha, beta, gIa, gIs, fsa, rW, rT, cutoff
        readonly np.ndarray rp0, Nh, Nw, Ntrans, drpdt, CMh, CMw, CMt, Dnm, distances
        readonly np.ndarray route_index, route_ratio
        readonly np.ndarray PWRh, PWRw, PP, FF, II, IT
        readonly np.ndarray iNh, iNw, iNtrans, indexJ, indexI, indexAGJ, aveS, aveI, Lambda
        
    def __init__(self, parameters, M, Nnode):
        self.alpha  = parameters.get('alpha')                    # fraction of asymptomatic infectives
        self.beta   = parameters.get('beta')                     # infection rate
        self.gIa    = parameters.get('gIa')                      # recovery rate of Ia
        self.gIs    = parameters.get('gIs')                      # recovery rate of Is
        self.fsa    = parameters.get('fsa')                      # the self-isolation parameter
        self.rW     = parameters.get('rW')                       # fitting parameter at work
        self.rT     = parameters.get('rT')                       # fitting parameter in trans
        self.cutoff = parameters.get('cutoff')                   # cutoff value of census data
        self.highSpeed = parameters.get('highspeed')             # flag of hispeed calculation        
        self.M      = M
        self.Nnode  = Nnode
        self.Nh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people living in node i
        self.Nw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people working at node i
        self.Ntrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# people comuteing i->j
        self.iNh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# inv people comuteing i->j

        self.CMh   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in HOME
        self.CMw   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in WORK
        self.CMt   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in TRANS
        self.Dnm   = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# census data matrix WR
        self.PWRh  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.PWRw  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.aveS  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average S at i node
        self.aveI  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average I at i node
        self.Lambda= np.zeros( (self.M, self.Nnode), dtype=DTYPE)     # effective infection rate
        self.drpdt = np.zeros( 3*self.Nnode*self.M, dtype=DTYPE)      # right hand side
        self.indexJ= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        self.indexI= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        #self.indexAGJ= np.zeros( (self.M, self.M, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij*Dnm_gam_ij at specifi alp, gam and i
        self.distances = np.zeros((self.Nnode,self.Nnode), DTYPE)
        self.II = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.IT = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.FF = np.zeros( (self.M, self.Nnode), dtype=DTYPE)               # Working memory
        self.PP = np.zeros( (self.Nnode, self.Nnode), dtype=np.int32)        # Working memory

    cdef rhs(self, rp, tt):
        cdef:
            int highSpeed=self.highSpeed
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, t_divid_100=int(tt/100)
            unsigned short i, j, k, alp, gam, age_id, ii, jj
            unsigned long t_i, t_j
            double alpha=self.alpha, beta=self.beta, gIa=self.gIa
            double aa=0.0, bb=0.0, cc=0.0, t_p_24 = tt%24
            double fsa=self.fsa, alphab=1-self.alpha, gIs=self.gIs
            double rW=self.rW, rT=self.rT
            double [:] S           = rp[0   :M1  ]
            double [:] Ia          = rp[M1  :2*M1]
            double [:] Is          = rp[2*M1:3*M1]
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:,:]   CMh     = self.CMh
            double [:,:]   CMw     = self.CMw
            double [:,:]   CMt     = self.CMt
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   aveS    = self.aveS
            double [:,:]   aveI    = self.aveI
            double [:,:,:]   II    = self.II
            double [:,:,:]   IT    = self.IT
            double [:,:]   FF      = self.FF
            double [:,:]   Lambda  = self.Lambda
            double [:]     X       = self.drpdt
            unsigned short [:,:,:]   indexJ    = self.indexJ
            unsigned short [:,:,:]   indexI    = self.indexI
            #unsigned short [:,:,:,:] indexAGJ  = self.indexAGJ
            unsigned short [:,:,:] route_index = self.route_index
            float [:,:,:]          route_ratio = self.route_ratio

        #t_divid_100 = int(tt/100)
        if t_divid_100 > self.t_old:
            print('Time', tt)
            self.t_old = t_divid_100
        #t_p_24 = tt%24
        if t_p_24 < 8.0 or t_p_24 > 18.0: #HOME
        #if True: #HOME
            #print("TIME_in_HOME", t_p_24)
            for i in range(Nnode):
                for alp in range(M):
                    t_i = alp*Nnode + i
                    aa = 0.0
                    if S[t_i] > 0.0:
                        bb = 0.0
                        for gam in range(M):
                            t_j = gam*Nnode + i
                            cc = Ia[t_j] + Is[t_j]
                            bb += CMh[alp,gam]*cc*iNh[gam,i]
                        aa = beta*bb*S[t_i]
                    X[t_i]        = -aa
                    X[t_i + M1]   = alpha *aa
                    X[t_i + 2*M1] = alphab*aa

                    if Ia[t_i] > 0.0:
                        X[t_i + M1]   += -gIa*Ia[t_i]
                    if Is[t_i] > 0.0:
                        X[t_i + 2*M1] += -gIs*Is[t_i]

        elif t_p_24  > 9.0 and t_p_24  < 17.0: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            if True: #distinguishable
                for i in range(Nnode):
                    for alp in range(M):
                        aveI[alp,i] = 0.0
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj = indexJ[alp,i,j]
                            t_j = alp*Nnode + jj
                            aveI[alp,i] += PWRh[alp,i,jj]*(Ia[t_j] + fsa*Is[t_j])
                    for alp in range(M):
                        Lambda[alp, i] = 0.0
                        for gam in range(M):
                            Lambda[alp, i] += CMw[alp,gam]*aveI[gam,i]*iNw[gam,i]
                        Lambda[alp, i] = rW*beta*Lambda[alp,i]

                        t_i = alp*Nnode + i
                            
                        X[t_i]        = 0.0
                        X[t_i + M1]   = 0.0
                        X[t_i + 2*M1] = 0.0
                            
                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        X[t_i + M1]   = -gIa*Ia[t_i]
                        X[t_i + 2*M1] = -gIs*Is[t_i]
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            if S[t_i] > 0.0:
                                aa = Lambda[alp,ii]*PWRh[alp,ii,i]*S[t_i]
                                X[t_i]        += -aa
                                X[t_i + M1]   += alpha *aa
                                X[t_i + 2*M1] += alphab*aa

        else: #TRANS
            #print("TIME_in_TRANS", t_p_24)
            if highSpeed == 1: # using averaged \hat{I}_{ij}
                for i in range(Nnode):
                    aveI[M,i] = 0.0
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aa = Ia[t_i] + Is[t_i]
                        #aveI[alp,i] = aa
                        aveI[M,i]  += aa
                        for j in range(Nnode):
                            II[M,i,j] = 0.0
                            IT[M,i,j] = 0.0
                        
                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        if route_index[i,jj,0] >= 2:
                            for k in range(1, route_index[i,jj,0]):
                                II[M,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[M,jj]

                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        for k in range(1, route_index[i,jj,0]):
                            IT[M,i,jj] += II[M,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        
                        cc = 0.0
                        for gam in range(M):
                            cc += CMt[alp,gam]

                        bb = 0.0
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            bb += IT[M,ii,i]*PWRh[alp,ii,i]*iNtrans[M,ii,i] # add i->j
                        bb += PWRh[M,i,i]*aveI[M,i]*PWRh[alp,i,i]*iNw[M,i] # add i->i
                        aa = rT*beta*cc*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = alpha *aa
                        X[t_i + 2*M1] = alphab*aa
                    
                        if Ia[t_i] + Is[t_i] > 0.0:
                            X[t_i + M1]   += -gIa*Ia[t_i]
                            X[t_i + 2*M1] += -gIs*Is[t_i]

            elif True: # using accurate \hat{I}_{ij}
                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aveI[alp,i] = Ia[t_i] + Is[t_i]
                    for j in range(Nnode):
                        for alp in range(M):
                            II[alp,i,j] = 0.0
                            IT[alp,i,j] = 0.0
                        
                for i in range(Nnode):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[alp,jj]

                for i in range(Nnode):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            for k in range(1, route_index[i,jj,0]):
                                IT[alp,i,jj] += II[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        bb = 0.0
                        for gam in range(M):
                            for j in range(1, indexI[alp,i,0] + 1):
                                ii = indexI[alp,i,j]
                                bb += CMt[alp,gam]*IT[gam,ii,i]*PWRh[alp,ii,i]*iNtrans[gam,ii,i]
                            t_j = gam*Nnode + i
                            bb += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_j] + Is[t_j])*iNw[gam,i]*PWRh[alp,i,i]
                        aa = rT*beta*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = alpha *aa
                        X[t_i + 2*M1] = alphab*aa
                    
                        if Ia[t_i] + Is[t_i] > 0.0:
                            X[t_i + M1]   += -gIa*Ia[t_i]
                            X[t_i + 2*M1] += -gIs*Is[t_i]

        return

    cdef prepare_fixed_variable(self, travel_restriction):
        cdef:
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, max_route_num
            unsigned short i, j, k, alp, index_i, index_j, index_agj, ii, count
            double cutoff=self.cutoff, cij, ccij, t_restriction=travel_restriction
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:,:] distances = self.distances
            unsigned short [:,:,:]   indexJ  = self.indexJ
            unsigned short [:,:,:]   indexI  = self.indexI
            #unsigned short [:,:,:,:] indexAGJ= self.indexAGJ
            unsigned short [:] route           # working memory
            double [:,:,:] C_Dnm   = self.IT
            unsigned short [:,:,:] route_index # the list of node in the route i -> j
            float [:,:,:] route_ratio          # the list of the distance ratio in the route i -> j
            double [:,:]  dij  = self.II[0]    # the distance between node i and j
            int [:,:]     pred = self.PP       # predecessor node belong the route i -> j

        #travel restriction
        for alp in range(M+1):
            for i in range(Nnode):
                for j in range(Nnode):
                    if i != j:
                        cij = Dnm[alp,i,j]
                        ccij = round(cij*t_restriction)
                        Dnm[alp,i,j] -= ccij
                        Dnm[alp,j,j] += ccij
    
        #cutoff
        cdef int nonzero_element = 0
        cdef double cutoff_total = 0.0
        C_Dnm = Dnm.copy()
        for alp in range(M+1):
            for i in range(Nnode):
                for j in range(Nnode):
                    cij = C_Dnm[alp,i,j]
                    if i != j:
                        if int(cij) > int(cutoff):
                            if alp != M:
                                nonzero_element += 1
                        else:
                            Dnm[alp,i,j] = 0.0
                            Dnm[alp,j,j] += cij
                    if alp != M:
                        if int(cij) > int(cutoff):
                            cutoff_total += cij
        print("Nonzero element " + str(nonzero_element) + '/' + str(M1**2) + ' ' + str(cutoff_total))

        for alp in range(M):
            for i in range(Nnode):
                Nh[alp,i] = 0.0 ## N^{H}_i residence in Node i and age gourp alp
                Nw[alp,i] = 0.0 ## N^{w}_i working in Node i and age group alp
                for j in range(Nnode):
                    Nh[alp,i] += Dnm[alp,j,i]
                    Nw[alp,i] += Dnm[alp,i,j]
                
        for alp in range(M):
            for i in range(Nnode):
                Nh[M,i] += Nh[alp,i] ## N^{H}_i residence in Node i
                Nw[M,i] += Nw[alp,i] ## N^{w}_i working in Node i

        #Generating the Ntrans from route and predecessor
        cdef N_Matrix = np.zeros((M,Nnode,Nnode), dtype=DTYPE)
        cdef route_num = np.zeros(Nnode*Nnode, dtype=int)
        cdef int total_route_index = 0
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:
                    #route = get_path(i, j, p)
                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1

                    count = 0
                    for k in range(len(route) - 1):
                        for alp in range(M):
                            N_Matrix[alp,route[k + 1],route[k]] += Dnm[alp,j,i]
                            count += 1
                    total_route_index += len(route)
                    route_num[i*Nnode + j] = len(route)
        route_num.sort() # should be improved
        max_route_num = route_num[Nnode**2 - 1]
        print("Max route number", route_num[0], max_route_num)
        print("Total index in all route", total_route_index, 1.0*total_route_index/Nnode**2)

        self.route_index = np.zeros((Nnode,Nnode,max_route_num + 1), dtype=np.uint16)
        self.route_ratio = np.zeros((Nnode,Nnode,max_route_num), dtype=np.float32)
        route_index = self.route_index
        route_ratio = self.route_ratio
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:

                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        
                    for k in range(0, len(route) - 1):
                        # N^{t}_{ji} the effective number of the people using the route i->j at age group alpha
                        for alp in range(M):
                            Ntrans[alp,j,i] += N_Matrix[alp,route[k + 1],route[k]]*dij[route[k + 1],route[k]]/dij[j,i]
                        route_index[j,i,k + 1] = np.uint16(route[k])
                        route_ratio[j,i,k + 1] = np.float32(dij[route[k + 1],route[k]]/dij[j,i])
                    route_ratio[j,i,0] = np.float32(len(route) - 1)
                    route_index[j,i,len(route)] = np.uint16(route[len(route)-1])
                    route_index[j,i,0] = np.uint16(len(route))
                else:
                    for alp in range(M):
                        Ntrans[alp,j,i] = Dnm[alp,j,i]

        for alp in range(M):
            for j in range(Nnode):
                for i in range(Nnode):
                    Ntrans[M,j,i] += Ntrans[alp,j,i] ## N^{t}_{ji} the effective number of the people using the route i->j

        for alp in range(M+1):
            for i in range(Nnode):
                if Nh[alp,i] != 0:
                    iNh[alp,i] = 1.0/Nh[alp,i]
                else:
                    iNh[alp,i] = 0.0
                
                if Nw[alp,i] != 0:
                    iNw[alp,i] = 1.0/Nw[alp,i]
                else:
                    iNw[alp, i] = 0.0
                
                index_j = 0
                index_i = 0
                for j in range(Nnode):
                    if Nh[alp,j] != 0:
                        PWRh[alp,i,j] = Dnm[alp,i,j]/Nh[alp,j]
                    else:
                        PWRh[alp,i,j] = 0.0
                
                    if Nw[alp,i] != 0:
                        PWRw[alp,i,j] = Dnm[alp,i,j]/Nw[alp,i]
                    else:
                        PWRw[alp,i,j] = 0.0
                    
                    if Ntrans[alp,i,j] != 0:
                        iNtrans[alp,i,j] = 1.0/Ntrans[alp,i,j]
                    else:
                        iNtrans[alp,i,j] = 0.0
                    
                    if int(Dnm[alp,i,j]) > cutoff or i == j:
                        indexJ[alp,i,index_j + 1] = j
                        index_j += 1
                    elif int(Dnm[alp,i,j]) != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,i,j]) + '\n')

                    if int(Dnm[alp,j,i]) > cutoff or i == j:
                        indexI[alp,i,index_i + 1] = j
                        index_i += 1
                    elif int(Dnm[alp,j,i]) != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,j,i]) + '\n')
                    
                indexJ[alp,i,0] = index_j
                indexI[alp,i,0] = index_i
                
        #for alp in range(M):
        #    for gam in range(M):
        #        for i in range(Nnode):
        #            index_agj = 0
        #            for j in range(Nnode):
        #                if Dnm[alp,i,j] > cutoff and Dnm[gam,i,j] > cutoff:
        #                    indexAGJ[alp,gam,i,index_agj + 1] = j
        #                    index_agj += 1
        #            indexAGJ[alp,gam,i,0] = index_agj


    def simulate(self, S0, Ia0, Is0, contactMatrix, workResidenceMatrix, distanceMatrix, travel_restriction, Tf, Nf, Ti=0, integrator='solve_ivp'):
        from scipy.integrate import solve_ivp
        from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
        from scipy.sparse import csr_matrix

        print('travel restriction', travel_restriction)
        print('cutoff', self.cutoff)
        print('highspeed', self.highSpeed)
        self.CMh  = contactMatrix(6.0)
        self.CMt  = contactMatrix(8.5)
        self.CMw  = contactMatrix(12.0)
        self.Dnm = workResidenceMatrix(0.0)
        self.distances = distanceMatrix(0.0)

        print('#Start finding the shortest path between each node')
        self.II[0], self.PP = shortest_path(distanceMatrix(0.0), return_predecessors=True)
        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(travel_restriction)
        print('#Calculation Start')
        
        def rhs0(t, rp):
            self.rhs(rp, t)
            return self.drpdt

        if integrator=='solve_ivp':
            time_points=np.linspace(Ti, Tf);  ## intervals at which output is returned by integrator.
            time_step = 1.0*Tf/Nf
            u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, Ia0, Is0)), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
            
            data={'X':u.y, 't':u.t, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        else:
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
            solver.set_initial_condition(np.concatenate((S0, Ia0, Is0)))
            u, time_points = solver.solve(time_points)

            data={'X':u, 't':time_points, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

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
        readonly int highSpeed
        readonly int M, Nnode, max_route_num, t_divid_100, t_old, ir
        readonly double beta, gE, gIa, gIs, gIh, gIc, fsa, fh, rW, rT, cutoff
        readonly np.ndarray alpha, hh, cc, mm, alphab
        readonly np.ndarray rp0, Nh, Nw, Ntrans, drpdt, CMh, CMw, CMt, Dnm, distances, RM
        readonly np.ndarray route_index, route_ratio
        readonly np.ndarray PWRh, PWRw, PP, FF, II, IT, TT
        readonly np.ndarray iNh, iNw, iNtrans, indexJ, indexI, indexAGJ, aveS, aveI, Lambda
        
    def __init__(self, parameters, M, Nnode):
        alpha       = parameters.get('alpha')             # fraction of asymptomatic infectives
        self.beta   = parameters.get('beta')              # infection rate
        self.gE     = parameters.get('gE')                # recovery rate of E class
        self.gIa    = parameters.get('gIa')               # recovery rate of Ia
        self.gIs    = parameters.get('gIs')               # recovery rate of Is
        self.gIh    = parameters.get('gIh')               # recovery rate of Ih
        self.gIc    = parameters.get('gIc')               # recovery rate of Ic
        self.fsa    = parameters.get('fsa')               # the self-isolation parameter
        self.fh     = parameters.get('fh')                # the hospital-isolation parameter
        self.rW     = parameters.get('rW')                # fitting parameter at work
        self.rT     = parameters.get('rT')                # fitting parameter in trans
        hh          = parameters.get('hh')                # hospital
        cc          = parameters.get('cc')                # ICU
        mm          = parameters.get('mm')                # mortality
        self.cutoff = parameters.get('cutoff')            # cutoff value of census data
        self.highSpeed = parameters.get('highspeed')      # flag of hispeed calculation        
        self.M      = M
        self.Nnode  = Nnode
        self.Nh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people living in node i
        self.Nw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people working at node i
        self.Ntrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# people comuteing i->j
        self.iNh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# inv people comuteing i->j

        self.RM    = np.random.rand(1000*self.M*self.Nnode) # random matrix 
        self.CMh   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in HOME
        self.CMw   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in WORK
        self.CMt   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in TRANS
        self.Dnm   = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# census data matrix WR
        self.PWRh  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.PWRw  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.aveS  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average S at i node
        self.aveI  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average I at i node
        self.Lambda= np.zeros( (self.M, self.Nnode), dtype=DTYPE)     # effective infection rate
        self.drpdt = np.zeros( 8*self.Nnode*self.M, dtype=DTYPE)      # right hand side
        self.indexJ= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        self.indexI= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        #self.indexAGJ= np.zeros( (self.M, self.M, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij*Dnm_gam_ij at specifi alp, gam and i
        self.distances = np.zeros((self.Nnode,self.Nnode), DTYPE)
        self.II = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.IT = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.TT = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.FF = np.zeros( (self.M, self.Nnode), dtype=DTYPE)               # Working memory
        self.PP = np.zeros( (self.Nnode, self.Nnode), dtype=np.int32)        # Working memory

        self.alpha  = np.zeros( self.M, dtype=DTYPE)
        self.alphab = np.zeros( self.M, dtype=DTYPE)
        if np.size(alpha)==1:
            self.alpha  = alpha*np.ones(M)
            self.alphab = (1.0 - alpha)*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha  = alpha
            self.alphab = np.ones(M) - self.alpha
            
        else:
            print('alpha can be a number or an array of size M')
            
        self.hh  = np.zeros( self.M, dtype=DTYPE)
        if np.size(hh)==1:
            self.hh  = hh*np.ones(M)
        elif np.size(hh)==M:
            self.hh  = hh
        else:
            print('hh can be a number or an array of size M')

        self.cc = np.zeros( self.M, dtype=DTYPE)
        if np.size(cc)==1:
            self.cc = cc*np.ones(M)
        elif np.size(cc)==M:
            self.cc= cc
        else:
            print('cc can be a number or an array of size M')

        self.mm = np.zeros( self.M, dtype=DTYPE)
        if np.size(mm)==1:
            self.mm = mm*np.ones(M)
        elif np.size(mm)==M:
            self.mm= mm
        else:
            print('mm can be a number or an array of size M')

        self.ir = 0

    cdef rhs(self, rp, tt):
        cdef:
            int highSpeed=self.highSpeed
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, t_divid_100=int(tt/100)
            int ir=self.ir
            unsigned short i, j, k, alp, gam, age_id, ii, jj
            unsigned long t_i, t_j
            double beta=self.beta, gIa=self.gIa
            double fsa=self.fsa, fh=self.fh, gE=self.gE
            double gIs=self.gIs, gIh=self.gIh, gIc=self.gIc
            double rW=self.rW, rT=self.rT
            double aa=0.0, bb=0.0, ccc=0.0, t_p_24 = tt%24
            double [:] S           = rp[0   :M1  ]
            double [:] E           = rp[M1  :2*M1]
            double [:] Ia          = rp[2*M1:3*M1]
            double [:] Is          = rp[3*M1:4*M1]
            double [:] Ih          = rp[4*M1:5*M1]
            double [:] Ic          = rp[5*M1:6*M1]
            double [:] Im          = rp[6*M1:7*M1]
            double [:] N           = rp[7*M1:8*M1] # muse be same as Nh
            double [:] ce1         = self.gE*self.alpha
            double [:] ce2         = self.gE*self.alphab
            double [:] hh          = self.hh
            double [:] cc          = self.cc
            double [:] mm          = self.mm
            double [:,:]   Nh      = self.Nh
            double [:,:]   Nw      = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh     = self.iNh
            double [:,:]   iNw     = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:]     RM      = self.RM
            double [:,:]   CMh     = self.CMh
            double [:,:]   CMw     = self.CMw
            double [:,:]   CMt     = self.CMt
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   aveS    = self.aveS
            double [:,:]   aveI    = self.aveI
            double [:,:,:] II      = self.II
            double [:,:,:] IT      = self.IT
            double [:,:,:] TT      = self.TT
            double [:,:]   FF      = self.FF
            double [:,:]   Lambda  = self.Lambda
            double [:]     X       = self.drpdt
            unsigned short [:,:,:]   indexJ    = self.indexJ
            unsigned short [:,:,:]   indexI    = self.indexI
            #unsigned short [:,:,:,:] indexAGJ  = self.indexAGJ
            unsigned short [:,:,:] route_index = self.route_index
            float [:,:,:]          route_ratio = self.route_ratio

        #t_divid_100 = int(tt/100)
        if t_divid_100 > self.t_old:
            print('Time', tt)
            self.t_old = t_divid_100
        #t_p_24 = tt%24
        if t_p_24 < 8.0 or t_p_24 > 18.0: #HOME
        #if True: #HOME
            #print("TIME_in_HOME", t_p_24)
            for i in range(Nnode):
                for alp in range(M):
                    t_i = alp*Nnode + i
                    aa = 0.0
                    if S[t_i] > 0.0:
                        bb = 0.0
                        for gam in range(M):
                            t_j = gam*Nnode + i
                            ccc = Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                            bb += CMh[alp,gam]*ccc*iNh[gam,i]
                        aa = beta*bb*S[t_i]
                    X[t_i]        = -aa
                    X[t_i + M1]   = aa - gE*E[t_i]
                    X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                    X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                    X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                    X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                    X[t_i + 6*M1] = mm[alp]*gIc*Ic[t_i]
                    aa = mm[alp]*gIc*Ic[t_i]
                    X[t_i + 7*M1] = -aa
                    if aa > 0.0: # procedure of decreasing the number of people
                        Nh[alp,i]  -= aa
                        if Nh[alp,i] > 0.0:
                            iNh[alp,i]  = 1.0/Nh[alp,i]
                            # determin decreased work place index
                            ccc = RM[ir]
                            ir += 1
                            ii = int(ccc*indexI[alp,i,0]) + 1 # decreased work index
                            k = indexI[alp,i,ii] # decreased work place
                            #print(i, ii, indexI[alp,i,0], k)
                            Nw[alp,k]    -= aa
                            if Nw[alp,k] > 0.0:
                                iNw[alp,k] = 1.0/Nw[alp,k]
                            else:
                                Nw[alp,k]  =0.0
                                iNw[alp,k] = 0.0
                            Dnm[alp,k,i] -= aa
                            if Dnm[alp,k,i] <= 0.0:
                                indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                indexI[alp,i,0] -= 1
                                Dnm[alp,k,i] = 0.0
                            PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                            PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                        else:
                            Nh[alp,i]  = 0.0
                            iNh[alp,i] = 0.0
                            indexI[alp,i,0] = 0
                        
        elif t_p_24  > 9.0 and t_p_24  < 17.0: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            if True: #distinguishable
                for i in range(Nnode):
                    for alp in range(M):
                        aveI[alp,i] = 0.0
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj = indexJ[alp,i,j]
                            t_j = alp*Nnode + jj
                            aveI[alp,i] += PWRh[alp,i,jj]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])
                    for alp in range(M):
                        Lambda[alp, i] = 0.0
                        for gam in range(M):
                            Lambda[alp,i] += CMw[alp,gam]*aveI[gam,i]*iNw[gam,i]
                        Lambda[alp, i] = rW*beta*Lambda[alp,i]

                        t_i = alp*Nnode + i
                            
                        X[t_i]        = 0.0
                        X[t_i + M1]   = 0.0
                        X[t_i + 2*M1] = 0.0
                        X[t_i + 3*M1] = 0.0
                        X[t_i + 4*M1] = 0.0
                        X[t_i + 5*M1] = 0.0
                        X[t_i + 6*M1] = 0.0
                        X[t_i + 7*M1] = 0.0
                            
                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        X[t_i + M1]   = - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 6*M1] = mm[alp]*gIc*Ic[t_i]
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            if S[t_i] > 0.0:
                                aa = Lambda[alp,ii]*PWRh[alp,ii,i]*S[t_i]
                                X[t_i]        += -aa
                                X[t_i + M1]   += aa
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 7*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

        else: #TRANS
            #print("TIME_in_TRANS", t_p_24)
            if highSpeed == 1: # using averaged \hat{I}_{ij}
                for i in range(Nnode):
                    aveI[M,i] = 0.0
                    Dnm[M,i,j] = 0.0
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aa = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                        #aveI[alp,i] = aa
                        aveI[M,i]  += aa
                        for j in range(Nnode):
                            II[M,i,j]   = 0.0
                            IT[M,i,j]   = 0.0
                            TT[M,i,j]   = 0.0
                            Dnm[M,i,j] += Dnm[alp,i,j]
                            Ntrans[M,i,j] = 0.0
                        
                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        if route_index[i,jj,0] >= 2:
                            for k in range(1, route_index[i,jj,0]):
                                II[M,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[M,jj]
                                TT[M,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*Dnm[M,i,jj]

                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        for k in range(1, route_index[i,jj,0]):
                            IT[M,i,jj] += II[M,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                            Ntrans[M,i,jj] += TT[M,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        
                        ccc = 0.0
                        for gam in range(M):
                            ccc += CMt[alp,gam]

                        bb = 0.0
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            if Ntrans[M,ii,i] > 0.0:
                                bb += IT[M,ii,i]*PWRh[alp,ii,i]/Ntrans[M,ii,i] # add i->j
                        bb += PWRh[M,i,i]*aveI[M,i]*PWRh[alp,i,i]*iNw[M,i] # add i->i
                        aa = rT*beta*ccc*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 6*M1] = mm[alp]*gIc*Ic[t_i]
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 7*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

            elif True: # using accurate \hat{I}_{ij}
                for i in prange(Nnode, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aveI[alp,i] = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                    for j in range(Nnode):
                        for alp in range(M):
                            II[alp,i,j] = 0.0
                            IT[alp,i,j] = 0.0
                            TT[alp,i,j] = 0.0
                            Ntrans[alp,i,j] = 0.0
                        
                for i in range(Nnode):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[alp,jj]
                                    TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += Dnm[alp,i,jj]

                for i in prange(Nnode, nogil=True):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            for k in range(1, route_index[i,jj,0]):
                                IT[alp,i,jj]     += II[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                                Ntrans[alp,i,jj] += TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        bb = 0.0
                        for gam in range(M):
                            for j in range(1, indexI[alp,i,0] + 1):
                                ii = indexI[alp,i,j]
                                if Ntrans[gam,ii,i] > 0.0:
                                    bb += CMt[alp,gam]*IT[gam,ii,i]*PWRh[alp,ii,i]/Ntrans[gam,ii,i]
                            t_j = gam*Nnode + i
                            bb += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])*iNw[gam,i]*PWRh[alp,i,i]
                        aa = rT*beta*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 6*M1] = mm[alp]*gIc*Ic[t_i]
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 7*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

        return

    cdef prepare_fixed_variable(self, travel_restriction):
        cdef:
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, max_route_num
            unsigned short i, j, k, alp, index_i, index_j, index_agj, ii, count
            double cutoff=self.cutoff, cij, ccij, t_restriction=travel_restriction
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:,:] distances = self.distances
            unsigned short [:,:,:]   indexJ  = self.indexJ
            unsigned short [:,:,:]   indexI  = self.indexI
            #unsigned short [:,:,:,:] indexAGJ= self.indexAGJ
            unsigned short [:] route           # working memory
            double [:,:,:] C_Dnm   = self.IT
            unsigned short [:,:,:] route_index # the list of node in the route i -> j
            float [:,:,:] route_ratio          # the list of the distance ratio in the route i -> j
            double [:,:]  dij  = self.II[0]    # the distance between node i and j
            int [:,:]     pred = self.PP       # predecessor node belong the route i -> j

        #travel restriction
        for alp in prange(M+1, nogil=True):
            for i in range(Nnode):
                for j in range(Nnode):
                    if i != j:
                        cij = Dnm[alp,i,j]
                        #ccij = round(cij*t_restriction)
                        ccij = cij*t_restriction
                        Dnm[alp,i,j] -= ccij
                        Dnm[alp,j,j] += ccij
    
        #cutoff
        cdef int nonzero_element = 0
        cdef double cutoff_total = 0.0
        C_Dnm = Dnm.copy()
        for alp in prange(M+1, nogil=True):
            for i in range(Nnode):
                for j in range(Nnode):
                    cij = C_Dnm[alp,i,j]
                    if i != j:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            if alp != M:
                                nonzero_element += 1
                        else:
                            Dnm[alp,i,j] = 0.0
                            Dnm[alp,j,j] += cij
                    if alp != M:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            cutoff_total += cij
        print("Nonzero element " + str(nonzero_element) + '/' + str(M1**2) + ' ' + str(cutoff_total))

        for alp in prange(M, nogil=True):
            for i in range(Nnode):
                Nh[alp,i] = 0.0 ## N^{H}_i residence in Node i and age gourp alp
                Nw[alp,i] = 0.0 ## N^{w}_i working in Node i and age group alp
                for j in range(Nnode):
                    Nh[alp,i] += Dnm[alp,j,i]
                    Nw[alp,i] += Dnm[alp,i,j]
                
        for alp in prange(M, nogil=True):
            for i in range(Nnode):
                Nh[M,i] += Nh[alp,i] ## N^{H}_i residence in Node i
                Nw[M,i] += Nw[alp,i] ## N^{w}_i working in Node i

        #Generating the Ntrans from route and predecessor
        cdef N_Matrix = np.zeros((M,Nnode,Nnode), dtype=DTYPE)
        cdef route_num = np.zeros(Nnode*Nnode, dtype=int)
        cdef int total_route_index = 0
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:
                    #route = get_path(i, j, p)
                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1

                    count = 0
                    for k in range(len(route) - 1):
                        for alp in range(M):
                            N_Matrix[alp,route[k + 1],route[k]] += Dnm[alp,j,i]
                            count += 1
                    total_route_index += len(route)
                    route_num[i*Nnode + j] = len(route)
        route_num.sort() # should be improved
        max_route_num = route_num[Nnode**2 - 1]
        print("Max route number", route_num[0], max_route_num)
        print("Total index in all route", total_route_index, 1.0*total_route_index/Nnode**2)

        self.route_index = np.zeros((Nnode,Nnode,max_route_num + 1), dtype=np.uint16)
        self.route_ratio = np.zeros((Nnode,Nnode,max_route_num), dtype=np.float32)
        route_index = self.route_index
        route_ratio = self.route_ratio
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:

                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        
                    for k in range(0, len(route) - 1):
                        # N^{t}_{ji} the effective number of the people using the route i->j at age group alpha
                        for alp in range(M):
                            Ntrans[alp,j,i] += N_Matrix[alp,route[k + 1],route[k]]*dij[route[k + 1],route[k]]/dij[j,i]
                        route_index[j,i,k + 1] = np.uint16(route[k])
                        route_ratio[j,i,k + 1] = np.float32(dij[route[k + 1],route[k]]/dij[j,i])
                    route_ratio[j,i,0] = np.float32(len(route) - 1)
                    route_index[j,i,len(route)] = np.uint16(route[len(route)-1])
                    route_index[j,i,0] = np.uint16(len(route))
                else:
                    for alp in range(M):
                        Ntrans[alp,j,i] = Dnm[alp,j,i]
        for alp in range(M):
            for j in range(Nnode):
                for i in range(Nnode):
                    Ntrans[M,j,i] += Ntrans[alp,j,i] ## N^{t}_{ji} the effective number of the people using the route i->j

        for alp in range(M+1):
            for i in range(Nnode):
                if Nh[alp,i] != 0:
                    iNh[alp,i] = 1.0/Nh[alp,i]
                else:
                    iNh[alp,i] = 0.0
                
                if Nw[alp,i] != 0:
                    iNw[alp,i] = 1.0/Nw[alp,i]
                else:
                    iNw[alp, i] = 0.0
                
                index_j = 0
                index_i = 0
                for j in range(Nnode):
                    if Nh[alp,j] != 0:
                        PWRh[alp,i,j] = Dnm[alp,i,j]/Nh[alp,j]
                    else:
                        PWRh[alp,i,j] = 0.0
                
                    if Nw[alp,i] != 0:
                        PWRw[alp,i,j] = Dnm[alp,i,j]/Nw[alp,i]
                    else:
                        PWRw[alp,i,j] = 0.0
                    
                    if Ntrans[alp,i,j] != 0:
                        iNtrans[alp,i,j] = 1.0/Ntrans[alp,i,j]
                    else:
                        iNtrans[alp,i,j] = 0.0
                    
                    if Dnm[alp,i,j] > cutoff or i == j:
                        indexJ[alp,i,index_j + 1] = j
                        index_j += 1
                    elif Dnm[alp,i,j] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,i,j]) + '\n')

                    if Dnm[alp,j,i] > cutoff or i == j:
                        indexI[alp,i,index_i + 1] = j
                        index_i += 1
                    elif Dnm[alp,j,i] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,j,i]) + '\n')
                    
                indexJ[alp,i,0] = index_j
                indexI[alp,i,0] = index_i
                
        #for alp in range(M):
        #    for gam in range(M):
        #        for i in range(Nnode):
        #            index_agj = 0
        #            for j in range(Nnode):
        #                if Dnm[alp,i,j] > cutoff and Dnm[gam,i,j] > cutoff:
        #                    indexAGJ[alp,gam,i,index_agj + 1] = j
        #                    index_agj += 1
        #            indexAGJ[alp,gam,i,0] = index_agj


    def simulate(self, S0, E0, Ia0, Is0, Ih0, Ic0, Im0, N0, contactMatrix, workResidenceMatrix, distanceMatrix, travel_restriction, Tf, Nf, Ti=0, integrator='solve_ivp'):
        from scipy.integrate import solve_ivp
        from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
        from scipy.sparse import csr_matrix

        print('travel restriction', travel_restriction)
        print('cutoff', self.cutoff)
        print('highspeed', self.highSpeed)
        self.CMh  = contactMatrix(6.0)
        self.CMt  = contactMatrix(8.5)
        self.CMw  = contactMatrix(12.0)
        self.Dnm = workResidenceMatrix(0.0)
        self.distances = distanceMatrix(0.0)

        print('#Start finding the shortest path between each node')
        self.II[0], self.PP = shortest_path(distanceMatrix(0.0), return_predecessors=True)
        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(travel_restriction)
        print('#Calculation Start')
        
        def rhs0(t, rp):
            self.rhs(rp, t)
            return self.drpdt

        if integrator=='solve_ivp':
            time_points=np.linspace(Ti, Tf);  ## intervals at which output is returned by integrator.
            time_step = 1.0*Tf/Nf
            u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, N0)), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
            
            data={'X':u.y, 't':u.t, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        else:
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
            solver.set_initial_condition(np.concatenate((S0, E0, Ia0, Is0, Ih0, Ic0, Im0, N0)))
            u, time_points = solver.solve(time_points)

            data={'X':u, 't':time_points, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        return data


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class SEI8R:
    """
    Susceptible, Exposed, Infected, Recovered (SEIR)
    The infected class has 8 groups:
    * Ia: asymptomatic
    * Is: symptomatic
    * Ih: hospitalized
    * Ic: ICU
    * Isd: symptomatic dash
    * Ihd: hospitalized dash
    * Icd: ICU dash

    * Im: Mortality

    S  ---> E
    E  ---> Ia, Is
    Ia ---> R
    Is ---> Ih, Isd
    Isd---> R
    Ih ---> Ic, Ihd
    Ihd---> R
    Ic ---> Im, Icd
    Icd---> R
    """
    cdef:
        readonly int highSpeed
        readonly int M, Nnode, max_route_num, t_divid_100, t_old, ir
        readonly double beta, gE, gIa, gIs, gIh, gIc, gIsd, gIhd, gIcd
        readonly double fsa, fh, rW, rT, cutoff
        readonly np.ndarray alpha, hh, cc, mm, alphab, hhb, ccb, mmb
        readonly np.ndarray rp0, Nh, Nw, Ntrans, drpdt, CMh, CMw, CMt, Dnm, distances, RM
        readonly np.ndarray route_index, route_ratio
        readonly np.ndarray PWRh, PWRw, PP, FF, II, IT, TT
        readonly np.ndarray iNh, iNw, iNtrans, indexJ, indexI, indexAGJ, aveS, aveI, Lambda
        
    def __init__(self, parameters, M, Nnode):
        alpha       = parameters.get('alpha')             # fraction of asymptomatic infectives
        self.beta   = parameters.get('beta')              # infection rate
        self.gE     = parameters.get('gE')                # recovery rate of E class
        self.gIa    = parameters.get('gIa')               # recovery rate of Ia
        self.gIs    = parameters.get('gIs')               # recovery rate of Is
        self.gIh    = parameters.get('gIh')               # recovery rate of Ih
        self.gIc    = parameters.get('gIc')               # recovery rate of Ic
        self.gIsd   = parameters.get('gIsd')              # recovery rate of Isd
        self.gIhd   = parameters.get('gIhd')              # recovery rate of Ihd
        self.gIcd   = parameters.get('gIcd')              # recovery rate of Icd
        self.fsa    = parameters.get('fsa')               # the self-isolation parameter
        self.fh     = parameters.get('fh')                # the hospital-isolation parameter
        self.rW     = parameters.get('rW')                # fitting parameter at work
        self.rT     = parameters.get('rT')                # fitting parameter in trans
        hh          = parameters.get('hh')                # hospital
        cc          = parameters.get('cc')                # ICU
        mm          = parameters.get('mm')                # mortality
        self.cutoff = parameters.get('cutoff')            # cutoff value of census data
        self.highSpeed = parameters.get('highspeed')      # flag of hispeed calculation        
        self.M      = M
        self.Nnode  = Nnode
        self.Nh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people living in node i
        self.Nw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)         # # people working at node i
        self.Ntrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# people comuteing i->j
        self.iNh    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# inv people comuteing i->j

        self.RM    = np.random.rand(1000*self.M*self.Nnode) # random matrix 
        self.CMh   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in HOME
        self.CMw   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in WORK
        self.CMt   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in TRANS
        self.Dnm   = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# census data matrix WR
        self.PWRh  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.PWRw  = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE)# probability of Dnm at w
        self.aveS  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average S at i node
        self.aveI  = np.zeros( (self.M+1, self.Nnode), dtype=DTYPE)   # average I at i node
        self.Lambda= np.zeros( (self.M, self.Nnode), dtype=DTYPE)     # effective infection rate
        self.drpdt = np.zeros( 11*self.Nnode*self.M, dtype=DTYPE)      # right hand side
        self.indexJ= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        self.indexI= np.zeros( (self.M+1, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        #self.indexAGJ= np.zeros( (self.M, self.M, self.Nnode, self.Nnode + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij*Dnm_gam_ij at specifi alp, gam and i
        self.distances = np.zeros((self.Nnode,self.Nnode), DTYPE)
        self.II = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.IT = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.TT = np.zeros( (self.M+1, self.Nnode, self.Nnode), dtype=DTYPE) # Working memory
        self.FF = np.zeros( (self.M, self.Nnode), dtype=DTYPE)               # Working memory
        self.PP = np.zeros( (self.Nnode, self.Nnode), dtype=np.int32)        # Working memory

        self.alpha  = np.zeros( self.M, dtype=DTYPE)
        self.alphab = np.zeros( self.M, dtype=DTYPE)
        if np.size(alpha)==1:
            self.alpha  = alpha*np.ones(M)
            self.alphab = (1.0 - alpha)*np.ones(M)
        elif np.size(alpha)==M:
            self.alpha  = alpha
            self.alphab = np.ones(M) - self.alpha
        else:
            print('alpha can be a number or an array of size M')
            
        self.hh   = np.zeros( self.M, dtype=DTYPE)
        self.hhb  = np.zeros( self.M, dtype=DTYPE)
        if np.size(hh)==1:
            self.hh  = hh*np.ones(M)
            self.hhb = (1.0 - hh)*np.ones(M)
        elif np.size(hh)==M:
            self.hh  = hh
            self.hhb = np.ones(M) - self.hh
        else:
            print('hh can be a number or an array of size M')

        self.cc  = np.zeros( self.M, dtype=DTYPE)
        self.ccb = np.zeros( self.M, dtype=DTYPE)
        if np.size(cc)==1:
            self.cc  = cc*np.ones(M)
            self.ccb = (1.0 - cc)*np.ones(M)
        elif np.size(cc)==M:
            self.cc  = cc
            self.ccb = np.ones(M) - self.cc
        else:
            print('cc can be a number or an array of size M')

        self.mm  = np.zeros( self.M, dtype=DTYPE)
        self.mmb = np.zeros( self.M, dtype=DTYPE)
        if np.size(mm)==1:
            self.mm  = mm*np.ones(M)
            self.mmb = (1.0 - mm)*np.ones(M)
        elif np.size(mm)==M:
            self.mm  = mm
            self.mmb = np.ones(M) - self.mm
        else:
            print('mm can be a number or an array of size M')

        self.ir = 0

    cdef rhs(self, rp, tt):
        cdef:
            int highSpeed=self.highSpeed
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, t_divid_100=int(tt/100)
            int ir=self.ir
            unsigned short i, j, k, alp, gam, age_id, ii, jj
            unsigned long t_i, t_j
            double beta=self.beta, gIa=self.gIa
            double fsa=self.fsa, fh=self.fh, gE=self.gE
            double gIs=self.gIs, gIh=self.gIh, gIc=self.gIc
            double gIsd=self.gIsd, gIhd=self.gIhd, gIcd=self.gIcd
            double rW=self.rW, rT=self.rT
            double aa=0.0, bb=0.0, ccc=0.0, t_p_24 = tt%24
            double [:] S           = rp[0   :M1  ]
            double [:] E           = rp[M1  :2*M1]
            double [:] Ia          = rp[2*M1:3*M1]
            double [:] Is          = rp[3*M1:4*M1]
            double [:] Isd         = rp[4*M1:5*M1]
            double [:] Ih          = rp[5*M1:6*M1]
            double [:] Ihd         = rp[6*M1:7*M1]
            double [:] Ic          = rp[7*M1:8*M1]
            double [:] Icd         = rp[8*M1:9*M1]
            double [:] Im          = rp[9*M1:10*M1]
            double [:] N           = rp[10*M1:11*M1] # muse be same as Nh
            double [:] ce1         = self.gE*self.alpha
            double [:] ce2         = self.gE*self.alphab
            double [:] hh          = self.hh
            double [:] cc          = self.cc
            double [:] mm          = self.mm
            double [:] hhb         = self.hhb
            double [:] ccb         = self.ccb
            double [:] mmb         = self.mmb
            double [:,:]   Nh      = self.Nh
            double [:,:]   Nw      = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh     = self.iNh
            double [:,:]   iNw     = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:]     RM      = self.RM
            double [:,:]   CMh     = self.CMh
            double [:,:]   CMw     = self.CMw
            double [:,:]   CMt     = self.CMt
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   aveS    = self.aveS
            double [:,:]   aveI    = self.aveI
            double [:,:,:] II      = self.II
            double [:,:,:] IT      = self.IT
            double [:,:,:] TT      = self.TT
            double [:,:]   FF      = self.FF
            double [:,:]   Lambda  = self.Lambda
            double [:]     X       = self.drpdt
            unsigned short [:,:,:]   indexJ    = self.indexJ
            unsigned short [:,:,:]   indexI    = self.indexI
            #unsigned short [:,:,:,:] indexAGJ  = self.indexAGJ
            unsigned short [:,:,:] route_index = self.route_index
            float [:,:,:]          route_ratio = self.route_ratio

        #t_divid_100 = int(tt/100)
        if t_divid_100 > self.t_old:
            print('Time', tt)
            self.t_old = t_divid_100
        #t_p_24 = tt%24
        if t_p_24 < 8.0 or t_p_24 > 18.0: #HOME
        #if True: #HOME
            #print("TIME_in_HOME", t_p_24)
            for i in range(Nnode):
                for alp in range(M):
                    t_i = alp*Nnode + i
                    aa = 0.0
                    if S[t_i] > 0.0:
                        bb = 0.0
                        for gam in range(M):
                            t_j = gam*Nnode + i
                            ccc = Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                            bb += CMh[alp,gam]*ccc*iNh[gam,i]
                        aa = beta*bb*S[t_i]
                    X[t_i]        = -aa
                    X[t_i + M1]   = aa - gE*E[t_i]
                    X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                    X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                    X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                    X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                    X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                    X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                    X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                    X[t_i + 9*M1] = mm[alp]*gIc*Ic[t_i]
                    aa = mm[alp]*gIc*Ic[t_i]
                    X[t_i + 10*M1] = -aa
                    if aa > 0.0: # procedure of decreasing the number of people
                        Nh[alp,i]  -= aa
                        if Nh[alp,i] > 0.0:
                            iNh[alp,i]  = 1.0/Nh[alp,i]
                            # determin decreased work place index
                            ccc = RM[ir]
                            ir += 1
                            ii = int(ccc*indexI[alp,i,0]) + 1 # decreased work index
                            k = indexI[alp,i,ii] # decreased work place
                            #print(i, ii, indexI[alp,i,0], k)
                            Nw[alp,k]    -= aa
                            if Nw[alp,k] > 0.0:
                                iNw[alp,k] = 1.0/Nw[alp,k]
                            else:
                                Nw[alp,k]  =0.0
                                iNw[alp,k] = 0.0
                            Dnm[alp,k,i] -= aa
                            if Dnm[alp,k,i] <= 0.0:
                                indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                indexI[alp,i,0] -= 1
                                Dnm[alp,k,i] = 0.0
                            PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                            PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                        else:
                            Nh[alp,i]  = 0.0
                            iNh[alp,i] = 0.0
                            indexI[alp,i,0] = 0
                        
        elif t_p_24  > 9.0 and t_p_24  < 17.0: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            if True: #distinguishable
                for i in range(Nnode):
                    for alp in range(M):
                        aveI[alp,i] = 0.0
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj = indexJ[alp,i,j]
                            t_j = alp*Nnode + jj
                            aveI[alp,i] += PWRh[alp,i,jj]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])
                    for alp in range(M):
                        Lambda[alp, i] = 0.0
                        for gam in range(M):
                            Lambda[alp,i] += CMw[alp,gam]*aveI[gam,i]*iNw[gam,i]
                        Lambda[alp, i] = rW*beta*Lambda[alp,i]

                        t_i = alp*Nnode + i
                            
                        X[t_i]        = 0.0
                        X[t_i + M1]   = 0.0
                        X[t_i + 2*M1] = 0.0
                        X[t_i + 3*M1] = 0.0
                        X[t_i + 4*M1] = 0.0
                        X[t_i + 5*M1] = 0.0
                        X[t_i + 6*M1] = 0.0
                        X[t_i + 7*M1] = 0.0
                        X[t_i + 8*M1] = 0.0
                        X[t_i + 9*M1] = 0.0
                        X[t_i + 10*M1] = 0.0
                            
                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        X[t_i + M1]   = - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                        X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                        X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                        X[t_i + 9*M1] = mm[alp]*gIc*Ic[t_i]
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            if S[t_i] > 0.0:
                                aa = Lambda[alp,ii]*PWRh[alp,ii,i]*S[t_i]
                                X[t_i]        += -aa
                                X[t_i + M1]   += aa
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 10*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

        else: #TRANS
            #print("TIME_in_TRANS", t_p_24)
            if highSpeed == 1: # using averaged \hat{I}_{ij}
                for i in range(Nnode):
                    aveI[M,i] = 0.0
                    Dnm[M,i,j] = 0.0
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aa = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                        #aveI[alp,i] = aa
                        aveI[M,i]  += aa
                        for j in range(Nnode):
                            II[M,i,j]   = 0.0
                            IT[M,i,j]   = 0.0
                            TT[M,i,j]   = 0.0
                            Dnm[M,i,j] += Dnm[alp,i,j]
                            Ntrans[M,i,j] = 0.0
                        
                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        if route_index[i,jj,0] >= 2:
                            for k in range(1, route_index[i,jj,0]):
                                II[M,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[M,jj]
                                TT[M,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*Dnm[M,i,jj]

                for i in range(Nnode):
                    for j in range(1, indexJ[M,i,0] + 1):
                        jj  = indexJ[M,i,j]
                        for k in range(1, route_index[i,jj,0]):
                            IT[M,i,jj] += II[M,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                            Ntrans[M,i,jj] += TT[M,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        
                        ccc = 0.0
                        for gam in range(M):
                            ccc += CMt[alp,gam]

                        bb = 0.0
                        for j in range(1, indexI[alp,i,0] + 1):
                            ii = indexI[alp,i,j]
                            if Ntrans[M,ii,i] > 0.0:
                                bb += IT[M,ii,i]*PWRh[alp,ii,i]/Ntrans[M,ii,i] # add i->j
                        bb += PWRh[M,i,i]*aveI[M,i]*PWRh[alp,i,i]*iNw[M,i] # add i->i
                        aa = rT*beta*ccc*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                        X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                        X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                        X[t_i + 9*M1] = mm[alp]*gIc*Ic[t_i]
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 10*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

            elif True: # using accurate \hat{I}_{ij}
                for i in prange(Nnode, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        aveI[alp,i] = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                    for j in range(Nnode):
                        for alp in range(M):
                            II[alp,i,j] = 0.0
                            IT[alp,i,j] = 0.0
                            TT[alp,i,j] = 0.0
                            Ntrans[alp,i,j] = 0.0
                        
                for i in range(Nnode):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[alp,jj]
                                    TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += Dnm[alp,i,jj]

                for i in prange(Nnode, nogil=True):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            for k in range(1, route_index[i,jj,0]):
                                IT[alp,i,jj]     += II[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                                Ntrans[alp,i,jj] += TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                

                for i in range(Nnode):
                    for alp in range(M):
                        t_i = alp*Nnode + i
                        bb = 0.0
                        for gam in range(M):
                            for j in range(1, indexI[alp,i,0] + 1):
                                ii = indexI[alp,i,j]
                                if Ntrans[gam,ii,i] > 0.0:
                                    bb += CMt[alp,gam]*IT[gam,ii,i]*PWRh[alp,ii,i]/Ntrans[gam,ii,i]
                            t_j = gam*Nnode + i
                            bb += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])*iNw[gam,i]*PWRh[alp,i,i]
                        aa = rT*beta*bb*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                        X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                        X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                        X[t_i + 9*M1] = mm[alp]*gIc*Ic[t_i]
                        aa = mm[alp]*gIc*Ic[t_i]
                        X[t_i + 10*M1] = -aa
                        if aa > 0.0: # procedure of decreasing the number of people
                            Nh[alp,i]  -= aa
                            if Nh[alp,i] > 0.0:
                                iNh[alp,i]  = 1.0/Nh[alp,i]
                                # determin decreased work place index
                                ccc = RM[ir]
                                ir += 1
                                ii = int(ccc*indexI[alp,i,0]) + 1 # decreased I-index
                                k = indexI[alp,i,ii] # decreased work node index
                                #print(i, ii, indexI[alp,i,0], k)
                                Nw[alp,k]    -= aa
                                if Nw[alp,k] > 0.0:
                                    iNw[alp,k] = 1.0/Nw[alp,k]
                                else:
                                    Nw[alp,k]  =0.0
                                    iNw[alp,k] = 0.0
                                Dnm[alp,k,i] -= aa
                                if Dnm[alp,k,i] <= 0.0:
                                    indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                                    indexI[alp,i,0] -= 1
                                    Dnm[alp,k,i] = 0.0
                                PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                                PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                            else:
                                Nh[alp,i]  = 0.0
                                iNh[alp,i] = 0.0
                                indexI[alp,i,0] = 0

        return

    cdef prepare_fixed_variable(self, travel_restriction):
        cdef:
            int M=self.M, Nnode=self.Nnode, M1=self.M*self.Nnode, max_route_num
            unsigned short i, j, k, alp, index_i, index_j, index_agj, ii, count
            double cutoff=self.cutoff, cij, ccij, t_restriction=travel_restriction
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:,:] distances = self.distances
            unsigned short [:,:,:]   indexJ  = self.indexJ
            unsigned short [:,:,:]   indexI  = self.indexI
            #unsigned short [:,:,:,:] indexAGJ= self.indexAGJ
            unsigned short [:] route           # working memory
            double [:,:,:] C_Dnm   = self.IT
            unsigned short [:,:,:] route_index # the list of node in the route i -> j
            float [:,:,:] route_ratio          # the list of the distance ratio in the route i -> j
            double [:,:]  dij  = self.II[0]    # the distance between node i and j
            int [:,:]     pred = self.PP       # predecessor node belong the route i -> j

        #travel restriction
        for alp in prange(M+1, nogil=True):
            for i in range(Nnode):
                for j in range(Nnode):
                    if i != j:
                        cij = Dnm[alp,i,j]
                        #ccij = round(cij*t_restriction)
                        ccij = cij*t_restriction
                        Dnm[alp,i,j] -= ccij
                        Dnm[alp,j,j] += ccij
    
        #cutoff
        cdef int nonzero_element = 0
        cdef double cutoff_total = 0.0
        C_Dnm = Dnm.copy()
        for alp in prange(M+1, nogil=True):
            for i in range(Nnode):
                for j in range(Nnode):
                    cij = C_Dnm[alp,i,j]
                    if i != j:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            if alp != M:
                                nonzero_element += 1
                        else:
                            Dnm[alp,i,j] = 0.0
                            Dnm[alp,j,j] += cij
                    if alp != M:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            cutoff_total += cij
        print("Nonzero element " + str(nonzero_element) + '/' + str(M1**2) + ' ' + str(cutoff_total))

        for alp in prange(M, nogil=True):
            for i in range(Nnode):
                Nh[alp,i] = 0.0 ## N^{H}_i residence in Node i and age gourp alp
                Nw[alp,i] = 0.0 ## N^{w}_i working in Node i and age group alp
                for j in range(Nnode):
                    Nh[alp,i] += Dnm[alp,j,i]
                    Nw[alp,i] += Dnm[alp,i,j]
                
        for alp in prange(M, nogil=True):
            for i in range(Nnode):
                Nh[M,i] += Nh[alp,i] ## N^{H}_i residence in Node i
                Nw[M,i] += Nw[alp,i] ## N^{w}_i working in Node i

        #Generating the Ntrans from route and predecessor
        cdef N_Matrix = np.zeros((M,Nnode,Nnode), dtype=DTYPE)
        cdef route_num = np.zeros(Nnode*Nnode, dtype=int)
        cdef int total_route_index = 0
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:
                    #route = get_path(i, j, p)
                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1

                    count = 0
                    for k in range(len(route) - 1):
                        for alp in range(M):
                            N_Matrix[alp,route[k + 1],route[k]] += Dnm[alp,j,i]
                            count += 1
                    total_route_index += len(route)
                    route_num[i*Nnode + j] = len(route)
        route_num.sort() # should be improved
        max_route_num = route_num[Nnode**2 - 1]
        print("Max route number", route_num[0], max_route_num)
        print("Total index in all route", total_route_index, 1.0*total_route_index/Nnode**2)

        self.route_index = np.zeros((Nnode,Nnode,max_route_num + 1), dtype=np.uint16)
        self.route_ratio = np.zeros((Nnode,Nnode,max_route_num), dtype=np.float32)
        route_index = self.route_index
        route_ratio = self.route_ratio
        for i in range(Nnode):
            for j in range(Nnode):
                if i != j and Dnm[M,j,i] > cutoff:

                    count = 0
                    ii = j
                    while ii != i and ii >= 0:
                        count += 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        count += 1

                    route = np.zeros(count, dtype=np.uint16)
                    count -= 1
                    ii = j
                    while ii != i and ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        ii = pred[i][ii]
                    if ii >= 0:
                        route[count] = np.uint16(ii)
                        count -= 1
                        
                    for k in range(0, len(route) - 1):
                        # N^{t}_{ji} the effective number of the people using the route i->j at age group alpha
                        for alp in range(M):
                            Ntrans[alp,j,i] += N_Matrix[alp,route[k + 1],route[k]]*dij[route[k + 1],route[k]]/dij[j,i]
                        route_index[j,i,k + 1] = np.uint16(route[k])
                        route_ratio[j,i,k + 1] = np.float32(dij[route[k + 1],route[k]]/dij[j,i])
                    route_ratio[j,i,0] = np.float32(len(route) - 1)
                    route_index[j,i,len(route)] = np.uint16(route[len(route)-1])
                    route_index[j,i,0] = np.uint16(len(route))
                else:
                    for alp in range(M):
                        Ntrans[alp,j,i] = Dnm[alp,j,i]
        for alp in range(M):
            for j in range(Nnode):
                for i in range(Nnode):
                    Ntrans[M,j,i] += Ntrans[alp,j,i] ## N^{t}_{ji} the effective number of the people using the route i->j

        for alp in range(M+1):
            for i in range(Nnode):
                if Nh[alp,i] != 0:
                    iNh[alp,i] = 1.0/Nh[alp,i]
                else:
                    iNh[alp,i] = 0.0
                
                if Nw[alp,i] != 0:
                    iNw[alp,i] = 1.0/Nw[alp,i]
                else:
                    iNw[alp, i] = 0.0
                
                index_j = 0
                index_i = 0
                for j in range(Nnode):
                    if Nh[alp,j] != 0:
                        PWRh[alp,i,j] = Dnm[alp,i,j]/Nh[alp,j]
                    else:
                        PWRh[alp,i,j] = 0.0
                
                    if Nw[alp,i] != 0:
                        PWRw[alp,i,j] = Dnm[alp,i,j]/Nw[alp,i]
                    else:
                        PWRw[alp,i,j] = 0.0
                    
                    if Ntrans[alp,i,j] != 0:
                        iNtrans[alp,i,j] = 1.0/Ntrans[alp,i,j]
                    else:
                        iNtrans[alp,i,j] = 0.0
                    
                    if Dnm[alp,i,j] > cutoff or i == j:
                        indexJ[alp,i,index_j + 1] = j
                        index_j += 1
                    elif Dnm[alp,i,j] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,i,j]) + '\n')

                    if Dnm[alp,j,i] > cutoff or i == j:
                        indexI[alp,i,index_i + 1] = j
                        index_i += 1
                    elif Dnm[alp,j,i] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,j,i]) + '\n')
                    
                indexJ[alp,i,0] = index_j
                indexI[alp,i,0] = index_i
                
        #for alp in range(M):
        #    for gam in range(M):
        #        for i in range(Nnode):
        #            index_agj = 0
        #            for j in range(Nnode):
        #                if Dnm[alp,i,j] > cutoff and Dnm[gam,i,j] > cutoff:
        #                    indexAGJ[alp,gam,i,index_agj + 1] = j
        #                    index_agj += 1
        #            indexAGJ[alp,gam,i,0] = index_agj


    def simulate(self, S0, E0, Ia0, Is0, Isd0, Ih0, Ihd0, Ic0, Icd0, Im0, N0, contactMatrix, workResidenceMatrix, distanceMatrix, travel_restriction, Tf, Nf, Ti=0, integrator='solve_ivp'):
        from scipy.integrate import solve_ivp
        from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
        from scipy.sparse import csr_matrix

        print('travel restriction', travel_restriction)
        print('cutoff', self.cutoff)
        print('highspeed', self.highSpeed)
        self.CMh  = contactMatrix(6.0)
        self.CMt  = contactMatrix(8.5)
        self.CMw  = contactMatrix(12.0)
        self.Dnm = workResidenceMatrix(0.0)
        self.distances = distanceMatrix(0.0)

        print('#Start finding the shortest path between each node')
        self.II[0], self.PP = shortest_path(distanceMatrix(0.0), return_predecessors=True)
        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(travel_restriction)
        print('#Calculation Start')
        
        def rhs0(t, rp):
            self.rhs(rp, t)
            return self.drpdt

        if integrator=='solve_ivp':
            time_points=np.linspace(Ti, Tf);  ## intervals at which output is returned by integrator.
            time_step = 1.0*Tf/Nf
            u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, E0, Ia0, Is0, Isd0, Ih0, Ihd0, Ic0, Icd0, Im0, N0)), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
            
            data={'X':u.y, 't':u.t, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        else:
            import odespy
            time_points=np.linspace(Ti, Tf, Nf);  ## intervals at which output is returned by integrator.
            solver = odespy.Vode(rhs0, method = 'bdf', atol=1E-7, rtol=1E-6, order=5, nsteps=10**6)
            solver.set_initial_condition(np.concatenate((S0, E0, Ia0, Is0, Isd0, Ih0, Ihd0, Ic0, Icd0, Im0, N0)))
            u, time_points = solver.solve(time_points)

            data={'X':u, 't':time_points, 'N':self.Nnode, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        return data
