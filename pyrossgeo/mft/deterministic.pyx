import  sys
import  numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, tanh, pow, log
from cython.parallel import prange
cdef double PI = 3.14159265359

DTYPE   = np.float
ctypedef np.float_t DTYPE_t
# wraps functions of type double->double
ctypedef double(*f_type)(double, double, double, double)
# here all possible functions double,double,double,double->double
cdef double scaling_none(double rho, double a, double b, double c):
    """1.0"""
    return 1.0

cdef double scaling_linear(double rho, double a, double b, double c):
    """a + b*rho"""
    return a + b*rho

cdef double scaling_powerlaw(double rho, double a, double b, double c):
    """a + b*rho^c"""
    return a + b * pow(rho, c)

cdef double scaling_exp(double rho, double a, double b, double c):
    """a * b^rho"""
    return a * pow(b, rho)

cdef double scaling_log(double rho, double a, double b, double c):
    """log(a + b*rho)"""
    return log(a + b*rho)

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

    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta: float
                rate of spread of infection.
            gE: float
                rate of removal from exposeds individuals.
            gIa: float
                rate of removal from asymptomatic individuals.
            gIs: float
                rate of removal from symptomatic individuals.
            gIsp: float
                rate of removal from symptomatic individuals towards buffer.
            gIh: float
                rate of removal for hospitalised individuals.
            gIhp: float
                rate of removal from hospitalised individuals towards buffer.
            gIc: float
                rate of removal for idividuals in intensive care.
            gIcp: float
                rate of removal from ICU individuals towards buffer.
            fsa: float
                fraction by which symptomatic individuals self isolate.
            fh  : float
                fraction by which hospitalised individuals are isolated.
            sa: float, np.array (M,)
                daily arrival of new susceptables.
                sa is rate of additional/removal of population by birth etc
            hh: float, np.array (M,)
                fraction hospitalised from Is
            cc: float, np.array (M,)
                fraction sent to intensive care from hospitalised.
            mm: float, np.array (M,)
                mortality rate in intensive care

    M: int
        Number of age groups
    Nd: int
        Number of geographical node. M*Nd is a number of compartments of individual for each class.
        I.e len(S), len(Ia), len(Is) and len(R)
    Dnm: np.array(M*Nd, M*Nd)
        Number of people living in node m and working at node n.
    dnm: np.array(Nd, Nd)
        Distance Matrix. dnm indicate the distance between node n and m. When there is no conectivity between node n and m, dnm equals as 0.
    travel_restriction: float
        Restriction of travel between nodes. 1.0 means everyone stay home
    cutoff: float
        Cutoff number. Dnm is ignored, when Dnm is less than this number.
    Methods
    -------
    initialize
    simulate
    """
    cdef:
        readonly int highSpeed
        readonly int M, Nd, max_route_num, t_divid_100, t_old, ir, seed
        readonly double beta, gE, gIa, gIs, gIh, gIc, gIsd, gIhd, gIcd
        readonly double fsa, fh, rW, rT, cutoff, travel_restriction
        readonly np.ndarray alpha, hh, cc, mm, alphab, hhb, ccb, mmb
        readonly np.ndarray rp0, Nh, Nw, Ntrans, Ntrans0, drpdt, CMh, CMw, CMt, Dnm, Dnm0, RM
        readonly np.ndarray route_index, route_ratio
        readonly np.ndarray PWRh, PWRw, PP, FF, II, IT, TT
        readonly np.ndarray iNh, iNw, iNtrans, indexJ, indexI, indexAGJ, aveS, aveI, Lambda
        
    def __init__(self, parameters, M, Nd, Dnm, dnm, travel_restriction, cutoff):
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
        self.cutoff = cutoff                              # cutoff value of census data
        self.M      = M
        self.Nd     = Nd
        self.travel_restriction = travel_restriction
        self.Nh     = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)         # # people living in node i
        self.Nw     = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)         # # people working at node i
        self.Ntrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# people comuteing i->j
        self.Ntrans0= np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# backup of Ntrans
        self.iNh    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# inv people comuteing i->j
        self.seed    = 1
        np.random.seed(self.seed)
        self.RM    = np.random.rand(5000*self.M*self.Nd) # random matrix 
        self.CMh   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in HOME
        self.CMw   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in WORK
        self.CMt   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in TRANS
        self.Dnm   = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# census data matrix WR
        self.Dnm   = Dnm
        self.Dnm0  = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# backup od Dnm
        self.PWRh  = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        self.PWRw  = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        self.aveS  = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)   # average S at i node
        self.aveI  = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)   # average I at i node
        self.Lambda= np.zeros( (self.M, self.Nd), dtype=DTYPE)     # effective infection rate
        self.drpdt = np.zeros( 11*self.Nd*self.M, dtype=DTYPE)      # right hand side
        self.indexJ= np.zeros( (self.M+1, self.Nd, self.Nd + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        self.indexI= np.zeros( (self.M+1, self.Nd, self.Nd + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        #self.indexAGJ= np.zeros( (self.M, self.M, self.Nd, self.Nd + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij*Dnm_gam_ij at specifi alp, gam and i
        #self.distances = np.zeros((self.Nd,self.Nd), DTYPE)
        #self.distances = dnm
        self.II = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        self.IT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        self.TT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        self.FF = np.zeros( (self.M, self.Nd), dtype=DTYPE)               # Working memory
        self.PP = np.zeros( (self.Nd, self.Nd), dtype=np.int32)        # Working memory

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

        from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
        from scipy.sparse import csr_matrix

        print('#Start finding the shortest path between each node')
        self.II[0], self.PP = shortest_path(dnm, return_predecessors=True)
        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(self.travel_restriction, 0)
        print('#Finish calculating fixed variables')

    def initialize(self, parameters, travel_restriction):
        """
        Parameters
        ----------
        parameters: dict
            Contains the following keys:
                alpha: float, np.array (M,)
                    fraction of infected who are asymptomatic.
                beta: float
                    rate of spread of infection.
                gE: float
                    rate of removal from exposeds individuals.
                gIa: float
                    rate of removal from asymptomatic individuals.
                gIs: float
                    rate of removal from symptomatic individuals.
                gIsp: float
                    rate of removal from symptomatic individuals towards buffer.
                gIh: float
                    rate of removal for hospitalised individuals.
                gIhp: float
                    rate of removal from hospitalised individuals towards buffer.
                gIc: float
                    rate of removal for idividuals in intensive care.
                gIcp: float
                    rate of removal from ICU individuals towards buffer.
                fsa: float
                    fraction by which symptomatic individuals self isolate.
                fh  : float
                    fraction by which hospitalised individuals are isolated.
                sa: float, np.array (M,)
                    daily arrival of new susceptables.
                    sa is rate of additional/removal of population by birth etc
                hh: float, np.array (M,)
                    fraction hospitalised from Is
                cc: float, np.array (M,)
                    fraction sent to intensive care from hospitalised.
                mm: float, np.array (M,)
                    mortality rate in intensive care

        travel_restriction: float
            Restriction of travel between nodes. 1.0 means everyone stay home
        """
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
        self.travel_restriction = travel_restriction
        self.Nh    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)         # # people living in node i
        self.Nw    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)         # # people working at node i
        self.Ntrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# people comuteing i->j
        self.iNh    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)        # inv people working at node i
        self.iNtrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# inv people comuteing i->j
        self.seed    = 1
        np.random.seed(self.seed)
        self.RM    = np.random.rand(5000*self.M*self.Nd) # random matrix 
        self.PWRh  = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        self.PWRw  = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        #self.aveS  = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)   # average S at i node
        #self.aveI  = np.zeros( (self.M+1, self.Nd), dtype=DTYPE)   # average I at i node
        #self.Lambda= np.zeros( (self.M, self.Nd), dtype=DTYPE)     # effective infection rate
        self.drpdt = np.zeros( 11*self.Nd*self.M, dtype=DTYPE)      # right hand side
        self.indexJ= np.zeros( (self.M+1, self.Nd, self.Nd + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        self.indexI= np.zeros( (self.M+1, self.Nd, self.Nd + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        #self.indexAGJ= np.zeros( (self.M, self.M, self.Nd, self.Nd + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij*Dnm_gam_ij at specifi alp, gam and i
        #self.distances = np.zeros((self.Nd,self.Nd), DTYPE)
        #self.distances = dnm
        #self.II = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        #self.IT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        #self.TT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
        #self.FF = np.zeros( (self.M, self.Nd), dtype=DTYPE)               # Working memory
        #self.PP = np.zeros( (self.Nd, self.Nd), dtype=np.int32)        # Working memory

        self.alpha  = np.zeros( self.M, dtype=DTYPE)
        self.alphab = np.zeros( self.M, dtype=DTYPE)
        if np.size(alpha)==1:
            self.alpha  = alpha*np.ones(self.M)
            self.alphab = (1.0 - alpha)*np.ones(self.M)
        elif np.size(alpha)==self.M:
            self.alpha  = alpha
            self.alphab = np.ones(self.M) - self.alpha
        else:
            print('alpha can be a number or an array of size M')
            
        self.hh   = np.zeros( self.M, dtype=DTYPE)
        self.hhb  = np.zeros( self.M, dtype=DTYPE)
        if np.size(hh)==1:
            self.hh  = hh*np.ones(self.M)
            self.hhb = (1.0 - hh)*np.ones(self.M)
        elif np.size(hh)==self.M:
            self.hh  = hh
            self.hhb = np.ones(self.M) - self.hh
        else:
            print('hh can be a number or an array of size M')

        self.cc  = np.zeros( self.M, dtype=DTYPE)
        self.ccb = np.zeros( self.M, dtype=DTYPE)
        if np.size(cc)==1:
            self.cc  = cc*np.ones(self.M)
            self.ccb = (1.0 - cc)*np.ones(self.M)
        elif np.size(cc)==self.M:
            self.cc  = cc
            self.ccb = np.ones(self.M) - self.cc
        else:
            print('cc can be a number or an array of size M')

        self.mm  = np.zeros( self.M, dtype=DTYPE)
        self.mmb = np.zeros( self.M, dtype=DTYPE)
        if np.size(mm)==1:
            self.mm  = mm*np.ones(self.M)
            self.mmb = (1.0 - mm)*np.ones(self.M)
        elif np.size(mm)==self.M:
            self.mm  = mm
            self.mmb = np.ones(self.M) - self.mm
        else:
            print('mm can be a number or an array of size M')

        self.ir = 0
        self.t_old = 0

        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(self.travel_restriction, 1)
        print('#Finish calculating fixed variables')

    cdef rhs(self, rp, tt):
        cdef:
            int highSpeed=self.highSpeed
            int M=self.M, Nd=self.Nd, M1=self.M*self.Nd, t_divid_100=int(tt/100)
            #unsigned short i, j, k, alp, gam, age_id, ii, jj
            int i, j, k, alp, gam, age_id, ii, jj
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
            for i in prange(Nd, nogil=True):
                for alp in range(M):
                    t_i = alp*Nd + i
                    aa = 0.0
                    if S[t_i] > 0.0:
                        FF[alp,i] = 0.0
                        #bb = 0.0
                        for gam in range(M):
                            t_j = gam*Nd + i
                            ccc = Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                            #bb += CMh[alp,gam]*ccc*iNh[gam,i]
                            FF[alp,i] += CMh[alp,gam]*ccc*iNh[gam,i]
                        aa = beta*FF[alp,i]*S[t_i]
                    X[t_i]        = -aa
                    X[t_i + M1]   = aa - gE*E[t_i]
                    X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                    X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                    X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                    X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                    X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                    X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                    X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                        
        elif t_p_24  > 9.0 and t_p_24  < 17.0: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            for i in prange(Nd, nogil=True):
                for alp in range(M):
                    aveI[alp,i] = 0.0
                    for j in range(1, indexJ[alp,i,0] + 1):
                        jj = indexJ[alp,i,j]
                        t_j = alp*Nd + jj
                        aveI[alp,i] += PWRh[alp,i,jj]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])
                for alp in range(M):
                    Lambda[alp, i] = 0.0
                    for gam in range(M):
                        Lambda[alp,i] += CMw[alp,gam]*aveI[gam,i]*iNw[gam,i]
                    Lambda[alp, i] = rW*beta*Lambda[alp,i]

                    t_i = alp*Nd + i
                            
                    X[t_i]        = 0.0
                    X[t_i + M1]   = - gE*E[t_i]
                    X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                    X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                    X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                    X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                    X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                    X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                    X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]
                        
            for i in prange(Nd, nogil=True):
                for alp in range(M):
                    t_i = alp*Nd + i
                    for j in range(1, indexI[alp,i,0] + 1):
                        ii = indexI[alp,i,j]
                        if S[t_i] > 0.0:
                            aa = Lambda[alp,ii]*PWRh[alp,ii,i]*S[t_i]
                            X[t_i]        += -aa
                            X[t_i + M1]   += aa

        else: #TRANS
            #print("TIME_in_TRANS", t_p_24)
            if highSpeed == 1: # using averaged \hat{I}_{ij}
                for alp in prange(M, nogil=True):
                    aveI[alp,0] = 0.0 # total I in age group alp
                    aveS[alp,0] = 0.0 # total Nh in age group alp
                    for i in range(Nd):
                        t_i = alp*Nd + i
                        aveI[alp,0] += Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                        aveS[alp,0] += Nh[alp,i]
                    aveI[alp,0] = aveI[alp,0]/aveS[alp,0] # total I/Nh in age group alp

                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        FF[alp,i] = 0.0
                        for gam in range(M):
                            FF[alp,i] += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i])*PWRh[alp,i,i]*iNw[gam,i] # add i->i
                            aa = aveI[gam,0]
                            age_id = alp
                            if indexI[alp,i,0] > indexI[gam,i,0]:
                                age_id = gam
                            for j in range(1, indexI[age_id,i,0] + 1):
                                ii = indexI[age_id,i,j]
                                t_j = gam*Nd + i
                                bb = Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                                FF[alp,i] += CMt[alp,gam]*(PWRh[gam,ii,i]*bb + (Ntrans[gam,ii,i] - Dnm[gam,ii,i])*aa)*PWRh[alp,ii,i]*iNtrans[gam,ii,i] # add i->j
                                
                        aa = rT*beta*FF[alp,i]*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                        X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                        X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]

            else: # using accurate \hat{I}_{ij}
                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        aveI[alp,i] = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                    for j in range(Nd):
                        for alp in range(M):
                            II[alp,i,j] = 0.0
                            IT[alp,i,j] = 0.0
                            TT[alp,i,j] = 0.0
                            Ntrans[alp,i,j] = 0.0
                        
                for i in range(Nd):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[alp,jj]
                                    TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += Dnm[alp,i,jj]

                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    IT[alp,i,jj]     += II[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                                    Ntrans[alp,i,jj] += TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                

                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        #bb = 0.0
                        FF[alp,i] = 0.0
                        for gam in range(M):
                            age_id = alp
                            if indexI[alp,i,0] > indexI[gam,i,0]:
                                age_id = gam
                            for j in range(1, indexI[age_id,i,0] + 1):
                                ii = indexI[age_id,i,j]
                                if Ntrans[gam,ii,i] > 0.0:
                                    FF[alp,i] += CMt[alp,gam]*IT[gam,ii,i]*PWRh[alp,ii,i]/Ntrans[gam,ii,i]
                            t_j = gam*Nd + i
                            FF[alp,i] += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])*iNw[gam,i]*PWRh[alp,i,i]
                        aa = rT*beta*FF[alp,i]*S[t_i]
                    
                        X[t_i]        = -aa
                        X[t_i + M1]   = aa - gE*E[t_i]
                        X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        X[t_i + 4*M1] = hhb[alp]*gIs*Is[t_i] - gIsd*Isd[t_i]
                        X[t_i + 5*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        X[t_i + 6*M1] = ccb[alp]*gIh*Ih[t_i] - gIhd*Ihd[t_i]
                        X[t_i + 7*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]
                        X[t_i + 8*M1] = mmb[alp]*gIc*Ic[t_i] - gIcd*Icd[t_i]

        # procedure of decreasing the number of people          
        for i in range(Nd):
            for alp in range(M):
                aa = 0.0
                t_i = alp*Nd + i
                if N[t_i] > 0.0:
                    aa = mm[alp]*gIc*Ic[t_i]
                X[t_i + 9*M1] = aa
                X[t_i + 10*M1] = -aa
                aa = Nh[alp,i] - N[t_i]
                if aa > 0.0 and N[t_i] > 0.0: # procedure of decreasing the number of people
                    Nh[M,i]    -= aa
                    Nh[alp,i]  -= aa
                    if Nh[alp,i] > 0.1:
                        iNh[alp,i]  = 1.0/Nh[alp,i]
                        # determin decreased work place index
                        ccc = RM[self.ir]
                        self.ir += 1        ##
                        ii = <int> (ccc*indexI[alp,i,0]) + 1 # decreased work index
                        k = indexI[alp,i,ii] # decreased work place ##
                        Nw[alp,k]    -= aa  ##
                        if Nw[alp,k] > 0.0:
                            iNw[alp,k] = 1.0/Nw[alp,k]
                        else:
                            Nw[alp,k]  = 0.0
                            iNw[alp,k] = 0.0
                        Dnm[alp,k,i] -= aa
                        if Dnm[alp,k,i] <= 0.0:
                            indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                            indexI[alp,i,0] -= 1
                            Dnm[alp,k,i] = 0.0
                        PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                        PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                    else:
                        N[t_i] = 0.0
                        Nh[alp,i]  = 0.0
                        iNh[alp,i] = 0.0
                        indexI[alp,i,0] = 0

        return

    cdef prepare_fixed_variable(self, _travel_restriction, _restart):
        cdef:
            int M=self.M, Nd=self.Nd, M1=self.M*self.Nd, max_route_num, restart=_restart
            #unsigned short i, j, k, alp, index_i, index_j, index_agj, ii, count, count0
            int i, j, k, alp, index_i, index_j, index_agj, ii, count, count0
            double cutoff=self.cutoff, cij, ccij, t_restriction=_travel_restriction
            double [:,:,:] Dnm     = self.Dnm
            double [:,:,:] Dnm0    = self.Dnm0
            double [:,:,:] PWRh    = self.PWRh
            double [:,:,:] PWRw    = self.PWRw
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:,:] Ntrans0 = self.Ntrans0
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            #double [:,:] distances = self.distances
            unsigned short [:,:,:]   indexJ  = self.indexJ
            unsigned short [:,:,:]   indexI  = self.indexI
            #unsigned short [:,:,:,:] indexAGJ= self.indexAGJ
            unsigned short [:] route           # working memory
            double [:,:,:] C_Dnm   = self.IT
            unsigned short [:,:,:] route_index # the list of node in the route i -> j
            float [:,:,:] route_ratio          # the list of the distance ratio in the route i -> j
            double [:,:]  dij  = self.II[0]    # the distance between node i and j
            int [:,:]     pred = self.PP       # predecessor node belong the route i -> j

        if restart == 0:
            for alp in prange(M+1, nogil=True):
                for i in range(Nd):
                    for j in range(Nd):
                        Dnm0[alp,i,j] = Dnm[alp,i,j]
        else:
            for alp in prange(M+1, nogil=True):
                for i in range(Nd):
                    for j in range(Nd):
                        Dnm[alp,i,j] = Dnm0[alp,i,j]
                        
        #travel restriction
        for alp in prange(M+1, nogil=True):
            for i in range(Nd):
                for j in range(Nd):
                    if i != j:
                        cij = Dnm[alp,i,j]
                        #ccij = round(cij*t_restriction)
                        ccij = cij*t_restriction
                        Dnm[alp,i,j] -= ccij
                        Dnm[alp,j,j] += ccij
    
        #cutoff
        cdef int nonzero_element = 0
        cdef double cutoff_total = 0.0
        #C_Dnm = Dnm.copy()
        for alp in prange(M+1, nogil=True):
            for i in range(Nd):
                for j in range(Nd):
                    cij = Dnm0[alp,i,j]
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
            for i in range(Nd):
                Nh[alp,i] = 0.0 ## N^{H}_i residence in Node i and age gourp alp
                Nw[alp,i] = 0.0 ## N^{w}_i working in Node i and age group alp
                for j in range(Nd):
                    Nh[alp,i] += Dnm[alp,j,i]
                    Nw[alp,i] += Dnm[alp,i,j]
                
        for i in prange(Nd, nogil=True):
            for alp in range(M):
                Nh[M,i] += Nh[alp,i] ## N^{H}_i residence in Node i
                Nw[M,i] += Nw[alp,i] ## N^{w}_i working in Node i

        #Generating the Ntrans from route and predecessor
        cdef N_Matrix = np.zeros((M,Nd,Nd), dtype=DTYPE)
        cdef route_num = np.zeros(Nd*Nd, dtype=int)
        cdef int total_route_index = 0
        if restart == 0:
            for i in range(Nd):
                for j in range(Nd):
                    if i != j and Dnm[M,j,i] > cutoff:
                        #route = get_path(i, j, p)
                        count = 0
                        ii = j
                        while ii != i and ii >= 0:
                            count += 1
                            ii = pred[i][ii]
                        if ii >= 0:
                            count += 1

                        count0 = count
                        route = np.zeros(count0, dtype=np.uint16)
                        count -= 1
                        ii = j
                        while ii != i and ii >= 0:
                            route[count] = np.uint16(ii)
                            count -= 1
                            ii = pred[i][ii]
                        if ii >= 0:
                            route[count] = np.uint16(ii)
                            count -= 1

                        #count = 0
                        for k in range(count0 - 1):
                            for alp in range(M):
                                N_Matrix[alp,route[k + 1],route[k]] += Dnm[alp,j,i]
                                #count += 1
                        total_route_index += count0
                        route_num[i*Nd + j] = count0
            route_num.sort() # should be improved
            max_route_num = route_num[Nd**2 - 1]
            print("Max route number", route_num[0], max_route_num)
            print("Total index in all route", total_route_index, 1.0*total_route_index/Nd**2)

            self.route_index = np.zeros((Nd,Nd,max_route_num + 1), dtype=np.uint16)
            self.route_ratio = np.zeros((Nd,Nd,max_route_num), dtype=np.float32)
            route_index = self.route_index
            route_ratio = self.route_ratio
            for i in range(Nd):
                for j in range(Nd):
                    if i != j and Dnm[M,j,i] > cutoff:

                        count = 0
                        ii = j
                        while ii != i and ii >= 0:
                            count += 1
                            ii = pred[i][ii]
                        if ii >= 0:
                            count += 1

                        count0 = count
                        route = np.zeros(count0, dtype=np.uint16)
                        count -= 1
                        ii = j
                        while ii != i and ii >= 0:
                            route[count] = np.uint16(ii)
                            count -= 1
                            ii = pred[i][ii]
                        if ii >= 0:
                            route[count] = np.uint16(ii)
                            count -= 1
                        
                        for k in range(0, count0 - 1):
                            # N^{t}_{ji} the effective number of the people using the route i->j at age group alpha
                            for alp in range(M):
                                Ntrans[alp,j,i] += N_Matrix[alp,route[k + 1],route[k]]*dij[route[k + 1],route[k]]/dij[j,i]
                            route_index[j,i,k + 1] = np.uint16(route[k])
                            route_ratio[j,i,k + 1] = np.float32(dij[route[k + 1],route[k]]/dij[j,i])
                        route_ratio[j,i,0] = np.float32(count0 - 1)
                        route_index[j,i,count0] = np.uint16(route[count0 - 1])
                        route_index[j,i,0] = np.uint16(count0)
                    else:
                        for alp in range(M):
                            Ntrans[alp,j,i] = 0.0 #Dnm[alp,j,i]
            for i in range(Nd):
                for j in range(Nd):
                    for alp in range(M):
                        Ntrans[M,i,j] += Ntrans[alp,i,j] ## N^{t}_{ij} the effective number of the people using the route j->i
                        Ntrans0[alp,i,j] = Ntrans[alp,i,j]
                    Ntrans0[M,i,j] = Ntrans[M,i,j]
        else:
            for alp in prange(M+1, nogil=True):
                for i in range(Nd):
                    for j in range(Nd):
                        Ntrans[alp,i,j] = Ntrans0[alp,i,j]      
        #END Generating the Ntrans from route and predecessor
        
        for alp in range(M+1):
            for i in range(Nd):
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
                for j in range(Nd):
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
        #        for i in range(Nd):
        #            index_agj = 0
        #            for j in range(Nd):
        #                if Dnm[alp,i,j] > cutoff and Dnm[gam,i,j] > cutoff:
        #                    indexAGJ[alp,gam,i,index_agj + 1] = j
        #                    index_agj += 1
        #            indexAGJ[alp,gam,i,0] = index_agj


    def simulate(self, S0, E0, Ia0, Is0, Isd0, Ih0, Ihd0, Ic0, Icd0, Im0, N0, contactMatrix, Tf, Nf, Ti=0, highSpeed=0):
        """
        Parameters
        ----------
        S0: np.array
            Initial number of susceptables.
        E0: np.array
            Initial number of exposeds.
        Ia0: np.array
            Initial number of asymptomatic infectives.
        Is0: np.array
            Initial number of symptomatic infectives.
        Ih0: np.array
            Initial number of hospitalized infectives.
        Ic0: np.array
            Initial number of ICU infectives.
        Im0: np.array
            Initial number of mortality.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        highSpeed: int, optional
            Flag of more coasening calculation.

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.
        """

        self.highSpeed = highSpeed # flag of hispeed calculation        
        print('travel restriction', self.travel_restriction)
        print('cutoff', self.cutoff)
        print('highspeed', self.highSpeed)
        self.CMh  = contactMatrix(6.0)
        self.CMt  = contactMatrix(8.5)
        self.CMw  = contactMatrix(12.0)
        #self.Dnm = workResidenceMatrix(0.0)
        #self.distances = distanceMatrix(0.0)

        print('#Calculation Start')
        
        def rhs0(t, rp):
            if self.ir > 4998*self.M*self.Nd:
                print("RM reset", t)
                self.seed += 1
                np.random.seed(self.seed)
                self.RM = np.random.rand(5000*self.M*self.Nd)
                self.ir = 0
            self.rhs(rp, t)
            return self.drpdt

        from scipy.integrate import solve_ivp
        time_points=np.linspace(Ti, Tf, int(Nf/24));  ## intervals at which output is returned by integrator.
        time_step = 1.0*Tf/Nf
        u = solve_ivp(rhs0, [Ti, Tf], np.concatenate((S0, E0, Ia0, Is0, Isd0, Ih0, Ihd0, Ic0, Icd0, Im0, N0)), method='RK23', t_eval=time_points, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
        
        data={'X':u.y, 't':u.t, 'N':self.Nd, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }

        return data

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class MeanFieldTheory:
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

    Parameters
    ----------
    parameters: dict
        Contains the following keys:
            alpha: float, np.array (M,)
                fraction of infected who are asymptomatic.
            beta: float
                rate of spread of infection.
            gE: float
                rate of removal from exposeds individuals.
            gIa: float
                rate of removal from asymptomatic individuals.
            gIs: float
                rate of removal from symptomatic individuals.
            gIh: float
                rate of removal for hospitalised individuals.
            gIc: float
                rate of removal for idividuals in intensive care.
            fsa: float
                fraction by which symptomatic individuals self isolate.
            fh  : float
                fraction by which hospitalised individuals are isolated.
            sa: float, np.array (M,)
                daily arrival of new susceptables.
                sa is rate of additional/removal of population by birth etc
            hh: float, np.array (M,)
                fraction hospitalised from Is
            cc: float, np.array (M,)
                fraction sent to intensive care from hospitalised.
            mm: float, np.array (M,)
                mortality rate in intensive care

    M: int
        Number of age groups
    Nd: int
        Number of geographical node. M*Nd is a number of compartments of individual for each class.
        I.e len(S), len(Ia), len(Is) and len(R)
    Dnm: np.array(M*Nd, M*Nd)
        Number of people living in node m and working at node n.
    dnm: np.array(Nd, Nd)
        Distance Matrix. dnm indicate the distance between node n and m. When there is no conectivity between node n and m, dnm equals as 0.
    travel_restriction: float
        Restriction of travel between nodes. 1.0 means everyone stay home
    cutoff: float
        Cutoff number. Dnm is ignored, when Dnm is less than this number.
    Methods
    -------
    initialize
    simulate
    """
    cdef:
        readonly int transport, highSpeed, mortal, fij_form
        readonly int geographical_model
        readonly int RapidTransport, SpacialCompartment
        readonly int M, Nd, model_dim, max_route_num, t_divid_100, t_old, ir, seed
        #readonly double beta, gE, gIa, gIs, gIh, gIc, fsa, fh
        readonly double rW, rT, w_period_ratio, h_period_ratio
        readonly double cutoff, travel_restriction
        #readonly np.ndarray alpha, hh, cc, mm, alphab
        readonly np.ndarray rp0, Nh, Nw, Ntrans, Ntrans0, drpdt
        readonly np.ndarray CMh, CMw, CMt, Dnm, Dnm0, RM
        readonly np.ndarray lCMh, lCMw, lCMt
        readonly np.ndarray route_index, route_ratio
        readonly np.ndarray PWRh, PWRw, PP, FF, II, IT, TT
        readonly np.ndarray iNh, iNw, iNtrans, indexJ, indexI, aveS, aveI, Lambda
        readonly np.ndarray i_clss, i_source, i_parameter
        readonly np.ndarray infect_source, infect_param , infected_class, infected_coeff
        readonly np.ndarray l_pos_source, l_neg_source, l_pos_param, l_neg_param
        readonly np.ndarray density_factor, norm_factor ## should be deleted
        
    def __init__(self, g_model, model, parameters, M, Nd, Dnm, dnm, Area, travel_restriction, cutoff, transport=0, highSpeed=0, fij_form=0):
        self.mortal = 0
        self.cutoff = cutoff                              # cutoff value of census data
        self.transport = transport
        self.highSpeed = highSpeed
        self.fij_form = fij_form
        self.RapidTransport = 0
        self.SpacialCompartment = 1
        self.M    = M
        self.Nd   = Nd
        self.travel_restriction = travel_restriction
        self.Nh   = np.zeros( (self.M, self.Nd), dtype=DTYPE)         # # people living in node i
        self.Nw   = np.zeros( (self.M, self.Nd), dtype=DTYPE)         # # people working at node i
        self.iNh  = np.zeros( (self.M, self.Nd), dtype=DTYPE)        # inv people living in node i
        self.iNw  = np.zeros( (self.M, self.Nd), dtype=DTYPE)        # inv people working at node i
        self.seed = 1
        np.random.seed(self.seed)
        self.RM    = np.random.rand(100*self.M*self.Nd) # random matrix 
        self.CMh   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in HOME
        self.CMw   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in WORK
        self.CMt   = np.zeros( (self.M, self.M), dtype=DTYPE) # contact matrix C in TRANS
        self.lCMh  = np.ones( (self.M, self.M, self.Nd), dtype=DTYPE) # local contact matrix C in HOME
        lCMh       = self.lCMh   
        self.lCMw  = np.ones( (self.M, self.M, self.Nd), dtype=DTYPE) # local contact matrix C in WORK
        lCMw       = self.lCMw   
        self.Dnm   = np.zeros( (self.M, self.Nd, self.Nd), dtype=np.uint16)# census data matrix WR
        self.Dnm   = Dnm
        #self.Dnm0  = np.zeros( (self.M, self.Nd, self.Nd), dtype=DTYPE)# backup of Dnm

        self.density_factor = np.zeros((2, self.Nd, self.M, self.M), dtype=DTYPE)
        self.norm_factor    = np.zeros((2, self.M, self.M), dtype=DTYPE)
        density_factor = self.density_factor
        norm_factor    = self.norm_factor

        self.aveS  = np.zeros( (self.M, self.Nd), dtype=DTYPE)   # average S at i node
        self.aveI  = np.zeros( (self.M, self.Nd), dtype=DTYPE)   # average I at i node
        self.Lambda= np.zeros( (self.M, self.Nd), dtype=DTYPE)   # effective infection rate
        #self.drpdt = np.zeros( 8*self.Nd*self.M, dtype=DTYPE)    # right hand side
        self.FF    = np.zeros( (self.M, self.Nd), dtype=DTYPE)       # Working memory
        self.ir    = 0

        self.define_model(model, parameters)

        if g_model == 'RapidTransport':
            self.geographical_model = self.RapidTransport
        elif g_model == 'SpacialCompartment':
            self.geographical_model = self.SpacialCompartment
        else:
            self.geographical_model = self.RapidTransport

        if g_model == 'SpacialCompartment':
            self.w_period_ratio = 10.0/24.0
            self.h_period_ratio = 14.0/24.0
        else:
            self.w_period_ratio = 1.0
            self.h_period_ratio = 0.0

        if transport == 1:
            self.Ntrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# people comuteing i->j
            self.Ntrans0= np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# people comuteing i->j
            self.iNtrans = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE)# inv people comuteing i->j
            self.II = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
            self.IT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
            self.TT = np.zeros( (self.M+1, self.Nd, self.Nd), dtype=DTYPE) # Working memory
            self.PP = np.zeros( (self.Nd, self.Nd), dtype=np.int32)   # Working memory

            from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
            from scipy.sparse import csr_matrix
        
            print('#Start finding the shortest path between each node')
            self.II[0], self.PP = shortest_path(dnm, return_predecessors=True)
        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(travel_restriction, 0)
        print('#Finish calculating fixed variables')
        
        #Set local contact matrix
        cdef scaling_params = np.zeros(3, dtype=DTYPE)
        cdef f_type gij
        cdef double aij_h, aij_w
        cdef double gij_h, gij_w
        cdef double fij_h, fij_w
        cdef double rho_ij_h, rho_ij_w
        cdef double N_ill
        cdef int i,j,k
        cdef iAreaSQ  = np.zeros( self.Nd, dtype=DTYPE)
        cdef Nh=self.Nh, iNh=self.iNh
        cdef Nw=self.Nw, iNw=self.iNw
        cdef Nh_total = np.zeros( self.M, dtype=DTYPE)
        cdef Nw_total = np.zeros( self.M, dtype=DTYPE)

        if 'contact_scaling' in model['settings'] and 'contact_scaling_parameters' in model['settings']:
            gij = self.functor_from_name(model['settings']['contact_scaling'])
            scaling_params = np.array(model['settings']['contact_scaling_parameters'], dtype=DTYPE)
            if model['settings']['contact_scaling'] != 'none':
                print("contact scaling", model['settings']['contact_scaling'])
                #print(scaling_params)
                #print(gij(4, scaling_params[0], scaling_params[1], scaling_params[2]))

                for i in range(Nd):
                    iAreaSQ[i]  = 1.0/Area[i]/Area[i]
                    for j in range(M):
                        Nh_total[j] += Nh[j,i]
                        Nw_total[j] += Nw[j,i]
                        
                for i in range(M):
                    for j in range(M):
                        aij_h = 0.0
                        aij_w = 0.0
                        for ll in range(Nd):
                            rho_ij_h = Nh[i,ll]*Nh[j,ll]*iAreaSQ[ll]
                            rho_ij_w = Nw[i,ll]*Nw[j,ll]*iAreaSQ[ll]
                    
                            gij_h = gij(rho_ij_h, scaling_params[0], scaling_params[1], scaling_params[2])
                            gij_w = gij(rho_ij_w, scaling_params[0], scaling_params[1], scaling_params[2])
                            density_factor[0,ll,i,j] = gij_h
                            density_factor[1,ll,i,j] = gij_w

                            bij_h = sqrt(Nh[i,ll]*Nh[j,ll]/Nh_total[i]/Nh_total[j])
                            bij_w = sqrt(Nw[i,ll]*Nw[j,ll]/Nw_total[i]/Nw_total[j])
                            if fij_form == 0:
                                bij_h = 1.0
                                bij_w = 1.0

                            fij_h =  Nh_total[i]*iNh[i,ll]
                            fij_w =  Nh_total[i]*iNw[i,ll]

                            lCMh[i,j,ll] = gij_h*fij_h*bij_h
                            lCMw[i,j,ll] = gij_w*fij_w*bij_w

                            aij_h += gij_h*bij_h
                            aij_w += gij_w*bij_w

                        norm_factor[0,i,j] = aij_h
                        norm_factor[1,i,j] = aij_w
                        
                        for ll in range(Nd):
                            lCMh[i,j,ll] /= aij_h
                            lCMw[i,j,ll] /= aij_w
            else:
                print("contact scaling", model['settings']['contact_scaling'])

    def spatial_CM(self, np.ndarray[np.float_t, ndim=2] CM):
        cdef:
            int i,j,k,alp,gam
            double [:,:]   Nh      = self.Nh
            double [:,:]   Nw      = self.Nw
            double [:,:]   iNh     = self.iNh
            double [:,:]   iNw     = self.iNw
            double [:,:,:] lCMh    = self.lCMh
            double [:,:,:] lCMw    = self.lCMw
            unsigned short[:,:,:]Dnm= self.Dnm
            np.ndarray[np.float_t, ndim=4] spatial_CM = np.zeros((self.Nd, self.M, self.Nd, self.M))
            
        for alp in prange(self.M, nogil=True):
            for i in range(self.Nd):
                for gam in range(self.M):
                    spatial_CM[i,alp,i,gam] += (14.0/24.0)*lCMh[alp,gam,i]*CM[alp,gam]
                    for j in range(self.Nd):
                        for k in range(self.Nd):
                            spatial_CM[i,alp,j,gam] += (10.0/24.0)*lCMw[alp,gam,k]*CM[alp,gam]*Dnm[alp,k,i]*Dnm[gam,k,j]*iNw[gam,k]*iNh[alp,i]
        return spatial_CM

    def initialize(self, model, parameters, travel_restriction):
        """
        Parameters
        ----------
        parameters: dict
            Contains the following keys:
                alpha: float, np.array (M,)
                    fraction of infected who are asymptomatic.
                beta: float
                    rate of spread of infection.
                gE: float
                    rate of removal from exposeds individuals.
                gIa: float
                    rate of removal from asymptomatic individuals.
                gIs: float
                    rate of removal from symptomatic individuals.
                gIh: float
                    rate of removal for hospitalised individuals.
                gIc: float
                    rate of removal for idividuals in intensive care.
                fsa: float
                    fraction by which symptomatic individuals self isolate.
                fh  : float
                    fraction by which hospitalised individuals are isolated.
                sa: float, np.array (M,)
                    daily arrival of new susceptables.
                    sa is rate of additional/removal of population by birth etc
                hh: float, np.array (M,)
                    fraction hospitalised from Is
                cc: float, np.array (M,)
                    fraction sent to intensive care from hospitalised.
                mm: float, np.array (M,)
                    mortality rate in intensive care

        travel_restriction: float
            Restriction of travel between nodes. 1.0 means everyone stay home
        """
        self.rW     = parameters.get('rW')                # fitting parameter at work
        self.rT     = parameters.get('rT')                # fitting parameter in trans
        self.travel_restriction = travel_restriction
        self.Nh    = np.zeros( (self.M, self.Nd), dtype=DTYPE)         # # people living in node i
        self.Nw    = np.zeros( (self.M, self.Nd), dtype=DTYPE)         # # people working at node i
        self.iNh    = np.zeros( (self.M, self.Nd), dtype=DTYPE)        # inv people living in node i
        self.iNw    = np.zeros( (self.M, self.Nd), dtype=DTYPE)        # inv people working at node i
        self.seed    = 1
        np.random.seed(self.seed)
        self.RM    = np.random.rand(100*self.M*self.Nd) # random matrix 
        #self.PWRh  = np.zeros( (self.M, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        #self.PWRw  = np.zeros( (self.M, self.Nd, self.Nd), dtype=DTYPE)# probability of Dnm at w
        self.drpdt = np.zeros( 8*self.Nd*self.M, dtype=DTYPE)      # right hand side
        #self.indexJ= np.zeros( (self.M, self.Nd, self.Nd + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i 
        #self.indexI= np.zeros( (self.M, self.Nd, self.Nd + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j

        self.ir = 0
        self.t_old = 0

        self.define_model(model, parameters)

        if self.transport == 1:
            self.Ntrans = np.zeros( (self.M, self.Nd, self.Nd), dtype=DTYPE)# people comuteing i->j
            self.iNtrans = np.zeros( (self.M, self.Nd, self.Nd), dtype=DTYPE)# inv people comuteing i->j

        print('#Start to calculate fixed variables')
        self.prepare_fixed_variable(self.travel_restriction, 1)
        print('#Finish calculating fixed variables')

    cdef define_model(self, model, parameters):
        cdef:
            int M=self.M, Nd=self.Nd
            
        self.rW = parameters.get('rW')                # fitting parameter at work
        self.rT = parameters.get('rT')                # fitting parameter in trans

        #Model definition
        model_classes = model['settings']['classes']
        self.model_dim = len(model_classes)
        model_dim = self.model_dim
        self.drpdt = np.zeros( model_dim*self.Nd*self.M, dtype=DTYPE)    # right hand side
        
        self.infect_source  = np.zeros(model_dim + 1, dtype=np.uint16)
        infect_source       = self.infect_source
        self.infect_param   = np.zeros(model_dim + 1)
        infect_param        = self.infect_param  
        self.infected_class = np.zeros(model_dim + 1, dtype=np.uint16)
        infected_class      = self.infected_class
        self.infected_coeff = np.ones( (model_dim + 1, self.M))
        infected_coeff      = self.infected_coeff

        self.l_pos_source   = np.zeros((model_dim, model_dim + 1), dtype=np.uint16)
        l_pos_source        = self.l_pos_source 
        self.l_neg_source   = np.zeros((model_dim, model_dim + 1), dtype=np.uint16)
        l_neg_source        = self.l_neg_source
        self.l_pos_param    = np.zeros((model_dim, model_dim + 1))
        l_pos_param         = self.l_pos_param
        self.l_neg_param    = np.zeros((model_dim, model_dim + 1))
        l_neg_param         = self.l_neg_param
        class_to_cindex = {}
        for i in range(model_dim):
            class_to_cindex[model_classes[i]] = i
        print(class_to_cindex)
                          
        i_class = []
        for class_name in model:
            if class_name == 'settings':
                continue
            dmy = []
            for coupling_class, model_param in model[class_name]['infection']:
                dmy.append(coupling_class)
            dmy.sort()
            if len(model[class_name]['infection']) != 0:
                i_class.append(dmy)
        num_infect = 1
        for i in range(len(i_class) - 1):
            if i_class[0] != i_class[i]:
                num_infect += 1
        #print("Infection relation", num_infect)
        if num_infect != 1:
            print("Error!!, The relationship of susceptible must be 1 in present implementation.")
            sys.exit()

        num_infected = 0
        for class_name in model:
            if class_name == 'settings':
                continue
        
            num_inf = 0
            for coupling_class, model_param in model[class_name]['infection']:
                if model_param[0] == '-':
                    num_inf += 1
                    infect_source[num_inf] = class_to_cindex[coupling_class]
                    infect_param[num_inf] = parameters.get(model_param.replace('-',''))
                    infect_source[0] = num_inf
                    #infect_param[0]  = num_inf
    
            if len(model[class_name]['infection']) != 0:
                if model[class_name]['infection'][0][1][0] != '-':
                    num_infected += 1
                    #print('Infected')
                    infected_class[num_infected] = class_to_cindex[class_name]
                    coeff = model[class_name].get("inf_coefficient", "Not_Found")
                    if coeff != "Not_Found":
                        if 'bar_' in coeff:
                            vcoeff = parameters.get(coeff.replace('bar_',''))
                            #print('vcoeff', vcoeff)
                            if np.size(vcoeff) == 1:
                                infected_coeff[num_infected] = (1.0 - vcoeff)*np.ones(M)
                            elif np.size(vcoeff) == M:
                                infected_coeff[num_infected] = np.ones(M) - vcoeff
                            else:
                                print('inf_coefficient can be a number or an array of size M')
                        else:
                            vcoeff = parameters.get(coeff)
                            if np.size(vcoeff) == 1:
                                infected_coeff[num_infected] = vcoeff*np.ones(M)
                            elif np.size(vcoeff) == M:
                                infected_coeff[num_infected] = vcoeff
                            else:
                                print('inf_coefficient can be a number or an array of size M')
            
            num_neg = 0
            num_pos = 0
            for coupling_class, model_param in model[class_name]['linear']:
                if model_param[0] == '-':
                    num_neg += 1
                else:
                    num_pos += 1
            l_pos_source[class_to_cindex[class_name], 0] = num_pos
            l_neg_source[class_to_cindex[class_name], 0] = num_neg
            l_pos_param[class_to_cindex[class_name], 0] = num_pos
            l_neg_param[class_to_cindex[class_name], 0] = num_neg

            num_neg = 0
            num_pos = 0
            for coupling_class, model_param in model[class_name]['linear']:
                if model_param[0] == '-':
                    num_neg += 1
                    l_neg_source[class_to_cindex[class_name], num_neg] = class_to_cindex[coupling_class]
                    l_neg_param[class_to_cindex[class_name], num_neg] = parameters.get(model_param.replace('-',''))
                else:
                    num_pos += 1
                    l_pos_source[class_to_cindex[class_name], num_pos] = class_to_cindex[coupling_class]
                    l_pos_param[class_to_cindex[class_name], num_pos] = parameters.get(model_param)
        infected_class[0] = num_infected
        #print(infect_source)
        #print(infect_param)
        #print(infected_class)
        #print(infected_coeff)
        #End Model definition
        
    # boiler-plate code for obtaining the right functor
    # uses NULL to signalize an error
    cdef f_type functor_from_name(self, fun_name) except NULL:
        if fun_name == 'none':
            return scaling_none
        elif fun_name == 'linear':
            return scaling_linear
        elif fun_name == "powerlaw":
            return scaling_powerlaw
        elif fun_name == "exp":
            return scaling_exp
        elif fun_name == "log":
            return scaling_log
        else:
            raise Exception("unknown function name '{0}'".format(fun_name))

    cdef prepare_fixed_variable(self, _travel_restriction, _restart):
        cdef:
            int M=self.M, Nd=self.Nd, M1=self.M*self.Nd, max_route_num, restart=_restart
            #unsigned short i, j, k, alp, index_i, index_j, index_agj, ii, count
            int i, j, k, alp, index_i, index_j, ii, count
            double cutoff=self.cutoff, cij, ccij, t_restriction=_travel_restriction
            unsigned short[:,:,:] Dnm     = self.Dnm
            #double [:,:,:] Dnm0    = self.Dnm0
            #double [:,:,:] PWRh    = self.PWRh
            #double [:,:,:] PWRw    = self.PWRw
            double [:,:]   Nh     = self.Nh
            double [:,:]   Nw     = self.Nw
            double [:,:]   iNh    = self.iNh
            double [:,:]   iNw    = self.iNw
            #double [:,:] distances = self.distances
            unsigned short [:,:,:]   indexJ#  = self.indexJ
            unsigned short [:,:,:]   indexI#  = self.indexI
            #unsigned short [:,:,:,:] indexAGJ= self.indexAGJ
            unsigned short [:] route           # working memory
            #double [:,:,:] C_Dnm   = self.IT
            unsigned short [:,:,:] route_index # the list of node in the route i -> j
            float [:,:,:] route_ratio          # the list of the distance ratio in the route i -> j
            double [:,:,:] iNtrans
            double [:,:,:] Ntrans
            double [:,:,:] Ntrans0
            double [:,:]  dij    # the distance between node i and j
            int [:,:]     pred   # predecessor node belong the route i -> j

        if self.transport == 1:
            N_Matrix = np.zeros((M,Nd,Nd), dtype=DTYPE)
            route_num = np.zeros(Nd*Nd, dtype=int)
            iNtrans = self.iNtrans
            Ntrans  = self.Ntrans
            Ntrans0 = self.Ntrans0
            dij  = self.II[0]    # the distance between node i and j
            pred = self.PP       # predecessor node belong the route i -> j

        #if restart == 0:
        #    for alp in prange(M, nogil=True):
        #        for i in range(Nd):
        #            for j in range(Nd):
        #                Dnm0[alp,i,j] = Dnm[alp,i,j]
        #else:
        #    for alp in prange(M, nogil=True):
        #        for i in range(Nd):
        #            for j in range(Nd):
        #                Dnm[alp,i,j] = Dnm0[alp,i,j]

        #Total number ofresidence people
        for alp in prange(M, nogil=True):
            for i in range(Nd):
                Nh[alp,i] = 0.0 ## N^{H}_i residence in Node i and age gourp alp
                for j in range(Nd):
                    Nh[alp,i] += <double> Dnm[alp,j,i]

        #travel restriction
        for alp in prange(M, nogil=True):
            for i in range(Nd):
                for j in range(Nd):
                    if i != j:
                        cij = Dnm[alp,i,j]
                        #ccij = round(cij*t_restriction)
                        ccij = cij*t_restriction
                        Dnm[alp,i,j] -= <unsigned short> ccij
                        Dnm[alp,j,j] += <unsigned short> ccij
    
        #cutoff
        cdef int nonzero_element = 0
        cdef double cutoff_total = 0.0
        #C_Dnm = Dnm.copy()
        for alp in prange(M, nogil=True):
            for i in range(Nd):
                for j in range(Nd):
                    cij = Dnm[alp,i,j]
                    if i != j:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            if alp != M:
                                nonzero_element += 1
                        else:
                            Dnm[alp,i,j] = 0
                            #Dnm[alp,j,j] += <int> cij
                    if alp != M:
                        #if int(cij) > int(cutoff):
                        if cij > cutoff:
                            cutoff_total += cij
        print("Nonzero element " + str(nonzero_element) + '/' + str(M1**2) + ' ' + str(cutoff_total))
        
        for alp in prange(M, nogil=True):
            for i in range(Nd):
                Dnm[alp,i,i] = <unsigned short> Nh[alp,i]
                for j in range(Nd):
                    if i != j:
                        Dnm[alp,i,i] -= Dnm[alp,j,i]

        #Total number of working people
        for alp in prange(M, nogil=True):
            for i in range(Nd):
                Nw[alp,i] = 0.0 ## N^{w}_i working in Node i and age group alp
                for j in range(Nd):
                    Nw[alp,i] += <double> Dnm[alp,i,j]

        #for i in prange(Nd, nogil=True):
        #    for alp in range(M):
        #        Nh[M,i] += Nh[alp,i] ## N^{H}_i residence in Node i
        #        Nw[M,i] += Nw[alp,i] ## N^{w}_i working in Node i
        #for i in prange(Nd, nogil=True):
        #    iNh[M,i] = 1.0/Nh[M,i] ## check divided zero
        #    iNw[M,i] = 1.0/Nw[M,i]
            
        #Generating the Ntrans from route and predecessor
        cdef total_route_index = 0
        if restart == 0 and self.transport == 1:
            for i in range(Nd):
                for j in range(Nd):
                    if i != j:# and Dnm[M,j,i] > cutoff:
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
                                if Dnm[alp,j,i] > cutoff:
                                    N_Matrix[alp,route[k + 1],route[k]] += Dnm[alp,j,i]
                                    count += 1
                        total_route_index += len(route)
                        route_num[i*Nd + j] = len(route)
            route_num.sort() # should be improved
            max_route_num = route_num[Nd**2 - 1]
            print("Max route number", route_num[0], max_route_num)
            print("Total index in all route", total_route_index, 1.0*total_route_index/Nd**2)

            self.route_index = np.zeros((Nd,Nd,max_route_num + 1), dtype=np.uint16)
            self.route_ratio = np.zeros((Nd,Nd,max_route_num), dtype=np.float32)
            route_index = self.route_index
            route_ratio = self.route_ratio
            for i in range(Nd):
                for j in range(Nd):
                    if i != j: # and Dnm[M,j,i] > cutoff:

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
            for i in prange(Nd, nogil=True):
                for j in range(Nd):
                    for alp in range(M):
                        #Ntrans[M,i,j] += Ntrans[alp,i,j] ## N^{t}_{ij} the effective number of the people using the route j->i
                        Ntrans0[alp,i,j] = Ntrans[alp,i,j]
                    #Ntrans0[M,i,j] = Ntrans[M,i,j]
        elif self.transport == 1:
            for alp in prange(M, nogil=True):
                for i in range(Nd):
                    for j in range(Nd):
                        Ntrans[alp,i,j] = Ntrans0[alp,i,j]
        #End Generating the Ntrans from route and predecessor

        cdef move_J_num = np.zeros(M*Nd, dtype=int)
        cdef move_I_num = np.zeros(M*Nd, dtype=int)
        
        for alp in range(M):
            for i in range(Nd):
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
                for j in range(Nd):
                    #if Nh[alp,j] != 0:
                    #    PWRh[alp,i,j] = Dnm[alp,i,j]/Nh[alp,j]
                    #else:
                    #    PWRh[alp,i,j] = 0.0
                
                    #if Nw[alp,i] != 0:
                    #    PWRw[alp,i,j] = Dnm[alp,i,j]/Nw[alp,i]
                    #else:
                    #    PWRw[alp,i,j] = 0.0

                    if self.transport == 1:
                        if Ntrans[alp,i,j] != 0:
                            iNtrans[alp,i,j] = 1.0/Ntrans[alp,i,j]
                        else:
                            iNtrans[alp,i,j] = 0.0
                    
                    if Dnm[alp,i,j] > cutoff or i == j:
                        #indexJ[alp,i,index_j + 1] = j
                        index_j += 1
                    elif Dnm[alp,i,j] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,i,j]) + '\n')

                    if Dnm[alp,j,i] > cutoff or i == j:
                        #indexI[alp,i,index_i + 1] = j
                        index_i += 1
                    elif Dnm[alp,j,i] != 0 and i != j:
                        print('Error!! ' + str(i) + ',' + str(j) + ' ' + str(Dnm[alp,j,i]) + '\n')
                    
                #indexJ[alp,i,0] = index_j
                #indexI[alp,i,0] = index_i

                move_J_num[Nd*alp + i] = <int> index_j
                move_I_num[Nd*alp + i] = <int> index_i

        move_J_num.sort() # should be improved
        max_move_J = move_J_num[M*Nd - 1]
        print("Max index J", max_move_J)
        self.indexJ = np.zeros( (self.M, self.Nd, max_move_J + 1), dtype=np.uint16) # the list j for non zero Dnm_alp_ij at specifi alp and i
        indexJ = self.indexJ
        
        move_I_num.sort() # should be improved
        max_move_I = move_I_num[M*Nd - 1]
        print("Max index I", max_move_I)
        self.indexI = np.zeros( (self.M, self.Nd, max_move_I + 1), dtype=np.uint16) # the list i for non zero Dnm_alp_ij at specifi alp and j
        indexI = self.indexI

        for alp in range(M):
            for i in range(Nd):
                index_j = 0
                index_i = 0
                for j in range(Nd):
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

        cdef double total_raw, total_J, total_I
        for alp in range(M):
            for i in range(Nd):
                total_raw = 0.0
                for j in range(Nd):
                    total_raw += Dnm[alp,i,j]
                total_J = 0.0
                for j in range(1, indexJ[alp,i,0] + 1):
                    total_J += Dnm[alp, i, indexJ[alp,i,j]]
                if total_raw != total_J:
                    print("Check_J", alp, i, total_raw, total_J)
                total_raw = 0.0
                for j in range(Nd):
                    total_raw += Dnm[alp,j,i]
                total_I = 0.0
                for j in range(1, indexI[alp,i,0] + 1):
                    total_I += Dnm[alp, indexI[alp,i,j], i]
                if total_raw != total_I:
                    print("Check_I", alp, i, total_raw, total_I)

    cdef rhs(self, rp, tt):
        cdef:
            int transport=self.transport, highSpeed=self.highSpeed, model_dim=self.model_dim
            int geographical_model = self.geographical_model
            int M=self.M, Nd=self.Nd, M1=self.M*self.Nd, t_divid_100=int(tt/100)
            #unsigned short i, j, k, alp, gam, age_id, ii, jj
            int i, j, k, alp, gam, age_id, ii, jj
            unsigned long t_i, t_j
            #double beta=self.beta
            #double gIa=self.gIa
            #double fsa=self.fsa, fh=self.fh, gE=self.gE
            #double gIs=self.gIs, gIh=self.gIh, gIc=self.gIc
            double rW=self.rW, rT=self.rT
            double w_period_ratio=self.w_period_ratio
            double h_period_ratio=self.h_period_ratio
            double aa=0.0, bb=0.0, ccc=0.0, t_p_24 = tt%24
            double [:] X0                     = rp
            unsigned short [:] infect_source  = self.infect_source
            double [:] infect_param           = self.infect_param
            unsigned short [:] infected_class = self.infected_class
            double [:,:] infected_coeff       = self.infected_coeff
            unsigned short [:,:] l_neg_source = self.l_neg_source
            unsigned short [:,:] l_pos_source = self.l_pos_source
            double [:,:] l_neg_param          = self.l_neg_param
            double [:,:] l_pos_param          = self.l_pos_param
            #double [:] E           = rp[M1  :2*M1]
            #double [:] Ia          = rp[2*M1:3*M1]
            #double [:] Is          = rp[3*M1:4*M1]
            #double [:] Ih          = rp[4*M1:5*M1]
            #double [:] Ic          = rp[5*M1:6*M1]
            #double [:] Im          = rp[6*M1:7*M1]
            #double [:] N           = rp[7*M1:8*M1] # must be same as Nh
            #double [:] ce1         = self.gE*self.alpha
            #double [:] ce2         = self.gE*self.alphab
            #double [:] hh          = self.hh
            #double [:] cc          = self.cc
            #double [:] mm          = self.mm
            double [:,:]   Nh      = self.Nh
            double [:,:]   Nw      = self.Nw
            #double [:]     Ni      = self.Ni
            double [:,:,:] Ntrans  = self.Ntrans
            double [:,:]   iNh     = self.iNh
            double [:,:]   iNw     = self.iNw
            double [:,:,:] iNtrans = self.iNtrans
            double [:]     RM      = self.RM
            double [:,:]   CMh     = self.CMh
            double [:,:]   CMw     = self.CMw
            double [:,:]   CMt     = self.CMt
            double [:,:,:] lCMh    = self.lCMh
            double [:,:,:] lCMw    = self.lCMw
            #double [:,:,:] Dnm     = self.Dnm
            unsigned short[:,:,:]Dnm= self.Dnm
            #double [:,:,:] PWRh    = self.PWRh
            #double [:,:,:] PWRw    = self.PWRw
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
        if (t_p_24 < 8.0 or t_p_24 > 18.0) and geographical_model == self.RapidTransport: #HOME
        #if True: #HOME
            #print("TIME_in_HOME", t_p_24)
            for i in prange(Nd, nogil=True):
                for alp in range(M):
                    t_i = alp*Nd + i
                    aa = 0.0
                    if X0[t_i] > 0.0:
                        FF[alp,i] = 0.0
                        #bb = 0.0
                        for gam in range(M):
                            t_j = gam*Nd + i
                            #ccc = 0.0
                            aveI[gam,i] = 0.0 #ave. infected source
                            for j in range(1,infect_source[0] + 1):
                                #ccc += X0[t_j + infect_source[j]*M1]*infect_param[j]
                                aveI[gam,i] += X0[t_j + infect_source[j]*M1]*infect_param[j]
                            FF[alp,i] += lCMh[alp,gam,i]*CMh[alp,gam]*aveI[gam,i]*iNh[gam,i]
                        aa = FF[alp,i]*X0[t_i]

                    #Linear
                    for j in range(model_dim):
                        X[t_i + j*M1] = 0.0
                        for k in range(1, l_pos_source[j, 0] + 1):
                            X[t_i + j*M1] += X0[t_i + l_pos_source[j, k]*M1]*l_pos_param[j, k]
                        for k in range(1, l_neg_source[j, 0] + 1):
                            if X0[t_i + j*M1] > 0.0:
                                X[t_i + j*M1] -= X0[t_i + l_neg_source[j, k]*M1]*l_neg_param[j, k]
                    #Infection
                    X[t_i] -= aa
                    for j in range(1, infected_class[0] + 1):
                        X[t_i + infected_class[j]*M1] += infected_coeff[j,alp]*aa

        elif (t_p_24  > 9.0 and t_p_24  < 17.0 or transport == 0) or geographical_model != self.RapidTransport: #WORK
        #elif True: #WORK
            #print("TIME_in_WORK", t_p_24)
            #for i in prange(Nd, nogil=True):
            for i in range(Nd):
                for alp in range(M):
                    aveI[alp,i] = 0.0
                    for j in range(1, indexJ[alp,i,0] + 1):
                        jj = indexJ[alp,i,j]
                        t_j = alp*Nd + jj
                        #aveI[alp,i] += PWRh[alp,i,jj]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])
                        #ccc = 0.0
                        FF[alp,i] = 0.0
                        for k in range(1,infect_source[0] + 1):
                            FF[alp,i] += X0[t_j + infect_source[k]*M1]*infect_param[k]
                        aveI[alp,i] += Dnm[alp,i,jj]*iNh[alp,jj]*FF[alp,i] ## \sum_j Pij Ij
                for alp in range(M):
                    Lambda[alp, i] = 0.0
                    for gam in range(M):
                        Lambda[alp,i] += lCMw[alp,gam,i]*CMw[alp,gam]*aveI[gam,i]*iNw[gam,i]
                    Lambda[alp, i] = rW*Lambda[alp,i]
                            
            #for i in prange(Nd, nogil=True):
            for i in range(Nd):
                for alp in range(M):
                    t_i = alp*Nd + i
                    for j in range(model_dim):
                        X[t_i + j*M1] = 0.0
                        for k in range(1, l_pos_source[j, 0] + 1):
                            X[t_i + j*M1] += X0[t_i + l_pos_source[j, k]*M1]*l_pos_param[j, k]
                        for k in range(1, l_neg_source[j, 0] + 1):
                            if X0[t_i + j*M1] > 0.0:
                                X[t_i + j*M1] -= X0[t_i + l_neg_source[j, k]*M1]*l_neg_param[j, k]
                    for j in range(1, indexI[alp,i,0] + 1):
                        ii = indexI[alp,i,j]
                        if X0[t_i] > 0.0:
                            #aa = Lambda[alp,ii]*PWRh[alp,ii,i]*S[t_i]
                            aa = Lambda[alp,ii]*Dnm[alp,ii,i]*iNh[alp,i]*X0[t_i]
                            #aa *= Ni[alp]*iNw[alp,ii] #Adapt I
                            aa *= w_period_ratio
                            X[t_i]        += -aa
                            for k in range(1, infected_class[0] + 1):
                                X[t_i + infected_class[k]*M1] += infected_coeff[k,alp]*aa

                    if geographical_model == self.SpacialCompartment:
                        aa = 0.0
                        if X0[t_i] > 0.0:
                            FF[alp,i] = 0.0
                            #bb = 0.0
                            for gam in range(M):
                                t_j = gam*Nd + i
                                #ccc = 0.0
                                aveI[alp,i] = 0.0 #ave. infected source
                                for j in range(1,infect_source[0] + 1):
                                    #ccc += X0[t_j + infect_source[j]*M1]*infect_param[j]
                                    aveI[alp,i] += X0[t_j + infect_source[j]*M1]*infect_param[j]
                                FF[alp,i] += lCMh[alp,gam,i]*CMh[alp,gam]*aveI[alp,i]*iNh[gam,i]
                            aa = FF[alp,i]*X0[t_i]
                            aa *= h_period_ratio
                            #aa *= Ni[alp]*iNh[alp,i] #Adapt I

                            #Infection
                            X[t_i] += -aa
                            for j in range(1, infected_class[0] + 1):
                                X[t_i + infected_class[j]*M1] += infected_coeff[j,alp]*aa

        else: #TRANS
            #print("TIME_in_TRANS", t_p_24)
            if highSpeed == 1: # using averaged \hat{I}_{ij}
                for alp in prange(M, nogil=True):
                    aveI[alp,0] = 0.0 # total I in age group alp
                    aveS[alp,0] = 0.0 # total Nh in age group alp
                    for i in range(Nd):
                        t_i = alp*Nd + i
                        for j in range(1,infect_source[0] + 1):
                            aveI[alp,0] += X0[t_i + infect_source[j]*M1]*infect_param[j]
                        #aveI[alp,0] += Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                        aveS[alp,0] += Nh[alp,i]
                    aveI[alp,0] = aveI[alp,0]/aveS[alp,0] # total I/Nh in age group alp

                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        FF[alp,i] = 0.0
                        for gam in range(M):
                            aveS[alp,i] = 0.0 #= Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                            for j in range(1,infect_source[0] + 1):
                                aveS[alp,i] += X0[t_i + infect_source[j]*M1]*infect_param[j]
                            FF[alp,i] += CMt[alp,gam]*Dnm[gam,i,i]*iNh[gam,i]*aveS[alp,i]*Dnm[alp,i,i]*iNh[alp,i]*iNw[gam,i] # add i->i
                            aa = aveI[gam,0]
                            age_id = alp
                            if indexI[alp,i,0] > indexI[gam,i,0]:
                                age_id = gam
                            for j in range(1, indexI[age_id,i,0] + 1):
                                ii = indexI[age_id,i,j]
                                t_j = gam*Nd + i
                                aveS[alp,i] = 0.0 #= Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                                for k in range(1,infect_source[0] + 1):
                                    aveS[alp,i] += X0[t_j + infect_source[k]*M1]*infect_param[k]
                                #bb = Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                                FF[alp,i] += CMt[alp,gam]*(Dnm[gam,ii,i]*iNh[gam,i]*aveS[alp,i] + (Ntrans[gam,ii,i] - Dnm[gam,ii,i])*aa)*Dnm[alp,ii,i]*iNh[alp,i]*iNtrans[gam,ii,i] # add i->j
                                
                        aa = rT*FF[alp,i]*X[t_i]

                        #Linear
                        for j in range(model_dim):
                            X[t_i + j*M1] = 0.0
                            for k in range(1, l_pos_source[j, 0] + 1):
                                X[t_i + j*M1] += X0[t_i + l_pos_source[j, k]*M1]*l_pos_param[j, k]
                            for k in range(1, l_neg_source[j, 0] + 1):
                                if X0[t_i + j*M1] > 0.0:
                                    X[t_i + j*M1] -= X0[t_i + l_neg_source[j, k]*M1]*l_neg_param[j, k]
                        #Infection
                        X[t_i]        = -aa
                        for j in range(1, infected_class[0] + 1):
                            X[t_i + infected_class[j]*M1] = infected_coeff[j,alp]*aa

                    
                        #X[t_i]        = -aa
                        #X[t_i + M1]   = aa - gE*E[t_i]
                        #X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        #X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        #X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        #X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]

            else: # using accurate \hat{I}_{ij}
                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        aveI[alp,i] = 0.0 #= Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                        for j in range(1, infect_source[0] + 1):
                            #aveI[alp,i] = Ia[t_i] + fsa*Is[t_i] + fh*Ih[t_i]
                            aveI[alp,i] += X0[t_i + infect_source[j]*M1]*infect_param[j]
                    for j in range(Nd):
                        for alp in range(M):
                            II[alp,i,j] = 0.0
                            IT[alp,i,j] = 0.0
                            TT[alp,i,j] = 0.0
                            Ntrans[alp,i,j] = 0.0
                        
                for i in range(Nd):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            if route_index[i,jj,0] >= 2:
                                for k in range(1, route_index[i,jj,0]):
                                    #II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += PWRh[alp,i,jj]*aveI[alp,jj]
                                    II[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += Dnm[alp,i,jj]*iNh[alp,jj]*aveI[alp,jj]
                                    TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]] += Dnm[alp,i,jj]

                for i in prange(Nd, nogil=True):
                    for alp in range(M):
                        for j in range(1, indexJ[alp,i,0] + 1):
                            jj  = indexJ[alp,i,j]
                            for k in range(1, route_index[i,jj,0]):
                                IT[alp,i,jj]     += II[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                                Ntrans[alp,i,jj] += TT[alp,route_index[i,jj,k+1],route_index[i,jj,k]]*route_ratio[i,jj,k]
                
                for i in range(Nd):
                    for alp in range(M):
                        t_i = alp*Nd + i
                        bb = 0.0
                        for gam in range(M):
                            for j in range(1, indexI[alp,i,0] + 1):
                                ii = indexI[alp,i,j]
                                if Ntrans[gam,ii,i] > 0.0:
                                    #bb += CMt[alp,gam]*IT[gam,ii,i]*PWRh[alp,ii,i]/Ntrans[gam,ii,i]
                                    bb += CMt[alp,gam]*IT[gam,ii,i]*Dnm[alp,ii,i]*iNh[alp,i]/Ntrans[gam,ii,i]
                            t_j = gam*Nd + i
                            #bb += CMt[alp,gam]*PWRh[gam,i,i]*(Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j])*iNw[gam,i]*PWRh[alp,i,i]
                            aveS[alp,i] = 0.0 #= Ia[t_j] + fsa*Is[t_j] + fh*Ih[t_j]
                            for j in range(1,infect_source[0] + 1):
                                aveS[alp,i] += X0[t_j + infect_source[j]*M1]*infect_param[j]
                            bb += CMt[alp,gam]*Dnm[gam,i,i]*iNh[gam,i]*(aveS[alp,i])*iNw[gam,i]*Dnm[alp,i,i]*iNh[alp,i]
                        aa = rT*bb*X0[t_i]

                        #Linear
                        for j in range(model_dim):
                            X[t_i + j*M1] = 0.0
                            for k in range(1, l_pos_source[j, 0] + 1):
                                X[t_i + j*M1] += X0[t_i + l_pos_source[j, k]*M1]*l_pos_param[j, k]
                            for k in range(1, l_neg_source[j, 0] + 1):
                                if X0[t_i + j*M1] > 0.0:
                                    X[t_i + j*M1] -= X0[t_i + l_neg_source[j, k]*M1]*l_neg_param[j, k]
                        #Infection
                        X[t_i]        = -aa
                        for j in range(1, infected_class[0] + 1):
                            X[t_i + infected_class[j]*M1] = infected_coeff[j,alp]*aa

                    
                        #X[t_i]        = -aa
                        #X[t_i + M1]   = aa - gE*E[t_i]
                        #X[t_i + 2*M1] = ce1[alp]*E[t_i] - gIa*Ia[t_i]
                        #X[t_i + 3*M1] = ce2[alp]*E[t_i] - gIs*Is[t_i]
                        #X[t_i + 4*M1] = hh[alp]*gIs*Is[t_i] - gIh*Ih[t_i]
                        #X[t_i + 5*M1] = cc[alp]*gIh*Ih[t_i] - gIc*Ic[t_i]

        # procedure of decreasing the number of people          
        for i in range(Nd):
            for alp in range(M):
                t_i = alp*Nd + i
                aa = Nh[alp,i] - X0[t_i + (model_dim - 1)*M1]
                if aa > 0.0 and self.mortal == 1: # procedure of decreasing the number of people
                    #Nh[M,i]    -= aa
                    Nh[alp,i]  -= aa
                    if Nh[alp,i] > 0.1:
                        iNh[alp,i]  = 1.0/Nh[alp,i]
                        # determin decreased work place index
                        ccc = RM[self.ir]
                        self.ir += 1        ##
                        ii = <int> (ccc*indexI[alp,i,0]) + 1 # decreased work index
                        k = indexI[alp,i,ii] # decreased work place ##
                        Nw[alp,k]    -= aa  ##
                        if Nw[alp,k] > 0.0:
                            iNw[alp,k] = 1.0/Nw[alp,k]
                        else:
                            Nw[alp,k]  = 0.0
                            iNw[alp,k] = 0.0
                        Dnm[alp,k,i] -= <int> aa
                        if Dnm[alp,k,i] <= 0.0:
                            indexI[alp,i,ii] = indexI[alp,i,indexI[alp,i,0]]
                            indexI[alp,i,0] -= 1
                            Dnm[alp,k,i] = 0
                        #PWRh[alp,k,i] = Dnm[alp,k,i]*iNh[alp,i]
                        #PWRw[alp,k,i] = Dnm[alp,k,i]*iNw[alp,k]
                    else:
                        X0[t_i + (model_dim - 1)*M1] = 0.0
                        Nh[alp,i]  = 0.0
                        iNh[alp,i] = 0.0
                        indexI[alp,i,0] = 0

        return

    def simulate(self, XX0, contactMatrix, Tf, Nf, Ti=0, method='default'):
        """
        Parameters
        ----------
        XX0: np.array
            Initial numbers of Model Variable.
        contactMatrix: python function(t)
             The social contact matrix C_{ij} denotes the
             average number of contacts made per day by an
             individual in class i with an individual in class j
        Tf: float
            Final time of integrator
        Nf: Int
            Number of time points to evaluate.
        Ti: float, optional
            Start time of integrator. The default is 0.
        #highSpeed: int, optional
        #    Flag of more coasening calculation.

        Returns
        -------
        dict
            'X': output path from integrator, 't': time points evaluated at,
            'param': input param to integrator.
        """
        
        if self.geographical_model == self.RapidTransport:
            print('RapidTransport', self.fij_form)
        elif self.geographical_model == self.SpacialCompartment:
            print('SpacialCompartment', self.fij_form)

        print('travel restriction', self.travel_restriction)
        print('cutoff', self.cutoff)
        print('highspeed', self.highSpeed)
        self.CMh  = contactMatrix(6.0)
        self.CMt  = contactMatrix(8.5)
        self.CMw  = contactMatrix(12.0)

        print("#Calculation Start")
        
        def rhs0(t, rp):
            if self.ir > 99*self.M*self.Nd:
                np.random.seed()
                self.RM    = np.random.rand(100*self.M*self.Nd)
                self.ir = 0
            self.rhs(rp, t)
            return self.drpdt

        from scipy.integrate import solve_ivp
        time_points=np.linspace(Ti, Tf, int(Tf/24 + 1));  ## intervals at which output is returned by integrator.
        time_step = 1.0*Tf/Nf
        if method == 'RK23':
            print('Numerical Integrate RK23')
            u = solve_ivp(rhs0, [Ti, Tf], XX0, method='RK23', t_eval=time_points, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
        elif method == 'RK45':
            print('Numerical Integrate RK45')
            u = solve_ivp(rhs0, [Ti, Tf], XX0, method='RK45', t_eval=time_points, dense_output=False, events=None, vectorized=False, args=None, max_step=time_step)
        else:
            print('Numerical Integrate default')
            u = solve_ivp(rhs0, [Ti, Tf], XX0, t_eval=time_points)

        
        #data={'X':u.y, 't':u.t, 'N':self.Nd, 'M':self.M,'alpha':self.alpha, 'beta':self.beta,'gIa':self.gIa, 'gIs':self.gIs }
        data={'X':u.y, 't':u.t, 'N':self.Nd, 'M':self.M}

        return data

