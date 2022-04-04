from cmath import tau
from inspect import trace
from logging import NullHandler
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

# general imports
import pystan
import numpy as np
from pathlib import Path
import pickle

# simulation parameters
T = 200              # number of time steps to simulate and record measurements for
n_u = 2
n_y = 2
n_x = 4

class Parameters:
    ...

class Data:
    ...

par = Parameters()

# input nonlinearity truths
par.sat_lower1 = -6.
par.sat_upper1 = 6.
par.dzone_left1 = -1.
par.dzone_right1 = 1.
par.alpha = np.array([par.sat_upper1,par.sat_lower1,par.dzone_left1,par.dzone_right1])

# output nonlinearity truths
par.dzone_left2 = -1.
par.dzone_right2 = 1.
par.sat_lower2 = -5
par.sat_upper2 = 5
par.beta = np.array([par.dzone_left2,par.dzone_right2,par.sat_upper2,par.sat_lower2])


# purturbed values acting as an initial 'estimate'
par.sat_lower1_start = par.sat_lower1 + np.random.uniform(-0.1,0.1)
par.sat_upper1_start = par.sat_upper1 + np.random.uniform(-0.1,0.1)
par.dzone_left1_start= par.dzone_left1 + np.random.uniform(-0.1,0.1)
par.dzone_right1_start = par.dzone_right1 + np.random.uniform(-0.1,0.1)
par.dzone_left2_start = par.dzone_left2 + np.random.uniform(-0.1,0.1)
par.dzone_right2_start = par.dzone_right2 + np.random.uniform(-0.1,0.1)
par.sat_lower2_start = par.sat_lower2 + np.random.uniform(-0.1,0.1)
par.sat_upper2_start = par.sat_upper2 + np.random.uniform(-0.1,0.1)

par.alpha_start = np.array([par.sat_upper1_start,par.sat_lower1_start,par.dzone_left1_start,par.dzone_right1_start])
par.beta_start = np.array([par.dzone_left2_start,par.dzone_right2_start,par.sat_upper2_start,par.sat_lower2_start])
par.alpha_start = par.alpha
par.beta_start = par.beta


# Converted 4th order state space from Automatica paper transfer function model via subspace + refinement
par.A = np.array([[-0.024523620064597,   0.234584563723593,  -0.305707254662992,   0.018543780534269],
                  [ 0.030847977137114,  -0.373575676737083,   0.328847355994889,  -0.054782036427097],
                  [-0.152531048119485,   1.017691071266128,   0.280089201091929,   0.010327770978548],
                  [-0.675824092453352,   0.297333057752354,  -0.662923652782273,   0.888010095736634]], dtype = float)

par.B = np.array([[ 0.015096988944748,   0.002076099180373],
                  [ 0.003278444657333,  -0.007844677565818],
                  [ 0.001411430165043,  -0.015489965684947],
                  [ 0.001569567819373,  -0.013878203630782]], dtype = float)

par.C = np.array([[ -2.243189117364193, -26.193714897189544,  20.792628497591011, -33.844112878994345],
                  [-18.898901075772695,  -0.345713205812388, -21.907191710967155,  30.035215453590148]], dtype = float)

par.D = np.array([[ 1.100006390153607,   0.000011904983774],
                  [-0.860024547340833,  -0.000045732186252]], dtype = float)

# 5% uniform error  --> starting at an 'estimate' that is close to the truth
par.A_start = par.A #+ np.multiply(0.02*np.abs(par.A),np.random.uniform(-1.,1.,(n_x,n_x)))
par.B_start = par.B #+ np.multiply(0.02*np.abs(par.B),np.random.uniform(-1.,1.,(n_x,n_u)))                
par.C_start = par.C #+ np.multiply(0.02*np.abs(par.C),np.random.uniform(-1.,1.,(n_y,n_x)))  
par.D_start = par.D #+ np.multiply(0.02*np.abs(par.D),np.random.uniform(-1.,1.,(n_y,n_u)))  

par.x_0 = np.array([[0.],[0.],[0.],[0.]]) # initial state assumed zero

# noise covariances
par.q = 0.1
par.sq = np.sqrt(par.q)
par.r = 0.1
par.sr = np.sqrt(par.r)
par.Q = par.q*np.eye(n_x) # ω (process noise) covariance
par.R = par.r*np.eye(n_y) # e (output error) covariance

def lti(x, u, ω, A,B,C,D):
    y = C@x + D@u
    x = A@x + B@u + ω
    return x,y

def sat(u,upper,lower):
    u = upper if u > upper else u
    u = lower if u < lower else u
    return u

def dzone(u,left,right):
    u = u - left if u < left else u
    u = 0 if left <= u <= right else u
    u = u - right if u > right else u
    return u
    
u = np.zeros((n_u,T))
w = np.zeros((n_u,T))
w_sim = np.zeros((n_u,T))
x = np.zeros((n_x,T+1))
x_sim = np.zeros((n_x,T+1))

# no noise 
s = np.zeros((n_y,T))
s_sim = np.zeros((n_y,T))

# noise corrupted output 
z = np.zeros((n_y,T))

ω = par.sq*np.random.standard_normal((n_y,T)) # process noise 
e = par.sr*np.random.standard_normal((n_y,T)) # output noise 

i = np.zeros((n_y,T))
y = np.zeros((n_y,T))

x[:,[0]] = par.x_0
x_sim[:,[0]] = par.x_0 # consistent 'pure' state 



# create some inputs that are random
u[0,:] = np.random.normal(0,7, T)
u[1,:] = np.random.normal(0,7, T)

# sim loop
for k in range(T):
    w[0,k] = sat(u[0,k],par.sat_upper1,par.sat_lower1)
    w[1,k] = dzone(u[1,k],par.dzone_left1,par.dzone_right1)
    w_sim[0,k] = sat(u[0,k],par.sat_upper1_start,par.sat_lower1_start)
    w_sim[1,k] = dzone(u[1,k],par.dzone_left1_start,par.dzone_right1_start)

    x[:,k+1],s[:,k] = lti(x[:,k],w[:,k], np.zeros(n_x), par.A,par.B,par.C,par.D)
    x_sim[:,k+1],s_sim[:,k] = lti(x_sim[:,k],w_sim[:,k], np.zeros(n_x),par.A_start,par.B_start,par.C_start,par.D_start)
    z[:,k] = s[:,k] + ω[:,k]
    i[0,k] = dzone(z[0,k],par.dzone_left2,par.dzone_right2)
    i[1,k] = sat(z[1,k],par.sat_upper2,par.sat_lower2)
    # i = s # uncomment to remove non-linearity
    y[:,k] = i[:,k] + e[:,k]

sim_data = Data() # stash the data 
sim_data.u = u
sim_data.w = w
sim_data.x = x
sim_data.s = s
sim_data.z = z
sim_data.s_sim = s_sim
sim_data.i = i
sim_data.y = y
sim_data.w_sim = w_sim
sim_data.x_sim = x_sim

plt.figure()
plt.plot(y[0,:])
plt.plot(z[0,:])

plt.figure()
plt.plot(y[1,:])
plt.plot(z[1,:])

plt.figure()
plt.plot(u[0,:])
plt.plot(w[0,:])

plt.figure()
plt.plot(u[1,:])
plt.plot(w[1,:])

#sampling parameters
M = 3000
wup = 1000
chains = 1
iter = wup + int(M/chains)
model_name = 'nd_hw_latent_z'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

def init_function(chain):
    initialiser = [] # empty list
    for ii in range(chains):
        init_dict = {'A': par.A_start, # start values that have a random uniform up to 5% error on every element
                     'B': par.B_start, 
                     'C': par.C_start, 
                     'D': par.D_start, 
                     'alpha': par.alpha_start, # randomly purturbed values 
                     'beta': par.beta_start,
                     'z': s_sim.transpose(), # look at initialising all parameters in a `consistent' max likelihood solution
                     # x_sim represents a value of the system state that is compatible with {A,B,C,D}_start and has no noise
                    #  'sq': par.sq*np.ones(n_x), # started at the truth --> will change with the representation of the system 
                     'x0_p': par.x_0.squeeze(),
                     'Q_tau': par.sq*np.ones(n_y), #np.random.uniform(0.1,0.3,(n_y)),
                     'Q_corr_chol': np.eye(n_y),
                     'R_tau': par.sr*np.ones(n_y), #np.random.uniform(0.1,0.3,(n_y)),
                     'R_corr_chol': np.eye(n_y),
        }
        initialiser.append(init_dict)
    output = dict(
        A = initialiser[chain]['A'],
        B = initialiser[chain]['B'],
        C = initialiser[chain]['C'],
        D = initialiser[chain]['D'],
        alpha = initialiser[chain]['alpha'],
        beta = initialiser[chain]['beta'],
        z = initialiser[chain]['z'],
        # sq = initialiser[chain]['sq'],
        x0_p = initialiser[chain]['x0_p'],
        Q_tau = initialiser[chain]['Q_tau'],
        Q_corr_chol = initialiser[chain]['Q_corr_chol'],
        R_tau = initialiser[chain]['R_tau'],
        R_corr_chol = initialiser[chain]['R_corr_chol']
    )
    return output

# four chains
init = [init_function(0)]#, init_function(1), init_function(2), init_function(3)]

stan_data = {
    'N': T,
    'n_u':n_u,
    'n_y':n_y,
    'n_x':n_x,
    'x0': par.x_0.squeeze(),
    'Q0': 0.1*np.eye(n_x),
    'y': y.transpose(),
    'u': u.transpose()
}

# with suppress_stdout_stderr():
fit = model.sampling(data=stan_data, warmup=wup, iter=iter, init=init,chains=chains, control=dict(metric = "diag_e")) # force dense mass matrix
traces = fit.extract()

# pickle run

run = 'results'

with open(run+'_traces.pkl','wb') as file:
    pickle.dump(traces,file)
with open(run+'_data.pkl','wb') as file:
    pickle.dump(sim_data,file)
with open(run+'_par.pkl','wb') as file:
    pickle.dump(par,file)
1 + 1




