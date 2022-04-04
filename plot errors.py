from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from pyparsing import line
import seaborn as sns
from helpers import plot_bode, plot_bode_mimo, plot_phase_mimo

class Parameters:
    ...

class Data:
    ...

run = 'run_latent_z_results_300_paper'
if Path(run + '_traces.pkl').is_file():
    traces = pickle.load(open(run + '_traces.pkl', 'rb'))

if Path(run + '_traces.pkl').is_file():
    par = pickle.load(open(run + '_par.pkl', 'rb'))

if Path(run + '_traces.pkl').is_file():
    sim_data = pickle.load(open(run + '_data.pkl', 'rb'))


y_hat = np.transpose(traces['y_hat_out'],(2,1,0))

ts = np.arange(y_hat.shape[1])
err1 = np.zeros((y_hat.shape[1],y_hat.shape[2]))
err2 = np.zeros((y_hat.shape[1],y_hat.shape[2]))
for ii in ts:
    err1[ii,:] = y_hat[0,ii,:] - sim_data.i[0,ii]
    err2[ii,:] = y_hat[1,ii,:] - sim_data.i[1,ii]

fig = plt.figure()
fig.set_size_inches(6.4, 4)
plt.subplot(2, 1, 1)
plt.plot(ts,err1.mean(axis = 1), color=u'#1f77b4',linewidth = 1,label='Mean of error')
plt.fill_between(ts,np.percentile(err1,97.5,axis=1),np.percentile(err1,2.5,axis=1),color=u'#1f77b4',alpha=0.15,label='95% interval')
plt.plot(ts,sim_data.y[0,:]-sim_data.i[0,:], color=u'#1f0000',linestyle='--',linewidth = 0.8,label='Measured value')
plt.axhline(y=0, color='r', linestyle='-', linewidth=0.8, label='True value')
plt.xlim([0,len(ts)])
plt.xlabel('Output 1')

plt.subplot(2, 1, 2)
plt.plot(ts,err2.mean(axis = 1), color=u'#1f77b4',linewidth = 1,label='Mean of error')
plt.fill_between(ts,np.percentile(err2,97.5,axis=1),np.percentile(err2,2.5,axis=1),color=u'#1f77b4',alpha=0.15,label='95% interval')
plt.plot(ts,sim_data.y[1,:]-sim_data.i[1,:], color=u'#1f0000',linestyle='--',linewidth = 0.8,label='Measured value')
plt.axhline(y=0, color='r', linestyle='-', linewidth=0.8,label= 'True value')
plt.xlim([0,len(ts)])
plt.suptitle('Prediction error')
plt.xlabel('Output 2')
plt.tight_layout()
plt.show()

omega = np.logspace(-2,1,200)
# plot_bode(traces['A'],traces['B'][:,:,[0]],traces['C'][:,[0],:],traces['D'][:,[0],[0]],par.A,par.B[:,[0]],par.C[[0],:],par.D[[0],[0]],omega)

plot_bode_mimo(traces['A'],traces['B'],traces['C'],traces['D'],par.A,par.B,par.C,par.D,2,2,omega)
plot_phase_mimo(traces['A'],traces['B'],traces['C'],traces['D'],par.A,par.B,par.C,par.D,2,2,omega)

f, axe = plt.subplots(ncols=2, nrows=2)

# f.set_size_inches(6.4, 6)
ax = sns.kdeplot(traces['alpha'][:,1], shade=True, ax = axe[0,0])
ax.axvline(par.sat_lower1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. min.')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,2], shade=True, ax = axe[0,1])
ax.axvline(par.dzone_left1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. left')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,3], shade=True, ax = axe[1,0])
ax.axvline(par.dzone_right1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. right')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['alpha'][:,0], shade=True, ax = axe[1,1])
ax.axvline(par.sat_upper1, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. max.')
ax.set(yticks=[])
ax.set(yticklabels=[])
plt.suptitle(r'Kernel density estimates for $\alpha$')
plt.tight_layout()
plt.show()

f2, axe = plt.subplots(ncols=2, nrows=2)
# f2.set_size_inches(6.4, 2)
ax = sns.kdeplot(traces['beta'][:,3], shade=True, ax = axe[0,0])
ax.axvline(par.sat_lower2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. min.')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,0], shade=True, ax = axe[0,1])
ax.axvline(par.dzone_left2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. left')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,1], shade=True, ax = axe[1,0])
ax.axvline(par.dzone_right2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Deadz. right')
ax.set(yticks=[])
ax.set(yticklabels=[])
ax = sns.kdeplot(traces['beta'][:,2], shade=True, ax = axe[1,1])
ax.axvline(par.sat_upper2, color='r', lw=1, linestyle='--')
ax.set(ylabel=None)
ax.set(xlabel='Sat. max.')
ax.set(yticks=[])
ax.set(yticklabels=[])
plt.suptitle(r'Kernel density estimates for $\beta$')
plt.tight_layout()
plt.show()


1+1
...