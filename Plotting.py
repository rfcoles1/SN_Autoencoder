import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats as stats

big = 32
med = 24
smol = 18

plt.rc('font', size=smol)
plt.rc('figure', titlesize=big)
plt.rc('legend', fontsize=smol)
plt.rc('axes', titlesize=med)
plt.rc('axes', labelsize=smol)
plt.rc('xtick', labelsize=smol)
plt.rc('ytick', labelsize=smol)

def plot_losses_recon(losses, ylim=1):
    iters = losses['iterations']
    losses = losses['loss']

    fig = plt.figure(figsize=(8,6))
    grid = gs.GridSpec(1,1, wspace=0.05)

    ax0 = fig.add_subplot(grid[0])
    ax0.set_title('Reconstruction Losses')
    ax0.set_xlabel('Batch Iterations')
    ax0.set_ylabel('Loss')
    ax0.plot(iters, losses, color='crimson', lw=2.5)
    #ax0.set_ylim(bottom=0, top=0.01)
    ax0.set_xlim(left=0)
    ax0.set_ylim(bottom=0, top=ylim)
    ax0.set_xlim(left=0)
    ax0.grid()

def plot_losses_synth(losses, ylim=[0.01, 0.1]):

    iters = losses['iterations']
    enc_y = losses['enc_y']
    enc_z = losses['enc_z']
    dec_x = losses['dec_x']

    fig = plt.figure(figsize=(16,6))
    grid = gs.GridSpec(1,2, wspace=0.05)

    ax0 = fig.add_subplot(grid[0])
    ax0.set_title('Regression Losses')
    ax0.set_xlabel('Batch Iterations')
    ax0.set_ylabel('Loss')
    ax0.plot(iters, enc_y, color='crimson', lw=2.5)
    #ax0.plot(iters, enc_z)
    ax0.set_ylim(bottom=0, top=ylim[0])
    ax0.set_xlim(left=0)
    ax0.grid()

    ax1 = fig.add_subplot(grid[1])
    ax1.set_title('Reconstruction Losses')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_xlabel('Batch Iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(iters, dec_x, color='slateblue', lw=2.5)
    ax1.set_ylim(bottom=0, top=ylim[1])
    ax1.set_xlim(left=0)
    ax1.grid()

def plot_losses_obs(losses, ylim=[.1,.1]):
    
    iters = losses['iterations']
    recons = losses['recon_loss']
    params = losses['param_loss']

    fig = plt.figure(figsize=(16,6))
    grid = gs.GridSpec(1,2, wspace=0.05)

    ax0 = fig.add_subplot(grid[0])
    ax0.set_title('Regression Losses')
    ax0.set_xlabel('Batch Iterations')
    ax0.set_ylabel('Loss')
    ax0.plot(iters, params, color='crimson', lw=2.5)
    ax0.set_ylim(bottom=0, top=ylim[0])
    ax0.set_xlim(left=0)
    ax0.grid()

    ax1 = fig.add_subplot(grid[1])
    ax1.set_title('Reconstruction Losses')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_xlabel('Batch Iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(iters, recons, color='slateblue', lw=2.5)
    ax1.set_ylim(bottom=0, top=ylim[1])
    ax1.set_xlim(left=0)
    ax1.grid()

def plot_losses_semi(losses, ylim=[0.01, 0.1]):

    iters = losses['iterations']
    obs_recon = losses['obs_recon_loss']
    synth_recon = losses['synth_recon_loss']
    param = losses['param_loss']

    fig = plt.figure(figsize=(16,6))
    grid = gs.GridSpec(1,2, wspace=0.05)

    ax0 = fig.add_subplot(grid[0])
    ax0.set_title('Regression Losses')
    ax0.set_xlabel('Batch Iterations')
    ax0.set_ylabel('Loss')
    ax0.plot(iters, param, color='crimson', lw=2.5)
    #ax0.plot(iters, enc_z)
    ax0.set_ylim(bottom=0, top=ylim[0])
    ax0.set_xlim(left=0)
    ax0.grid()

    ax1 = fig.add_subplot(grid[1])
    ax1.set_title('Reconstruction Losses')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_xlabel('Batch Iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(iters, obs_recon, label='Observations', color='slateblue', lw=2.5)
    ax1.plot(iters, synth_recon, label='Synthetic', color='peru', lw=2.5)
    ax1.set_ylim(bottom=0, top=ylim[1])
    ax1.set_xlim(left=0)
    ax1.grid()
    ax1.legend()

def plot_params_synth(true, pred, y_min, y_max):

    label_names = [r'$T_{\mathrm{eff}}$',r'$\log(g)$',r'$v_{micro}$',r'$[C/H]$', \
        r'$[N/H]$',r'$[O/H]$',r'$[Na/H]$',r'$[Mg/H]$',r'$[Al/H]$', \
        r'$[Si/H]$',r'$[P/H]$',r'$[S/H]$',r'$[K/H]$',r'$[Ca/H]$', \
        r'$[Ti/H]$',r'$[V/H]$',r'[$Cr/H$]',r'$[Mn/H]$',r'$[Fe/H]$', \
        r'$[Co/H]$',r'$[Ni/H]$',r'[$Cu/H$]',r'$[Ge/H]$',r'$[C12/C13]$',r'$v_{macro}$'] 


    #x_data = x_data*x_std.numpy() + x_mean.numpy()
    true = (true+0.5)*(y_max.numpy() - y_min.numpy()) + y_min.numpy()
    pred = (pred+0.5)*(y_max.numpy() - y_min.numpy()) + y_min.numpy()
    
    resid = true - pred
    bias = np.median(resid, axis=0)
    std = np.std(resid,axis=0)

    fig = plt.figure(figsize=(10,200))
    grid = gs.GridSpec(25,2, width_ratios=[4,1])

    xlims = [[2800,8000],[-.5,5.5],[-.3,3.6],*np.tile([-2.1,0.8],(20,1)),[-25,105],[-2,38]]
    
    ylims = [1000, 2., 5, *np.tile(2,20), 80, 10]
    ylims = [[-x,x] for x in ylims]


    for i in range(25):
        ax0 = fig.add_subplot(grid[i,0])
        ax0.set_xlabel(r'%s' %label_names[i])
        ax0.set_ylabel(r'$\Delta $%s' %label_names[i]) 
       
        ax0.scatter(true[:,i], resid[:,i], color='crimson')
        ax0.plot([xlims[i][0], xlims[i][1]], [0,0], color='dimgrey', linestyle='--')
        
        ax0.set_xlim(xlims[i][0], xlims[i][1])
        ax0.set_ylim(ylims[i][0], ylims[i][1])
        ax0.grid()
        
        ax1 = fig.add_subplot(grid[i,1])
        
        n, bin_edges = np.histogram(resid[:,i], 50)
        probs = n/np.shape(resid[:,i])[0]
        bin_mid = (bin_edges[1:]+bin_edges[:-1])/2.
        bin_wid = bin_edges[1]-bin_edges[0]
        (mu, sigma) = stats.norm.fit(resid[:,i])
        y = stats.norm.pdf(bin_mid, mu, sigma)*bin_wid

        ax1.plot(y, bin_mid, color='darkred', lw=2)
        ax1.plot([0,max(y)], [0,0], color='dimgrey', linestyle='--')
        
        ax1.set_xlim(0,max(y))
        ax1.set_ylim(ylims[i][0], ylims[i][1])
      
        ax1.get_xaxis().set_visible(False)
        ax1.grid(axis='y')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position('right')  


def plot_spectra_err(true, pred, mask=0):

    wavegrid = np.load('./data/wave_grid_ASPCAP.npy')
    start_w = len(wavegrid) - 7167
    wavegrid = wavegrid[start_w:]

    N = np.shape(true)[0]
    if len(np.shape(mask)) == 0:
        mask = np.ones([N,7167])

    idx = np.where(mask == 0)
    thistrue = np.copy(true) 
    thistrue[idx] = 0
    
    resid = thistrue - pred

    fig = plt.figure(figsize=(15,4))
    grid = gs.GridSpec(1,1)
    ax = fig.add_subplot(grid[0])
    ax.set_ylabel(r'Reconstruction Error')
    ax.set_xlabel(r'Wavelength')
    
    xlims = [15150, 16990]
    ax.set_xlim(left=xlims[0], right=xlims[1])
    ax.plot([xlims[0], xlims[1]], [0,0], color='k', linestyle='--')
    
    mean = np.mean(resid, axis=0)
    std = np.std(resid, axis=0)

    ax.fill_between(wavegrid[:2872], (mean+std).flatten()[:2872], (mean-std).flatten()[:2872], color='slateblue', alpha=1)
    ax.fill_between(wavegrid[2873:5272], (mean+std).flatten()[2873:5272], (mean-std).flatten()[2873:5272], color='slateblue', alpha=1)
    ax.fill_between(wavegrid[5273:], (mean+std).flatten()[5273:], (mean-std).flatten()[5273:], color='slateblue', alpha=1)
   
    ax.fill_between(wavegrid[:2872], (mean+2*std).flatten()[:2872], (mean-2*std).flatten()[:2872], color='slateblue', alpha=0.5)
    ax.fill_between(wavegrid[2873:5272], (mean+2*std).flatten()[2873:5272], (mean-2*std).flatten()[2873:5272], color='slateblue', alpha=0.5)
    ax.fill_between(wavegrid[5273:], (mean+2*std).flatten()[5273:], (mean-2*std).flatten()[5273:], color='slateblue', alpha=0.5)
    
    ax.fill_between(wavegrid[:2872], (mean+3*std).flatten()[:2872], (mean-3*std).flatten()[:2872], color='slateblue', alpha=0.25)
    ax.fill_between(wavegrid[2873:5272], (mean+3*std).flatten()[2873:5272], (mean-3*std).flatten()[2873:5272], color='slateblue', alpha=0.25)
    ax.fill_between(wavegrid[5273:], (mean+3*std).flatten()[5273:], (mean-3*std).flatten()[5273:], color='slateblue', alpha=0.25)
    


def plot_reconstruct(true, pred, mask = 0, N=4):
    #TODO indicate errornous pixels using the mask

    wavegrid = np.load('./data/wave_grid_ASPCAP.npy')
    start_w = len(wavegrid) - 7167
    wavegrid = wavegrid[start_w:]

    fig = plt.figure(figsize=(15,N*6))
    outer = gs.GridSpec(N,1)
   
    xlims = [15150, 16990]
    
    if len(np.shape(mask)) == 0:
        mask = np.ones([N,7167])

    for i in range(N):
        inner = gs.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[i], hspace=0.0)
        ax0 = fig.add_subplot(inner[0])
        ax0.set_ylabel('Target')
        ax0.get_xaxis().set_visible(False)
        ax0.set_xlim(left=xlims[0], right=xlims[1])
 
        idx = np.where(mask[i] == 0)
        
        #ax0.plot(wavegrid[:2872],true[i][:2872], color='slateblue')
        #ax0.plot(wavegrid[2873:5272],true[i][2873:5272], color='slateblue')
        #ax0.plot(wavegrid[5273:],true[i][5273:], color='slateblue')
        #ax0.scatter(wavegrid[idx],true[i][idx], color = 'r')

        thistrue = np.copy(true[i]) 
        thistrue[idx] = 0
        ax0.plot(wavegrid[:2872],thistrue[:2872], color='slateblue')
        ax0.plot(wavegrid[2873:5272],thistrue[2873:5272], color='slateblue')
        ax0.plot(wavegrid[5273:],thistrue[5273:], color='slateblue')


        ax1 = fig.add_subplot(inner[1])
        ax1.set_ylabel('Predicted')
        ax1.set_xlabel('Wavelength')
        ax1.set_xlim(left=xlims[0], right=xlims[1])

        ax1.plot(wavegrid[:2872],pred[i][:2872], color='darkblue')
        ax1.plot(wavegrid[2873:5272],pred[i][2873:5272], color='darkblue')
        ax1.plot(wavegrid[5273:],pred[i][5273:], color='darkblue')

        lim0 = ax0.get_ylim()
        lim1 = ax1.get_ylim()

        ax0.set_ylim(bottom=min(lim0[0], lim1[0]), top=max(lim0[1], lim1[1]))
        ax1.set_ylim(bottom=min(lim0[0], lim1[0]), top=max(lim0[1], lim1[1]))

def plot_lime(mean,std, N=2):
    wavegrid = np.load('./data/wave_grid_ASPCAP.npy')
    start_w = len(wavegrid) - 7167
    wavegrid = wavegrid[start_w:]

    fig = plt.figure(figsize=(15,N*6))
    outer = gs.GridSpec(N,1)
   
    xlims = [15150, 16990]

    ymax = np.max(mean+std)
    ymin = np.min(mean-std)

    for i in range(N):
        thismean = np.array(mean[:,i]).reshape(7167,1)
        thisstd = np.array(std[:,i]).reshape(7167,1)
    
        ax = fig.add_subplot(outer[i])
        ax.set_xlim(left=xlims[0], right=xlims[1])
 
        ax.plot(wavegrid[:2872],thismean[:2872], color='crimson')
        ax.plot(wavegrid[2873:5272],thismean[2873:5272], color='crimson')
        ax.plot(wavegrid[5273:],thismean[5273:], color='crimson')

        ax.fill_between(wavegrid[:2872], (thismean+thisstd).flatten()[:2872], \
            (thismean-thisstd).flatten()[:2872], color='k', alpha=0.4)
        ax.fill_between(wavegrid[2873:5272], (thismean+thisstd).flatten()[2873:5272], \
            (thismean-thisstd).flatten()[2873:5272], color='k', alpha=0.4)
        ax.fill_between(wavegrid[5273:], (thismean+thisstd).flatten()[5273:], \
            (thismean-thisstd).flatten()[5273:], color='k', alpha=0.4)
   
        ax.set_ylim(bottom=ymin, top=ymax)
        
