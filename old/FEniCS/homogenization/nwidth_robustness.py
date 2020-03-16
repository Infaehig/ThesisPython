import matplotlib.pyplot as plt
import numpy as np

import sa_utils

nwidth_keys = ['nwidthE', 'nwidth1', 'nwidth0']
nwidth_letters = ['E', '1', '0']
nwidth_keynames = [r'$d^E_{l, n}$', r'$d^1_{l, n}$', r'$d^0_{l, n}$']
nwidth_colors = ['r-', 'b-', 'g-']

supinf_keys = ['supinfE', 'supinf1', 'supinf0']
supinf_keynames = [r'$\Psi^E_l(V_l)$', r'$\Psi^1_l(V_l)$', r'$\Psi^0_l(V_l)$']
supinf_colors = ['r:', 'b:', 'g:'] 

keys = [nwidth_keys, supinf_keys]
keynames = [nwidth_keynames, supinf_keynames]

assert(len(nwidth_keys)==len(supinf_keys))
key_len = len(nwidth_keys)
type_len = len(keys)

def plot_multiple(basename, nwidth_files=[], nwidth_names=[], supinf_files=[], supinf_names=[], outdir='.', colors=[nwidth_colors, supinf_colors], normal_colors=False, legend=True):
    assert(len(nwidth_files) <= len(nwidth_names))
    num_nwidth_files = len(nwidth_files)

    assert(len(supinf_names) <= len(supinf_files))
    num_supinf_files = len(supinf_files)

    files = [nwidth_files, supinf_files]
    num_files = np.array([num_nwidth_files, num_supinf_files])
    max_num_files = np.max(num_files)
    print(max_num_files)
    names = [nwidth_names, supinf_names]

    arrays = [[] for ii in range(type_len)]
    steps = [[] for ii in range(type_len)]
    lengths = [[] for ii in range(type_len)]
    minxs = []
    maxxs = []
    minys = [[] for ii in range(key_len)]
    maxys = [[] for ii in range(key_len)]

    active_num = len(np.where(num_files > 0)[0])

    for kk in range(type_len):
        for ii in range(num_files[kk]):
            arrays[kk].append(np.genfromtxt(files[kk][ii], delimiter=', ', names=True))
            length = len(arrays[kk][-1]['dof'])
            lengths[kk].append(length)
        if num_files[kk]:
            minxs.append(np.min([np.min(arrays[kk][ii]['dof'][1:]) for ii in range(num_files[kk])]))
            maxxs.append(np.max([np.max(arrays[kk][ii]['dof'][1:]) for ii in range(num_files[kk])]))
            for jj in range(key_len):
                minys[jj].append(np.min([np.min(arrays[kk][ii][keys[kk][jj]][1:]) for ii in range(num_files[kk])]))
                maxys[jj].append(np.max([np.max(arrays[kk][ii][keys[kk][jj]][1:]) for ii in range(num_files[kk])]))
    if num_files[1]:
        minx = minxs[1]; maxx = maxxs[1]
    else:
        minx = minxs[0]; maxx = maxxs[0]
    minys = [np.min(minys[ii]) for ii in range(key_len)]
    maxys = [np.max(maxys[ii]) for ii in range(key_len)]

    for jj in range(key_len):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = [[] for ii in range(type_len)]
        file_count = 0
        widths = np.zeros(type_len)

        for kk in range(type_len):
            if not files[kk]:
                continue

            for ii in range(num_files[kk]):
                key = keys[kk][jj]
                label = keynames[kk][jj]+', '+names[kk][ii]
                handles[kk].append(*ax.semilogy(arrays[kk][ii]['dof'], arrays[kk][ii][key],
                                                sa_utils.line_styles[kk%sa_utils.num_line_styles]+sa_utils.marker_styles[kk%sa_utils.num_marker_styles], color=sa_utils.color_styles[ii%sa_utils.num_color_styles],
                                                mec=sa_utils.color_styles[ii%sa_utils.num_color_styles], mfc='none', label=label))
                file_count += 1
                widths[kk] += len(label)
        width = np.max(widths)
        if handles[0] and handles[1]:
            aa = handles[0]
            bb = handles[1]
            handles = []
            for ii in range(num_files[0]):
                handles.append(bb[ii])
                handles.append(aa[ii])
        else:
            handles = handles[0]+handles[1]
        blubb=outdir+'/'+basename+'_'+key
        ax.grid(True, which='major')
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, minys[jj], maxys[jj])
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$d^'+str(nwidth_letters[jj])+r'_{l, n}$')
        fig.savefig(blubb+'_semilogy.pdf')

        if legend:
            figlegend = plt.figure(figsize=(width/15*sa_utils.legendx, active_num*1.05*sa_utils.legendy), frameon=False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc=10, ncol=max_num_files)
            figlegend.savefig(blubb+'_legend.pdf', bbox_extra_artists=(lgd, ))

        plt.close('all')
