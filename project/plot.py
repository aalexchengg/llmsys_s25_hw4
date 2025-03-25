import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def plot_throughput(throughputs, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(throughputs)), throughputs, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Throughput (tokens per second)')
    ax.set_xticks(np.arange(len(throughputs)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = 35.79226880073547, 0.1031459913907234
    device0_mean, device0_std =  21.19192373752594, 2.424188866289119
    device1_mean, device1_std =  23.927924060821532, 2.611800016808934
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png')
    
    single_through, single_through_std = 105004.92052385394, 311.14472730235883
    double_through, double_through_std = 207868.775011, 1026.05163292
    plot([double_through, single_through],
        [double_through_std, single_through_std],
        ['Data Parallel - 2GPUs', 'Single GPU'],
        'ddp_vs_rn_thru.png')

    pp_mean, pp_std = 23.73433482646942, 0.07547318935394287
    mp_mean, mp_std = 23.764028072357178, 0.13997578620910645
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')

    mp_through, mp_through_std = 26932.39561751172, 158.6382257914065
    pp_through, pp_through_std = 26965.426936906566, 85.74779062058951
    plot([pp_through, mp_through],
        [pp_through_std, mp_through_std],
         ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp_thru.png')