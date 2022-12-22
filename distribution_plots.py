import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.data_utils import ALPolicy
from tqdm import tqdm


# inverse distribs
# def distribs():
#     pre_sld_al = data['al_probs'][:, 1]
#     pre_sld_rand = data['rand_probs'][:, 1]
#     post_sld_al = data['al_sld_posteriors'][:, 1]
#     post_sld_rand = data['rand_sld_posteriors'][:, 1]
#     bins = np.linspace(0., 1., 11)
#     al_hist, _ = np.histogram(pre_sld_al)
#     sld_al_hist, _ = np.histogram(post_sld_al)
#     rand_hist, _ = np.histogram(pre_sld_rand)
#     sld_rand_hist, _ = np.histogram(post_sld_rand)
#     return al_hist, sld_al_hist, rand_hist, sld_rand_hist

if __name__ == '__main__':
    policies = [ALPolicy.from_string('ALvRUS')]
    plot_save = 'plots/distrib_plots'
    path = '.data/active_learning/results/'
    r_classes = ['C17', 'CCAT']
    sizes = [2000, 23149]
    x_lim = [-1e-2, 1.]
    os.makedirs(plot_save, exist_ok=True)
    for cls in tqdm(r_classes):
        for size in sizes:
            for policy in policies:
                if policy is ALPolicy.RELEVANCE_UNCERTAINTY_SAMPLING:
                    with open(os.path.join(path, str(policy), f'{policy}_{cls}_{size}size_SVM_results.pkl'), 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(os.path.join(path, str(policy), 'all_classes', f'{policy}_{cls}_{size}size_SVM_results.pkl'), 'rb') as f:
                        data = pickle.load(f)
                pre_sld_al = data['al_probs'][:, 1]
                post_sld_al = data['al_sld_posteriors'][:, 1]
                rand_pre = data['rand_probs'][:, 1]
                rand_post = data['rand_sld_posteriors'][:, 1]
                prev = data['al_y_te'].mean()

                df = pd.DataFrame({f'{policy} Pre-SLD': pre_sld_al, f'{policy} Post-SLD': post_sld_al, f'Rand ({policy.compact_str()}) Pre-SLD': rand_pre, f'Rand ({policy.compact_str()}) Post-SLD': rand_post})
                sns.histplot(df[[f'{policy} Pre-SLD', f'{policy} Post-SLD']], log_scale=(False, True), element='step')
                plt.title(f'{policy} posteriors. Tr. size {size}; Te prev. {prev:.3f}')
                plt.ylabel('Log Count')
                plt.xlabel(f'Pr$(y_c=1|x)$; $c = {cls}$')
                if x_lim:
                    plt.xlim(x_lim)
                plt.savefig(os.path.join(plot_save, f'{policy}_{size}_{cls}.png'))
                plt.show()
                plt.close()
                sns.histplot(df[[f'Rand ({policy.compact_str()}) Pre-SLD', f'Rand ({policy.compact_str()}) Post-SLD']], log_scale=(False, True), element='step')
                plt.title(f'$Rand$ ({policy.compact_str()}) posteriors. Tr. size {size}; Te prev. {prev:.3f}')
                plt.ylabel('Log Count')
                plt.xlabel(f'Pr$(y_c=1|x)$; $c = {cls}$')
                if x_lim:
                    plt.xlim(x_lim)
                plt.savefig(os.path.join(plot_save, f'rand{policy.compact_str()}_{size}_{cls}.png'))
                plt.show()
                plt.close()

