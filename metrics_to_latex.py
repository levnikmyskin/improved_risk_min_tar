"""
This file is very messy and it was used to generate the tables in our paper. The code simply transforms the
tsv files coming from `results_analysis.py` into latex tables. We recommend to carefully look and adapt this script to
your needs should you want to reproduce results.
"""
import pandas as pd


if __name__ == '__main__':
    sizes = [2000, 4000, 8000, 16000]
    #labels = 'minecore'
    labels = 'all'
    ks = {'sf1': 'SoftF1', '1-brier': '(1-Brier)'}
    #path = f'.data/ALvUS/ALvUS_avg_aggregated-metrics_SVM_{labels}-labels.tsv'
    #df = pd.read_csv(path, sep='\t', index_col=0)[['AL MLE', 'AL SLD', 'Rand MLE', 'Rand SLD', 'Tr prev.', 'Te prev.']]

    #df = df.applymap(lambda r: r.split('$')[0])[['AL MLE', 'AL SLD', 'Rand MLE', 'Rand SLD', 'Tr prev.', 'Te prev.']]
    #print(df.rename(index=lambda i: ks[i] if i in {'sf1', '1-brier'} else i.capitalize()).to_latex(escape=False))
        #formatters=[lambda i: f'{float(i.split("$")[0]):.3f}'] * 6))
    path = '.data/ALvUS/ALvUS_{}_aggregated-metrics_SVM_{}-labels.tsv'
    for size in sizes:
        print(f'##### SIZE {size} #######')
        df = pd.read_csv(path.format(size, labels), sep='\t', index_col=0)[['AL MLE', 'AL SLD', 'Rand MLE', 'Rand SLD', 'Tr prev.', 'Te prev.']]
        #df = df.applymap(lambda r: r.split('$')[0])[['AL MLE', 'AL SLD', 'Rand MLE', 'Rand SLD', 'Tr prev.', 'Te prev.']]
        print(df.rename(index=lambda i: ks[i] if i in {'sf1', '1-brier'} else i.capitalize()).to_latex(escape=False))
