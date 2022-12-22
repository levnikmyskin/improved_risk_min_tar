import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


def write_dot_product_on_disk(x: csr_matrix, name=''):
    """
    The QUIRE strategy for Active Learning requires an n x n similarity matrix. This is clearly unfeasible when n is
    too large (in our paper, we work with 100.000 documents from RCV1). The only solution here is to use np.memmap and
    work from disk.
    """
    rows_in_slice = 10_000
    slice_start = 0
    slice_end = slice_start + rows_in_slice
    fp = np.memmap(name, dtype='float64', mode='w+', shape=(x.shape[0], x.shape[0]))
    pbar = tqdm(total=x.shape[0] // rows_in_slice)
    while slice_end <= x.shape[0]:
        fp[slice_start:slice_end] = x[slice_start:slice_end].dot(x.T).todense()
        slice_start += rows_in_slice
        slice_end = slice_start + rows_in_slice
        pbar.update(1)
    fp.flush()
    pbar.close()


if __name__ == '__main__':
    from sklearn.datasets import fetch_rcv1
    r = fetch_rcv1()
    x = r.data[:100_000]
    write_dot_product_on_disk(x, 'sim_100k_rcv1.dat')
