from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np
from tqdm import tqdm


def map_parallel(func, args, method="multithreading", workers=-1, prog_bar=True):
    """
    Map a function to a list of arguments in parallel.
    RH 2022

    Args:
        func (function):
            Function to map.
        args (list):
            List of arguments to map the function to.
            len(args) should be equal to the number of arguments.
            If the function takes multiple arguments, args should be an
             iterable (e.g. list, tuple, generator) of length equal to
             the number of arguments. Each element can then be an iterable
             for each iteration of the function.
        method (str):
            Method to use for parallelization. Options are:
                'multithreading': Use multithreading from concurrent.futures.
                'multiprocessing': Use multiprocessing from concurrent.futures.
                # 'mpire': Use mpire
                # 'joblib': Use joblib.Parallel
                'serial': Use list comprehension
        workers (int):
            Number of workers to use. If -1, use all available.
        prog_bar (bool):
            Whether to show a progress bar with tqdm.

    Returns:
        output (list):
            List of results from mapping the function to the arguments.
    """
    if workers == -1:
        workers = mp.cpu_count()

    ## Get number of arguments. If args is a generator, make None.
    n_args = len(args[0]) if hasattr(args, "__len__") else None

    if method == "multithreading":
        executor = ThreadPoolExecutor
    elif method == "multiprocessing":
        executor = ProcessPoolExecutor
    # elif method == "mpire":
    #     import mpire
    #     executor = mpire.WorkerPool
    # elif method == 'joblib':
    #     import joblib
    #     return joblib.Parallel(n_jobs=workers)(joblib.delayed(func)(arg) for arg in tqdm(args, total=n_args, disable=prog_bar!=True))
    elif method == "serial":
        # return [func(*arg) for arg in tqdm(args, disable=prog_bar!=True)]
        return list(tqdm(map(func, *args), total=n_args, disable=prog_bar != True))
    else:
        raise ValueError(f"method {method} not recognized")

    with executor(workers) as ex:
        return list(tqdm(ex.map(func, *args), total=n_args, disable=prog_bar != True))


def multiprocessing_pool_along_axis(x_in, function, n_workers=None, axis=0, **kwargs):
    pool = mp.Pool(processes=n_workers)
    if axis == 0:
        results = pool.map(
            partial(function, **kwargs), [x_in[ii] for ii in range(x_in.shape[0])]
        )
        pool.close()
        pool.join()
        return np.row_stack(results)
    elif axis == 1:
        results = pool.map(
            partial(function, **kwargs), [x_in[:, ii] for ii in range(x_in.shape[1])]
        )
        pool.close()
        pool.join()
        return np.column_stack(results)
