import os
import pandas as pd
import time
import tqdm

n_threads = [1, 2, 3, 4]
n_bodies = [10, 20, 50, 100, 500, 1000]
algo_names = ["row", "column", "block"]


for algo_name in algo_names:
    d = {}
    print(algo_name)
    for i in tqdm.tqdm(n_bodies):
        for n in n_threads:
            st = time.time()
            c = os.popen(
                f"""mpiexec --allow-run-as-root -np {n} ./main {i} {algo_name}"""
            ).read()
            res = time.time() - st
            try:
                d[i].append(int(c.replace("correct ", "")))
            except Exception:
                d[i] = [float(c.replace("correct ", ""))]
    df = pd.DataFrame(d, index=n_threads)
    print(df.to_markdown())
