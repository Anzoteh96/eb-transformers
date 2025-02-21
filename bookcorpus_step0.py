import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    parquets = []
    mydir = 'bookcorpusopen/data'
    for filename in os.listdir(mydir):
        p = pd.read_parquet(os.path.join(mydir, filename))
        parquets.append(p)

    for i, row in tqdm(enumerate(parquets_df.iterrows())):
        filename = row[0]
        txt = row[1].loc['text']
        outfile = os.path.join('bookcorpusopen/files', filename)
        with open(outfile, 'w') as f:
            f.write(txt)
