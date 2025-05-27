import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from tqdm import tqdm


tiktoken_cache_dir = "../tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
stop_words = set(stopwords.words('english'))

def get_token_stats(fullpath, plotpath, num_word_threshold = None):
    # Readthe file first.
    # Let L be the number of lines, and T be the number of distinct tokens.
    with open(fullpath, "r") as f:
         words = [line for line in f] # length L.
    # Next, we want to count the tokens, with stopwords removed.
    vectorizer = CountVectorizer(stop_words='english')
    docs  = vectorizer.fit_transform(words) # L x T, M[i, j] = frequency of token j in line i.
    features = vectorizer.get_feature_names_out() # size T.
    visualizer = FreqDistVisualizer(features=features, orient='v', n = 50)
    if num_word_threshold is not None:
        num_tokens = np.array(docs.sum(axis = 1)).flatten()
        partition_ind = np.sum(np.cumsum(num_tokens) < num_word_threshold)
        docs_X = docs[:partition_ind]
        docs_Y = docs[partition_ind:]
        len_X = docs_X.shape[0] # partition_ind
        len_Y = docs_Y.shape[0]
        freq_x = np.array(docs_X.sum(axis = 0)).flatten() # size T.
        freq_y = np.array(docs_Y.sum(axis = 0)).flatten()
        freq_df_x = pd.DataFrame({"Tokens": features, "Frequency": freq_x})
        # freq_df_x = freq_df_x[freq_df_x["Frequency"] > 0] # Maybe we will keep those with Freq == 0? 
        freq_df_y = pd.DataFrame({"Tokens": features, "Frequency": freq_y})
        freq_df = pd.merge(freq_df_x, freq_df_y, on = "Tokens")
        freq_df["len_X"] = len_X
        freq_df["len_Y"] = len_Y
        #visualizer.fit(docs_X)
    else:
        freq_df = pd.DataFrame({"Tokens": features, "Frequency": freq}).sort_values(by = 'Frequency', ascending = False)
        #visualizer.fit(docs)


    #visualizer.show()
    #plt.savefig(plotpath)
    #plt.clf()
    return freq_df


if __name__ == "__main__":
    input_folder = "../bookcorpusopen/files"
    
    plot_folder = "bookcorpus/plot"
    df_folder = "bookcorpus/dataframes_countvec"
    
    for file in tqdm(os.listdir(input_folder)):
        fullpath = os.path.join(input_folder, file)
        file_toks = file.split(".")
        file_name = file_toks[0]
        plotpath = os.path.join(plot_folder, "{}.png".format(file_name))
        dfpath = os.path.join(df_folder, "{}.csv".format(file_name))
    #    freq_df = get_token_stats(fullpath, plotpath, 2000)
    #    freq_df.to_csv(dfpath)
        try:
            freq_df = get_token_stats(fullpath, plotpath, 2000) 
            freq_df.to_csv(dfpath)
        except:
            print(file_name)
