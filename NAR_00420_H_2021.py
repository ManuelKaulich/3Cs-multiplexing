'''
Copyright 2021 Martin Wegner (ma.wegner@em.uni-frankfurt.de)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def line_intersection(line1, line2):
    '''Find the intersection of two lines.'''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def lorenz_plot(dataframe, column):
    '''Takes a pandas dataframe and plots the cumulative distribution of a given column incl. an ideal distribution.'''
    data = dataframe[[column]].copy()
    data["ideal"] = 1
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    
    sample_auc = -1
    for col in data.columns:
        df = data[col].fillna(0)
        df.sort_values(inplace=True, ascending=False)
        x = [x / df.shape[0] for x in range(df.shape[0])]
        y = np.cumsum(df / df.sum())
        auc = np.trapz(y, [i * (1.0 / df.shape[0]) for i in range(df.shape[0])])

        x.append(1.0)
        y = list(pd.concat([pd.Series([0.0]), np.cumsum(df / df.sum())]).values)

        if col == "ideal":
            my_y = 0.5
            mycolor = "black"
        else:
            my_y = 0.25
            mycolor = "blue"
        if col == "ideal":
            ax.plot(x, y, label="{}, AUC={}".format(col, 0.5), color=mycolor)
        else:
            sample_auc = auc
            ax.plot(x, y, label="{}, AUC={}".format(col, round(auc, 3)), color=mycolor)

        h_x = None
        for i in range(0, len(y) - 1):
            if y[i] == 0.9:
                h_x = y[i]
                break
            elif y[i] < 0.9 and y[i + 1] > 0.9:
                h_x = line_intersection(([0.0, 0.9], [1.0, 0.9]), ([x[i], y[i]], [x[i + 1], y[i + 1]]))
                break

        if h_x:
            ax.plot([0.0, 0.9], [0.9, 0.9], linestyle="--", marker="o", color="black")  # horizontal line
            ax.plot([h_x[0], h_x[0]], [0.9, my_y + 0.05], linestyle="--", marker="o", color=mycolor)  # vertical sample line
            ax.text(h_x[0] - 0.029, my_y, "{}%".format(int(round(h_x[0] * 100))))
    ax.set_xlabel("fraction of NGS reads, ranked by abundance")
    ax.set_ylabel("cumulative fraction of NGS reads")

    ax.legend()
    ax.set_title(f"Lorenz plot (cumulative distribution) of {column}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def aggregate_normalize_genes(dataframe, samples, agg_method):
    '''Aggregate dataframe on two columns "Gene_1" and "Gene_2" using "agg_method" and normalize to counts per million.'''
    df = dataframe[["Gene_1", "Gene_2"] + samples].groupby(["Gene_1", "Gene_2"]).agg(agg_method)
    for sample in samples:
        if "CTRL" not in sample:
            df[f"{sample}_norm"] = ((df[sample] / df[sample].sum()) * 1_000_000) + 1
    return df


def single_counts(genes_dataframe):
    '''Extract single counts per gene from a dataframe using the boolean columns "CTRL_1" and "CTRL_2" and merge with combinatorial counts.'''
    gene_nht_combis = genes_dataframe[~(genes_df["CTRL_1"]) & (genes_dataframe["CTRL_2"])].groupby(level=0).agg("median")
    gene_nht_combis = gene_nht_combis[[c for c in gene_nht_combis.columns if "CTRL" not in c]]
    gene_nht_combis.columns = [c + "_left" for c in gene_nht_combis.columns]

    nht_gene_combis = genes_dataframe[(genes_df["CTRL_1"]) & ~(genes_dataframe["CTRL_2"])].groupby(level=1).agg("median")
    nht_gene_combis = nht_gene_combis[[c for c in nht_gene_combis.columns if "CTRL" not in c]]
    nht_gene_combis.columns = [c + "_right" for c in nht_gene_combis.columns]
        
    genes_dataframe = pd.merge(genes_dataframe, gene_nht_combis, left_on="Gene_1", right_index=True, how="outer")
    genes_dataframe = pd.merge(genes_dataframe, nht_gene_combis, left_on="Gene_2", right_index=True, how="outer")
    
    return genes_dataframe


def models(genes_dataframe, ctrl, treatment):
    '''Compute genetic interaction models for two given samples in a dataframe.'''
    genes_dataframe[f"LFC_{ctrl}_{treatment}"] = np.log2(genes_dataframe[f"{treatment}_norm"] / genes_dataframe[f"{ctrl}_norm"])
    genes_dataframe[f"LFC_{ctrl}_{treatment}_left"] = np.log2(genes_dataframe[f"{treatment}_norm_left"] / genes_dataframe[f"{ctrl}_norm_left"])
    genes_dataframe[f"LFC_{ctrl}_{treatment}_right"] = np.log2(genes_dataframe[f"{treatment}_norm_right"] / genes_dataframe[f"{ctrl}_norm_right"])
    
    genes_dataframe[f"max_model_{ctrl}_{treatment}"] = genes_dataframe[[f"LFC_{ctrl}_{treatment}_left", f"LFC_{ctrl}_{treatment}_right"]].max(axis=1)
    genes_dataframe[f"max_dLFC_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}"] - genes_df[f"max_model_{ctrl}_{treatment}"]

    genes_dataframe[f"sum_model_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}_left"] + genes_dataframe[f"LFC_{ctrl}_{treatment}_right"]
    genes_dataframe[f"sum_dLFC_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}"] - genes_dataframe[f"sum_model_{ctrl}_{treatment}"]

    genes_dataframe[f"min_model_{ctrl}_{treatment}"] = genes_dataframe[[f"LFC_{ctrl}_{treatment}_left", f"LFC_{ctrl}_{treatment}_right"]].min(axis=1)
    genes_dataframe[f"min_dLFC_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}"] - genes_dataframe[f"min_model_{ctrl}_{treatment}"]

    genes_dataframe[f"mult_model_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}_left"] * genes_dataframe[f"LFC_{ctrl}_{treatment}_right"]
    genes_dataframe[f"mult_dLFC_{ctrl}_{treatment}"] = (genes_dataframe[f"LFC_{ctrl}_{treatment}"] - genes_dataframe[f"mult_model_{ctrl}_{treatment}"])

    genes_dataframe[f"log_model_{ctrl}_{treatment}"] = np.log2(((2**genes_dataframe[f"LFC_{ctrl}_{treatment}_left"]) - 1) * ((2**genes_dataframe[f"LFC_{ctrl}_{treatment}_right"]) - 1) + 1)
    genes_dataframe[f"log_dLFC_{ctrl}_{treatment}"] = genes_dataframe[f"LFC_{ctrl}_{treatment}"] - genes_dataframe[f"log_model_{ctrl}_{treatment}"]
    
    return genes_dataframe
