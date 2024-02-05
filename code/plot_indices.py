import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_indices():
    data = pd.read_csv("data/all_indices.csv", header=[0,1], index_col=[0,1])
    data = data.fillna(0)

    for idx_lv0 in data.index.levels[0]:
        for col_lv0 in data.columns.levels[1]: 

            km = data.loc[idx_lv0][("km", col_lv0)]
            location = np.arange(len(km))
            
            fig, ax = plt.subplots(figsize=(10,5))
            
            rect1 = ax.bar(location, km.values, width=0.3, label="No Byzantines", color="green")
            ax.bar_label(rect1, fmt='%.2f', rotation=90, label_type="center", padding=20)
            
            by = data.loc[idx_lv0][("by", col_lv0)]
            rect2 = ax.bar(location+0.3, by.values, width=0.3, label=f"Byzantines", color="red")
            ax.bar_label(rect2, fmt='%.2f', rotation=90, label_type="center", padding=20)
            
            co = data.loc[idx_lv0][("co", col_lv0)]
            rect3 = ax.bar(location+0.6, co.values, width=0.3, label="Correction", color="blue")
            ax.bar_label(rect3, fmt='%.2f', rotation=90, label_type="center", padding=20)

            ax.legend() # loc='upper left', ncols=2
            ax.set_ylabel("Score")
            ax.set_xticks(location+0.3, location+1)
            ax.set_xlabel("Number of Byzantine workers")
            plt.title('%s -- %s indice'%(idx_lv0, col_lv0));
            plt.savefig(f"../figures/{idx_lv0}_{col_lv0}.jpg", 
                format='jpg', dpi=500, bbox_inches='tight');
            plt.close(fig)
            
if __name__ == "__main__":
    plot_indices();