# CALCULATE FEATURES/MEASURES OF CENTRALITY

#### import libraries
import numpy as np # data management
#import seaborn as sns # data visualization
import matplotlib.pyplot as plt # plotting
import pandas as pd
import bct # brain connectivity toolbox
import os # to select directory

#### Select path/directory where files are located
os.chdir('/Users/Lore/Desktop/brainhack/project')

#### Import community index vector
civ = pd.read_table('network_roi_7networks.txt')

#### Load FC data
FCData = np.loadtxt('my_test.txt')

#### Transform to weighted and binary adjecent matrices
adj_wei = FCData - np.eye(FCData.shape[0])
adj_bin = bct.utils.binarize(bct.utils.threshold_proportional(adj_wei, 0.2))

#### Plot weighted adjacency matrix
fig, ax = plt.subplots(figsize=(7, 7))
plot_wei = ax.imshow(adj_wei,cmap = 'viridis')
plt.title('Weighted adjacency matrix')
fig.colorbar(plot_wei)
plt.savefig('plot_wei.png')

#### Plot binary adjecent matrix 
fig, ax = plt.subplots(figsize=(7, 7))
plot_bin = ax.imshow(adj_bin, cmap = 'viridis')
plt.title('Binary adjacency matrix')
fig.colorbar(plot_bin)
plt.savefig('plot_bin.png')

#### Calculate within module degree z-scores
mod_z = bct.centrality.module_degree_zscore(adj_wei,civ.network_group)
np.savetxt('within_module_z.txt', mod_z, fmt='%f')

#### Calculate participation coefficients
part_coef = bct.centrality.participation_coef(adj_wei,civ.network_group)
np.savetxt('participation_coeff.txt', part_coef, fmt='%f')

#### Calculate node strength
strength = bct.degree.strengths_und(adj_wei)
np.savetxt('node_strength.txt', strength, fmt='%f')

