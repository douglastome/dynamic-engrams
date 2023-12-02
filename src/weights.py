'''
Copyright 2023 Douglas Feitosa Tomé

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#matplotlib.use('Agg')

class Weights:
 def __init__(self):
  self.random_trial = []
  self.pre_region = []
  self.pre_neuron_type = []
  self.pre_neuron_index = []
  self.pre_cluster = []
  self.post_region = []
  self.post_neuron_type = []
  self.post_neuron_index = []
  self.post_cluster = []
  self.learn_time = []
  self.consolidation_time = []
  self.phase = []
  self.weight = []


 def add_weight(self, random_trial, pre_region, pre_neuron_type, pre_neuron_index, pre_cluster, post_region, post_neuron_type, post_neuron_index, post_cluster, learn_time, consolidation_time, phase, weight):
  self.random_trial.append(random_trial)
  self.pre_region.append(pre_region)
  self.pre_neuron_type.append(pre_neuron_type)
  self.pre_neuron_index.append(pre_neuron_index)
  self.pre_cluster.append(pre_cluster)
  self.post_region.append(post_region)
  self.post_neuron_type.append(post_neuron_type)
  self.post_neuron_index.append(post_neuron_index)
  self.post_cluster.append(post_cluster)
  self.learn_time.append(learn_time)
  self.consolidation_time.append(consolidation_time)
  self.phase.append(phase)
  self.weight.append(weight)


 def build_data_frame(self):
  dd = {'random_trial' : self.random_trial,
        'pre_region' : self.pre_region,
        'pre_neuron_type' : self.pre_neuron_type,
        'pre_neuron_index' : self.pre_neuron_index,
        'pre_cluster' : self.pre_cluster,
        'post_region' : self.post_region,
        'post_neuron_type' : self.post_neuron_type,
        'post_neuron_index' : self.post_neuron_index,
        'post_cluster' : self.post_cluster,
        'learn_time' : self.learn_time,
        'consolidation_time' : self.consolidation_time,
        'phase' : self.phase,
        'weight' : weight}
  self.df = pd.DataFrame(dd)


 def save_data_frame(self, datadir, filename):
  #filename = filename + '.csv'
  self.df.to_csv(os.path.join(datadir, filename))
  print("Saved results data frame in %s"%filename)


 def load_df(self, datadir, filename):
  self.df = pd.read_csv(os.path.join(datadir, filename))


 def select_rows(self, *args):
  select_rows = np.full((len(self.df)), True)
  '''
  print('select_rows')
  print('args', type(args))  
  print(args)
  print('select', type(select), select.shape)
  print(select)
  '''
  for col,values in args:
   '''print ('col,value', col, value)'''
   select = np.full((len(self.df)), False)
   for value in values:
    select = select | (self.df[col] == value)
   select_rows = select_rows & select
  return self.df[select_rows]


 def plot_line(self, x, y, hue, style, ci, data, datadir, filename, xlabel, ylabel):
  fig = plt.figure()
  plt.rcParams.update({'font.size': 13})
  ax = None
  if style == None:
   ax= sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data)
  else:
   ax = sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data)
   
  #ax.get_legend().remove()
  
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
  ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

  plt.xlabel(xlabel + ' time (min)')
  plt.ylabel(ylabel)
  ax.set_xticks(list(set(data[x])))
  
  #plt.title("")
  plt.ylim((0, 1.05))
  sns.despine()
  plt.savefig(os.path.join(datadir, filename))
  print('Saved results plot', filename)
  plt.close(fig)


 def plot(self, x, y, hue, style, ci, datadir, filename, xlabel, ylabel, *args):
  '''
  print('plot args', type(args))
  print(args)
  '''
  data = self.select_rows(*args)
  data[x] = data[x] / 60
  '''
  print("\nFiltered data frame for plot %s:"%filename)
  print(data)
  '''
  self.plot_line(x, y, hue, style, ci, data, datadir, filename, xlabel, ylabel)
