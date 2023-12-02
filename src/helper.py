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

import sys
import os
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from itertools import combinations
from random import sample
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

# Import Auryn tools
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *

from recall_metrics import RecallMetrics

#matplotlib.use('Agg')


def save_pickled_object(object, path):
 max_bytes = 2**31 - 1
 bytes_out = pickle.dumps(object, protocol=2)
 n_bytes = sys.getsizeof(bytes_out)
 with open(path, 'wb') as f_out:
  for idx in range(0, n_bytes, max_bytes):
   f_out.write(bytes_out[idx:idx+max_bytes])


def load_pickled_object(path):
 max_bytes = 2**31 - 1
 try:
  input_size = os.path.getsize(path)
  bytes_in = bytearray(0)
  with open(path, 'rb') as f_in:
   for _ in range(0, input_size, max_bytes):
    bytes_in += f_in.read(max_bytes)
    object = pickle.loads(bytes_in)
 except OSError as err:
  print('OS error: {0}'.format(err))
 except:
  print('Unexpected error:', sys.exc_info()[0])
  raise
 return object


def parse_stim_pat(arg):
 '''
 Parse stimulus-pattern mapping passed as a command-line argument.
 Expected format of argument: list of (stimulus,pattern) tuples as a string.
 Example of expected format of argument: "[(0,0),(1,1),(2,2),(3,3)]"

 Returns: dictionary where each key is a stimulus id and each value is the respective pattern.
 '''
 stim_pat = {}
 pairs = arg.split("),(")
 for pair in pairs:
  if pair[0] == '(':
   pair = pair[1:]
  if pair[-1] == ')':
   pair = pair[:-1]
  stim, pat = pair.split(',')
  stim_pat[int(stim)] = int(pat)
 return stim_pat


def get_stim_data(datadir, prefix, nb_stim):
 '''
 Extract stimulus presentation data.

 Returns: stimtime -> 1D array with the stimulus presentation time series.
          stimdata -> 2D array where row indicates time and column indicates stimulus.
                      each cell denotes whether the stimulus is active or inactive (1 vs 0).
 '''
 stimfile = np.loadtxt("%s/%s.0.stimtimes"%(datadir,prefix))
 #stimfile = np.loadtxt("%s/%s.0.reptimes"%(datadir,prefix))
 stimtime = np.zeros(len(stimfile))
 stimdata = np.zeros((len(stimfile),nb_stim))
 for i,row in enumerate(stimfile):
  t,a,s = row
  stimtime[i] = t
  stimdata[i,int(s)] = a

 return stimtime, stimdata


def get_rep_data(datadir, prefix, nb_patterns):
 '''
 Extract replay presentation data.

 Returns: reptime -> 1D array with the replay presentation time series.
          repdata -> 2D array where row indicates time and column indicates stimulus.
                      each cell denotes whether the stimulus is active or inactive (1 vs 0).
 '''
 #print ("Loading replay presentation times file: %s/%s.0.reptimes"%(datadir,prefix))
 repfile = np.loadtxt("%s/%s.0.reptimes"%(datadir,prefix))
 #print ("repfile", repfile.shape)
 reptime = np.zeros(len(repfile))
 #print ("reptime", reptime.shape)
 #print ("nb_stim", nb_stim)
 repdata = np.zeros((len(repfile),nb_patterns))
 for i,row in enumerate(repfile):
  t,a,s = row
  reptime[i] = t
  repdata[i,int(s)] = a

 return reptime, repdata


def get_stim_intervals(nb_stim, stimtime, stimdata, t_start, t_stop):
 '''
 Extracts the stimulus presentation time intervals for t_start <= time < t_stop.

 Returns: nested list with one list per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
          each stimulus list contains time intervals as tuples (start, end).
 '''
 stim_intervals = []
 
 for stim in range(nb_stim):
  stim_intervals.append([])
  is_stim_start = True
  for idx in range(stimdata.shape[0]):
   if stimtime[idx] < t_start:
    continue
   if stimtime[idx] >= t_stop and is_stim_start:
    break
   if stimdata[idx, stim] == 1 and is_stim_start:
    stim_start = stimtime[idx]
    is_stim_start = False
   if stimdata[idx, stim] == 0 and not is_stim_start:
    stim_end = stimtime[idx]
    stim_intervals[stim].append((stim_start, stim_end))
    is_stim_start = True
    
 return stim_intervals


def get_active_stim(stimdata):
 '''
 Extracts the active stimulus in each time point.

 Returns: list whose elements are the active stimulus: None or (0,...,nb_stim=stimdata.shape[1]).
 '''
 active_stimuli = []
 for row in stimdata:
  active_stimulus = None
  for stim in range(row.shape[0]):
   if row[stim] == 1:
    active_stimulus = stim
    break
  active_stimuli.append(active_stimulus)
 return active_stimuli


def get_inter_stim_intervals(stimtime, stimdata, t_start, t_stop):
 '''
 Extracts the inter-stimulus intervals for t_start <= time < t_stop.

 Returns: list whose elements are inter-stimulus intervals as tuples (start, end).
 '''
 inter_stim_intervals = []
 
 start_idx = None
 stop_idx = stimdata.shape[0] - 1
 for idx in range(stimdata.shape[0]):
  if stimtime[idx] < t_start:
   continue
  elif start_idx == None:
   start_idx = idx
  if stimtime[idx] >= t_stop:
   stop_idx = idx
   break

 #print('t_start', t_start)
 #print('t_stop', t_stop)
 #print('start_idx',start_idx)
 #print('stop_idx',stop_idx)
 
 if start_idx == None:
  return inter_stim_intervals
 
 active_stimuli = get_active_stim(stimdata)
 if active_stimuli[start_idx] != None:
  is_new_inter_stim = True
  #active_stimulus = active_stimuli[start_idx]
 else:
  inter_stim_start = stimtime[start_idx]
  is_new_inter_stim = False
 for idx in range(start_idx + 1, stop_idx + 1):
  if active_stimuli[idx] != None and (not is_new_inter_stim):
   inter_stim_end = stimtime[idx]
   inter_stim_intervals.append((inter_stim_start, inter_stim_end))
   is_new_inter_stim = True
   #active_stimulus = active_stimuli[idx]
  if active_stimuli[idx] == None and is_new_inter_stim:
   inter_stim_start = stimtime[idx]
   is_new_inter_stim = False
    
 return inter_stim_intervals


def sort_intervals(intervals):
 '''
 Sort a list of (start, end) intervals in ascending order according to interval duration end-start.
 '''
 intervals.sort(key=lambda x:x[1]-x[0])


def get_max_inter_stim_interval(stimtime, stimdata, t_start, t_stop):
 '''
 Extract the maximum-duration inter-stimulus interval for t_start <= time < t_stop.

 Return: maximum-duration interval as a tuple (start, end).
 '''
 inter_stim_intervals = get_inter_stim_intervals(stimtime, stimdata, t_start, t_stop)
 sort_intervals(inter_stim_intervals)
 #print('t_start - t_stop', t_start, '-', t_stop)
 #print('inter_stim_intervals', inter_stim_intervals)
 if len(inter_stim_intervals) > 0:
  return inter_stim_intervals[-1]
 return (None, None)


def get_rates_across_stim_presentations_old(nb_stim, nb_neurons, stim_intervals, sfo):
 '''
 DEPRECATED: use get_rates_across_stim_presentations(nb_stim, stim_intervals, stim_spikes) instead.

 Computes the stimulus-evoked average firing rates of nb_neurons across the specified intervals.
 The firing rate is averaged across all presentations of a given stimulus.
 Only one average firing rate is computed per stimulus per neuron.
 
 Return: list with one array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         each array contains the average firing rate of each neuron and has shape (1, nb_neurons).
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  spike_counts = np.zeros((1,nb_neurons))
  stim_duration = 0
  for stim_start,stim_stop in stim_intervals[stim]:
   spikes = sfo.time_binned_spike_counts(stim_start, stim_stop, bin_size=stim_stop-stim_start,
                                         max_neuron_id=nb_neurons);
   spike_counts += spikes
   stim_duration += stim_stop - stim_start
  if stim_duration == 0:
   print("%d stimulus was not presented at all! Stimulus-evoked rates are not applicable!"%stim)
   raise ValueError
  stim_rates.append(spike_counts / stim_duration)

 return stim_rates


def get_rates_per_stim_presentation_old(nb_stim, nb_neurons, stim_intervals, sfo):
 '''
 DEPRECATED: use get_rates_per_stim_presentation(nb_stim, stim_intervals, stim_spikes) instead.

 Computes the stimulus-evoked average firing rates of nb_neurons for each specified interval.
 The firing rate is averaged for each presentation of a given stimulus individually.
 One average firing rate is computed per stimulus presentation per neuron.

 Return: list with one 2D array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         the first dimension of each array indicates the stimulus presentation (1st...Nth)
         while the second dimension denotes the firing rate of each neuron (0...nb_neurons).
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  nb_stim_presentations = len(stim_intervals[stim])
  rates = np.zeros((nb_stim_presentations, nb_neurons))
  for presentation in range(nb_stim_presentations):
   stim_start, stim_stop = stim_intervals[stim][presentation]
   stim_spikes = sfo.time_binned_spike_counts(stim_start,
                                              stim_stop,
                                              bin_size=stim_stop-stim_start,
                                              max_neuron_id=nb_neurons)
   rates[presentation, :] = stim_spikes / (stim_stop - stim_start)
  stim_rates.append(rates)

 return stim_rates


def get_stim_spikes(nb_stim, nb_neurons, stim_intervals, sfo):
 '''
 Computes the stimulus-evoked spike counts of nb_neurons for each specified interval.
 One spike count is computed per stimulus presentation per neuron.

 Return: list with one 2D array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         the first dimension of each array indicates the stimulus presentation (1st...Nth)
         while the second dimension denotes the spike count of each neuron (0...nb_neurons).
 '''
 stim_spikes = []
 
 for stim in range(nb_stim):
  nb_stim_presentations = len(stim_intervals[stim])
  spikes = np.zeros((nb_stim_presentations, nb_neurons))
  for presentation in range(nb_stim_presentations):
   stim_start, stim_stop = stim_intervals[stim][presentation]
   spikes[presentation, :] = sfo.time_binned_spike_counts(stim_start,
                                              stim_stop,
                                              bin_size=stim_stop-stim_start,
                                              max_neuron_id=nb_neurons)
  stim_spikes.append(spikes)

 return stim_spikes


def get_rates_across_stim_presentations(nb_stim, stim_intervals, stim_spikes):
 '''
 Computes the stimulus-evoked average firing rates across the specified stimulus intervals.
 The firing rate is averaged across all presentations of a given stimulus.
 Only one average firing rate is computed per stimulus per neuron.
 
 Return: list with one array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         each array contains the average firing rate of each neuron.
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  stim_duration = 0
  for presentation in range(len(stim_intervals[stim])):
   stim_start, stim_stop = stim_intervals[stim][presentation]
   stim_duration += stim_stop - stim_start
  stim_rates.append(np.sum(stim_spikes[stim], axis=0) / stim_duration)

 return stim_rates


def get_max_rates_across_stim_presentations(nb_stim, rates_per_stim):
 '''
 Computes the maximum stimulus-evoked average firing rates across the specified stimulus intervals.
 The maximum firing rate a neuron displayed across all presentations of a given stimulus is retained.
 Only one maximum firing rate is computed per stimulus per neuron.
 
 Return: list with one array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         each array contains the maximum firing rate of each neuron.
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  stim_rates.append(np.amax(rates_per_stim[stim], axis=0))

 return stim_rates


def get_rates_across_pat_presentations(nb_patterns, stim_pat, nb_neurons, stim_intervals, stim_spikes):
 '''
 Computes the pattern-evoked average firing rates across the respective stimulus intervals.
 The firing rate is averaged across all presentations of the stimuli matching a given pattern.
 Only one average firing rate is computed per pattern per neuron.
 
 Return: list with one array per pattern in an ordered fashion (i.e. pattern 0...nb_patterns).
         each array contains the average firing rate of each neuron.
 '''
 pat_rates = []
 
 for pattern in range(nb_patterns):
  stimuli_duration = 0
  stimuli_spikes = np.zeros(nb_neurons)
  for stim,pat in stim_pat.items():
   if pat == pattern:
    for presentation in range(len(stim_intervals[stim])):
     stim_start, stim_stop = stim_intervals[stim][presentation]
     stimuli_duration += stim_stop - stim_start
     stimuli_spikes += np.sum(stim_spikes[stim], axis=0)
  pat_rates.append(stimuli_spikes / stimuli_duration)
 return pat_rates


def get_rates_across_pat_presentations_new(nb_patterns, stim_pat, nb_neurons, stim_intervals, stim_spikes):
 '''
 Computes the pattern-evoked average firing rates across the respective stimulus intervals.
 The firing rate is averaged across all presentations of the stimuli matching a given pattern.
 Only one average firing rate is computed per pattern per neuron.
 
 Return: list with one array per pattern in an ordered fashion (i.e. pattern 0...nb_patterns).
         each array contains the average firing rate of each neuron.
 '''
 
 for pattern in range(nb_patterns):
  stimuli_duration = 0
  stimuli_spikes = np.zeros(nb_neurons)
  for stim,pat in stim_pat.items():
   if pat == pattern:
    for presentation in range(len(stim_intervals[stim])):
     stim_start, stim_stop = stim_intervals[stim][presentation]
     stimuli_duration += stim_stop - stim_start
     stimuli_spikes += np.sum(stim_spikes[stim], axis=0)
  pat_rates.append(stimuli_spikes / stimuli_duration)
 return pat_rates


def get_rates_percentile_across_stim_presentations(nb_stim, rates_per_stim, percentile):
 '''
 Computes the stimulus-evoked percentile firing rates across the specified stimulus intervals.
 The percentile firing rate is computed across all presentations of a given stimulus.
 Only one percentile firing rate is computed per stimulus per neuron.
 
 Return: list with one array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         each array contains the percentile firing rate of each neuron.
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  stim_rates.append(np.percentile(rates_per_stim[stim], percentile, axis=0))

 return stim_rates


def get_rates_per_stim_presentation(nb_stim, stim_intervals, stim_spikes):
 '''
 Computes the stimulus-evoked average firing rates for each specified interval.
 The firing rate is averaged for each presentation of a given stimulus individually.
 One average firing rate is computed per stimulus presentation per neuron.

 Return: list with one 2D array per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
         the first dimension of each array indicates the stimulus presentation (1st...Nth)
         while the second dimension denotes the firing rate of each neuron.
 '''
 stim_rates = []
 
 for stim in range(nb_stim):
  rates = np.zeros(stim_spikes[stim].shape)
  for presentation in range(len(stim_intervals[stim])):
   stim_start, stim_stop = stim_intervals[stim][presentation]
   rates[presentation, :] = stim_spikes[stim][presentation, :] / (stim_stop - stim_start)
  stim_rates.append(rates)

 return stim_rates


def get_activated_neurons(stim_rates, min_rate):
 '''
 Identify stimulus-evoked activated neurons using firing rate. A neuron is considered activated
 if its stimulus-evoked firing rate averaged across all stimulus presentations is above min_rate.

 Returns: list of the ids of the neurons that respond to stimulus presentations.
 '''
 
 activated_neurons = []
 avg_stim_rates = np.zeros(stim_rates[0].shape)
 for stim_rate in stim_rates:
  avg_stim_rates = avg_stim_rates + stim_rate
 avg_stim_rates = avg_stim_rates / float(len(stim_rates))
 for idx,rate in enumerate(avg_stim_rates.flatten()):
  if rate > min_rate:
   activated_neurons.append(idx)

 return activated_neurons


def get_cell_assembly_rate(nb_stim, stim_rates, min_rate):
 '''
 Identify stimulus-evoked cell assemblies using firing rate.

 Returns: nested list with one list per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
          each list contains the ids of the neurons that respond selectively to the respective
          stimulus.
 '''
 
 cell_assemblies = []
 for stim in range(nb_stim):
  cell_assembly = []
  for idx,rate in enumerate(stim_rates[stim].flatten()):
   if rate > min_rate:
    cell_assembly.append(idx)
  cell_assemblies.append(cell_assembly)

 return cell_assemblies


def get_cell_assembly_nmf(nb_patterns, nmf, min_weight):
 '''
 Identify stimulus-evoked cell assemblies using NMF.

 Returns: nested list with one list per stimulus in an ordered fashion (i.e. stimulus 0...nb_stim).
          each list contains the ids of the neurons that respond selectively to the respective
          stimulus.
 '''

 cell_assemblies = []
 order = [i for i in range(nb_patterns)]
 for comp in nmf.components_[order]:
  cell_assembly = []
  for idx,weight in enumerate(comp):
   if weight > min_weight:
    cell_assembly.append(idx)
  cell_assemblies.append(cell_assembly)


 return cell_assemblies


def get_lst_seq_count(lst):
 '''
 Counts the number of elements in each sequence in lst.
 lst is a list where each element is a sequence (e.g., list or tuple)

 Returns: array where each element i contains the number of elements in the sequence i of lst
 '''
 return np.array([len(seq) for seq in lst])
 '''
 count = []
 for seq in lst:
  print('seq', seq)
  count.append(len(seq))
 return np.array(count)
 '''


def get_neuron_membership(nb_neurons, cell_assemblies):
 '''
 Extracts the cell assemblies to which each neuron belongs.

 Returns: dictionary where the key is the neuron id and the value is the list of cell assemblies
          the neuron belongs to
 '''
 neuron_membership = {}
 
 for neuron in range(nb_neurons):
  neuron_membership[neuron] = []
  
 for stim,ca in enumerate(cell_assemblies):
  for neuron in ca:
   neuron_membership[neuron].append(stim)
   '''
   if neuron not in neuron_membership:
    neuron_membership[neuron] = [stim]
   else:
    neuron_membership[neuron].append(stim)
   '''
 return neuron_membership


def get_stim_combinations(nb_stim):
 '''
 Generates all possible combinations of stimuli

 Returns: list of tuples where each tuple is a different combination of stimuli
 '''
 stim = list(range(nb_stim))
 stim_combinations = []
 for length in range(0, nb_stim + 1):
  comb = combinations(stim, length)
  stim_combinations.extend(list(comb))
 return stim_combinations

  
def get_ca_partitions(nb_neurons, nb_stim, cell_assemblies):
 '''
 Extract the neurons that belong to every possible cell assembly partition.
 Cell assembly partition is a unique stimulus combination.
 For example, (0,1,3) refers to cells that simultaneously encode stimuli 0, 1, and 3.

 Returns: stim_combinations: list of all possible stimulus combinations
          ca_partitions: list where each element i is a tuple of the neurons that simultaneously 
                         belong to the cell assembly of each stimulus in stim_combinations[i]
 '''
 neuron_membership = get_neuron_membership(nb_neurons, cell_assemblies)
 stim_combinations = get_stim_combinations(nb_stim)
 ca_partitions_dic = {}
 for sc in stim_combinations:
  ca_partitions_dic[sc] = []
 for neuron,cas in neuron_membership.items():
  ca_partitions_dic[tuple(cas)].append(neuron)
 ca_partitions = []
 for sc in stim_combinations:
  ca_partitions.append(ca_partitions_dic[sc])
 return stim_combinations, ca_partitions
 
 


def get_neuron_ca_count(nb_neurons, cell_assemblies):
 '''
 Counts the number of cell assemblies each neuron is a member of.

 Returns: array where each element i contains the cell assembly count of neuron i.
 '''
 neuron_ca_count = np.zeros(nb_neurons, dtype=int)
 for assembly in cell_assemblies:
  for neuron in assembly:
   neuron_ca_count[neuron] += 1

 return neuron_ca_count


def get_neuron_ca_count_freq_old(nb_stim, neuron_ca_count):
 '''
 DEPRECATED: use get_neuron_ca_count_freq instead
 Computes the frequency distribution of neuronal cell assembly membership count.

 Returns: array where each element i has the number of neurons that are members of i assemblies.
 '''
 neuron_ca_count_freq = np.zeros(nb_stim + 1)
 for count in list(neuron_ca_count):
  neuron_ca_count_freq[count] += 1

 return neuron_ca_count_freq


def get_neuron_ca_count_freq(nb_patterns, stim_combinations, ca_partitions):
 '''
 Computes the frequency distribution of neuronal cell assembly membership count.

 Returns: array where each element i has the number of neurons that are members of i assemblies.
 '''
 neuron_ca_count_freq = np.zeros(1 + nb_patterns)
 for sc,cap in zip(stim_combinations,ca_partitions):
  neuron_ca_count_freq[len(sc)] += len(cap)
 return neuron_ca_count_freq


def plot_ca_count(nb_stim, colors, ca_count, nb_neurons, brain_area, neuron_type, datadir, filename):
 '''
 Plot and save the cell assembly membership count.
 '''
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

 print('ca_count')
 for stim in range(nb_stim):
  print(stim, ca_count[stim])
 #fig = plt.figure()
 #fig = plt.figure(figsize=(2.5,2.0))
 fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
 #fig = plt.figure(figsize=(1.12,0.84))
 ax = fig.add_subplot(111)
 ax.bar(np.arange(-1, nb_stim), ca_count/float(nb_neurons)*100, color=["black"] + colors)
 #ax.get_xaxis().set_visible(False)
 #ax.set_xticklabels([])
 ax.set_xticks([])
 plt.xlabel("engram")
 plt.ylabel("fraction of\n neurons (%)")
 #plt.title("%s %s cell assembly membership"%(brain_area, neuron_type))
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print("Saved %s %s cell assembly membership plot"%(brain_area, neuron_type))
 plt.close(fig)

 
def plot_ca_partition_count(stim_combinations, ca_partition_count, brain_area, neuron_type, datadir, filename):
 '''
 Plot and save the cell assembly partition membership count.
 '''
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 fig = plt.figure(figsize=(6.4,7.8))
 bars = [str(sc) for sc in stim_combinations]
 y_pos = np.arange(len(bars))
 plt.bar(y_pos, ca_partition_count, color="white", edgecolor="black")
 plt.xticks(y_pos, bars, rotation="vertical")
 plt.ylabel("Neurons")
 plt.title("%s %s cell assembly partition membership"%(brain_area, neuron_type))
 sns.despine()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print("Saved %s %s cell assembly partition membership plot"%(brain_area, neuron_type))
 plt.close(fig)

 
def plot_neuron_ca_count_freq(nb_stim, neuron_ca_count_freq, nb_neurons, brain_area, neuron_type, datadir, filename):
 '''
 Plot and save the frequency distribution of neuronal cell assembly membership count.
 '''
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 #print('ca', type(neuron_ca_count_freq))
 #fig = plt.figure(figsize=(6.4,4.8))
 #fig = plt.figure(figsize=(3.7,2.8))
 fig = plt.figure(figsize=(3.2,2.4))
 #fig = plt.figure(figsize=(1.12,0.84))
 #fig = plt.figure(figsize=(7.7,5.8))
 plt.bar(np.arange(0, nb_stim + 1), neuron_ca_count_freq/float(nb_neurons)*100, color="white", edgecolor="black")
 plt.xlabel("# of engrams")
 plt.ylabel("fraction of\n neurons (%)")
 plt.ylim((0,105))
 ax = fig.gca()
 ax.xaxis.set_major_locator(MaxNLocator(integer=True))
 #plt.title("%s %s neuron cell assembly membership"%(brain_area, neuron_type))
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print("Saved %s %s cell assembly membership distribution plot"%(brain_area, neuron_type))
 plt.close(fig)

 
def save_cell_assembly_file(brain_area, cell_assemblies, datadir, filename):
 '''
 Save previously identified cell assemblies in a file for future monitoring.
 '''
 with open(os.path.join(datadir, filename), "w") as f:
  for assembly in cell_assemblies:
   for neuron in assembly:
    f.write("%i\n"%neuron)
   f.write("\n\n")
 print ("Saved %s cell assemblies"%brain_area)


def generate_replay_pat(brain_area, cell_assemblies, nb_neurons_stim_replay, datadir, filename):
 '''
 Generate replay pattern based on cell assemblies in a file for future replay.

 nb_neurons_stim_replay: total number of neurons in the replay pattern per stimulus
 '''
 with open(os.path.join(datadir, filename), "w") as f:
  neuron_id = 0
  for ca in range(len(cell_assemblies)):
   f.write("# Replay pattern for stimulus %i\n"%(ca))
   for _ in range(nb_neurons_stim_replay):
    f.write("%i 1.000000\n"%neuron_id)
    neuron_id += 1
   f.write("\n\n")
 print ("Saved replay pattern in %s"%brain_area)


def load_cell_assemblies(datadir, filename):
 '''
 Load previously identified cell assemblies saved in a file.
 '''
 cell_assemblies = []
 with open(os.path.join(datadir, filename), "r") as f:
  lines = f.readlines()
  is_new_assemb = True
  for line in lines:
   if is_new_assemb:
    if line != '\n':
     cell_assemblies.append([])
     cell_assemblies[-1].append(int(line.split()[0]))
     is_new_assemb = False
   else:
    if line != '\n':
     cell_assemblies[-1].append(int(line.split()[0]))
    else:
     is_new_assemb = True

 return cell_assemblies


def get_ca_rates_old(spike_counts, cell_assemblies, bin_size):
 '''
 DEPRECATED: use get_ca_rates(neuron_rates, cell_assemblies) instead.

 Compute the firing rate of cell assemblies.

 Returns: list containing the firing rate of cell assemblies in an ordered manner (0...#cas).
          each list element is a 1D array.
 '''
 ca_spike_counts = []
 ca_rates = []
 for idx in range(len(cell_assemblies)):
  ca_spike_counts.append(spike_counts[:, cell_assemblies[idx]])
  ca_rates.append(np.sum(ca_spike_counts[idx], axis=1)/float(len(cell_assemblies[idx]) * bin_size))

 return ca_rates


def get_ca_rates(neuron_rates, cell_assemblies):
 '''
 Compute the firing rate of cell assemblies.

 Returns: list containing the firing rate of cell assemblies in an ordered manner (0...#cas).
          each list element is a 1D array.
 '''
 ca_rates = []
 for ca in cell_assemblies:
  ca_neuron_rates = neuron_rates[:, ca]
  if ca_neuron_rates.shape[0] == 0:
   print('Error when attempting to compute cell assembly rates!')
   print('Shape must be (nb_presentations>0, len(cell_assembly)>0')
   print('Shape of ca_neuron_rates is', ca_neuron_rates.shape)
   raise ValueError
  ca_rates.append(np.average(ca_neuron_rates, axis=1))

 return ca_rates


def get_ca_percentile_rates(neuron_rates, cell_assemblies, percentile):
 '''
 Compute the firing rate of cell assemblies.

 Returns: list containing the firing rate of cell assemblies in an ordered manner (0...#cas).
          each list element is a 1D array.
 '''
 ca_rates = []
 for ca in cell_assemblies:
  ca_neuron_rates = neuron_rates[:, ca]
  if ca_neuron_rates.shape[0] == 0:
   print('Error when attempting to compute cell assembly rates!')
   print('Shape must be (nb_presentations>0, len(cell_assembly)>0')
   print('Shape of ca_neuron_rates is', ca_neuron_rates.shape)
   raise ValueError
  ca_rates.append(np.percentile(ca_neuron_rates, percentile, axis=1))

 return ca_rates


def get_ca_stim_rates_old(rates_per_stim_presentation, cell_assemblies):
 '''
 DEPRECATED: use get_ca_stim_rates(rates_per_stim_presentation, cell_assemblies) instead.

 Compute the stimulus-evoked firing rate of cell assemblies.
 One firing rate is computed per cell assembly per stimulus presentation.

 Returns: list with one 2D array per stimulus in an ordered manner.
          each array row indicates stimulus presentation while column indicates cell assembly.
 '''
 ca_stim_rates = []
 for stim in range(len(rates_per_stim_presentation)):
  ca_rates_list = get_ca_rates(rates_per_stim_presentation[stim], cell_assemblies)
  nb_stim_presentations = rates_per_stim_presentation[stim].shape[0]
  ca_rates_array = np.zeros((nb_stim_presentations, len(cell_assemblies)))
  for ca in range(len(cell_assemblies)):
   ca_rates_array[:, ca] = ca_rates_list[ca]
  ca_stim_rates.append(ca_rates_array)

 return ca_stim_rates


def get_ca_stim_rates(rates_per_stim_presentation, cell_assemblies):
 '''
 Compute the stimulus-evoked firing rate of cell assemblies.
 One firing rate is computed per cell assembly per stimulus presentation.

 Returns: list with one 2D array per stimulus in an ordered manner.
          each array row indicates stimulus presentation while column indicates cell assembly.
 '''
 ca_stim_rates = []
 for stim in range(len(rates_per_stim_presentation)):
  ca_rates = get_ca_rates(rates_per_stim_presentation[stim], cell_assemblies)
  ca_stim_rates.append(np.stack(ca_rates, axis=-1))

 return ca_stim_rates


def get_ca_stim_fraction(rates_per_stim_presentation, cell_assemblies, threshold, ensembles=None):
 '''
 Compute the fraction of neurons in a cell assembly that fire above threshold in response 
 to stimuli. If ca_fraction=False, activated fraction is computed relative to nb_neurons.
 One fraction is computed per cell assembly per stimulus presentation.

 Returns: list with one 2D array per stimulus in an ordered manner.
          each array row indicates stimulus presentation while column indicates cell assembly.
 '''
 ca_stim_fraction = []
 for stim in range(len(rates_per_stim_presentation)):
  ca_fraction = []
  if type(ensembles) == type(None):
   ensembles = cell_assemblies
  for ca,en in zip(cell_assemblies,ensembles):
   ca_neuron_rates = rates_per_stim_presentation[stim][:, ca]
   ca_fraction.append(np.sum(ca_neuron_rates > threshold, axis=1) / float(len(en)))
   #ca_fraction.append(np.sum(ca_neuron_rates > threshold, axis=1) / float(len(ca)))
   '''
   if ca_fraction:
    ca_fraction.append(np.sum(ca_neuron_rates > threshold, axis=1) / float(len(ca)))
   else:
    ca_fraction.append(np.sum(ca_neuron_rates > threshold, axis=1) / float(nb_neurons))
   '''
  ca_stim_fraction.append(np.stack(ca_fraction, axis=-1))

 return ca_stim_fraction


def get_ca_percentile_stim_rates(rates_per_stim_presentation, cell_assemblies, percentile):
 '''
 Compute the stimulus-evoked firing rate of cell assemblies.
 One firing rate is computed per cell assembly per stimulus presentation.

 Returns: list with one 2D array per stimulus in an ordered manner.
          each array row indicates stimulus presentation while column indicates cell assembly.
 '''
 ca_stim_rates = []
 for stim in range(len(rates_per_stim_presentation)):
  ca_rates = get_ca_percentile_rates(rates_per_stim_presentation[stim], cell_assemblies, percentile)
  ca_stim_rates.append(np.stack(ca_rates, axis=-1))

 return ca_stim_rates


def get_ca_pat_rates_old(rates_across_pat_presentation, cell_assemblies):
 '''
 Compute the pattern-evoked firing rate of cell assemblies.
 One firing rate is computed per cell assembly per pattern.

 Returns: list with one array per pattern in an ordered manner.
          each array element indicates the firing rate of a cell assembly in an ordered manner.
 '''
 ca_pat_rates = []
 for pat in range(len(rates_across_pat_presentation)):
  ca_rates = np.zeros(len(cell_assemblies))
  for ca_idx,ca in enumerate(cell_assemblies):
   ca_neuron_rates = rates_across_pat_presentation[pat][ca]
   print('ca_neuron_rates.shape', ca_neuron_rates.shape)
   print('ca_neuron_rates\n', ca_neuron_rates)
   if ca_neuron_rates.shape[0] == 0:
    print('Error when attempting to compute cell assembly rates!')
    print('Shape must be (len(cell_assembly)>0)')
    print('Shape of ca_neuron_rates is', ca_neuron_rates.shape)
    raise ValueError
   ca_rates[ca_idx] = np.average(ca_neuron_rates)
  ca_pat_rates.append(ca_rates)

 print('type(ca_pat_rates)', type(ca_pat_rates))
 print('ca_pat_rates\n', ca_pat_rates)
 return ca_pat_rates


def get_ca_pat_rates(nb_patterns, stim_pat, ca_stim_rates):
 '''
 Compute the average stimulus-evoked firing rate of cell assemblies across stimulus presentations
 of a given pattern.
 One firing rate is computed per cell assembly per pattern.

 Returns: list with one array per pattern in an ordered manner.
          each array element indicates the firing rate of a cell assembly in an ordered manner.
 '''
 ca_pat_rates = []
 for pattern in range(nb_patterns):
  ca_rates = []
  for stim,pat in stim_pat.items():
   if pat == pattern:
    ca_rates.append(ca_stim_rates[stim])
  pat_rates = np.concatenate(ca_rates, axis=0)
  #print('pat_rates.shape', pat_rates.shape)
  #print('pat_rates\n', pat_rates)
  ca_pat_rates.append(np.average(pat_rates, axis=0))

 #print('type(ca_pat_rates)', type(ca_pat_rates))
 #print('ca_pat_rates\n', ca_pat_rates)
 return ca_pat_rates


def plot_stim_activity(nb_patterns, nb_stim, stim_pat, t_start, t_stop, time, min_rate, area_color, colors, cell_assemb_method, datadir, filename, brain_area, neuron_type, stimtime, stimdata, rates, binned_spike_counts=None):
 '''
 Plot and save stimulus and activity of specified brain area.

 binned_spike_counts is required if cell_assemb_method is nmf.
 '''

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting stimulus and %s %s activity..."%(brain_area, neuron_type))
 # set up plot area
 gs = GridSpec(2,1,height_ratios=[1,8])
 #fig = plt.figure(figsize=(16,4), dpi=300)
 #fig = plt.figure(figsize=(6.4,3.2), dpi=300)
 #fig = plt.figure(figsize=(2,1.2), dpi=300)
 #fig = plt.figure(figsize=(3,1.32), dpi=300)
 fig = plt.figure(figsize=(3.5,2.0), dpi=300) # recent and remote separate
 #fig = plt.figure(figsize=(3,1), dpi=300) # recent and remote together
 
 # plot stimulus
 ax = plt.subplot(gs[0])

 if cell_assemb_method == "rate":
  for stim in range(nb_stim):
   #plt.plot(stimtime, stimdata[:, stim], color=colors[stim_pat[stim]])
   stimact = stimdata[:, stim]
   zeros = np.zeros(stimact.shape)
   ax.fill_between(stimtime, stimact, where=stimact > zeros, color=colors[stim_pat[stim]])
   
 if cell_assemb_method == "nmf":
  # note that these colors might be out of order due to permutation invariance of NMF 
  # which is why we make them black below
  order = [i for i in range(nb_stim)]
  plt.plot(stimtime, stimdata[:, order], color="black")

 plt.xlim((t_start,t_stop))

 ax.axis('off')
 
 #ax.set_xticklabels([])
 #ax.set_yticklabels([])
 
 #ax.get_xaxis().set_visible(False)
 #ax.get_yaxis().set_visible(False)
 
 #plt.xlabel("Time (s)")
 #plt.ylabel("Stimulus")
 sns.despine()
 
 # plot population activity
 ax = plt.subplot(gs[1])

 if  len(rates) == 1:
  plt.plot(time, rates[0], color=area_color) # area rate
 elif len(rates) == nb_patterns:
  for ca in range(nb_patterns):
   plt.plot(time, rates[ca], color=colors[ca]) # rates of cell assemblies
 elif len(rates) == nb_patterns + 1:
   plt.plot(time, rates[0], color='black') # rate of cluster not sensitive to any stimulus
   for ca in range(nb_patterns):
    plt.plot(time, rates[ca+1], color=colors[ca]) # rates of cell assemblies

 ax.axhline(y=min_rate, color='gray', linestyle='--', linewidth=0.75)
 #plt.xlim((t_start,t_stop))
 ax.set_xlim(left=t_start, right=t_stop)
 ax.set_xticks((t_start, t_stop))
 plt.xlabel("testing time (s)")
 plt.ylabel("population activity (Hz)")
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ('Saved plot of stimulus and %s %s activity'%(brain_area, neuron_type))
 plt.close(fig)


def plot_stim_spike_activity(nb_patterns, nb_stim, stim_pat, t_start, t_stop, time, min_rate, area_color, colors, cell_assemb_method, datadir, filename, brain_area, neuron_type, stimtime, stimdata, rates, spikes):
 '''
 Plot and save stimulus, spikes, and activity of specified brain area.

 binned_spike_counts is required if cell_assemb_method is nmf.
 '''

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting stimulus and %s %s spikes and activity..."%(brain_area, neuron_type))
 # set up plot area
 gs = GridSpec(3,1,height_ratios=[1,20,14])
 #fig = plt.figure(figsize=(16,8))
 #fig = plt.figure(figsize=(3.5,1.75))
 fig = plt.figure(figsize=(3.5,2.291))
 
 # plot stimulus
 ax = plt.subplot(gs[0])

 if cell_assemb_method == "rate":
  for stim in range(nb_stim):
   #plt.plot(stimtime, stimdata[:, stim], color=colors[stim_pat[stim]])
   stimact = stimdata[:, stim]
   zeros = np.zeros(stimact.shape)
   ax.fill_between(stimtime, stimact, where=stimact > zeros, color=colors[stim_pat[stim]])
   
 if cell_assemb_method == "nmf":
  # note that these colors might be out of order due to permutation invariance of NMF 
  # which is why we make them black below
  order = [i for i in range(nb_stim)]
  plt.plot(stimtime, stimdata[:, order], color="black")
 
 ax.axis('off')
 
 #ax.set_xticklabels([])
 #ax.set_yticklabels([])
 
 #ax.get_xaxis().set_visible(False)
 #ax.get_yaxis().set_visible(False)
 
 #plt.xlabel("Time (s)")
 #plt.ylabel("Stimulus")
 
 plt.xlim((t_start,t_stop))
 sns.despine()

 # plot spike raster
 ax = plt.subplot(gs[1])

 #for neuron in range(spikes.shape[1]):
  #plt.plot(time, spikes[:, neuron], color='black', marker='o', markersize=1, linestyle='')

 #print('spikes.shape', spikes.shape)
 #plt.scatter(spikes[:,0], spikes[:,1], c='black', s=1) # s=1

 '''
 print('spikes', spikes.shape)
 print(spikes)
 np.random.shuffle(spikes)
 print("SHUFFLED")
 print('spikes', spikes.shape)
 print(spikes)
 '''

 '''
 plot_idx = np.random.permutation(spikes.shape[0])
 print('plot_idx', type(plot_idx))
 print(plot_idx)
 plt.scatter(spikes[plot_idx[::5],0], spikes[plot_idx[::5],1], c='black', s=1.0, marker='.', edgecolors=None, linewidths=0)
 '''

 #print('spikes', spikes.shape)
 #print(spikes)
 
 neuron_map = {}
 neurons = list(np.random.choice(256, size=256))
 neuron_count = 0
 for x in spikes:
  #print(x[0], x[1])
  if x[1] not in neuron_map:
   neuron_map[x[1]] = neurons[neuron_count]
   neuron_count += 1
  x[1] = neuron_map[x[1]]

 #print('spikes', spikes.shape)
 #print(spikes)
 
 plt.scatter(spikes[::5,0], spikes[::5,1], c='black', s=1.0, marker='.', edgecolors=None, linewidths=0)
 
 
 plt.xlim((t_start,t_stop))
 plt.ylabel("256 Neurons")
 #ax.get_xaxis().set_visible(False)
 #ax.set_yticklabels([])
 ax.axis('off')
 sns.despine()
 
 # plot population activity
 ax = plt.subplot(gs[2])

 if  len(rates) == 1:
  plt.plot(time, rates[0], color=area_color) # area rate
 elif len(rates) == nb_patterns:
  for ca in range(nb_patterns):
   plt.plot(time, rates[ca], color=colors[ca]) # rates of cell assemblies
 elif len(rates) == nb_patterns + 1:
   plt.plot(time, rates[0], color='black') # rate of cluster not sensitive to any stimulus
   for ca in range(nb_patterns):
    plt.plot(time, rates[ca+1], color=colors[ca]) # rates of cell assemblies

 #ax.axhline(y=min_rate, color='gray', linestyle='--', linewidth=0.75)
 ax.axhline(y=4, color='gray', linestyle='--', linewidth=0.75)
 ax.axhline(y=10, color='gray', linestyle='--', linewidth=0.75)
 
 ax.set_xticks([t_start, t_stop])
 ax.set_yticks([0, 4, 10])
 
 plt.xlim((t_start,t_stop))
 
 plt.xlabel("Consolidation time (HH:MM)")
 plt.ylabel("Population activity (Hz)")
 
 sns.despine()
 
 plt.savefig(os.path.join(datadir, filename))
 print ('Saved plot of stimulus and %s %s spikes and activity'%(brain_area, neuron_type))
 plt.close(fig)


def plot_activity(nb_patterns, t_start, t_stop, t, min_rate, area_color, colors, cell_assemb_method, datadir, filename, brain_area, neuron_type, rates):
 '''
 Plot and save cell assembly activity of specified brain area.

 binned_spike_counts is required if cell_assemb_method is nmf.
 '''
 
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting %s %s activity..."%(brain_area, neuron_type))
 
 #t = t / 3600.0
 #t_start = t_start / 3600.0
 #t_stop = t_stop / 3600.0
 
 # set up plot area
 #gs = GridSpec(1,1)
 
 #fig = plt.figure(figsize=(6.2,2.0))

 # plot population activity
 #ax = plt.subplot(gs[0])
 #fig, ax = plt.subplots(figsize=(6.2,2.0))
 #fig, ax = plt.subplots(figsize=(3.1,2.0))
 #fig, ax = plt.subplots(figsize=(3.2,2.4), dpi=300)
 fig, ax = plt.subplots(figsize=(2.24,1.68), dpi=300)
 #fig, ax = plt.subplots(figsize=(1.12,0.84), dpi=300)

 if  len(rates) == 1:
  ax.plot(t, rates[0], color=area_color) # area rate
 elif len(rates) == nb_patterns:
  for ca in range(nb_patterns):
   ax.plot(t, rates[ca], color=colors[ca]) # rates of cell assemblies
 elif len(rates) == nb_patterns + 1:
   ax.plot(t, rates[0], color='black') # rate of cluster not sensitive to any stimulus
   for ca in range(nb_patterns):
    ax.plot(t, rates[ca+1], color=colors[ca]) # rates of cell assemblies

 ax.axhline(y=min_rate, color='gray', linestyle='--', linewidth=1.00)
 #ax.xlim((t_start,t_stop))
 ax.set_xlim(left=t_start, right=t_stop)
 ax.set_xticks((t_start, t_stop))
 #formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
 #formatter = FuncFormatter(lambda s, x: '{}h {}min'.format(int(s // 3600.0), int((s % 3600.0) / 60)))
 formatter = FuncFormatter(lambda s, x: '{:.{prec}}'.format(s / 3600.0, prec=3))
 #formatter = FuncFormatter(lambda s, x: s / 3600.0)
 ax.xaxis.set_major_formatter(formatter)
 
 plt.xlabel("consolidation time (h)")
 plt.ylabel("population\n activity (Hz)")
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ('Saved plot of %s %s activity'%(brain_area, neuron_type))
 plt.close(fig)
 
 
def plot_spike_activity(nb_patterns, t_start, t_stop, time, min_rate, area_color, colors, cell_assemb_method, datadir, filename, brain_area, neuron_type, rates, spikes):
 '''
 Plot and save spike raster and cell assembly activity of specified brain area.

 binned_spike_counts is required if cell_assemb_method is nmf.
 '''
 
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting %s %s spike raster and activity..."%(brain_area, neuron_type))
 # set up plot area
 gs = GridSpec(2,1,height_ratios=[20,14])
 #fig = plt.figure(figsize=(16,8), dpi=300)
 #fig = plt.figure(figsize=(6.4,3.2), dpi=300)
 #fig = plt.figure(figsize=(6.4,2.5), dpi=300)
 #fig = plt.figure(figsize=(5.12,2.5), dpi=300)
 fig = plt.figure(figsize=(5.12,1.75), dpi=300)

 # plot spike raster
 ax = plt.subplot(gs[0])

 #for neuron in range(spikes.shape[1]):
  #plt.plot(time, spikes[:, neuron], color='black', marker='o', markersize=1, linestyle='')

 #print("spikes shape:", np.shape(spikes))

 plt.scatter(spikes[::5,0], spikes[::5,1], c='black', s=0.075, marker=',')
 
 plt.xlim((t_start,t_stop))
 #plt.ylabel("256 Neurons")
 #ax.get_xaxis().set_visible(False)
 #ax.set_yticklabels([])
 ax.axis('off')
 sns.despine()
 
 # plot population activity
 ax = plt.subplot(gs[1])

 if  len(rates) == 1:
  plt.plot(time, rates[0], color=area_color) # area rate
 elif len(rates) == nb_patterns:
  for ca in range(nb_patterns):
   plt.plot(time, rates[ca], color=colors[ca]) # rates of cell assemblies
 elif len(rates) == nb_patterns + 1:
   plt.plot(time, rates[0], color='black') # rate of cluster not sensitive to any stimulus
   for ca in range(nb_patterns):
    plt.plot(time, rates[ca+1], color=colors[ca]) # rates of cell assemblies

 #ax.axhline(y=min_rate, color='gray', linestyle='--', linewidth=0.75)
 ax.axhline(y=4, color='black', linestyle='--', linewidth=0.75)
 #plt.xlim((t_start,t_stop))
 ax.set_xlim(left=t_start, right=t_stop)
 ax.set_xticks((t_start, t_stop))
 #formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
 #formatter = FuncFormatter(lambda s, x: '{}h {}min'.format(int(s // 3600.0), int((s % 3600.0) / 60)))
 formatter = FuncFormatter(lambda s, x: '{}:{}:{}'.format(int(s // 3600.0), int((s % 3600.0) / 60), int((s % 3600.0) % 60)))
 #formatter = FuncFormatter(lambda s, x: '{:.{prec}}'.format(s / 3600.0, prec=3))
 #formatter = FuncFormatter(lambda s, x: s / 3600.0)
 ax.xaxis.set_major_formatter(formatter)
 
 plt.xlabel("consolidation time (HH:MM:SS)")
 plt.ylabel("population activity (Hz)")
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ('Saved plot of %s %s spike raster and activity'%(brain_area, neuron_type))
 plt.close(fig)

#''' 
def plot_spike_stats(spikes, color, brain_area, neuron_type, datadir, filename):
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting %s %s spike stats..."%(brain_area, neuron_type))
 fig = plt.figure()
 #fig = plt.figure(figsize=(10,3))
 #fig = plt.figure(figsize=(7.0,1.68))
 #fig = plt.figure(figsize=(3.5,0.84))
 #fig = plt.figure(figsize=(1.75,0.84))
 
 plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
 
 plt.subplot(131)
 #plt.tick_params( axis='y', which='both', left=False, right=False, labelleft=False)
 rates = stats.rates(spikes)
 #print ("rates", type(rates))
 #print (rates.shape)
 #rates = rates[rates < 1 + 1/30]
 #print(rates.shape)
 #exit()
 #plt.hist(rates, bins=np.arange(0, 1 + 1/10, 1/10), color=color, edgecolor="black")
 plt.hist(rates, bins=30, color=color, edgecolor="black")
 #stats.rate_hist(spikes, color="#00548c", bins=30)
 plt.xlabel("Rate (Hz)")

 plt.subplot(132)
 plt.tick_params( axis='y', which='both', left=False, right=False, labelleft=False)
 isis = stats.isis(spikes)
 plt.hist(isis, bins=np.logspace(np.log10(1e-3), np.log10(10.0), 30), color=color, edgecolor="black")
 #stats.isi_hist(spikes, color="#ffcc00", bins=np.logspace(np.log10(1e-3), np.log10(10.0), 30))
 plt.gca().set_xscale("log")
 plt.xlabel("ISI (s)")

 plt.subplot(133)
 plt.tick_params( axis='y', which='both', left=False, right=False, labelleft=False)
 cvisis = stats.cvisis(spikes)
 #plt.hist(cvisis, bins=30, color=color, edgecolor="black")
 plt.hist(cvisis, bins=np.arange(0, 2 + 2/30, 2/30), color=color, edgecolor="black")
 #stats.cvisi_hist(spikes, color="#c4000a", bins=30)
 plt.xlabel("CV ISI")

 plt.tight_layout()
 sns.despine()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved %s %s spike stats plot"%(brain_area, neuron_type))
 plt.close(fig)
#'''

'''
def plot_spike_stats(spikes, color, brain_area, neuron_type, datadir, filename):
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 print("Plotting %s %s spike stats..."%(brain_area, neuron_type))
 #fig = plt.figure(figsize=(10,3))
 #fig = plt.figure(figsize=(3.5,0.84))
 #fig = plt.figure(figsize=(1.75,0.84))
 #fig = plt.figure(figsize=(1.2,0.84))

 #f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(1.2,0.84))
 f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

 
 
 #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
 
 #plt.subplot(131)
 #plt.tick_params( axis='y', which='both', left=False, right=False, labelleft=False)
 rates = stats.rates(spikes)
 #print ("rates", type(rates))
 #print (rates.shape)
 #rates = rates[rates < 1 + 1/30]
 #print(rates.shape)
 #exit()
 #plt.hist(rates, bins=np.arange(0, 1 + 1/10, 1/10), color=color, edgecolor="black")
 #stats.rate_hist(spikes, color="#00548c", bins=30)

 ax.hist(rates, bins=30, color=color, edgecolor="black")
 ax2.hist(rates, bins=30, color=color, edgecolor="black")

 ax.set_ylim(0.5, 1)
 ax2.set_ylim(0, 0.5)

 
 ax.spines['bottom'].set_visible(False)
 ax2.spines['top'].set_visible(False)
 ax.xaxis.tick_top()
 ax.tick_params(labeltop=False)  # don't put tick labels at the top
 ax2.xaxis.tick_bottom()
 
 plt.xlabel("Rate (Hz)")

 plt.tight_layout()
 sns.despine()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved %s %s spike stats plot"%(brain_area, neuron_type))
 plt.close(f)
 exit()
'''


def plot_weight_distribution(area, connection, weights, datadir, filename, color=None):
 print ("Plotting %s %s weight distribution..."%(area,connection))

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
 #fig = plt.figure()
 fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
 #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
 #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
 
 if type(color) == type(None):
  wh = plt.hist(weights.data, bins=100, log=True)
 else:
  wh = plt.hist(weights.data, bins=100, log=True, color=color)
 
 #plt.title("%s %s weight distribution"%(area,connection))
 plt.xlabel('weight')
 plt.ylabel('# of synapses')
 #ax = plt.gca()
 #ax.axes.get_yaxis().set_visible(False)
 #plt.axis('off')
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved plot of %s %s weight distribution"%(area,connection))
 plt.close(fig)


def plot_rf_weight_distribution(area, connection, weights, datadir, filename, color=None):
 print ("Plotting %s %s weight distribution..."%(area,connection))

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
 #fig = plt.figure()
 fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
 #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
 #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
 
 if type(color) == type(None):
  wh = plt.hist(weights)
 else:
  wh = plt.hist(weights, color=color)
 
 #plt.title("%s %s weight distribution"%(area,connection))
 plt.xlabel('total weight')
 plt.ylabel('# of neurons')
 #ax = plt.gca()
 #ax.axes.get_yaxis().set_visible(False)
 #plt.axis('off')
 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved plot of %s %s weight distribution"%(area,connection))
 plt.close(fig)


def chain_ca_partitions(ca_partitions):
 '''
 Chain the neurons in each cell assembly partition in sequence.

 Returns: list with neurons in the order they appear in ca_partitions
 '''
 neurons = []
 for cap in ca_partitions:
  neurons.extend(cap)
 return neurons



def plot_weight_matrix(area, connection, wmatrix, ordered_pre, ordered_post, datadir, filename, cmap=None):
 print ("Plotting %s %s weight matrix..."%(area,connection))

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
 #fig, ax = plt.subplots()
 #fig, ax = plt.subplots(figsize=(3.2,2.4), dpi=300)
 #fig, ax = plt.subplots(figsize=(2.24,1.68), dpi=300)
 fig = plt.figure(figsize=(1.568, 1.176), dpi=300)
 #fig, ax = plt.subplots(figsize=(1.68,1.26), dpi=300)
 #fig, ax = plt.subplots(figsize=(1.12,0.84), dpi=300)
 
 if type(cmap) == type(None):
  plt.imshow(wmatrix, origin='lower')
  #plt.imshow(mean_wmat, origin='lower', norm=Normalize(vmin=0.000, vmax=0.050))
  #plt.imshow(mean_wmat, origin='lower', interpolation='nearest', vmin=0, vmax=0.015)
 else:
  plt.imshow(wmatrix, cmap=plt.get_cmap(cmap), origin='lower')
  #plt.imshow(mean_wmat, cmap=plt.get_cmap(cmap), origin='lower', norm=Normalize(vmin=0.000, vmax=0.050))
  #plt.imshow(mean_wmat, cmap=plt.get_cmap(cmap), origin='lower', interpolation='nearest', vmin=0, vmax=0.015)
  
 plt.axis('off')
 plt.colorbar()
 #plt.xlabel('Post')
 #plt.ylabel('Pre')
 #plt.title("%s %s weight matrix. ordered: pre %s / post %s"%(area,connection,ordered_pre,ordered_post))
 #sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved plot of %s %s weight matrix"%(area,connection))
 plt.close(fig)


def plot_mean_weight_matrix(area, connection, mean_wmat, pre_cells, post_cells, datadir, filename, clustering, cmap=None):
 print ("Plotting mean %s %s weight matrix..."%(area,connection))

 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 
 #fig, ax = plt.subplots(figsize=(3.2,2.4), dpi=300)
 #fig, ax = plt.subplots(figsize=(2.24,1.68), dpi=300)
 #fig, ax = plt.subplots(figsize=(1.68,1.26), dpi=300)
 fig, ax = plt.subplots(figsize=(1.568,1.176), dpi=300)
 #fig, ax = plt.subplots(figsize=(1.12,0.84), dpi=300)
 
 if type(cmap) == type(None):
  plt.imshow(mean_wmat, origin='lower')
  #plt.imshow(mean_wmat, origin='lower', norm=Normalize(vmin=0.000, vmax=0.085))
  #plt.imshow(mean_wmat, origin='lower', interpolation='nearest', vmin=0.000, vmax=0.085)
 else:
  plt.imshow(mean_wmat, cmap=plt.get_cmap(cmap), origin='lower')
  #plt.imshow(mean_wmat, cmap=plt.get_cmap(cmap), origin='lower', norm=Normalize(vmin=0.000, vmax=0.085))
  #plt.imshow(mean_wmat, cmap=plt.get_cmap(cmap), origin='lower', interpolation='nearest', vmin=0.000, vmax=0.085)
 
 plt.yticks(list(range(len(pre_cells))))
 plt.xticks(list(range(len(post_cells))))
 ax.set_yticklabels([])
 ax.set_xticklabels([])
 
 colorbar = plt.colorbar(format='%.3f')
 
 #plt.xlabel('Postsynaptic ' + clustering)
 #plt.ylabel('Presynaptic ' + clustering)
 #plt.title("%s %s %s mean weight"%(area,connection,clustering))

 #sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved plot of mean %s %s weight matrix"%(area,connection))
 plt.close(fig)


def get_recall_metrics(nb_stim, nb_patterns, stim_pat, min_rate, ca_stim_rates):
 all_pat_metrics = RecallMetrics(0, 0, 0, 0, 0, 0)
 ind_pat_metrics = [RecallMetrics(0, 0, 0, 0, 0, 0) for _ in range(nb_patterns)]
 for stim in range(nb_stim):
  pos_label = stim_pat[stim]
  neg_labels = [i for i in range(nb_patterns) if i != pos_label]
  nb_presentations = ca_stim_rates[stim].shape[0]
  ind_pat_metrics[pos_label].add_samples(nb_presentations)
  all_pat_metrics.add_samples(nb_presentations)
  '''
  print('stim', stim)
  print('pos_label', pos_label)
  print('neg_labels', neg_labels)
  print ('nb_presentations', nb_presentations)
  '''
  for presentation in range(nb_presentations):
   fp = 0
   tn = 0
   for nl in neg_labels:
    if ca_stim_rates[stim][presentation, nl] > min_rate:
     fp += 1
    else:
     tn += 1
   ind_pat_metrics[pos_label].add_fp(fp)
   ind_pat_metrics[pos_label].add_tn(tn)
   all_pat_metrics.add_fp(fp)
   all_pat_metrics.add_tn(tn)
   if ca_stim_rates[stim][presentation, pos_label] > min_rate:
    ind_pat_metrics[pos_label].add_tp(1)
    all_pat_metrics.add_tp(1)
    if fp == 0:
     ind_pat_metrics[pos_label].add_matches(1)
     all_pat_metrics.add_matches(1)
   else:
    ind_pat_metrics[pos_label].add_fn(1)
    all_pat_metrics.add_fn(1)

 return all_pat_metrics, ind_pat_metrics

 
def generate_replay_wmat(brain_area, nb_area_neurons, nb_neurons_stim_replay, nb_neuron_cons_stim_replay, cell_assemblies, datadir, filename, weight=1):
 '''
 Generate the receptive field weight matrix for replay->brain_area.

 nb_neuron_cons_stim_replay: connection probability from a replay pattern to the respective cell assembly
 '''
 nb_pre_neurons = nb_neurons_stim_replay * len(cell_assemblies)
 nb_post_neurons = nb_area_neurons
 all_post_neurons = list(range(nb_post_neurons))
 #wmat = np.zeros((nb_pre_neurons, nb_post_neurons), dtype=int)
 wmat = np.zeros((nb_pre_neurons, nb_post_neurons))

 for stim,ca in enumerate(cell_assemblies):
  pre_neurons = list(range(stim*nb_neurons_stim_replay, (stim+1)*nb_neurons_stim_replay))
  for pre_neuron in pre_neurons:
   #post_neurons = sample(all_post_neurons, int(nb_neuron_cons_stim_replay*nb_post_neurons))
   if brain_area in ['ctx','hpc','thl']:
    post_neurons = sample(ca, int(nb_neuron_cons_stim_replay*len(ca)))
   else:
    post_neurons = sample(all_post_neurons, int(nb_neuron_cons_stim_replay*len(nb_post_neurons)))
   for post_neuron in post_neurons:
    wmat[pre_neuron, post_neuron] = weight
  

  #all_pre_neurons = list(range(stim*nb_neurons_stim_replay, (stim+1)*nb_neurons_stim_replay))
  #for post_neuron in ca:
  # pre_neurons = sample(all_pre_neurons, nb_neuron_cons_stim_replay)
  # for pre_neuron in pre_neurons:
  #  wmat[pre_neuron, post_neuron] = weight
 
 swmat = csr_matrix(wmat) # ensures row major format in COO output
 mmwrite(os.path.join(datadir, filename), swmat)
 print("Saved connection matrix for replay")


'''
def plot_stim_evoked_ca_rate_hist(rates, colors, cell_assembly, datadir, filename):
 fig = plt.figure()
 max_rate = rates.max()
 max_bin = int(math.ceil(max_rate / 10))
 bins = [i*10 for i in range(max_bin + 1)]
 plt.hist(rates, bins=bins, weights=np.ones(len(rates)) / len(rates), color=colors[cell_assembly])
 plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
 #plt.ylim((0,1))
 plt.ylabel('fraction of cell assembly (%)')
 plt.xlabel('firing rate (Hz)')
 plt.grid(True)
 plt.savefig(os.path.join(datadir, filename))
 plt.close(fig)
'''


def get_size_intersection(sequence1, sequence2):
 return len(set(sequence1) & set(sequence2))


def plot_weight_cdf(area, connection, incoming_weights, colors, datadir, filename, n_bins=50):
 fig, ax = plt.subplots()
 
 for cluster,weights in enumerate(incoming_weights):
  if cluster == 0:
   color = 'black'
  else:
   color = colors[cluster - 1]
  n, bins, patches = ax.hist(weights, n_bins, density=True, histtype='step', cumulative=True, label='cluster ' + str(cluster), color=color)
  
 # tidy up the figure
 ax.grid(True)
 
 #ax.legend(loc='right')
 legend = ax.legend()
 legend.remove()
 #ax.get_legend().remove()
 
 #ax.set_title('Cumulative step histograms')
 ax.set_xlabel(connection + ' incoming weights (1)')
 ax.set_ylabel('Cumulative')
 
 plt.savefig(os.path.join(datadir, filename))
 print ("Saved cdf plot of %s %s incoming weights"%(area,connection))
 plt.close(fig)
  

 
def plot_comp_weight_cdf(area, connection, learn_incoming_weights, con_incoming_weights, colors, datadir, filename, n_bins=50):
 plt.rcParams.update({'font.size': 7})
 plt.rcParams.update({'font.family': 'sans-serif'})
 plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
 fig, ax = plt.subplots(dpi=300)
 #fig = plt.figure(figsize=(2.5,2.0))
 #fig.set_size_inches(3.7, 2.8)
 #fig.set_size_inches(3.4, 2.0)
 fig.set_size_inches(2.24, 1.68)
 #fig.set_size_inches(1.12, 0.84)
 #fig.set_size_inches(2.5, 2.0)
 #fig.set_size_inches(11.1, 8.4)

 '''
 for cluster,weights in enumerate(incoming_weights):
  if cluster == 0:
   color = 'black'
  else:
   color = colors[cluster - 1]
  n, bins, patches = ax.hist(weights, n_bins, density=True, histtype='step', cumulative=True, label='cluster ' + str(cluster), color=color)
 '''

 n, bins, patches = ax.hist(learn_incoming_weights, n_bins, density=True, histtype='step', cumulative=True, label='$t_{consolidation}=0h$', color='black', linewidth=1.2)
 n, bins, patches = ax.hist(con_incoming_weights, n_bins, density=True, histtype='step', cumulative=True, label='$t_{consolidation}=24h$', color='gray', linewidth=1.2)



 # tidy up the figure
 #ax.grid(True)

 #ax.legend(loc='right')
 #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
 #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


 #legend = ax.legend()
 #legend.remove()

 #ax.get_legend().remove()

 #ax.set_title('Cumulative step histograms')
 #ax.set_xlabel('feedforward\nweights (1)')
 #ax.set_xlabel('feedforward weights')
 #ax.set_xlabel('inhibitory weights')
 ax.set_xlabel('excitatory weights')
 ax.set_ylabel('cumulative')

 sns.despine()
 plt.tight_layout()
 plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
 print ("Saved cdf plot of %s %s comparative incoming weights in %s"%(area,connection,os.path.join(datadir, filename)))
 plt.close(fig)
