'''
Copyright 2023  Douglas Feitosa Tomé

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
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

import random
from scipy import stats
#from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat

matplotlib.use('Agg')


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)



def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def plot_compare_cdfs(data1,
                      data2,
                      label1,
                      label2,
                      color1,
                      color2,
                      xlabel,
                      datadir,
                      filename,
                      n_bins=50,
                      log=False):
    '''
    Plot the CDF of data1 and data2.
    '''

    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

    fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
    #fig, ax = plt.subplots(dpi=300)
    #fig.set_size_inches(1.12, 0.84)
    #fig.set_size_inches(3.7, 2.8)
    #fig.set_size_inches(3.4, 2.0)
    #fig.set_size_inches(2.5, 2.0)
    #fig.set_size_inches(11.1, 8.4)
 
    n, bins, patches = ax.hist(data1,
                               n_bins,
                               density=True,
                               histtype='step',
                               cumulative=True,
                               log=log,
                               label=label1,
                               color=color1,
                               linewidth=1.2)
    
    n, bins, patches = ax.hist(data2,
                               n_bins,
                               density=True,
                               histtype='step',
                               cumulative=True,
                               log=log,
                               label=label2,
                               color=color2,
                               linewidth=1.2)
    
    fix_hist_step_vertical_line_at_end(ax)

    # tidy up the figure
 
    #ax.grid(True)

    #ax.legend(loc='right')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #legend = ax.legend()
    #legend.remove()

    #ax.get_legend().remove()

    #ax.set_title('Cumulative Distribution Functions')

    #plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)
    #ax.vlines(10, 0, 1, colors='black', linestyles='--', linewidth=0.25)
    #ax.vlines(0.2, 0, 1, colors='black', linestyles='--', linewidth=0.25)
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative')

    #ax.set_xticks([0, 10, 20, 30, 40])
    #ax.set_xlim(-2,40)
    
    #ax.set_yticks([0.95, 1])
    #ax.set_yticks([], minor=True)
 
    #ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
 
    #ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    #ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    if log:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('% 1.2f'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('% 1.2f'))
 
    #ax.ticklabel_format(axis='y',style='plain')
 
    #ax.get_yaxis().set_ticks([])
    #ax.set_yticks([0.95, 1])

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
    print ("Saved plot of comparative CDFs")
    plt.close(fig)


class Animal:
        
    def __init__(self, animal, number, sessions, freezing, datadir):
        '''
        Load calcium imaging data ΔF/F of each separate session for a given animal
        
        animal: species
        number: animal number
        sessions: list of experimental sessions
        freezing: freezing behaviour for cfc, neutral1, training1, neutral2, and training2
        datadir: directory to csv files with calcium imaging data
        '''

        self.animal = animal

        self.number = number

        self.sessions = sessions

        self.freezing = freezing

        self.data = [np.loadtxt(os.path.join(datadir, f), delimiter=",") for f in self.get_files()]

        '''
        self.data = []
        for f in self.get_files():
            d = np.loadtxt(os.path.join(datadir, f), delimiter=",")
            self.data.append(d / np.average(d))
        '''

        '''
        self.data = []
        for f in self.get_files():
            d = np.loadtxt(os.path.join(datadir, f), delimiter=",")
            self.data.append(d - np.amin(d))
        '''

        self.number_neurons = self.data[0].shape[1]

        #'''
        print('\n\n', self.animal+self.number, 'data', type(self.data), len(self.data))
        
        for s,d in zip(self.sessions, self.data):
            print('\n' + s)
            print(type(d), d.shape)
            #print(d)

        print('\n\nnumber of neurons', self.number_neurons)
        #'''


    def get_files(self):

        return [self.animal + self.number + '_' + s + '.csv' for s in self.sessions]


    def get_avg_neuron(self):
        '''
        Compute average ΔF/F of each individual neuron for each individual session
        '''
        
        self.avg_neuron = [np.average(d, 0) for d in self.data]

        animal = []
        number = []
        session = []
        timepoint = []
        neuron = []
        avg_neuron = []
        t = -1
        for s,a in zip(self.sessions, self.avg_neuron):
            t = t + 1
            for n in range(self.number_neurons):
                animal.append(self.animal)
                number.append(self.number)
                session.append(s)
                timepoint.append(t)
                neuron.append(n)
                avg_neuron.append(a[n])

        self.avg_neuron_df = build_data_frame([('animal',animal),
                                               ('number',number),
                                               ('session',session),
                                               ('timepoint',timepoint),
                                               ('neuron',neuron),
                                               ('avg_neuron',avg_neuron)])
        
        '''
        print('\n\navg_neuron_df', type(self.avg_neuron_df))
        print(self.avg_neuron_df)
        '''

        '''
        print('\n\navg_neuron', type(self.avg_neuron), len(self.avg_neuron))
        #print(self.avg_neuron)
        for s,a in zip(self.sessions, self.avg_neuron):
            print('\n' + s)
            print(type(a), a.shape)
            print('min', np.amin(a))
            print('max', np.amax(a))
            print('median', np.median(a))
            print('average', np.average(a))
        #'''


    def get_activated_neuron(self, snr):
        '''
        Extract activated neurons based on average firing rate of individual neurons

        snr: signal-to-noise ratio to identify activated neurons
        '''

        #print('\n\nExtract activated neurons based on average firing rate of individual neurons')

        self.activated_neuron = []
        self.non_activated_neuron = []
        neuron = set(list(range(self.number_neurons)))
        #print('neuron', type(neuron))
        #print(neuron)

        # Activated neurons during hc0
        threshold = snr * np.average(self.avg_neuron[0])
        index = np.where(self.avg_neuron[0] > threshold)
        self.activated_neuron_hc0 = index[0]
        self.non_activated_neuron_hc0 = np.array(list(neuron.difference(set(self.activated_neuron_hc0.flatten()))))

        '''
        print('\nactivated_neuron_hc0')
        print('activated_neuron_hc0', type(self.activated_neuron_hc0), self.activated_neuron_hc0.shape)
        print(self.activated_neuron_hc0)
        print('non_activated_neuron_hc0', type(self.non_activated_neuron_hc0), self.non_activated_neuron_hc0.shape)
        print(self.non_activated_neuron_hc0)
        '''
        
        # Activated neurons during cfc
        threshold = snr * np.average(self.avg_neuron[1])
        index = np.where(self.avg_neuron[1] > threshold)
        self.activated_neuron_cfc = index[0]
        self.non_activated_neuron_cfc = np.array(list(neuron.difference(set(self.activated_neuron_cfc.flatten()))))

        '''
        print('\nactivated_neuron_cfc')
        print('activated_neuron_cfc', type(self.activated_neuron_cfc), self.activated_neuron_cfc.shape)
        print(self.activated_neuron_cfc)
        print('non_activated_neuron_cfc', type(self.non_activated_neuron_cfc), self.non_activated_neuron_cfc.shape)
        print(self.non_activated_neuron_cfc)
        '''

        # Activated neurons during hc1
        threshold = snr * np.average(self.avg_neuron[2])
        index = np.where(self.avg_neuron[2] > threshold)
        self.activated_neuron_hc1 = index[0]
        self.non_activated_neuron_hc1 = np.array(list(neuron.difference(set(self.activated_neuron_hc1.flatten()))))

        '''
        print('\nactivated_neuron_hc1')
        print('activated_neuron_hc1', type(self.activated_neuron_hc1), self.activated_neuron_hc1.shape)
        print(self.activated_neuron_hc1)
        print('non_activated_neuron_hc1', type(self.non_activated_neuron_hc1), self.non_activated_neuron_hc1.shape)
        print(self.non_activated_neuron_hc1)
        '''

        # Activated neurons during neutral1
        threshold = snr * np.average(self.avg_neuron[3])
        index = np.where(self.avg_neuron[3] > threshold)
        self.activated_neuron_neutral1 = index[0]
        self.non_activated_neuron_neutral1 = np.array(list(neuron.difference(set(self.activated_neuron_neutral1.flatten()))))

        '''
        print('\nactivated_neuron_neutral1')
        print('activated_neuron_neutral1', type(self.activated_neuron_neutral1), self.activated_neuron_neutral1.shape)
        print(self.activated_neuron_neutral1)
        print('non_activated_neuron_neutral1', type(self.non_activated_neuron_neutral1), self.non_activated_neuron_neutral1.shape)
        print(self.non_activated_neuron_neutral1)
        '''

        # Activated neurons during training1
        threshold = snr * np.average(self.avg_neuron[4])
        index = np.where(self.avg_neuron[4] > threshold)
        self.activated_neuron_training1 = index[0]
        self.non_activated_neuron_training1 = np.array(list(neuron.difference(set(self.activated_neuron_training1.flatten()))))

        '''
        print('\nactivated_neuron_training1')
        print('activated_neuron_training1', type(self.activated_neuron_training1), self.activated_neuron_training1.shape)
        print(self.activated_neuron_training1)
        print('non_activated_neuron_training1', type(self.non_activated_neuron_training1), self.non_activated_neuron_training1.shape)
        print(self.non_activated_neuron_training1)
        '''

        # Activated neurons during hc2
        threshold = snr * np.average(self.avg_neuron[5])
        index = np.where(self.avg_neuron[5] > threshold)
        self.activated_neuron_hc2 = index[0]
        self.non_activated_neuron_hc2 = np.array(list(neuron.difference(set(self.activated_neuron_hc2.flatten()))))

        '''
        print('\nactivated_neuron_hc2')
        print('activated_neuron_hc2', type(self.activated_neuron_hc2), self.activated_neuron_hc2.shape)
        print(self.activated_neuron_hc2)
        print('non_activated_neuron_hc2', type(self.non_activated_neuron_hc2), self.non_activated_neuron_hc2.shape)
        print(self.non_activated_neuron_hc2)
        '''

        # Activated neurons during neutral2
        threshold = snr * np.average(self.avg_neuron[6])
        index = np.where(self.avg_neuron[6] > threshold)
        self.activated_neuron_neutral2 = index[0]
        self.non_activated_neuron_neutral2 = np.array(list(neuron.difference(set(self.activated_neuron_neutral2.flatten()))))

        '''
        print('\nactivated_neuron_neutral2')
        print('activated_neuron_neutral2', type(self.activated_neuron_neutral2), self.activated_neuron_neutral2.shape)
        print(self.activated_neuron_neutral2)
        print('non_activated_neuron_neutral2', type(self.non_activated_neuron_neutral2), self.non_activated_neuron_neutral2.shape)
        print(self.non_activated_neuron_neutral2)
        '''

        # Activated neurons during training2
        threshold = snr * np.average(self.avg_neuron[7])
        index = np.where(self.avg_neuron[7] > threshold)
        self.activated_neuron_training2 = index[0]
        self.non_activated_neuron_training2 = np.array(list(neuron.difference(set(self.activated_neuron_training2.flatten()))))

        '''
        print('\nactivated_neuron_training2')
        print('activated_neuron_training2', type(self.activated_neuron_training2), self.activated_neuron_training2.shape)
        print(self.activated_neuron_training2)
        print('non_activated_neuron_training2', type(self.non_activated_neuron_training2), self.non_activated_neuron_training2.shape)
        print(self.non_activated_neuron_training2)
        '''

        self.activated_neuron_sequence = [self.activated_neuron_hc0,
                                          self.activated_neuron_cfc,
                                          self.activated_neuron_hc1,
                                          self.activated_neuron_neutral1,
                                          self.activated_neuron_training1,
                                          self.activated_neuron_hc2,
                                          self.activated_neuron_neutral2,
                                          self.activated_neuron_training2]

        self.non_activated_neuron_sequence = [self.non_activated_neuron_hc0,
                                              self.non_activated_neuron_cfc,
                                              self.non_activated_neuron_hc1,
                                              self.non_activated_neuron_neutral1,
                                              self.non_activated_neuron_training1,
                                              self.non_activated_neuron_hc2,
                                              self.non_activated_neuron_neutral2,
                                              self.non_activated_neuron_training2]



    def get_engram_cell_snr_neuron(self):
        '''
        Extract engram cells using individual neurons' snr based on average firing rate

        snr: signal-to-noise ratio to identify engram cells
        '''

        self.engram_cell_cfc = self.activated_neuron_cfc
        self.non_engram_cell_cfc = self.non_activated_neuron_cfc
        
        self.engram_cell_training1 = self.activated_neuron_training1
        self.non_engram_cell_training1 = self.non_activated_neuron_training1
        
        self.engram_cell_training2 = self.activated_neuron_training2
        self.non_engram_cell_training2 = self.non_activated_neuron_training2

        # TO-DO: include engram_type as in get_engram_cell_discrimination_hc_neuron()
        # Dynamic engram
        self.engram_cell_sequence = [self.activated_neuron_cfc,
                                     self.activated_neuron_cfc,
                                     self.activated_neuron_training1,
                                     self.activated_neuron_training1,
                                     self.activated_neuron_training1,
                                     self.activated_neuron_training2,
                                     self.activated_neuron_training2,
                                     self.activated_neuron_training2]

        self.non_engram_cell_sequence = [self.non_activated_neuron_cfc,
                                         self.non_activated_neuron_cfc,
                                         self.non_activated_neuron_training1,
                                         self.non_activated_neuron_training1,
                                         self.non_activated_neuron_training1,
                                         self.non_activated_neuron_training2,
                                         self.non_activated_neuron_training2,
                                         self.non_activated_neuron_training2]


    def compute_discrimination(self, response1, response2):
        
        return (response1 - response2) / (response1 + response2)


    def get_discrimination_hc_neuron(self):

        self.cfc_discrimination_hc = self.compute_discrimination(self.avg_neuron[1],
                                                                 self.avg_neuron[0])

        self.neutral1_discrimination_hc = self.compute_discrimination(self.avg_neuron[3],
                                                                      self.avg_neuron[2])

        self.training1_discrimination_hc = self.compute_discrimination(self.avg_neuron[4],
                                                                       self.avg_neuron[2])

        self.neutral2_discrimination_hc = self.compute_discrimination(self.avg_neuron[6],
                                                                      self.avg_neuron[5])

        self.training2_discrimination_hc = self.compute_discrimination(self.avg_neuron[7],
                                                                       self.avg_neuron[5])

        self.discrimination_hc = [self.cfc_discrimination_hc,
                                  self.neutral1_discrimination_hc,
                                  self.training1_discrimination_hc,
                                  self.neutral2_discrimination_hc,
                                  self.training2_discrimination_hc]

        self.discrimination_hc_sessions = list(self.sessions[i] for i in [1,3,4,6,7])


    def get_discrimination_recall_neuron(self):

        self.discrimination_recall1_neuron = self.compute_discrimination(self.avg_neuron[4],
                                                                         self.avg_neuron[3])

        self.discrimination_recall2_neuron = self.compute_discrimination(self.avg_neuron[7],
                                                                         self.avg_neuron[6])


    def get_discrimination_recall_freezing(self):

        self.discrimination_recall1_freezing = self.compute_discrimination(self.freezing[2],
                                                                           self.freezing[1])

        self.discrimination_recall2_freezing = self.compute_discrimination(self.freezing[4],
                                                                           self.freezing[3])

        '''
        print('discrimination_recall1_freezing', type(self.discrimination_recall1_freezing))
        print(self.discrimination_recall1_freezing)

        print('discrimination_recall2_freezing', type(self.discrimination_recall2_freezing))
        print(self.discrimination_recall2_freezing)
        '''


    def get_discrimination_recall_engram(self):

        self.discrimination_recall1_engram = self.compute_discrimination(self.avg_engram[4],
                                                                         self.avg_engram[3])

        self.discrimination_recall2_engram = self.compute_discrimination(self.avg_engram[7],
                                                                         self.avg_engram[6])

        '''
        print('discrimination_recall1_engram', type(self.discrimination_recall1_engram))
        print(self.discrimination_recall1_engram)

        print('discrimination_recall2_engram', type(self.discrimination_recall2_engram))
        print(self.discrimination_recall2_engram)
        '''


    def get_engram_cell_discrimination_hc_neuron(self, threshold, engram_type):

        #self.engram_cell = []
        #self.non_engram_cell = []
        
        neuron = set(list(range(self.number_neurons)))
        
        '''
        print('neuron', type(neuron))
        print(neuron)
        '''

        # Engram cells during cfc
        index = np.where(self.cfc_discrimination_hc > threshold)
        self.engram_cell_cfc = index[0]
        self.non_engram_cell_cfc = np.array(list(neuron.difference(set(self.engram_cell_cfc.flatten()))))

        '''
        print('\nengram_cell_cfc')
        print('engram_cell_cfc',
              type(self.engram_cell_cfc),
              self.engram_cell_cfc.shape)
        print(self.engram_cell_cfc)
        print('non_engram_cell_cfc',
              type(self.non_engram_cell_cfc),
              self.non_engram_cell_cfc.shape)
        print(self.non_engram_cell_cfc)
        '''

        # Engram cells during training1
        index = np.where(self.training1_discrimination_hc > threshold)
        self.engram_cell_training1 = index[0]
        self.non_engram_cell_training1 = np.array(list(neuron.difference(set(self.engram_cell_training1.flatten()))))

        '''
        print('\nengram_cell_training1')
        print('engram_cell_training1',
              type(self.engram_cell_training1),
              self.engram_cell_training1.shape)
        print(self.engram_cell_training1)
        print('non_engram_cell_training1',
              type(self.non_engram_cell_training1),
              self.non_engram_cell_training1.shape)
        print(self.non_engram_cell_training1)
        '''

        # Engram cells during training2
        index = np.where(self.training2_discrimination_hc > threshold)
        self.engram_cell_training2 = index[0]
        self.non_engram_cell_training2 = np.array(list(neuron.difference(set(self.engram_cell_training2.flatten()))))

        '''
        print('\nengram_cell_training2')
        print('engram_cell_training2',
              type(self.engram_cell_training2),
              self.engram_cell_training2.shape)
        print(self.engram_cell_training2)
        print('non_engram_cell_training2',
              type(self.non_engram_cell_training2),
              self.non_engram_cell_training2.shape)
        print(self.non_engram_cell_training2)
        '''

        if engram_type == 'dynamic':
            
            # Dynamic Engram
            self.engram_cell_sequence = [self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_training1,
                                         self.engram_cell_training1,
                                         self.engram_cell_training1,
                                         self.engram_cell_training2,
                                         self.engram_cell_training2,
                                         self.engram_cell_training2]

            self.non_engram_cell_sequence = [self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training2,
                                             self.non_engram_cell_training2,
                                             self.non_engram_cell_training2]
        elif engram_type == 'stable':

            # Stable Engram
            self.engram_cell_sequence = [self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc]

            self.non_engram_cell_sequence = [self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc]

        elif engram_type == 'random':

            # Random Neurons
            random.seed(10)
        
            self.engram_cell_random = np.array(sorted(random.sample(list(range(self.number_neurons)), self.engram_cell_cfc.size)))

            self.non_engram_cell_random = np.array(list(neuron.difference(set(self.engram_cell_random.flatten()))))

            '''
            print('\nengram_cell_random')
            print('engram_cell_random',
                  type(self.engram_cell_random),
                  self.engram_cell_random.shape)
            print(self.engram_cell_random)
            print('non_engram_cell_random',
                  type(self.non_engram_cell_random),
                  self.non_engram_cell_random.shape)
            print(self.non_engram_cell_random)
            '''

            self.engram_cell_sequence = [self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random,
                                         self.engram_cell_random]

            self.non_engram_cell_sequence = [self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random,
                                             self.non_engram_cell_random]

        else:
            raise Exception('engram_type must be either dynamic, stable or random.')


    def get_random_neurons(self, engram):

        #print('number_neurons', self.number_neurons)
        #print('engram.size', engram.size)
        return np.array(sorted(random.sample(list(range(self.number_neurons)), engram.size)))

    
    def get_random_engram(self):

        #random.seed(10)

        #print('get_random_engram')

        #print('cfc')
        random_engram_cfc = self.get_random_neurons(self.engram_cell_cfc)

        #print('training1')
        random_engram_training1 = self.get_random_neurons(self.engram_cell_training1)
        #random_engram_training1 = self.get_random_neurons(self.engram_cell_cfc)

        #print('training2')
        random_engram_training2 = self.get_random_neurons(self.engram_cell_training2)
        #random_engram_training2 = self.get_random_neurons(self.engram_cell_cfc)

        '''
        self.random_engram_sequence = [self.random_engram_cfc,
                                       self.random_engram_cfc,
                                       self.random_engram_training1,
                                       self.random_engram_training1,
                                       self.random_engram_training1,
                                       self.random_engram_training2,
                                       self.random_engram_training2,
                                       self.random_engram_training2]
        '''

        return random_engram_cfc,random_engram_training1,random_engram_training2


    def get_session_engram_cell_nmf(self, data, threshold, threshold_type):
        nd = data - np.amin(data)

        #'''
        #model = NMF(n_components=1)
        model = NMF(n_components=1, max_iter=10000)
        W = model.fit_transform(nd)
        H = model.components_
        if threshold_type == 'multiplier':
            index = np.where(H.flatten() > threshold * np.average(H))
        elif threshold_type == 'quantile':
            index = np.where(H.flatten() > np.quantile(H, threshold))
        elif threshold_type == 'absolute':
            index = np.where(H.flatten() > threshold_absolute)
        else:
            raise Exception('threshold_type must be either multiplier, quantile or absolute.')
        neuron = set(list(range(self.number_neurons)))
        engram_cell = index[0]
        non_engram_cell = np.array(list(neuron.difference(set(engram_cell.flatten()))))
        #'''

        '''
        model = NMF(n_components=2)
        W = model.fit_transform(nd)
        H = model.components_
        cluster0 = []
        cluster1 = []
        #engram_cell = []
        #non_engram_cell = []
        for i in range(self.number_neurons):
            if H[0,i] > H[1,i]:
                cluster0.append(i)
            else:
                cluster1.append(i)
        if np.average(data[:,cluster0]) > np.average(data[:,cluster1]):
            engram_cell = np.array(cluster0)
            non_engram_cell = np.array(cluster1)
        else:
            engram_cell = np.array(cluster1)
            non_engram_cell = np.array(cluster0)
        #'''
            
        #'''
        print('W', type(W), W.shape)
        print('H', type(H), H.shape)
        print('engram_cell', type(engram_cell), engram_cell.shape)
        print(engram_cell)
        print('non_engram_cell', type(non_engram_cell), non_engram_cell.shape)
        print(non_engram_cell)
        #'''

        return engram_cell,non_engram_cell,H
        
    def get_engram_cell_nmf(self, threshold, threshold_type, engram_type):
        
        #self.engram_cell = []
        #self.non_engram_cell = []
        #self.H = []
        
        print('Engram cell identification using NMF')

        # NMF input: Training context
        '''
        print('\ncfc')
        self.engram_cell_cfc,self.non_engram_cell_cfc,self.H_cfc = self.get_session_engram_cell_nmf(self.data[1], threshold, threshold_type)
        
        print('\ntraining1')
        self.engram_cell_training1,self.non_engram_cell_training1,self.H_training1 = self.get_session_engram_cell_nmf(self.data[4], threshold, threshold_type)
        
        print('\ntraining2')
        self.engram_cell_training2,self.non_engram_cell_training2,self.H_training2 = self.get_session_engram_cell_nmf(self.data[7], threshold, threshold_type)
        #'''
        
        # NMF input: Concatenate home cage and training context
        '''
        print('\ncfc')
        self.engram_cell_cfc,self.non_engram_cell_cfc,self.H_cfc = self.get_session_engram_cell_nmf(np.concatenate((self.data[0],self.data[1])), threshold, threshold_type)

        print('\ntraining1')
        self.engram_cell_training1,self.non_engram_cell_training1,self.H_training1 = self.get_session_engram_cell_nmf(np.concatenate((self.data[2],self.data[4])), threshold, threshold_type)

        print('\ntraining2')
        self.engram_cell_training2,self.non_engram_cell_training2,self.H_training2 = self.get_session_engram_cell_nmf(np.concatenate((self.data[5],self.data[7])), threshold, threshold_type)
        #'''

        # NMF input: Normalized neuronal activation
        #'''
        print('\ncfc')
        #data = (self.data[1] - self.avg_neuron[0])
        #data = (self.data[1] - self.avg_neuron[0]) / (self.avg_neuron[0])
        #data = (self.data[1]) / (self.avg_neuron[0])
        data = (self.data[1] - self.avg_neuron[0]) / (self.avg_neuron[1] + self.avg_neuron[0])
        self.engram_cell_cfc,self.non_engram_cell_cfc,self.H_cfc = self.get_session_engram_cell_nmf(data, threshold, threshold_type)
        
        print('\ntraining1')
        #data = (self.data[4] - self.avg_neuron[2])
        #data = (self.data[4] - self.avg_neuron[2]) / (self.avg_neuron[2])
        #data = (self.data[4]) / (self.avg_neuron[2])
        data = (self.data[4] - self.avg_neuron[2]) / (self.avg_neuron[4] + self.avg_neuron[2])
        self.engram_cell_training1,self.non_engram_cell_training1,self.H_training1 = self.get_session_engram_cell_nmf(data, threshold, threshold_type)
        
        print('\ntraining2')
        #data = (self.data[7] - self.avg_neuron[5])
        #data = (self.data[7] - self.avg_neuron[5]) / (self.avg_neuron[5])
        #data = (self.data[7]) / (self.avg_neuron[5])
        data = (self.data[7] - self.avg_neuron[5]) / (self.avg_neuron[7] + self.avg_neuron[5])
        self.engram_cell_training2,self.non_engram_cell_training2,self.H_training2 = self.get_session_engram_cell_nmf(data, threshold, threshold_type)
        #'''

        if engram_type == 'dynamic':
            self.engram_cell_sequence = [self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_training1,
                                         self.engram_cell_training1,
                                         self.engram_cell_training1,
                                         self.engram_cell_training2,
                                         self.engram_cell_training2,
                                         self.engram_cell_training2]

            self.non_engram_cell_sequence = [self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training1,
                                             self.non_engram_cell_training2,
                                             self.non_engram_cell_training2,
                                             self.non_engram_cell_training2]
            
            self.H = [self.H_cfc,
                      self.H_cfc,
                      self.H_training1,
                      self.H_training1,
                      self.H_training1,
                      self.H_training2,
                      self.H_training2,
                      self.H_training2]
            
        elif engram_type == 'stable':
            self.engram_cell_sequence = [self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc,
                                         self.engram_cell_cfc]

            self.non_engram_cell_sequence = [self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc,
                                             self.non_engram_cell_cfc]

            self.H = [self.H_cfc,
                      self.H_cfc,
                      self.H_cfc,
                      self.H_cfc,
                      self.H_cfc,
                      self.H_cfc,
                      self.H_cfc,
                      self.H_cfc]
    
    
    def compute_overlap(self, set1, set2, is_normalized=True):
        
        if is_normalized:
            return len(set(set1).intersection(set(set2))) / float(len(set(set1)))
        else:
            return len(set(set1).intersection(set(set2)))

    
    def get_activated_neuron_overlap(self):
        
        self.activated_neuron_overlap_training_h = []
        self.activated_neuron_overlap_training_h.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_hc1))
        self.activated_neuron_overlap_training_h.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_hc2))

        self.activated_neuron_overlap_training_n = []
        self.activated_neuron_overlap_training_n.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_neutral1))
        self.activated_neuron_overlap_training_n.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_neutral2))

        self.activated_neuron_overlap_training_t = []
        self.activated_neuron_overlap_training_t.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_training1))
        self.activated_neuron_overlap_training_t.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_training2))

        self.activated_neuron_overlap_recall_h = []
        self.activated_neuron_overlap_recall_h.append(self.compute_overlap(self.activated_neuron_hc1, self.activated_neuron_cfc))
        self.activated_neuron_overlap_recall_h.append(self.compute_overlap(self.activated_neuron_hc2, self.activated_neuron_cfc))

        self.activated_neuron_overlap_recall_n = []
        self.activated_neuron_overlap_recall_n.append(self.compute_overlap(self.activated_neuron_neutral1, self.activated_neuron_cfc))
        self.activated_neuron_overlap_recall_n.append(self.compute_overlap(self.activated_neuron_neutral2, self.activated_neuron_cfc))

        self.activated_neuron_overlap_recall_t = []
        self.activated_neuron_overlap_recall_t.append(self.compute_overlap(self.activated_neuron_training1, self.activated_neuron_cfc))
        self.activated_neuron_overlap_recall_t.append(self.compute_overlap(self.activated_neuron_training2, self.activated_neuron_cfc))

        self.activated_neuron_overlap_neuron_h = []
        self.activated_neuron_overlap_neuron_h.append(self.compute_overlap(self.activated_neuron_hc1, self.activated_neuron_cfc, False) / float(self.number_neurons))
        self.activated_neuron_overlap_neuron_h.append(self.compute_overlap(self.activated_neuron_hc2, self.activated_neuron_cfc, False) / float(self.number_neurons))

        self.activated_neuron_overlap_neuron_n = []
        self.activated_neuron_overlap_neuron_n.append(self.compute_overlap(self.activated_neuron_neutral1, self.activated_neuron_cfc, False) / float(self.number_neurons))
        self.activated_neuron_overlap_neuron_n.append(self.compute_overlap(self.activated_neuron_neutral2, self.activated_neuron_cfc, False) / float(self.number_neurons))

        self.activated_neuron_overlap_neuron_t = []
        self.activated_neuron_overlap_neuron_t.append(self.compute_overlap(self.activated_neuron_training1, self.activated_neuron_cfc, False) / float(self.number_neurons))
        self.activated_neuron_overlap_neuron_t.append(self.compute_overlap(self.activated_neuron_training2, self.activated_neuron_cfc, False) / float(self.number_neurons))

        self.activated_neuron_overlap_seq_back = []
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_hc0, self.activated_neuron_cfc))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_hc1))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_hc1, self.activated_neuron_neutral1))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_neutral1, self.activated_neuron_training1))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_training1, self.activated_neuron_hc2))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_hc2, self.activated_neuron_neutral2))
        self.activated_neuron_overlap_seq_back.append(self.compute_overlap(self.activated_neuron_neutral2, self.activated_neuron_training2))

        self.activated_neuron_overlap_seq_fwd = []
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_cfc, self.activated_neuron_hc0))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_hc1, self.activated_neuron_cfc))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_neutral1, self.activated_neuron_hc1))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_training1, self.activated_neuron_neutral1))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_hc2, self.activated_neuron_training1))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_neutral2, self.activated_neuron_hc2))
        self.activated_neuron_overlap_seq_fwd.append(self.compute_overlap(self.activated_neuron_training2, self.activated_neuron_neutral2))   

        '''
        print('\n\nactivated_neuron_overlap_training_h')
        print(self.activated_neuron_overlap_training_h)

        print('\n\nactivated_neuron_overlap_training_n')
        print(self.activated_neuron_overlap_training_n)

        print('\n\nactivated_neuron_overlap_training_t')
        print(self.activated_neuron_overlap_training_t)

        print('\nactivated_neuron_overlap_recall_h')
        print(self.activated_neuron_overlap_recall_h)

        print('\nactivated_neuron_overlap_recall_n')
        print(self.activated_neuron_overlap_recall_n)

        print('\nactivated_neuron_overlap_recall_t')
        print(self.activated_neuron_overlap_recall_t)

        print('\nactivated_neuron_overlap_neuron_h')
        print(self.activated_neuron_overlap_neuron_h)

        print('\nactivated_neuron_overlap_neuron_n')
        print(self.activated_neuron_overlap_neuron_n)

        print('\nactivated_neuron_overlap_neuron_t')
        print(self.activated_neuron_overlap_neuron_t)

        print('\nactivated_neuron_overlap_seq_back')
        print(self.activated_neuron_overlap_seq_back)
        
        print('\nactivated_neuron_overlap_seq_fwd')
        print(self.activated_neuron_overlap_seq_fwd)
        '''


    def get_engram_cell_overlap(self):
        
        self.engram_cell_overlap_training = []
        self.engram_cell_overlap_training.append(self.compute_overlap(self.engram_cell_cfc,
                                                                      self.engram_cell_training1))
        self.engram_cell_overlap_training.append(self.compute_overlap(self.engram_cell_cfc,
                                                                      self.engram_cell_training2))

        self.engram_cell_overlap_recall = []
        self.engram_cell_overlap_recall.append(self.compute_overlap(self.engram_cell_training1,
                                                                    self.engram_cell_cfc))
        self.engram_cell_overlap_recall.append(self.compute_overlap(self.engram_cell_training2,
                                                                    self.engram_cell_cfc))

        self.engram_cell_overlap_neuron = []
        self.engram_cell_overlap_neuron.append(self.compute_overlap(self.engram_cell_training1, self.engram_cell_cfc, False) / float(self.number_neurons))
        self.engram_cell_overlap_neuron.append(self.compute_overlap(self.engram_cell_training2, self.engram_cell_cfc, False) / float(self.number_neurons))

        self.engram_cell_overlap_seq_back = []
        self.engram_cell_overlap_seq_back.append(self.compute_overlap(self.engram_cell_cfc,
                                                                      self.engram_cell_training1))
        self.engram_cell_overlap_seq_back.append(self.compute_overlap(self.engram_cell_training1,
                                                                      self.engram_cell_training2))

        self.engram_cell_overlap_seq_fwd = []
        self.engram_cell_overlap_seq_fwd.append(self.compute_overlap(self.engram_cell_training1,
                                                                     self.engram_cell_cfc))
        self.engram_cell_overlap_seq_fwd.append(self.compute_overlap(self.engram_cell_training2,
                                                                     self.engram_cell_training1))

        self.engram_cell_overlap_seq_neuron = []
        self.engram_cell_overlap_seq_neuron.append(self.compute_overlap(self.engram_cell_cfc, self.engram_cell_training1, False) / float(self.number_neurons))
        self.engram_cell_overlap_seq_neuron.append(self.compute_overlap(self.engram_cell_training1, self.engram_cell_training2, False) / float(self.number_neurons))

        self.engram_cell_overlap = [self.engram_cell_overlap_training,
                                    self.engram_cell_overlap_recall,
                                    self.engram_cell_overlap_neuron,
                                    self.engram_cell_overlap_seq_back,
                                    self.engram_cell_overlap_seq_fwd,
                                    self.engram_cell_overlap_seq_neuron]
        
        self.engram_cell_overlap_type = ['training',
                                         'recall',
                                         'neuron',
                                         'seq_back',
                                         'seq_fwd',
                                         'seq_neuron']

        '''
        print('\n\nengram_cell_overlap_training')
        print(self.engram_cell_overlap_training)

        print('\nengram_cell_overlap_recall')
        print(self.engram_cell_overlap_recall)

        print('\nengram_cell_overlap_neuron')
        print(self.engram_cell_overlap_neuron)

        print('\nengram_cell_overlap_seq_back')
        print(self.engram_cell_overlap_seq_back)
        
        print('\nengram_cell_overlap_seq_fwd')
        print(self.engram_cell_overlap_seq_fwd)

        print('\nengram_cell_overlap_seq_neuron')
        print(self.engram_cell_overlap_seq_neuron)
        '''


    def get_random_engram_overlap(self):

        random_engram_cfc,random_engram_training1,random_engram_training2 = self.get_random_engram()
        
        random_engram_overlap_training = []
        random_engram_overlap_training.append(self.compute_overlap(random_engram_cfc, random_engram_training1))
        random_engram_overlap_training.append(self.compute_overlap(random_engram_cfc, random_engram_training2))
        
        random_engram_overlap_recall = []
        random_engram_overlap_recall.append(self.compute_overlap(random_engram_training1, random_engram_cfc))
        random_engram_overlap_recall.append(self.compute_overlap(random_engram_training2, random_engram_cfc))
        
        random_engram_overlap_neuron = []
        random_engram_overlap_neuron.append(self.compute_overlap(random_engram_training1, random_engram_cfc, False) / float(self.number_neurons))
        random_engram_overlap_neuron.append(self.compute_overlap(random_engram_training2, random_engram_cfc, False) / float(self.number_neurons))
        
        random_engram_overlap_seq_back = []
        random_engram_overlap_seq_back.append(self.compute_overlap(random_engram_cfc, random_engram_training1))
        random_engram_overlap_seq_back.append(self.compute_overlap(random_engram_training1, random_engram_training2))
        
        random_engram_overlap_seq_fwd = []
        random_engram_overlap_seq_fwd.append(self.compute_overlap(random_engram_training1, random_engram_cfc))
        random_engram_overlap_seq_fwd.append(self.compute_overlap(random_engram_training2, random_engram_training1))

        random_engram_overlap_seq_neuron = []
        random_engram_overlap_seq_neuron.append(self.compute_overlap(random_engram_training1, random_engram_cfc, False) / float(self.number_neurons))
        random_engram_overlap_seq_neuron.append(self.compute_overlap(random_engram_training2, random_engram_training1, False) / float(self.number_neurons))
        
        random_engram_overlap = [random_engram_overlap_training,
                                 random_engram_overlap_recall,
                                 random_engram_overlap_neuron,
                                 random_engram_overlap_seq_back,
                                 random_engram_overlap_seq_fwd,
                                 random_engram_overlap_seq_neuron]
        
        random_engram_overlap_type = ['training',
                                      'recall',
                                      'neuron',
                                      'seq_back',
                                      'seq_fwd',
                                      'seq_neuron']
        
        return random_engram_overlap_type,random_engram_overlap


    def get_avg_engram(self):
        '''
        Compute average ΔF/F of each engram cell ensemble for each individual session
        '''

        #print('\n\nCompute avg ΔF/F of each engram cell ensemble for each individual session')

        self.avg_engram = []

        for d,e in zip(self.data, self.engram_cell_sequence):
            self.avg_engram.append(np.average(d[:,e]))

        animal = []
        number = []
        session = []
        timepoint = []
        neuron = []
        avg_neuron = []
        t = -1

        for s,a,e in zip(self.sessions, self.avg_neuron, self.engram_cell_sequence):
            t = t + 1
            for n in e:
                animal.append(self.animal)
                number.append(self.number)
                session.append(s)
                timepoint.append(t)
                neuron.append(n)
                avg_neuron.append(a[n])

        self.avg_engram_df = build_data_frame([('animal',animal),
                                               ('number',number),
                                               ('session',session),
                                               ('timepoint',timepoint),
                                               ('neuron',neuron),
                                               ('avg_neuron',avg_neuron)])

        '''
        print('avg_engram', type(self.avg_engram), len(self.avg_engram))
        print(self.avg_engram)
        '''

        '''
        print('avg_engram_df')
        print(self.avg_engram_df)
        '''


    def get_avg_non_engram(self):
        '''
        Compute average ΔF/F of each non-engram cell ensemble for each individual session
        '''

        #print('\n\nCompute avg ΔF/F of each non-engram cell ensemble for each individual session')

        self.avg_non_engram = []

        for d,e in zip(self.data, self.non_engram_cell_sequence):
            self.avg_non_engram.append(np.average(d[:,e]))

        animal = []
        number = []
        session = []
        timepoint = []
        neuron = []
        avg_neuron = []
        t = -1

        for s,a,e in zip(self.sessions, self.avg_neuron, self.non_engram_cell_sequence):
            t = t + 1
            for n in e:
                animal.append(self.animal)
                number.append(self.number)
                session.append(s)
                timepoint.append(t)
                neuron.append(n)
                avg_neuron.append(a[n])

        self.avg_non_engram_df = build_data_frame([('animal',animal),
                                                   ('number',number),
                                                   ('session',session),
                                                   ('timepoint',timepoint),
                                                   ('neuron',neuron),
                                                   ('avg_neuron',avg_neuron)])

        '''
        print('avg_non_engram', type(self.avg_non_engram), len(self.avg_non_engram))
        print(self.avg_non_engram)
        '''

        '''
        print('avg_non_engram_df')
        print(self.avg_non_engram_df)
        '''


    def compare_discrimination_hc_neuron(self, datadir):
        # Statistical test and plot: CDFs of pairs of cfc_discrimination_hc distributions

        # Load engram cell ensembles
        engrams = [self.engram_cell_cfc, self.engram_cell_training1, self.engram_cell_training2]
        neurons = set(range(0,self.number_neurons))
        overlap = set(engrams[0])
        new_engram_cells = set([])
        for engram in engrams[1:]:
            overlap = overlap & set(engram)
            new_engram_cells = new_engram_cells | (set(engram) - set(engrams[0]))

        print('engrams', type(engrams), len(engrams))
        print('neurons', type(neurons), len(neurons))
        print('engrams[0]', type(engrams[0]), len(engrams[0]))
        print('overlap', type(overlap), len(overlap))
        print('new_engram_cells', type(new_engram_cells), len(new_engram_cells))

        compare_data = self.cfc_discrimination_hc
        #compare_data = self.avg_neuron[1]
        #compare_data = self.avg_neuron[1] - self.avg_neuron[0]
        print('compare_data', type(compare_data), compare_data.shape)

        if len(overlap) > 1:
            print('\nOverlap vs rest of neurons')
            disc1 = compare_data[sorted(list(overlap))]
            disc2 = compare_data[sorted(list(neurons - overlap))]
            test = stats.kstest(disc1, disc2, alternative='two-sided', mode='auto')
            #test = stats.cramervonmises_2samp(disc1, disc2, method='auto')
            #test = stats.anderson_ksamp([disc1, disc2])
            #test = statistic(disc1, disc2, 0)
            '''
            test = stats.permutation_test((disc1,disc2),
                                          statistic,
                                          vectorized=True,
                                          n_resamples=1000000,
                                          permutation_type='independent',
                                          alternative='two-sided')
            #'''
            print('disc1', type(disc1), disc1.shape, np.mean(disc1))
            print('disc2', type(disc2), disc2.shape, np.mean(disc2))
            print(test.statistic)
            print(test.pvalue)
            filename = 'cfc_discrimination_hc_' + self.animal + self.number + '_overlap_vs_rest-neurons.svg'
            plot_compare_cdfs(disc1,
                              disc2,
                              'Engram cells',
                              'Non-engram cells',
                              '#ed553b',
                              'black',
                              'Cell Discrimination',
                              datadir,
                              filename)
            
            print('\nOverlap vs rest of training')
            disc1 = compare_data[sorted(list(overlap))]
            disc2 = compare_data[sorted(list(set(engrams[0]) - overlap))]
            test = stats.kstest(disc1, disc2, alternative='two-sided', mode='auto')
            #test = stats.cramervonmises_2samp(disc1, disc2, method='auto')
            #test = stats.anderson_ksamp([disc1, disc2])
            #test = statistic(disc1, disc2, 0)
            '''
            test = stats.permutation_test((disc1,disc2),
                                          statistic,
                                          vectorized=True,
                                          n_resamples=1000000,
                                          permutation_type='independent',
                                          alternative='two-sided')
            #'''
            print('disc1', type(disc1), disc1.shape, np.mean(disc1))
            print('disc2', type(disc2), disc2.shape, np.mean(disc2))
            print(test.statistic)
            print(test.pvalue)
            filename = 'cfc_discrimination_hc_' + self.animal + self.number + '_overlap_vs_rest-training.svg'
            plot_compare_cdfs(disc1,
                              disc2,
                              'Engram cells',
                              'Non-engram cells',
                              '#ed553b',
                              'black',
                              'Cell Discrimination',
                              datadir,
                              filename)

        if len(new_engram_cells) > 1:
            print('\nNew engram cells vs rest of neurons')
            disc1 = compare_data[sorted(list(new_engram_cells))]
            disc2 = compare_data[sorted(list(neurons - new_engram_cells))]
            test = stats.kstest(disc1, disc2, alternative='two-sided', mode='auto')
            #test = stats.cramervonmises_2samp(disc1, disc2, method='auto')
            #test = stats.anderson_ksamp([disc1, disc2])
            #test = statistic(disc1, disc2, 0)
            '''
            test = stats.permutation_test((disc1,disc2),
                                          statistic,
                                          vectorized=True,
                                          n_resamples=1000000,
                                          permutation_type='independent',
                                          alternative='two-sided')
            #'''
            print('disc1', type(disc1), disc1.shape, np.mean(disc1))
            print('disc2', type(disc2), disc2.shape, np.mean(disc2))
            print(test.statistic)
            print(test.pvalue)
            filename = 'cfc_discrimination_hc_' + self.animal + self.number + '_new-engram-cells_vs_rest-neurons.svg'
            plot_compare_cdfs(disc1,
                              disc2,
                              'Engram cells',
                              'Non-engram cells',
                              '#ed553b',
                              'black',
                              'Cell Discrimination',
                              datadir,
                              filename)

            print('\nNew engram cells vs non-training')
            disc1 = compare_data[sorted(list(new_engram_cells))]
            disc2 = compare_data[sorted(list((neurons - set(engrams[0])) - new_engram_cells))]
            test = stats.kstest(disc1, disc2, alternative='two-sided', mode='auto')
            #test = stats.cramervonmises_2samp(disc1, disc2, method='auto')
            #test = stats.anderson_ksamp([disc1, disc2])
            #test = statistic(disc1, disc2, 0)
            '''
            test = stats.permutation_test((disc1,disc2),
                                          statistic,
                                          vectorized=True,
                                          n_resamples=1000000,
                                          permutation_type='independent',
                                          alternative='two-sided')
            #'''
            print('disc1', type(disc1), disc1.shape, np.mean(disc1))
            print('disc2', type(disc2), disc2.shape, np.mean(disc2))
            print(test.statistic)
            print(test.pvalue)
            filename = 'cfc_discrimination_hc_' + self.animal + self.number + '_new-engram-cells_vs_rest-non-training.svg'
            plot_compare_cdfs(disc1,
                              disc2,
                              'Engram cells',
                              'Non-engram cells',
                              '#ed553b',
                              'black',
                              'Cell Discrimination',
                              datadir,
                              filename)


class Experiment:

    def __init__(self,
                 animals,
                 engram_id,
                 engram_type,
                 discrimination_threshold,
                 snr,
                 nmf_threshold_type,
                 nmf_threshold_multiplier,
                 nmf_threshold_quantile,
                 nmf_threshold_absolute,
                 datadir):
        
        self.animals = animals
        self.engram_id = engram_id
        self.engram_type = engram_type
        self.discrimination_threshold = discrimination_threshold
        self.snr = snr
        self.nmf_threshold_type = nmf_threshold_type
        self.nmf_threshold_multiplier = nmf_threshold_multiplier
        self.nmf_threshold_quantile = nmf_threshold_quantile
        self.nmf_threshold_absolute = nmf_threshold_absolute
        self.datadir = datadir


    def analyze_animals(self, confidence_interval, palette_pop, palette_engram):

        for mouse in self.animals:

            print('\n\n' + mouse.animal + mouse.number)
    
            mouse.get_avg_neuron()
        
            mouse.get_discrimination_hc_neuron()
        
            mouse.get_discrimination_recall_neuron()
        
            mouse.get_discrimination_recall_freezing()

            if self.engram_id == 'discrimination':
                mouse.get_engram_cell_discrimination_hc_neuron(self.discrimination_threshold,
                                                               self.engram_type)
                if self.engram_type == 'dynamic':
                    if mouse.number == '1':
                        mouse.compare_discrimination_hc_neuron(self.datadir)
            
            elif self.engram_id == 'snr':
                mouse.get_activated_neuron(self.snr)
                mouse.get_engram_cell_snr_neuron()
                mouse.get_activated_neuron_overlap()
            
            elif self.engram_id == 'nmf':
                if self.nmf_threshold_type == 'multiplier':
                    threshold = self.nmf_threshold_multiplier
                elif self.nmf_threshold_type == 'quantile':
                    threshold = self.nmf_threshold_quantile
                elif self.nmf_threshold_type == 'absolute':
                    threshold = self.nmf_threshold_absolute
                mouse.get_engram_cell_nmf(threshold,
                                          self.nmf_threshold_type,
                                          self.engram_type)
            
            else:
                raise Exception('engram_id must be either discrimination, snr, or nmf')
        
            mouse.get_avg_engram()
        
            mouse.get_avg_non_engram()

            mouse.get_discrimination_recall_engram()
        
            mouse.get_engram_cell_overlap()

            '''
            plot_histogram(mouse.avg_neuron,
                           mouse.sessions,
                           mouse.animal,
                           mouse.number,
                           50,
                           False,
                           'Average \u0394F/F',
                           None,
                           self.datadir,
                           'hist_avg')
            #'''

            '''
            plot_histogram(mouse.discrimination_hc,
                           mouse.discrimination_hc_sessions,
                           mouse.animal,
                           mouse.number,
                           50,
                           False,
                           'Discrimination',
                           None,
                           self.datadir,
                           'hist_discrimination_hc')
            #'''

            '''
            if self.engram_id == 'nmf':
                plot_histogram(mouse.H,
                               mouse.sessions,
                               mouse.animal,
                               mouse.number,
                               50, #50
                               False,
                               'NMF Coefficient',
                               None,
                               self.datadir,
                               'hist_nmf_coeff')
            #'''

            '''
            plot_line(mouse.avg_neuron_df,
                      'timepoint',
                      'avg_neuron',
                      'neuron',
                      None,
                      confidence_interval,
                      self.datadir,
                      'mouse' + mouse.number + '_avg_neuron_ind.svg',
                      'Session',
                      '\u0394F/F',
                      None,
                      None,
                      None,
                      None,
                      None,
                      True)

            plot_line(mouse.avg_neuron_df,
                      'timepoint',
                      'avg_neuron',
                      'number',
                      None,
                      confidence_interval,
                      self.datadir,
                      'mouse' + mouse.number + '_avg_neuron_pop.svg',
                      'Session',
                      '\u0394F/F',
                      None,
                      None,
                      None,
                      None,
                      palette_pop,
                      True)

            plot_line(mouse.avg_engram_df,
                      'timepoint',
                      'avg_neuron',
                      'number',
                      None,
                      confidence_interval,
                      self.datadir,
                      'mouse' + mouse.number + '_avg_engram.svg',
                      'Session',
                      '\u0394F/F',
                      None,
                      None,
                      None,
                      None,
                      palette_engram,
                      True)
            #'''


    def get_avg_neuron(self):

        animal = []
        number = []
        session = []
        timepoint = []
        neuron = []
        avg_neuron = []
    
        for m in self.animals:
            t = -1
            for s,a in zip(m.sessions, m.avg_neuron):
                t = t + 1
                for n in range((m.number_neurons)):
                    animal.append(m.animal)
                    number.append(m.number)
                    session.append(s)
                    timepoint.append(t)
                    neuron.append(n)
                    avg_neuron.append(a[n])

        self.avg_neuron_df = build_data_frame([('animal',animal),
                                               ('number',number),
                                               ('session',session),
                                               ('timepoint',timepoint),
                                               ('neuron',neuron),
                                               ('avg_neuron',avg_neuron)])
        '''
        print('\n\navg_neuron_df')
        print(self.avg_neuron_df)
        '''


    def get_avg_engram(self):

        animal = []
        number = []
        session = []
        timepoint = []
        neuron = []
        avg_neuron = []
    
        for m in self.animals:
            t = -1
            for s,a,e in zip(m.sessions, m.avg_neuron, m.engram_cell_sequence):
                t = t + 1
                for n in e:
                    animal.append(m.animal)
                    number.append(m.number)
                    session.append(s)
                    timepoint.append(t)
                    neuron.append(n)
                    avg_neuron.append(a[n])

        self.avg_engram_df = build_data_frame([('animal',animal),
                                               ('number',number),
                                               ('session',session),
                                               ('timepoint',timepoint),
                                               ('neuron',neuron),
                                               ('avg_neuron',avg_neuron)])
        '''
        print('\n\navg_engram_df')
        print(self.avg_engram_df)
        '''


    def get_engram_cell_overlap(self):
        
        animal = []
        number = []
        overlap = []
        timepoint = []
        metric = []
        
        for m in self.animals:
            for tp,o in zip(m.engram_cell_overlap_type, m.engram_cell_overlap):
                t = -22
                for i in o:
                    t = t + 23
                    animal.append(m.animal)
                    number.append(m.number)
                    overlap.append(tp)
                    timepoint.append(t)
                    metric.append(i)

        self.engram_cell_overlap_df = build_data_frame([('animal',animal),
                                                        ('number',number),
                                                        ('overlap',overlap),
                                                        ('timepoint',timepoint),
                                                        ('metric',metric)])

        '''
        print('\n\nengram_cell_overlap_df')
        print(self.engram_cell_overlap_df)
        '''


    def get_random_engram_overlap(self):
        
        animal = []
        number = []
        overlap = []
        timepoint = []
        metric = []

        N = 10

        random.seed(10)
        
        for _ in range(N):
            for m in self.animals:
                random_engram_overlap_type,random_engram_overlap = m.get_random_engram_overlap()
                for tp,o in zip(random_engram_overlap_type, random_engram_overlap):
                    t = -22
                    for i in o:
                        t = t + 23
                        animal.append(m.animal)
                        number.append(m.number)
                        overlap.append(tp)
                        timepoint.append(t)
                        metric.append(i)

        self.random_engram_overlap_df = build_data_frame([('animal',animal),
                                                          ('number',number),
                                                          ('overlap',overlap),
                                                          ('timepoint',timepoint),
                                                          ('metric',metric)])
        
        '''
        print('\n\nrandom_engram_overlap_df')
        print(self.random_engram_overlap_df)
        '''


    def get_discrimination(self):
        
        animal = []
        number = []
        timepoint = []
        discrimination_engram = []
        discrimination_freezing = []
        
        for m in self.animals:
            animal.append(m.animal)
            number.append(m.number)
            timepoint.append('1')
            discrimination_engram.append(m.discrimination_recall1_engram)
            discrimination_freezing.append(m.discrimination_recall1_freezing)

            animal.append(m.animal)
            number.append(m.number)
            timepoint.append('24')
            discrimination_engram.append(m.discrimination_recall2_engram)
            discrimination_freezing.append(m.discrimination_recall2_freezing)

        self.discrimination_df = build_data_frame([('animal',animal),
                                                   ('number',number),
                                                   ('timepoint',timepoint),
                                                   ('discrimination_engram',discrimination_engram),
                                                   ('discrimination_freezing',discrimination_freezing)])

        '''
        print('\n\ndiscrimination_df')
        print(self.discrimination_df)
        #'''


    def get_freezing(self):
        
        animal = []
        number = []
        timepoint = []
        freezing = []
        
        for m in self.animals:
            t = -1
            for f in m.freezing:
                t = t + 1
                animal.append(m.animal)
                number.append(m.number)
                timepoint.append(t)
                freezing.append(f)

        self.freezing_df = build_data_frame([('animal',animal),
                                             ('number',number),
                                             ('timepoint',timepoint),
                                             ('freezing',freezing)])

        '''
        print('\n\nfreezing_df')
        print(self.freezing_df)
        '''


def build_data_frame(data):
    '''
    data: list of column,value pairs
    '''

    #self.columns = []
    #for col in columns:
    #    self.columns.append(col)
    #    setattr(self, col, [])
    
    d = {}
    for col,val in data:
        d[col] = val
    dframe = pd.DataFrame(d)
    '''
    print('data_frame')
    print(dframe)
    '''
    return dframe


def plot_histogram(data,
                   sessions,
                   animal,
                   number,
                   bins,
                   log,
                   xlabel,
                   ylabel,
                   datadir,
                   filename,
                   color=None):

    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
    #fig = plt.figure()
    #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
    #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
    #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

    for s,d in zip(sessions, data):
        #fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
        fig, ax = plt.subplots(dpi=300)
        
        if type(color) == type(None):
            hist = plt.hist(d, bins=bins, log=log)
        else:
            hist = plt.hist(d, bins=bins, log=log, color=color)

        #plt.title(title)
        
        if type(xlabel) != type(None):
            plt.xlabel(xlabel)
        if type(ylabel) != type(None):
            plt.ylabel(ylabel)
        
        #ax = plt.gca()
        #ax.axes.get_yaxis().set_visible(False)
        #plt.axis('off')

        #ax.set_xticks([0, 50, 100, 150, 200])
            
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(datadir, filename + '_' + animal + number + '_' + s + '.svg'),
                    format='svg',
                    dpi=300)
        print ("\n\nSaved plot of histogram")
        plt.close(fig)


def select_rows(df, *args):
    '''
    df: data frame

    args: list of (column, values) tuples. values for all columns must be provided.
    values is a tuple of the form (value1, value2, ...).
    if len(values) == 1, values is of the form (value,)
    '''

    '''
    print('select_rows function')
    print('df', type(df))
    print(df)
    print('args', type(args), len(args))
    print(args)
    '''

    select_rows = np.full((len(df)), True)

    '''
    print('select_rows0')
    print(select_rows)
    '''
    
    '''
    print('select_rows', type(select_rows), select_rows.shape)
    print(select_rows)
    print('args', type(args))  
    print(args)
    '''
    
    for col,values in args:
        '''print ('col,value', col, value)'''
        select = np.full((len(df)), False)
        for value in values:
            select = select | (df[col] == value)
        select_rows = select_rows & select

    '''
    print('select_rows1')
    print(select_rows)
    '''
    
    return df[select_rows]


def plot_line(data,
              x,
              y,
              hue,
              style,
              ci,
              datadir,
              filename,
              xlabel,
              ylabel,
              xmult,
              ymult,
              xlim,
              ylim,
              palette,
              has_axhline,
              xticklabels=None):

    if type(xmult) != type(None):
        data[x] = data[x] * xmult
    if type(ymult) != type(None):
        data[y] = data[y] * ymult

    '''
    print('data')
    print(data)
    '''
    
    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

    #fig = plt.figure(dpi=300)
    #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
    #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
    #fig = plt.figure(figsize=(1.568, 1.176), dpi=300)
    #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

    #fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
    fig, ax = plt.subplots(figsize=(2.352, 1.764), dpi=300)
    #fig, ax = plt.subplots(dpi=300)

    linewidth = 1.0
        
    if type(style) != type(None):
        sns.lineplot(x=x,
                     y=y,
                     hue=hue,
                     style=style,
                     ci=ci,
                     data=data,
                     palette=palette,
                     ax=ax)
    else:
        if type(palette) != type(None):
            sns.lineplot(x=x,
                         y=y,
                         hue=hue,
                         ci=ci,
                         data=data,
                         palette=palette,
                         ax=ax,
                         linewidth=linewidth)
        else:
            sns.lineplot(x=x,
                         y=y,
                         hue=hue,
                         ci=ci,
                         data=data,
                         ax=ax)

    if has_axhline:
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)

        
    ax.get_legend().remove()
  
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
    #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.set_xticks(list(set(data[x])))

    #ax.set_xticklabels(['HC','FT','HC','NC','TC','HC','NC','TC'])
    #plt.axvline(x=2, color='black', linestyle='--', linewidth=0.25)
    #plt.axvline(x=4, color='black', linestyle='--', linewidth=0.25)
    #plt.axvline(x=5, color='black', linestyle='--', linewidth=0.25)
    #plt.axvline(x=7, color='black', linestyle='--', linewidth=0.25)

    if type(xticklabels) != type(None):
        ax.set_xticklabels(xticklabels)
  
    #plt.title("")

    if type(xlim) != type(None):
        plt.xlim(xlim)
    if type(ylim) != type(None):
        plt.ylim(ylim)
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
    print('Saved line plot', filename)
    plt.close(fig)


def plot_swarm(data,
               x,
               y,
               hue,
               style,
               ci,
               datadir,
               filename,
               xlabel,
               ylabel,
               xmult,
               ymult,
               xlim,
               ylim,
               palette,
               has_axhline,
               order=None,
               marker=None):
        
    if type(xmult) != type(None):
        data[x] = data[x] * xmult
    if type(ymult) != type(None):
        data[y] = data[y] * ymult
        
    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

    #fig = plt.figure(dpi=300)
    #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
    #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
    #fig = plt.figure(figsize=(1.568, 1.176), dpi=300)
    #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

    #fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
    fig, ax = plt.subplots(figsize=(2.352, 1.764), dpi=300)
    #fig, ax = plt.subplots(dpi=300)

    #order = ['training']
    #order = ['recall-training','recall-novel']
    #order = ['recall-discrimination']
    #order = ['home-cage', 'recall-training','recall-novel']
    #order = ['control', 'cck-inhibition']
    #order = ['1', '2', '3', '4']
    size = 7.5 # 7.5 5.0 3.5 2.5

    if type(palette) != type(None):
        if type(marker) != type(None):
            sns.swarmplot(x=x,
                          y=y,
                          hue=hue,
                          data=data,
                          palette=palette,
                          size=size,
                          ax=ax,
                          marker=marker,
                          hue_order=order)
        else:
            sns.swarmplot(x=x,
                          y=y,
                          hue=hue,
                          data=data,
                          palette=palette,
                          size=size,
                          ax=ax)
        '''
        sns.swarmplot(x=x,
                      y=y,
                      hue=hue,
                      hue_order=order,
                      data=data,
                      palette=palette,
                      size=size,
                      ax=ax)
        '''
    else:
        sns.swarmplot(x=x, y=y, hue=hue, data=data, size=size, ax=ax)
        #sns.swarmplot(x=x, y=y, hue=hue, hue_order=order, data=data, size=size, ax=ax)

    if has_axhline:
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)
        
    ax.get_legend().remove()
  
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
    #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    #ax.set_xticks(list(set(data[x])))
    #ax.set_xticks([0.0, 0.20])

    ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
  
    #plt.title("")

    if type(xlim) != type(None):
        plt.xlim(xlim)
    if type(ylim) != type(None):
        plt.ylim(ylim)
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
    print('Saved swarm plot', filename)
    plt.close(fig)

def plot_error_bar(data,
                   x,
                   y,
                   hue,
                   order,
                   ci,
                   datadir,
                   filename,
                   xlabel,
                   ylabel,
                   xmult,
                   ymult,
                   xlim,
                   ylim,
                   palette,
                   errwidth,
                   scale,
                   capsize):
    
    if type(xmult) != type(None):
        data[x] = data[x] * xmult
    if type(ymult) != type(None):
        data[y] = data[y] * ymult
    
    '''
    print("\nFiltered data frame for bar plot of %s:"%filename)
    print(data)
    #'''
    
    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
    
    #fig = plt.figure(dpi=300)
    #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
    #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
    #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
    #fig = plt.figure(figsize=(0.896, 0.84), dpi=300)
    
    #fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
    fig, ax = plt.subplots(figsize=(2.352, 1.764), dpi=300)
    #fig, ax = plt.subplots(dpi=300)
    
    #ax = None
    
    sns.pointplot(x=x,
                  y=y,
                  hue=hue,
                  order=order,
                  errorbar=('ci', ci),
                  errwidth=errwidth,
                  data=data,
                  join=False,
                  estimator=np.mean,
                  palette=palette,
                  scale=scale,
                  capsize=capsize,
                  markers='_',
                  ax=ax)
    
    #plt.axhline(y=10, color='black', linestyle='--', linewidth=0.25)
    
    if type(ax.get_legend()) != type(None):
        ax.get_legend().remove()
    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
    #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)
    
    if type(xlabel) != type(None):
        plt.xlabel(xlabel)
    if type(ylabel) != type(None):
        plt.ylabel(ylabel)
    
    #ax.set_xticks([])
    #ax.get_xaxis().set_visible(False)
    
    #ax.set_yticks([0, 20, 40, 60, 80])
    
    #plt.title("")
    
    if type(xlim) != type(None):
        plt.xlim(xlim)
    if type(ylim) != type(None):
        plt.ylim(ylim)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
    print('Saved error bar plot', filename)
    plt.close(fig)


def shapiro(data):
    '''
    Perform the Shapiro-Wilk test for normality.
    null hypothesis: data was drawn from a normal distribution.
    '''
        
    return stats.shapiro(data)


def normaltest(data):
    '''
    Test whether a sample differs from a normal distribution.
    null hypothesis: data was drawn from a normal distribution.
    '''

    if len(data) >= 8:
        return stats.normaltest(data)
    else:
        print("normaltest only valid with at least 8 samples. %d were given."%(len(data)))
        return None

    
def levene_2(data1, data2):
    '''
    Perform Levene test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.levene(data1, data2)
    

def levene_3(data1, data2, data3):
    '''
    Perform Levene test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.levene(data1, data2, data3)


def levene_4(data1, data2, data3, data4):
    '''
    Perform Levene test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.levene(data1, data2, data3, data4)


def bartlett_2(data1, data2):
    '''
    Perform Bartlett test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.bartlett(data1, data2)
    

def bartlett_3(data1, data2, data3):
    '''
    Perform Bartlett test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.bartlett(data1, data2, data3)


def bartlett_4(data1, data2, data3, data4):
    '''
    Perform Bartlett test for equal variances.
    null hypothesis: all input samples are from populations with equal variances.
    '''
        
    return stats.bartlett(data1, data2, data3, data4)


def unpaired_t(data1, data2, equal_var=True, alternative='two-sided'):
    '''
    Perform t-test for the means of two independent samples.
    null hypothesis: two independent samples have identical average (expected) values.
    '''

    return stats.ttest_ind(data1, data2, equal_var=equal_var)
    #return stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)


def mannwhitneyu(data1, data2, alternative='two-sided'):
    '''
    Perform the Mann-Whitney U rank test on two independent samples.
    null hypothesis: two independent samples have the same underlying distribution.
    '''

    return stats.mannwhitneyu(data1, data2, alternative=alternative)


def paired_t(data1, data2, alternative='two-sided'):
    '''
    Perform t-test for the means of two related samples.
    null hypothesis: two related samples have identical average (expected) values.
    '''

    return stats.ttest_rel(data1, data2)
    #return stats.ttest_rel(data1, data2, alternative=alternative)


def wilcoxon(data1, data2, alternative='two-sided'):
    '''
    Perform Wilcoxon signed-rank test for the means of two related samples.
    null hypothesis: two related samples have the same underlying distribution.
    '''
        
    return stats.wilcoxon(data1, y=data2, alternative=alternative)


def friedman(data1, data2, data3, data4):
    '''
    Perform Friedman test for repeated measurements.
    null hypothesis: multiple samples have the same underlying distribution.
    '''
        
    return stats.friedmanchisquare(data1, data2, data3, data4)


def one_samp_t(data1, popmean, alternative='two-sided'):
    '''
    Perform t-test for the mean of ONE group of scores.
    null hypothesis: mean of a sample of independent observations is equal to the given 
                     population mean, popmean.
    '''

    return stats.ttest_1samp(data1, popmean)
    #return stats.ttest_1samp(data1, popmean, alternative=alternative)



def one_samp_wilcoxon(data1, alternative='two-sided'):
    '''
    Perform Wilcoxon signed-rank test for the difference of two related samples.
    null hypothesis: difference of two related samples is symmetric about zero.
    '''
        
    return stats.wilcoxon(data1, alternative=alternative)


def one_way_anova(data1, data2, data3):
    '''
    Perform one-way ANOVA.
    null hypothesis: group means are equal.
    '''
        
    return stats.f_oneway(data1, data2, data3)


def build_ols_model(data1, data2, data3, data1_name, data2_name, data3_name):
    d1 = data1
    d2 = data2
    d3 = data3
        
    #print('d1', type(d1), d1.shape)
    #print('d2', type(d2), d2.shape)
    #print('d3', type(d3), d3.shape)

    data = {}
    data[data1_name] = d1
    data[data2_name] = d2
    data[data3_name] = d3
    df = pd.DataFrame(data)

    #print('\n\ndf')
    #print(df)

    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[data1_name, data2_name, data3_name])
    df_melt.columns = ['index', 'treatments', 'value']
    #print('\n\ndf_melt')
    #print(df_melt)

    model = ols('value ~ C(treatments)', data=df_melt).fit()

    return df_melt,model


def one_way_anova_lm(model):
    '''
    Perform one-way ANOVA for a fitted linear model.
    null hypothesis: group means are equal.
    '''

    return sm.stats.anova_lm(model, typ=2)


def tukey_hsd(df_melt):
    '''
    Perform Tukey's HSD post hoc test.
    For unequal sample sizes, perform Tukey-Kramer test.
    '''

    res_tukey = stat()
    res_tukey.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)', ss_typ=2)
    return res_tukey.tukey_summary


def residual_shapiro(model):
    '''
    Check normality of residuals in one-way ANOVA
    '''

    return stats.shapiro(model.resid)


def test(d1, d2, normality=True, variance=True, pairwise=True):

    if np.array(d1).shape == np.array(d2).shape:
        diff = np.array(d2) - np.array(d1)
    else:
        diff = None
    
    print('d1')
    print(d1)
    print('average d1', np.average(d1))
    print('d2')
    print(d2)
    print('average d2', np.average(d2))
    if type(diff) != type(None):
        print('diff')
        print(np.transpose(diff))

    if  normality:
        # Shapiro-Wilk test for normality
        shapiro1 = shapiro(d1)
        shapiro2 = shapiro(d2)
        if type(diff) != type(None):
            shapirodiff = shapiro(diff)
        print('\nd1', 'shapiro')
        print(shapiro1)
        print('\nd2', 'shapiro')
        print(shapiro2)
        if type(diff) != type(None):
            print('\ndiff', 'shapiro')
            print(shapirodiff)

        # D’Agostino and Pearson’s test for normality
        normaltest1 = normaltest(d1)
        normaltest2 = normaltest(d2)
        if type(diff) != type(None):
            normaltestdiff = normaltest(diff)
        print('\nd1', "d'agostino and pearson")
        print(normaltest1)
        print('\nd2', "d'agostino and pearson")
        print(normaltest2)
        if type(diff) != type(None):
            print('\ndiff', "d'agostino and pearson")
            print(normaltestdiff)

    if variance:
        # Levene's test and Bartlett's test for equality of variances
        levene = levene_2(d1, d2)
        bartlett = bartlett_2(d1, d2)
        print('\nlevene')
        print(levene)
        print('\nbartlett')
        print(bartlett)

    if pairwise:
        if type(diff) != type(None):
            # Statistical tests of related samples
            # Paired t-test for equality of means: related samples with the same variance
            paired_t_res = paired_t(d1, d2)
            #paired_t_res = paired_t(d1, d2, alternative='two-sided')
            print('\npaired t-test')
            print(paired_t_res)

            # Wilcoxon signed-rank test: paired samples whose underlying distributions are not
            #                            normal
            wilcoxon_res = wilcoxon(d1, d2, alternative='two-sided')
            print('\nwilcoxon t-test')
            print(wilcoxon_res)

            # Friedman test: paired samples whose underlying distributions are not normal
            #friedman = friedman(data1, data2, data3, data4)
            #print('\nfriedman t-test')
            #print(friedman)

        # Statistical tests of independent samples
        # Unpaired t-test for equality of means: independent samples with the same variance
        unpaired_t_res = unpaired_t(d1, d2, equal_var=True)
        #unpaired_t_res = unpaired_t(d1, d2, equal_var=True, alternative='two-sided')
        print('\nunpaired t-test')
        print(unpaired_t_res)

        # Welch's unpaired t-test: independent samples with different variances
        welch_unpaired_t_res = unpaired_t(d1, d2, equal_var=False)
        #welch_unpaired_t_res = unpaired_t(d1, d2, equal_var=False, alternative='two-sided')
        print('\nwelch_unpaired t-test')
        print(welch_unpaired_t_res)

        # Mann-Whitney U test: independent samples whose underlying distributions are not normal
        mannwhitneyu_res = mannwhitneyu(d1, d2, alternative='two-sided')
        print('\nmannwhitneyu')
        print(mannwhitneyu_res)

        # One-sample t-test for the mean of d1 vs. the mean of d2
        one_samp_t_res = one_samp_t(d1, np.average(d2))
        print('\none-sample t-test')
        print(one_samp_t_res)



def main(argv):

    ##############################################################################################
    # Enter initialization variables

    # directory where imaging data is stored and where plots will be saved
    datadir = "~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/data"
    datadir = os.path.expanduser(datadir)

    # engram identification method: discrimination / snr / nmf
    engram_id = 'discrimination'
    #engram_id = 'nmf'
    #engram_id = 'snr'

    # engram type: dynamic / stable / random
    engram_type = 'dynamic'
    #engram_type = 'stable'
    #engram_type = 'random'
    
    # discrimination threshold for idenfitying engram cells
    discrimination_threshold = 0.2

    # NMF coefficient threshold type for identifying engram cells
    #nmf_threshold_type ='multiplier'
    nmf_threshold_type ='quantile'
    #nmf_threshold_type = 'absolute'

    # NMF coefficient threshold (average multiplier) for identiyfing engram cells
    nmf_threshold_multiplier = 1.1

    # NMF coefficient threshold (quantile) for identiyfing engram cells
    nmf_threshold_quantile = 0.99

    # NMF coefficient threshold (absolute value) for identiyfing engram cells
    nmf_threshold_absolute = 2.5

    # signal-to-noise ratio for identifying engram cells
    snr = 1.5

    ##############################################################################################
    # Enter experimental data

    animal = 'mouse'

    sessions = ['hc0', 'cfc', 'hc1', 'neutral1', 'training1', 'hc2', 'neutral2', 'training2']

    # mouse1 freezing for cfc, neutral1, training1, neutral2, and training2
    mouse1_freezing = [117.55, 194.19, 171.79, 51.38, 160.59]

    # mouse2 freezing for cfc, neutral1, training1, neutral2, and training2
    mouse2_freezing = [104.11, 104.51, 175.37, 46.99, 88.70]

    # mouse3 freezing for cfc, neutral1, training1, neutral2, and training2
    mouse3_freezing = [92.12, 177.57, 159.44, 45.50, 129.28]

    # mouse4 freezing for cfc, neutral1, training1, neutral2, and training2
    mouse4_freezing = [83.68, 142.70, 166.92, 65.22, 123.75]

    ##############################################################################################
    # Enter plotting variables
    
    confidence_interval = 95

    '''
    palette_freezing_training = sns.color_palette(['sandybrown'])
    palette_freezing_training_novel = sns.color_palette(['sandybrown', 'darkseagreen'])
    palette_discrimination_control = sns.color_palette(['darkseagreen'])
    palette_discrimination_control_manipulation = sns.color_palette(['darkseagreen', 'olive'])
    palette_labelling = sns.color_palette(['darkgray', 'sandybrown', 'darkseagreen'])

    # simulation all stimuli: square, circle, pentagon, hexagon
    palette_sim_all_stim = sns.color_palette(['#ed553b', '#20639b', '#f6d55c', '#956cb4'])

    # simulation training stimulus: square
    palette_sim_train_stim = sns.color_palette(['#ed553b'])
        
    # simulation novel stimuli: circle, pentagon, hexagon
    palette_sim_novel_stim = sns.color_palette(['#20639b', '#f6d55c', '#956cb4'])
    '''
    
    # simlulation engram vs. non-engram cells
    palette_dual = sns.color_palette(['#ed553b', 'black'])

    # engram cells
    #palette_engram = sns.color_palette(['#ed553b'])
    #palette_engram = sns.color_palette(['#e78ac3'])
    #palette_engram = sns.color_palette(['#d55e00'])
    #palette_engram = sns.color_palette(['#ca9161'])
    palette_engram = sns.color_palette(['#fbafe4'])

    # non-engram cells
    palette_non_engram = sns.color_palette(['black'])

    # neuronal population
    #palette_pop = sns.color_palette(['blue'])
    palette_pop = sns.color_palette(['gray'])

    #palette_mouse = sns.color_palette(['#4878d0', '#ee854a', '#6acc64'])
    #palette_mouse = sns.color_palette(['#4878d0', '#ee854a', 'darkcyan'])
    palette_mouse = 'colorblind' # 'Pastel2' 'Dark2' 'Set2'

    palette_random = sns.color_palette(['gray'])
    
    ##############################################################################################
    # Load and analyze experimental data
    print('Loading data')

    mouse1 = Animal(animal, '1', sessions, mouse1_freezing, datadir)
    
    mouse2 = Animal(animal, '2', sessions, mouse2_freezing, datadir)
    
    mouse3 = Animal(animal, '3', sessions, mouse3_freezing, datadir)

    mouse4 = Animal(animal, '4', sessions, mouse4_freezing, datadir)

    print('\n\nAnalyzing data')

    exp = Experiment([mouse1, mouse2, mouse3, mouse4],
                     engram_id,
                     engram_type,
                     discrimination_threshold,
                     snr,
                     nmf_threshold_type,
                     nmf_threshold_multiplier,
                     nmf_threshold_quantile,
                     nmf_threshold_absolute,
                     datadir)

    exp.analyze_animals(confidence_interval, palette_pop, palette_engram)

    exp.get_avg_neuron()
    
    exp.get_avg_engram()

    exp.get_engram_cell_overlap()

    exp.get_random_engram_overlap()

    exp.get_discrimination()

    exp.get_freezing()

    ##############################################################################################
    # Plot and test experimental data
    print('\n\nPlotting and testing data')

    # Freezing for each individual mouse
    #'''
    plot_line(exp.freezing_df,
              'timepoint',
              'freezing',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_freezing.svg',
              None,
              'Freezing (s)',
              None,
              None,
              None,
              [0,200],
              palette_mouse,
              False,
              ['FT','NC','TC','NC','TC'])
    #'''

    
    # ΔF/F
    # ΔF/F of neural population across mice
    '''
    plot_line(exp.avg_neuron_df,
              'timepoint',
              'avg_neuron',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_avg_neuron.svg',
              'Session',
              '\u0394F/F',
              None,
              None,
              None,
              [0,200],
              palette_pop,
              False,
              ['HC','FT','HC','NC','TC','HC','NC','TC'])
    '''

    # ΔF/F of neural population for each individual mouse
    '''
    plot_line(exp.avg_neuron_df,
              'timepoint',
              'avg_neuron',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_avg_neuron.svg',
              None,
              '\u0394F/F',
              None,
              None,
              None,
              [0,200],
              palette_mouse,
              False,
              ['HC','FT','HC','NC','TC','HC','NC','TC'])
    '''

    # ΔF/F of engram cells across mice
    '''
    plot_line(exp.avg_engram_df,
              'timepoint',
              'avg_neuron',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_avg_engram.svg',
              'Session',
              '\u0394F/F',
              None,
              None,
              None,
              [0,200],
              palette_engram,
              False,
              ['HC','FT','HC','NC','TC','HC','NC','TC'])
    '''

    # ΔF/F of engram cells for each individual mouse
    #'''
    plot_line(exp.avg_engram_df,
              'timepoint',
              'avg_neuron',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_avg_engram.svg',
              None,
              '\u0394F/F',
              None,
              None,
              None,
              None,  #[0,200]
              palette_mouse,
              False,
              ['HC','FT','HC','NC','TC','HC','NC','TC'])
    #'''

    # Freezing Discrimination vs. Engram Discrimination

    # Circle marker
    #'''
    plot_swarm(exp.discrimination_df,
               'discrimination_engram',
               'discrimination_freezing',
               'number', #number animal
               None,
               confidence_interval,
               datadir,
               'mouse_discrimination_circle.svg',
               'Engram Discrimination',
               'Freezing Discrimination',
               None,
               None,
               None,
               [-0.1,0.6], #ylim
               palette_mouse, #palette_mouse palette_non_engram
               False,
               ['1', '2', '3', '4'],
               'o')
    #'''

    # Cross marker
    #'''
    plot_swarm(exp.discrimination_df,
               'discrimination_engram',
               'discrimination_freezing',
               'number', #number animal
               None,
               confidence_interval,
               datadir,
               'mouse_discrimination_cross.svg',
               'Engram Discrimination',
               'Freezing Discrimination',
               None,
               None,
               None,
               [-0.1,0.6], #ylim
               palette_mouse, #palette_mouse palette_non_engram
               False,
               ['1', '2', '3', '4'],
               'P')
    #'''

    #'''
    print('\n\nFreezing Discrimination vs. Engram Discrimination')
    print(exp.discrimination_df)
    
    x = exp.discrimination_df['discrimination_engram']
    y = exp.discrimination_df['discrimination_freezing']

    test(x,y,pairwise=False)
    
    #corr = pearsonr(x, y)
    #print('\n\npearsonr')
    #print(corr)

    corr = stats.spearmanr(x, y)
    print('\n\nspearmanr')
    print(corr)

    #corr = stats.kendalltau(x, y)
    #print('\n\nkendalltau')
    #print(corr)
    #'''

    # Engram overlap
    '''
    otype = 'training'
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (% of training)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,100]

    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    '''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))

    #print('\n\nrandom engram overlap (% of training)')
    #print('df_random')
    #print(df_random)

    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest training overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of training$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    '''
    

    # Engram overlap
    '''
    otype = 'recall'
    
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (% of recall)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,100]

    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    '''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))

    #print('\n\nrandom engram overlap (% of recall)')
    #print('df_random')
    #print(df_random)

    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest recall overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of recall$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    '''
    
    
    '''
    otype = 'neuron'
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (% of neurons)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,30]

    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of neurons)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of neurons)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    '''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))

    #print('\n\nrandom engram overlap (% of neurons)')
    #print('df_random')
    #print(df_random)

    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest neurons overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
              'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'recall$^+$ ∩ training$^+$\n(% of neurons$^+$)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    #'''
    otype = 'seq_back'
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (seq_back)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,100]

    plot_error_bar(df,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_engram,
                   1.0,
                   0.75,
                   0.15)
    
    '''
    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    #'''

    '''
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    #'''
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    #'''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))
    
    #print('\n\nrandom engram overlap (% of training)')
    #print('df_random')
    #print(df_random)
    
    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest seq_back overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)

    plot_error_bar(df_random,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_random_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_random,
                   1.0,
                   0.75,
                   0.15)

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    #'''

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    #'''
    otype = 'seq_fwd'
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (seq_back)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,100]

    plot_error_bar(df,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_engram,
                   1.0,
                   0.75,
                   0.15)
    
    '''
    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    #'''

    '''
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    #'''
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    #'''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))
    
    #print('\n\nrandom engram overlap (% of training)')
    #print('df_random')
    #print(df_random)
    
    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest seq_fwd overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)

    plot_error_bar(df_random,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_random_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_random,
                   1.0,
                   0.75,
                   0.15)

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    #'''

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''
    
    #'''
    otype = 'seq_neuron'
    df = select_rows(exp.engram_cell_overlap_df, ('overlap',(otype,)))

    #print('\n\nengram cell overlap (seq_back)')
    #print('df')
    #print(df)

    t1 = select_rows(df, ('timepoint',(1,)))['metric']
    t24 = select_rows(df, ('timepoint',(24,)))['metric']

    ymult = 100
    df['metric'] = df['metric'] * ymult
    ylim = [0,100]

    plot_error_bar(df,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_engram,
                   1.0,
                   0.75,
                   0.15)
    
    '''
    plot_line(df,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_engram,
              False)
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_engram,
               False,
               None,
               'P')
    #'''

    '''
    plot_line(df,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_engram_cell_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    #'''
    
    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_engram_cell_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''

    #'''
    df_random = select_rows(exp.random_engram_overlap_df, ('overlap',(otype,)))
    
    #print('\n\nrandom engram overlap (% of training)')
    #print('df_random')
    #print(df_random)
    
    t1_random = select_rows(df_random, ('timepoint',(1,)))['metric']
    t24_random = select_rows(df_random, ('timepoint',(24,)))['metric']
    
    df_random['metric'] = df_random['metric'] * ymult

    print('\n\n\ntest seq_neuron overlap: engram cells vs. random neurons')
    print('1h')
    test(t1, t1_random)
    print('24h')
    test(t24, t24_random)

    plot_error_bar(df_random,
                   'timepoint',
                   'metric',
                   'animal',
                   None,
                   confidence_interval,
                   datadir,
                   'mouse_all_random_engram_cell_overlap_bar_' + otype + '.svg',
                   'Delay time (h)',
                   'longitudinal engram overlap\n(%)',
                   None,
                   None,
                   None,
                   ylim,
                   palette_random,
                   1.0,
                   0.75,
                   0.15)

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'animal',
              None,
              confidence_interval,
              datadir,
              'mouse_all_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_random,
              False)

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'animal',
               None,
               confidence_interval,
               datadir,
               'mouse_all_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_random,
               False,
               None,
               'P')
    '''

    '''
    plot_line(df_random,
              'timepoint',
              'metric',
              'number',
              None,
              confidence_interval,
              datadir,
              'mouse_ind_random_engram_overlap_' + otype + '.svg',
              'Delay time (h)',
              'longitudinal engram overlap\n(%)',
              None,
              None,
              None,
              ylim,
              palette_mouse,
              False)
    
    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_circle.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'o')

    plot_swarm(df_random,
               'timepoint',
               'metric',
               'number',
               None,
               confidence_interval,
               datadir,
               'mouse_ind_random_engram_overlap_swarm_' + otype + '_cross.svg',
               'Delay time (h)',
               'longitudinal engram overlap\n(%)',
               None,
               None,
               None,
               ylim,
               palette_mouse,
               False,
               None,
               'P')
    #'''


if __name__ == "__main__":
 main(sys.argv[1:])
