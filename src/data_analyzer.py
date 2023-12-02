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
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
import pingouin as pg

from helper import load_pickled_object,load_cell_assemblies

matplotlib.use('Agg')

class DataTester:
    
    
    def __init__(self, datadir=None, filename=None, columns=None):
        '''
        Provide either datadir and filename or columns for initialization.
        
        datadir: directory to csv file with tabular data
        filename: name of csv file with tabular data
        columns: list of column names
        '''

        if type(datadir) != type(None) and type(filename) != type(None):
            self.load_data_frame(datadir, filename)
        
        elif type(columns) != type(None):
            self.columns = []
            for col in columns:
                self.columns.append(col)
                setattr(self, col, [])
                
        else:
            sys.exit("Error: Provide either datadir and filename or columns for initialization.")
            


    def get_discrimination(self, trials, region, neuron_type, learn_time, con_times, phase, ref_prefix, comparison_prefixes, pattern, metric, columns, datadir, filename):
        print("Computing discrimination index...")
        disc_tester = DataTester(columns=columns)
        for con_time in con_times:
            #print('con_time', type(con_time), con_time)
            for trial in trials:
                #print('trial', type(trial), trial)
                ref_score = self.get_data('score',
                                          ('random_trial', (trial,)),
                                          ('region', (region,)),
                                          ('neuron_type',(neuron_type,)),
                                          ('learn_time', (learn_time,)),
                                          ('consolidation_time', (con_time,)),
                                          ('phase', (phase,)),
                                          ('base_prefix', (ref_prefix,)),
                                          ('pattern',(pattern,)),
                                          ('metric',(metric,)))
                #print('ref_score', type(ref_score), ref_score)

                for comp_prefix in comparison_prefixes:
                    #print('comp_prefix', type(comp_prefix), comp_prefix)
                    comp_score = self.get_data('score',
                                               ('random_trial', (trial,)),
                                               ('region', (region,)),
                                               ('neuron_type',(neuron_type,)),
                                               ('learn_time', (learn_time,)),
                                               ('consolidation_time', (con_time,)),
                                               ('phase', (phase,)),
                                               ('base_prefix', (comp_prefix,)),
                                               ('pattern',(pattern,)),
                                               ('metric',(metric,)))
                    #print('comp_score', type(comp_score), comp_score)

                    disc = (ref_score - comp_score) / float(ref_score + comp_score)
                    disc_tester.add_data([('random_trial', trial),
                                          ('region', region),
                                          ('neuron_type',neuron_type),
                                          ('learn_time', learn_time),
                                          ('consolidation_time', con_time),
                                          ('phase', phase),
                                          ('base_prefix', ref_prefix + '-' +comp_prefix),
                                          ('pattern', pattern),
                                          ('metric', metric),
                                          ('score', disc[0])])
        disc_tester.build_data_frame()
        disc_tester.save_data_frame(datadir, filename)
        print ("Saved discrimination index file")


    def add_data(self, data):
        '''
        data: list of (column, value) tuples. values for all columns must be provided.
        '''

        col_check = 0
        for col, val in data:
            if col in self.columns:
                col_check += 1
            else:
                sys.exit("Error: attempted to add data with unknown column '" + col + "'")
            getattr(self, col).append(val)
        if col_check != len(self.columns):
            sys.exit("Error: attempted to add data with missing columns")


    def build_data_frame(self):
        data = {}
        for col in self.columns:
            data[col] = getattr(self, col)
        self.df = pd.DataFrame(data)

    
    def save_data_frame(self, datadir, filename):
        self.df.to_csv(os.path.join(datadir, filename))


    def load_data_frame(self, datadir, filename):
        #self.df = pd.read_csv(os.path.join(datadir, filename))
        self.df = pd.read_csv(os.path.join(datadir, filename), low_memory=False)


    def select_rows(self, *args):
        '''
        args: list of (column, values) tuples. values for all columns must be provided.
              values is a tuple of the form (value1, value2, ...).
              if len(values) == 1, values is of the form (value,)
        '''
        
        select_rows = np.full((len(self.df)), True)
        '''
        print('select_rows', type(select_rows), select_rows.shape)
        print(select_rows)
        print('args', type(args))  
        print(args)
        '''
        for col,values in args:
            '''print ('col,value', col, value)'''
            select = np.full((len(self.df)), False)
            for value in values:
                select = select | (self.df[col] == value)
            select_rows = select_rows & select
        return self.df[select_rows]


    def get_data(self, data_column, *args):
        '''
        Retrieve data in specified column and rows.

        data_column: column containing desired data
        args: as in select_rows
        '''

        return self.select_rows(*args)[data_column].to_numpy()

    
    def store_data(self, data_name, data_column, *args):
        '''
        Store data in specified column and rows as a new object property.
        
        data_name: name of data store
        data_column: column containing desired data
        args: as in select_rows
        '''

        setattr(self, data_name, self.get_data(data_column, *args))
        
        data = getattr(self, data_name)
        print(data_name, type(data), data.shape)
        print(data)


    def shapiro(self, data_name):
        '''
        Perform the Shapiro-Wilk test for normality.

        null hypothesis: data was drawn from a normal distribution.
        '''
        
        return stats.shapiro(getattr(self, data_name))


    def normaltest(self, data_name):
        '''
        Test whether a sample differs from a normal distribution.

        null hypothesis: data was drawn from a normal distribution.
        '''

        data = getattr(self, data_name)
        if len(data) >= 8:
            return stats.normaltest(data)
        else:
            print("normaltest only valid with at least 8 samples. %d were given."%(len(data)))
            return None

    
    def levene_2(self, data1_name, data2_name):
        '''
        Perform Levene test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.levene(getattr(self, data1_name), getattr(self, data2_name))
    

    def levene_3(self, data1_name, data2_name, data3_name):
        '''
        Perform Levene test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.levene(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name))


    def levene_4(self, data1_name, data2_name, data3_name, data4_name):
        '''
        Perform Levene test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.levene(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name), getattr(self, data4_name))


    def bartlett_2(self, data1_name, data2_name):
        '''
        Perform Bartlett test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.bartlett(getattr(self, data1_name), getattr(self, data2_name))
    

    def bartlett_3(self, data1_name, data2_name, data3_name):
        '''
        Perform Bartlett test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.bartlett(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name))


    def bartlett_4(self, data1_name, data2_name, data3_name, data4_name):
        '''
        Perform Bartlett test for equal variances.

        null hypothesis: all input samples are from populations with equal variances.
        '''
        
        return stats.bartlett(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name), getattr(self, data4_name))


    def unpaired_t(self, data1_name, data2_name, equal_var=True, alternative='two-sided'):
        '''
        Perform t-test for the means of two independent samples.

        null hypothesis: two independent samples have identical average (expected) values.
        '''

        return stats.ttest_ind(getattr(self, data1_name), getattr(self, data2_name), equal_var=equal_var)
        #return stats.ttest_ind(getattr(self, data1_name), getattr(self, data2_name), equal_var=equal_var, alternative=alternative)


    def one_samp_t(self, data1_name, popmean, alternative='two-sided'):
        '''
        Perform t-test for the mean of ONE group of scores.

        null hypothesis: mean of a sample of independent observations is equal to the given 
                         population mean, popmean.
        '''

        return stats.ttest_1samp(getattr(self, data1_name), popmean)
        #return stats.ttest_1samp(getattr(self, data1_name), popmean, alternative=alternative)
    

    def mannwhitneyu(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform the Mann-Whitney U rank test on two independent samples.

        null hypothesis: two independent samples x have the same underlying distribution.
        '''

        return stats.mannwhitneyu(getattr(self, data1_name), getattr(self, data2_name), alternative=alternative)



    def paired_t(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform t-test for the means of two related samples.

        null hypothesis: two related samples have identical average (expected) values.
        '''

        return stats.ttest_rel(getattr(self, data1_name), getattr(self, data2_name))
        #return stats.ttest_rel(getattr(self, data1_name), getattr(self, data2_name), alternative=alternative)


    
    def wilcoxon(self, data1_name, data2_name, alternative='two-sided'):
        '''
        Perform Wilcoxon signed-rank test for the means of two related samples.

        null hypothesis: two related samples have the same underlying distribution.
        '''
        
        return stats.wilcoxon(getattr(self, data1_name), y=getattr(self, data2_name), alternative=alternative)


    
    def friedman(self, data1_name, data2_name, data3_name, data4_name):
        '''
        Perform Friedman test for repeated measurements.

        null hypothesis: multiple samples have the same underlying distribution.
        '''
        
        return stats.friedmanchisquare(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name), getattr(self, data4_name))


    def one_samp_wilcoxon(self, data1_name, alternative='two-sided'):
        '''
        Perform Wilcoxon signed-rank test for the difference of two related samples.

        null hypothesis: difference of two related samples is symmetric about zero.
        '''
        
        return stats.wilcoxon(getattr(self, data1_name), alternative=alternative)


    def one_way_anova(self, data1_name, data2_name, data3_name):
        '''
        Perform one-way ANOVA.

        null hypothesis: group means are equal.
        '''
        
        return stats.f_oneway(getattr(self, data1_name), getattr(self, data2_name), getattr(self, data3_name))


    def build_ols_model(self, data1_name, data2_name, data3_name):
        d1 = getattr(self, data1_name)
        d2 = getattr(self, data2_name)
        d3 = getattr(self, data3_name)
        
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

        self.df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[data1_name, data2_name, data3_name])
        self.df_melt.columns = ['index', 'treatments', 'value']
        #print('\n\ndf_melt')
        #print(df_melt)

        self.model = ols('value ~ C(treatments)', data=self.df_melt).fit()


    def one_way_anova_lm(self):
        '''
        Perform one-way ANOVA for a fitted linear model.

        null hypothesis: group means are equal.
        '''
        
        return sm.stats.anova_lm(self.model, typ=2)


    def tukey_hsd(self):
        '''
        Perform Tukey's HSD post hoc test
        For unequal sample sizes, perform Tukey-Kramer test
        '''

        res_tukey = stat()
        res_tukey.tukey_hsd(df=self.df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)', ss_typ=2)
        return res_tukey.tukey_summary


    def residual_shapiro(self):
        '''
        Check normality of residuals in one-way ANOVA
        '''

        return stats.shapiro(self.model.resid)


    def plot_line(self,
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
                  *args):
        '''
        args: as in select_rows(*args)
        '''
        
        data = self.select_rows(*args)
        #data.sort_values(x, inplace=True)

        #'''
        if type(xmult) != type(None):
            data[x] = data[x] * xmult
        if type(ymult) != type(None):
            data[y] = data[y] * ymult
        #'''
        
        '''
        print("\nFiltered data frame for plot %s:"%filename)
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

        fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
        #fig, ax = plt.subplots(dpi=300)
        
        #order = ['recall-training','recall-novel']
        #size = 1.25
        
        #ax = None

        #ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, palette=palette, size=size)
        #sns.swarmplot(x=x, y=y, hue=hue, data=data, palette=palette, size=size, ax=ax)
        #sns.swarmplot(x=x, y=y, hue=hue, hue_order=order, data=data, palette=palette, size=size, ax=ax)
        
        if type(style) != type(None):
            #ax= sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data, palette=palette)
            sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data, palette=palette, ax=ax)
            #sns.lineplot(x=x, y=y, hue=hue, hue_order=order, style=style, ci=ci, data=data, palette=palette, ax=ax)
        else:
            #ax = sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data, palette=palette)
            sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data, palette=palette, ax=ax)
            #sns.lineplot(x=x, y=y, hue=hue, hue_order=order, ci=ci, data=data, palette=palette, ax=ax)

        #sns.swarmplot(x=x, y=y, hue=hue, data=data, palette=palette, size=size, ax=ax)
        #sns.swarmplot(x=x, y=y, hue=hue, hue_order=order, data=data, palette=palette, size=size, ax=ax)
        if has_axhline:
            plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)
        
        ax.get_legend().remove()
  
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
        #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        x_sort = sorted(list(set(data[x])))
        if x_sort[0] == 0 and x_sort[-1] == 24:
            ax.set_xticks([0, 24])
        else:
            ax.set_xticks(list(set(data[x])))
  
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
        
    
    def plot_swarm(self,
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
                   order,
                   *args):
        '''
        args: as in select_rows(*args)
        '''
        
        data = self.select_rows(*args)
        data.sort_values(x, inplace=True)

        #'''
        if type(xmult) != type(None):
            data[x] = data[x] * xmult
        if type(ymult) != type(None):
            data[y] = data[y] * ymult
        #'''
        
        '''
        print("\nFiltered data frame for plot %s:"%filename)
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

        fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
        #fig, ax = plt.subplots(dpi=300)

        #order = ['training']
        #order = ['recall-training','recall-novel']
        #order = ['recall-discrimination']
        #order = ['home-cage', 'recall-training','recall-novel']
        #order = ['control', 'cck-inhibition']
        size = 1.25 #1.25 3.0

        #sns.swarmplot(x=x, y=y, hue=hue, data=data, palette=palette, size=size, ax=ax)

        #'''
        sns.swarmplot(x=x,
                      y=y,
                      hue=hue,
                      hue_order=order,
                      data=data,
                      palette=palette,
                      size=size,
                      ax=ax)
        #'''

        '''
        sns.swarmplot(x=x,
                      y=y,
                      hue=hue,
                      hue_order=order,
                      data=data,
                      palette=palette,
                      size=size,
                      linewidth=0.5,
                      edgecolor='black',
                      ax=ax)
        #'''

        #plt.scatter(data[x], data[y], s=5, facecolors='darkseagreen', edgecolors='darkseagreen', linestyle='-')
        #plt.scatter(data[x], data[y], s=5, facecolors='none', edgecolors='darkseagreen', linestyle='-')

        if has_axhline:
            plt.axhline(y=0, color='black', linestyle='--', linewidth=0.25)
        
        ax.get_legend().remove()
  
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
        #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

        if type(xlabel) != type(None):
            plt.xlabel(xlabel)
        if type(ylabel) != type(None):
            plt.ylabel(ylabel)
        
        #ax.set_xticks(list(set(data[x])))
        #ax.set_xticks([])
        #ax.get_xaxis().set_visible(False)
  
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


    def plot_bar(self,
                 x,
                 y,
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
                 errcolor,
                 errwidth,
                 palette,
                 *args):
        
        '''
        args: as in select_rows(*args)
        '''

        data = self.select_rows(*args)
        #data.sort_values(x, inplace=True)

        if type(xmult) != type(None):
            data[x] = data[x] * xmult
        if type(ymult) != type(None):
            data[y] = data[y] * ymult
        
        '''
        print("\nFiltered data frame for bar plot of %s:"%filename)
        print(data)
        '''
        
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.family': 'sans-serif'})
        plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

        #fig = plt.figure(dpi=300)
        #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
        #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
        #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
        #fig = plt.figure(figsize=(0.896, 0.84), dpi=300)

        fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)

        size = 1.25 # 1.5
                
        #ax = None
        #ax = sns.barplot(x=x, y=y, order=order, ci=ci, data=data, estimator=np.mean, palette=palette, errcolor=errcolor, errwidth=errwidth)
        #ax = sns.barplot(x=x, y=y, order=order, ci=ci, data=data, estimator=np.mean, palette=palette, errcolor=errcolor)
        #ax = sns.barplot(x=x, y=y, order=order, ci=ci, data=data, estimator=np.mean, palette=palette, errwidth=errwidth)
        #ax = sns.barplot(x=x, y=y, order=order, ci=ci, data=data, estimator=np.mean, palette=palette)

        sns.barplot(x=x,
                    y=y,
                    order=order,
                    ci=ci,
                    data=data,
                    estimator=np.mean,
                    palette=palette,
                    ax=ax)
        #sns.barplot(x=x, y=y, order=order, ci=ci, data=data, estimator=np.mean, palette=palette, ax=ax, errwidth=errwidth)

        #===================UNCOMMENT THIS FOR INDIVIDUAL DATA POINTS========================
        #sns.swarmplot(x=x, y=y, hue=x, hue_order=order, data=data, palette=palette, size=size, ax=ax)
        #sns.scatterplot(x=x, y=y, hue=x, hue_order=order, data=data, palette=palette, size=x, sizes=[2.5, 2.5], ax=ax)

        #plt.axhline(y=10, color='black', linestyle='--', linewidth=0.25)

        #if type(ax.get_legend()) != type(None):
        #    ax.get_legend().remove()
        
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
        #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

        if type(xlabel) != type(None):
            plt.xlabel(xlabel)
        if type(ylabel) != type(None):
            plt.ylabel(ylabel)
        
        #ax.set_xticks([])
        ax.get_xaxis().set_visible(False)

        #ax.set_yticks([0, 20, 40, 60, 80])
  
        #plt.title("")

        if type(xlim) != type(None):
            plt.xlim(xlim)
        if type(ylim) != type(None):
            plt.ylim(ylim)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
        print('Saved bar plot', filename)
        plt.close(fig)

    def plot_error_bar(self,
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
                       capsize,
                       *args):
        
        '''
        args: as in select_rows(*args)
        '''

        data = self.select_rows(*args)
        #data.sort_values(x, inplace=True)

        if type(xmult) != type(None):
            data[x] = data[x] * xmult
        if type(ymult) != type(None):
            data[y] = data[y] * ymult
        
        '''
        print("\nFiltered data frame for bar plot of %s:"%filename)
        print(data)
        '''
        
        plt.rcParams.update({'font.size': 7})
        plt.rcParams.update({'font.family': 'sans-serif'})
        plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

        #fig = plt.figure(dpi=300)
        #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
        #fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
        #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
        #fig = plt.figure(figsize=(0.896, 0.84), dpi=300)

        fig, ax = plt.subplots(figsize=(1.568, 1.176), dpi=300)
                
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
    ax.vlines(10, 0, 1, colors='black', linestyles='--', linewidth=0.25)
 
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

 
def main(argv):
    # Parameters

    # Choose the data to be analyzed
    has_simulation_analysis = True
    has_experiment_analysis = False
    
    # For analysis of simulation output:
    datadir_sim = "~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001"
    datadir_sim = os.path.expanduser(datadir_sim)

    # For analysis of experimental data:
    datadir = "~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/data"
    datadir = os.path.expanduser(datadir)
    filename = 'experimental-data.csv'
    datatester = DataTester(datadir=datadir, filename=filename)

    # set confidence interval
    confidence_interval = 99

    # palettes for experimental data
    palette_freezing_training = sns.color_palette(['sandybrown'])
    palette_freezing_training_novel = sns.color_palette(['sandybrown', 'darkseagreen'])
    palette_discrimination_control = sns.color_palette(['darkseagreen'])
    palette_discrimination_control_manipulation = sns.color_palette(['darkseagreen', 'olive'])
    palette_cal_light_reactivation = sns.color_palette(['#27aae1ff', '#ec1c24ff'])
    palette_cfos_labeling = sns.color_palette(['#d4aa00ff', '#784421ff'])
    palette_eyfp_labeling = sns.color_palette(['#d4aa00ff'])
    palette_white = sns.color_palette(['#ffffffff', '#ffffffff'])
    palette_labelling = sns.color_palette(['darkgray', 'sandybrown', 'darkseagreen'])

    # palette for simulation with all the stimuli: square, circle, pentagon, hexagon
    palette_sim_all_stim = sns.color_palette(['#ed553b', '#20639b', '#f6d55c', '#956cb4'])

    # palette for simulation with only the training stimulus: square
    palette_sim_train_stim = sns.color_palette(['#ed553b'])
        
    # palette for simulation with only the novel stimuli: circle, pentagon, hexagon
    palette_sim_novel_stim = sns.color_palette(['#20639b', '#f6d55c', '#956cb4'])

    # palette for engram vs. non-engram cells
    palette_dual = sns.color_palette(['#ed553b', 'black'])

    #palette = sns.color_palette("muted", n_colors=4)
    #palette = sns.color_palette(['#4878d0', '#ee854a', '#6acc64', '#8c613c'])
    #palette = sns.color_palette(['sandybrown', 'darkseagreen', 'thistle', 'royalblue']) # p1
    #palette = sns.color_palette(['#20639b', '#ed553b', '#f6d55c']) # p2
    #palette = sns.color_palette(['magenta']) # training
        
    #palette = sns.color_palette(['#ed553b', '#20639b']) # training and novel recall
    #palette = sns.color_palette(['thistle']) # # discrimination
    #palette = sns.color_palette(['lightcoral', 'sandybrown', 'darkseagreen',]) # labelling

    #palette = sns.color_palette(['darkgray', 'black']) # engram vs. non-engram cells

    if has_simulation_analysis:
        # Compute discrimination index of simulation data
        #'''
        filename = 'metrics-all.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        columns = ['random_trial',
                   'region',
                   'neuron_type',
                   'learn_time',
                   'consolidation_time',
                   'phase',
                   'base_prefix',
                   'pattern',
                   'metric',
                   'score']
        
        disc_filename = 'discrimination.csv'
        
        trials = 10
        
        neuron_type = 'exc' # 'exc' / 'inh'
        
        datatester_sim.get_discrimination(list(range(trials)),
                                          'hpc',
                                          neuron_type,
                                          300,
                                          list(range(0,90000,3600)),
                                          'test',
                                          'rft-0',
                                          ['rft-1', 'rft-2', 'rft-3'],
                                          '0',
                                          'tpr',
                                          columns,
                                          datadir_sim,
                                          disc_filename)
        #'''
        
        
        # Plot simulation recall discrimination
        #'''
        filename = 'discrimination.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        neuron_type = 'exc' # 'exc' / 'inh'
        
        filename = neuron_type + '_neurons-tpr-phase_test-pat_ind-rfl-300-discrimination.svg'
        datatester_sim.plot_line('consolidation_time',
                                 'score',
                                 'base_prefix',
                                 None,
                                 confidence_interval,
                                 datadir_sim,
                                 filename,
                                 'Consolidation time (h)',
                                 'Discrimination',
                                 1/float(3600),
                                 1,
                                 None,
                                 (-1.01, 1.01),
                                 palette_sim_novel_stim,
                                 True,
                                 ('region', ('hpc',)),
                                 ('neuron_type',(neuron_type,)),
                                 ('learn_time', (300,)),
                                 ('phase', ('test',)),
                                 ('pattern',(0,)),
                                 ('metric',('tpr',)))
        #'''
    

        # Plot simulation recall metrics
        # Plot simulation recall
        '''
        filename = 'metrics-all.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        filename = 'exc_neurons-tpr-phase_test-pat_ind-rfl-300.svg'
        datatester_sim.plot_line('consolidation_time',
                                 'score',
                                 'base_prefix',
                                 None,
                                 confidence_interval,
                                 datadir_sim,
                                 filename,
                                 'Consolidation time (h)',
                                 'Recall (%)',
                                 1/float(3600),
                                 100,
                                 None,
                                 (0, 100.05),
                                 palette_sim_all_stim,
                                 False,
                                 ('region', ('hpc',)),
                                 ('neuron_type',('exc',)),
                                 ('learn_time', (300,)),
                                 ('phase', ('test',)),
                                 ('pattern',('0',)),
                                 ('metric',('tpr',)))
        #'''
        
        
        # Plot simulation stimulus overlap
        #'''
        filename = 'stimulus_overlap.csv'
        datatester_sim = DataTester(datadir=datadir, filename=filename)
        
        filename = 'stimulus_overlap.svg'
        datatester_sim.plot_bar('stimulus',
                                'overlap',
                                ['circle', 'pentagon', 'hexagon'],
                                None,
                                datadir,
                                filename,
                                None,
                                'Overlap (%)',
                                None,
                                100,
                                None,
                                (0, 100.05),
                                None,
                                None,
                                palette_sim_novel_stim)
        #'''
        
        
        # Plot simulation recall
        '''
        filename = 'metrics-all.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        filename = 'exc_neurons-tpr-phase_test-pat_ind-rfl-300.svg'
        datatester_sim.plot_bar('consolidation_time',
                                'score',
                                #'base_prefix',
                                [24],
                                #None,
                                'sd',
                                datadir_sim,
                                filename,
                                'Consolidation time (h)',
                                'Recall (%)',
                                1/float(3600),
                                100,
                                None,
                                (0, 100.05),
                                None,
                                None,
                                palette_sim_train_stim,
                                ('region', ('hpc',)),
                                ('neuron_type',('exc',)),
                                ('learn_time', (300,)),
                                ('phase', ('test',)),
                                ('pattern',('0',)),
                                ('base_prefix',('rft-0',)),
                                ('metric',('tpr',)))    
        #'''
        
        '''
        filename = 'metrics-all.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        filename = 'exc_neurons-avg_ca_rate-phase_test-pat_ind-rfl-300.svg'
        datatester_sim.plot_bar('consolidation_time',
                                'score',
                                [24],
                                'sd',
                                datadir_sim,
                                filename,
                                'Consolidation time (h)',
                                'Recall rate (Hz)',
                                1/float(3600),
                                None,
                                None,
                                None,
                                None,
                                None,
                                palette_sim_train_stim,
                                ('region', ('hpc',)),
                                ('neuron_type',('exc',)),
                                ('learn_time', (300,)),
                                ('phase', ('test',)),
                                ('pattern',('0-0',)),
                                ('base_prefix',('rft-0',)),
                                ('metric',('avg_ca_rate',)))
        #'''
        
        # Plot simulation recall in control vs. manipulation
        '''
        filename = 'metrics-compare.csv'
        datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
        
        filename = 'recall.svg'
        datatester_sim.plot_bar('group',
                                'recall',
                                ['control', 'block-ltp'],
                                'sd',
                                datadir_sim,
                                filename,
                                None,
                                'Recall (%)',
                                None,
                                100,
                                None,
                                (0, 100.05),
                                None,
                                None,
                                palette_discrimination_control_manipulation)
        #'''

        # Simulation plot: CDFs of pairs of stimulus-evoked firing rate distributions
        #'''
        datadir_sim = "~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/0"
        datadir_sim = os.path.expanduser(datadir_sim)
        
        # Load stimulus-evoked firing rates of neurons during training
        rates_file = 'exc-neuron-rates-across-stim-presentations-rfl-120-300-hpc.pkl'
        rates = load_pickled_object(os.path.join(datadir_sim, rates_file))[0]
        print('rates', type(rates), rates.shape)
        
        # Load engram cell ensembles
        engrams = []
        engrams.append(load_cell_assemblies(datadir_sim, 'rfl-300.rate.pat.exc.hpc')[0])
        neurons = set(range(0,4096))
        overlap = set(engrams[0])
        new_engram_cells = set([])
        for i in range(0, 90000, 3600):
            engrams.append(load_cell_assemblies(datadir_sim, 'rfp-' + str(i) + '.rate.pat.exc.hpc')[0])
            overlap = overlap & set(engrams[-1])
            new_engram_cells = new_engram_cells | (set(engrams[-1]) - set(engrams[0]))

        print('engrams', type(engrams), len(engrams))
        print('neurons', type(neurons), len(neurons))
        print('overlap', type(overlap), len(overlap))
        print('new_engram_cells', type(new_engram_cells), len(new_engram_cells))
        
        print('\nOverlap vs rest of neurons')
        rate1 = rates[sorted(list(overlap))]
        rate2 = rates[sorted(list(neurons - overlap))]
        ks = stats.kstest(rate1, rate2, alternative='two-sided', mode='auto')
        print('rate1', type(rate1), rate1.shape)
        print('rate2', type(rate2), rate2.shape)
        print(ks)
        filename = 'training_stimulus_evoked_firing_rate_overlap_vs_rest-neurons.svg'
        plot_compare_cdfs(rate1,
                          rate2,
                          'Engram cells',
                          'Non-engram cells',
                          '#ed553b',
                          'black',
                          'Firing rate (Hz)',
                          datadir_sim,
                          filename)
        
        print('\nOverlap vs rest of training')
        rate1 = rates[sorted(list(overlap))]
        rate2 = rates[sorted(list(set(engrams[0]) - overlap))]
        ks = stats.kstest(rate1, rate2, alternative='two-sided', mode='auto')
        print('rate1', type(rate1), rate1.shape)
        print('rate2', type(rate2), rate2.shape)
        print(ks)
        filename = 'training_stimulus_evoked_firing_rate_overlap_vs_rest-training.svg'
        plot_compare_cdfs(rate1,
                          rate2,
                          'Engram cells',
                          'Non-engram cells',
                          '#ed553b',
                          'black',
                          'Firing rate (Hz)',
                          datadir_sim,
                          filename)
        
        print('\nNew engram cells vs rest of neurons')
        rate1 = rates[sorted(list(new_engram_cells))]
        rate2 = rates[sorted(list(neurons - new_engram_cells))]
        ks = stats.kstest(rate1, rate2, alternative='two-sided', mode='auto')
        print('rate1', type(rate1), rate1.shape)
        print('rate2', type(rate2), rate2.shape)
        print(ks)
        filename = 'training_stimulus_evoked_firing_rate_new-engram-cells_vs_rest-neurons.svg'
        plot_compare_cdfs(rate1,
                          rate2,
                          'Engram cells',
                          'Non-engram cells',
                          '#ed553b',
                          'black',
                          'Firing rate (Hz)',
                          datadir_sim,
                          filename)
        
        print('\nNew engram cells vs non-training')
        rate1 = rates[sorted(list(new_engram_cells))]
        rate2 = rates[sorted(list((neurons - set(engrams[0])) - new_engram_cells))]
        ks = stats.kstest(rate1, rate2, alternative='two-sided', mode='auto')
        print('rate1', type(rate1), rate1.shape)
        print('rate2', type(rate2), rate2.shape)
        print(ks)
        filename = 'training_stimulus_evoked_firing_rate_new-engram-cells_vs_rest-non-training.svg'
        plot_compare_cdfs(rate1,
                          rate2,
                          'Engram cells',
                          'Non-engram cells',
                          '#ed553b',
                          'black',
                          'Firing rate (Hz)',
                          datadir_sim,
                          filename)
        #'''


    # Plot experimental data
    if has_experiment_analysis:
        '''
        filename = 'exp_A-freezing-training-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('A',)),
                             ('group',('behavior',)),
                             ('phase',('training',)),
                             ('metric', ('freezing',)))

        filename = 'exp_A-freezing-training-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training,
                              True,
                              ['training'],
                              ('experiment',('A',)),
                              ('group',('behavior',)),
                              ('phase',('training',)),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_A-freezing-recall_training-recall_novel-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('A',)),
                             ('group',('behavior',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))
    
        filename = 'exp_A-freezing-recall_training-recall_novel-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.05),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('A',)),
                                  ('group',('behavior',)),
                                  ('phase',('recall-training','recall-novel')),
                                  ('metric', ('freezing',)))

        filename = 'exp_A-freezing-recall_training-recall_novel-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('A',)),
                              ('group',('behavior',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_A-freezing-discrimination-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Discrimination',
                             1,
                             1,
                             None,
                             (-1.01, 1.01),
                             palette_discrimination_control,
                             True,
                             ('experiment',('A',)),
                             ('group',('behavior',)),
                             ('phase',('recall-discrimination',)),
                             ('metric', ('freezing',)))

        filename = 'exp_A-freezing-discrimination-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Discrimination',
                                  None,
                                  None,
                                  None,
                                  (-1.01, 1.01),
                                  palette_discrimination_control,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('A',)),
                                  ('group',('behavior',)),
                                  ('phase',('recall-discrimination',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_A-freezing-discrimination-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Discrimination',
                              1,
                              1,
                              None,
                              (-1.01, 1.01),
                              palette_discrimination_control,
                              True,
                              ['recall-discrimination'],
                              ('experiment',('A',)),
                              ('group',('behavior',)),
                              ('phase',('recall-discrimination',)),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_A-labelling-of-EGFP-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                             1,
                             1,
                             None,
                             (0, 25),
                             palette_labelling,
                             False,
                             ('experiment',('A',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training','recall-novel','home-cage')),
                             ('metric', ('c-Fos+EGFP+ofEGFP+',)))


        filename = 'exp_A-labelling-of-EGFP-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 25),
                                  palette_labelling,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('A',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training','recall-novel','home-cage')),
                                  ('metric', ('c-Fos+EGFP+ofEGFP+',)))

    
        filename = 'exp_A-labelling-of-EGFP-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                              1,
                              1,
                              None,
                              (0, 25),
                              palette_labelling,
                              False,
                              ['home-cage', 'recall-training','recall-novel'],
                              ('experiment',('A',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training','recall-novel','home-cage')),
                              ('metric', ('c-Fos+EGFP+ofEGFP+',)))
        #'''

        #'''
        filename = 'exp_A-labelling-of-c-Fos-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                             1,
                             1,
                             None,
                             (0, 10.05),
                             palette_labelling,
                             False,
                             ('experiment',('A',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training','recall-novel','home-cage')),
                             ('metric', ('c-Fos+EGFP+ofc-Fos+',)))


        filename = 'exp_A-labelling-of-c-Fos-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 10.05),
                                  palette_labelling,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('A',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training','recall-novel','home-cage')),
                                  ('metric', ('c-Fos+EGFP+ofc-Fos+',)))

    
        filename = 'exp_A-labelling-of-c-Fos-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                              1,
                              1,
                              None,
                              (0, 10.05),
                              palette_labelling,
                              False,
                              ['home-cage', 'recall-training','recall-novel'],
                              ('experiment',('A',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training','recall-novel','home-cage')),
                              ('metric', ('c-Fos+EGFP+ofc-Fos+',)))
        #'''

        #'''
        filename = 'exp_A-labelling-of-DAPI-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                             1,
                             1,
                             None,
                             (0, 6),
                             palette_labelling,
                             False,
                             ('experiment',('A',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training','recall-novel','home-cage')),
                             ('metric', ('c-Fos+EGFP+ofDAPI+',)))


        filename = 'exp_A-labelling-of-DAPI-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 6),
                                  palette_labelling,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('A',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training','recall-novel','home-cage')),
                                  ('metric', ('c-Fos+EGFP+ofDAPI+',)))

    
        filename = 'exp_A-labelling-of-DAPI-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                              1,
                              1,
                              None,
                              (0, 6),
                              palette_labelling,
                              False,
                              ['home-cage', 'recall-training','recall-novel'],
                              ('experiment',('A',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training','recall-novel','home-cage')),
                              ('metric', ('c-Fos+EGFP+ofDAPI+',)))

        #'''


        '''
        filename = 'exp_B-freezing-training-control-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('B',)),
                             ('group',('control',)),
                             ('phase',('training',)),
                             ('metric', ('freezing',)))

        filename = 'exp_B-freezing-training-control-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training,
                              True,
                              ['training'],
                              ('experiment',('B',)),
                              ('group',('control',)),
                              ('phase',('training',)),
                              ('metric', ('freezing',)))

        filename = 'exp_B-freezing-training-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('B',)),
                             ('group',('cck-inhibition',)),
                             ('phase',('training',)),
                             ('metric', ('freezing',)))
        
        filename = 'exp_B-freezing-training-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training,
                              True,
                              ['training'],
                              ('experiment',('B',)),
                              ('group',('cck-inhibition',)),
                              ('phase',('training',)),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_B-freezing-recall_training-recall_novel-control-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('B',)),
                             ('group',('control',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))

        filename = 'exp_B-freezing-recall_training-recall_novel-control-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('B',)),
                                  ('group',('control',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))

        filename = 'exp_B-freezing-recall_training-recall_novel-control-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('B',)),
                              ('group',('control',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))
        #'''

        #'''
        filename = 'exp_B-freezing-recall_training-recall_novel-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('B',)),
                             ('group',('cck-inhibition',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))

        filename = 'exp_B-freezing-recall_training-recall_novel-cck-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('B',)),
                                  ('group',('cck-inhibition',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))

        filename = 'exp_B-freezing-recall_training-recall_novel-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('B',)),
                              ('group',('cck-inhibition',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_B-freezing-discrimination-control-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'group',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Discrimination',
                             1,
                             1,
                             None,
                             (-1.01, 1.01),
                             palette_discrimination_control_manipulation,
                             True,
                             ('experiment',('B',)),
                             ('phase',('recall-discrimination',)),
                             ('metric', ('freezing',)))
        
        filename = 'exp_B-freezing-discrimination-control-cck-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'group',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Discrimination',
                                  None,
                                  None,
                                  None,
                                  (-1.01, 1.01),
                                  palette_discrimination_control_manipulation,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('B',)),
                                  ('phase',('recall-discrimination',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_B-freezing-discrimination-control-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'group',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Discrimination',
                              1,
                              1,
                              None,
                              (-1.01, 1.01),
                              palette_discrimination_control_manipulation,
                              True,
                              ['control', 'cck-inhibition'],
                              ('experiment',('B',)),
                              ('phase',('recall-discrimination',)),
                              ('metric', ('freezing',)))
        
        #'''

        '''
        filename = 'exp_C-freezing-training-control-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('C',)),
                             ('group',('control',)),
                             ('phase',('training',)),
                             ('metric', ('freezing',)))

        filename = 'exp_C-freezing-training-control-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training,
                              True,
                              ['training'],
                              ('experiment',('C',)),
                              ('group',('control',)),
                              ('phase',('training',)),
                              ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-training-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('C',)),
                             ('group',('cck-inhibition',)),
                             ('phase',('training',)),
                             ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-training-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training,
                              True,
                              ['training'],
                              ('experiment',('C',)),
                              ('group',('cck-inhibition',)),
                              ('phase',('training',)),
                              ('metric', ('freezing',)))
        
        #'''

        #'''
        filename = 'exp_C-freezing-recall_training-recall_novel-control-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('C',)),
                             ('group',('control',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-recall_training-recall_novel-control-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('C',)),
                                  ('group',('control',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-recall_training-recall_novel-control-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('C',)),
                              ('group',('control',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))
        #'''
        
        #'''
        filename = 'exp_C-freezing-recall_training-recall_novel-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('C',)),
                             ('group',('cck-inhibition',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-recall_training-recall_novel-cck-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('C',)),
                                  ('group',('cck-inhibition',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-recall_training-recall_novel-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('C',)),
                              ('group',('cck-inhibition',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_C-freezing-discrimination-control-cck-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'group',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Discrimination',
                             1,
                             1,
                             None,
                             (-1.01, 1.01),
                             palette_discrimination_control_manipulation,
                             True,
                             ('experiment',('C',)),
                             ('phase',('recall-discrimination',)),
                             ('metric', ('freezing',)))

        filename = 'exp_C-freezing-discrimination-control-cck-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'group',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Discrimination',
                                  None,
                                  None,
                                  None,
                                  (-1.01, 1.01),
                                  palette_discrimination_control_manipulation,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('C',)),
                                  ('phase',('recall-discrimination',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_C-freezing-discrimination-control-cck-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'group',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Discrimination',
                              1,
                              1,
                              None,
                              (-1.01, 1.01),
                              palette_discrimination_control_manipulation,
                              True,
                              ['control', 'cck-inhibition'],
                              ('experiment',('C',)),
                              ('phase',('recall-discrimination',)),
                              ('metric', ('freezing',)))

        #'''

        #'''
        filename = 'exp_D-freezing-recall_training-recall_novel-control-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('D',)),
                             ('group',('control',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))
        
        filename = 'exp_D-freezing-recall_training-recall_novel-control-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('D',)),
                                  ('group',('control',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_D-freezing-recall_training-recall_novel-control-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('D',)),
                              ('group',('control',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))
        #'''

        #'''
        filename = 'exp_D-freezing-recall_training-recall_novel-pv-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_freezing_training_novel,
                             False,
                             ('experiment',('D',)),
                             ('group',('pv-inhibition',)),
                             ('phase',('recall-training','recall-novel')),
                             ('metric', ('freezing',)))

        filename = 'exp_D-freezing-recall_training-recall_novel-pv-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.5),
                                  palette_freezing_training_novel,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('D',)),
                                  ('group',('pv-inhibition',)),
                                  ('phase',('recall-training','recall-novel',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_D-freezing-recall_training-recall_novel-pv-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_freezing_training_novel,
                              True,
                              ['recall-training','recall-novel'],
                              ('experiment',('D',)),
                              ('group',('pv-inhibition',)),
                              ('phase',('recall-training','recall-novel')),
                              ('metric', ('freezing',)))
        
        #'''

        #'''
        filename = 'exp_D-freezing-discrimination-control-pv-inhibition-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'group',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Discrimination',
                             1,
                             1,
                             None,
                             (-1.01, 1.01),
                             palette_discrimination_control_manipulation,
                             True,
                             ('experiment',('D',)),
                             ('phase',('recall-discrimination',)),
                             ('metric', ('freezing',)))
        
        filename = 'exp_D-freezing-discrimination-control-pv-inhibition-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'group',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Discrimination',
                                  None,
                                  None,
                                  None,
                                  (-1.01, 1.01),
                                  palette_discrimination_control_manipulation,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('D',)),
                                  ('phase',('recall-discrimination',)),
                                  ('metric', ('freezing',)))
        
        filename = 'exp_D-freezing-discrimination-control-pv-inhibition-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'group',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Discrimination',
                              1,
                              1,
                              None,
                              (-1.01, 1.01),
                              palette_discrimination_control_manipulation,
                              True,
                              ['control', 'pv-inhibition'],
                              ('experiment',('D',)),
                              ('phase',('recall-discrimination',)),
                              ('metric', ('freezing',)))
        
        #'''

        #'''
        filename = 'exp_F-freezing-neutral-context-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'group',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'Freezing (%)',
                             1,
                             1,
                             None,
                             (0, 100.05),
                             palette_cal_light_reactivation,
                             False,
                             ('experiment',('F',)))

        filename = 'exp_F-freezing-neutral-context-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'group',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'Freezing (%)',
                                  None,
                                  None,
                                  None,
                                  (0, 100.05),
                                  palette_cal_light_reactivation,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('F',)))

        filename = 'exp_F-freezing-neutral-context-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'group',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'Freezing (%)',
                              1,
                              1,
                              None,
                              (0, 100.05),
                              palette_cal_light_reactivation,
                              False,
                              ['mCh', 'Chrimson-mCh'],
                              ('experiment',('F',)))
        
        #'''
        
        #'''
        filename = 'exp_G-pv_vs_gc-bar.svg'
        datatester.plot_bar('metric',
                            'score',
                            ['c-Fos+PV+ofc-Fos+', 'c-Fos+GC+ofc-Fos+'],
                            'sd',
                            datadir,
                            filename,
                            None,
                            '% of c-Fos$^+$',
                            None,
                            None,
                            None,
                            (0, 100.05),
                            None,
                            None,
                            palette_cfos_labeling,
                            ('experiment',('G',)),
                            ('group',('PV',)))

        '''
        filename = 'exp_G-pv_vs_gc-swarm.svg'
        datatester.plot_swarm('metric',
                              'score',
                              'metric',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              None,
                              '% of c-Fos$^+$',
                              None,
                              None,
                              None,
                              (0, 100.05),
                              palette_white, #palette_cfos_labeling palette_white
                              False,
                              ['c-Fos+PV+ofc-Fos+', 'c-Fos+GC+ofc-Fos+'],
                              ('experiment',('G',)),
                              ('group',('PV',)))
        #'''
        
        filename = 'exp_G-cck_vs_gc-bar.svg'
        datatester.plot_bar('metric',
                            'score',
                            ['c-Fos+CCK+ofc-Fos+', 'c-Fos+GC+ofc-Fos+'],
                            'sd',
                            datadir,
                            filename,
                            None,
                            '% of c-Fos$^+$',
                            None,
                            None,
                            None,
                            (0, 100.05),
                            None,
                            None,
                            palette_cfos_labeling,
                            ('experiment',('G',)),
                            ('group',('CCK',)))

        '''
        filename = 'exp_G-cck_vs_gc-swarm.svg'
        datatester.plot_swarm('metric',
                              'score',
                              'metric',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              None,
                              '% of c-Fos$^+$',
                              None,
                              None,
                              None,
                              (0, 100.05),
                              palette_white, #palette_cfos_labeling palette_white
                              False,
                              ['c-Fos+CCK+ofc-Fos+', 'c-Fos+GC+ofc-Fos+'],
                              ('experiment',('G',)),
                              ('group',('CCK',)))
        #'''
        #'''
        
        #'''
        filename = 'exp_H-cck-bar.svg'
        datatester.plot_bar('metric',
                            'score',
                            ['eYFP+CCK+ofeYFP+'],
                            'sd',
                            datadir,
                            filename,
                            None,
                            '% of eYFP$^+$',
                            None,
                            None,
                            None,
                            (0, 100.05),
                            None,
                            None,
                            palette_eyfp_labeling,
                            ('experiment',('H',)),
                            ('group',('CCK',)))

        '''
        filename = 'exp_H-cck-swarm.svg'
        datatester.plot_swarm('metric',
                              'score',
                              'metric',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              None,
                              '% of eYFP$^+$',
                              None,
                              None,
                              None,
                              (0, 100.05),
                              palette_white, #palette_cfos_labeling palette_white
                              False,
                              ['eYFP+CCK+ofeYFP+'],
                              ('experiment',('H',)),
                              ('group',('CCK',)))
        #'''
        
        filename = 'exp_H-gad1-bar.svg'
        datatester.plot_bar('metric',
                            'score',
                            ['eYFP+GAD1+ofeYFP+'],
                            'sd',
                            datadir,
                            filename,
                            None,
                            '% of eYFP$^+$',
                            None,
                            None,
                            None,
                            (0, 100.05),
                            None,
                            None,
                            palette_eyfp_labeling,
                            ('experiment',('H',)),
                            ('group',('GAD1',)))

        '''
        filename = 'exp_H-gad1-swarm.svg'
        datatester.plot_swarm('metric',
                              'score',
                              'metric',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              None,
                              '% of eYFP$^+$',
                              None,
                              None,
                              None,
                              (0, 100.05),
                              palette_white, #palette_cfos_labeling palette_white
                              False,
                              ['eYFP+GAD1+ofeYFP+'],
                              ('experiment',('H',)),
                              ('group',('GAD1',)))
        #'''
        #'''
        
        #'''
        filename = 'exp_I-labelling-of-EGFP-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                             None,
                             None,
                             None,
                             (0, 25),
                             palette_freezing_training,
                             False,
                             ('experiment',('I',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training',)),
                             ('metric', ('c-Fos+EGFP+ofEGFP+',)))
        
        filename = 'exp_I-labelling-of-EGFP-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 25),
                                  palette_freezing_training,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('I',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training',)),
                                  ('metric', ('c-Fos+EGFP+ofEGFP+',)))
        
        filename = 'exp_I-labelling-of-EGFP-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of EGFP$^+$)',
                              None,
                              None,
                              None,
                              (0, 25),
                              palette_freezing_training,
                              False,
                              ['recall-training'],
                              ('experiment',('I',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training',)),
                              ('metric', ('c-Fos+EGFP+ofEGFP+',)))
        #'''
        
        #'''
        filename = 'exp_I-labelling-of-c-Fos-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                             None,
                             None,
                             None,
                             (0, 10.05),
                             palette_freezing_training,
                             False,
                             ('experiment',('I',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training',)),
                             ('metric', ('c-Fos+EGFP+ofc-Fos+',)))
        
        filename = 'exp_I-labelling-of-c-Fos-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 10.05),
                                  palette_freezing_training,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('I',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training',)),
                                  ('metric', ('c-Fos+EGFP+ofc-Fos+',)))
        
        filename = 'exp_I-labelling-of-c-Fos-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of c-Fos$^+$)',
                              None,
                              None,
                              None,
                              (0, 10.05),
                              palette_freezing_training,
                              False,
                              ['recall-training'],
                              ('experiment',('I',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training',)),
                              ('metric', ('c-Fos+EGFP+ofc-Fos+',)))
        #'''
        
        #'''
        filename = 'exp_I-labelling-of-DAPI-line.svg'
        datatester.plot_line('hours_post_cfc',
                             'score',
                             'phase',
                             None,
                             confidence_interval,
                             datadir,
                             filename,
                             'Delay time (h)',
                             'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                             None,
                             None,
                             None,
                             (0, 6),
                             palette_freezing_training,
                             False,
                             ('experiment',('I',)),
                             ('group',('labelling',)),
                             ('phase',('recall-training',)),
                             ('metric', ('c-Fos+EGFP+ofDAPI+',)))
        
        filename = 'exp_I-labelling-of-DAPI-bar.svg'
        datatester.plot_error_bar('hours_post_cfc',
                                  'score',
                                  'phase',
                                  None,
                                  confidence_interval,
                                  datadir,
                                  filename,
                                  'Delay time (h)',
                                  'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                                  None,
                                  None,
                                  None,
                                  (0, 6),
                                  palette_freezing_training,
                                  1.0,
                                  0.75,
                                  0.15,
                                  ('experiment',('I',)),
                                  ('group',('labelling',)),
                                  ('phase',('recall-training',)),
                                  ('metric', ('c-Fos+EGFP+ofDAPI+',)))
        
        filename = 'exp_I-labelling-of-DAPI-swarm.svg'
        datatester.plot_swarm('hours_post_cfc',
                              'score',
                              'phase',
                              None,
                              confidence_interval,
                              datadir,
                              filename,
                              'Delay time (h)',
                              'c-Fos$^+$ ∩ EGFP$^+$\n(% of DAPI$^+$)',
                              None,
                              None,
                              None,
                              (0, 6),
                              palette_freezing_training,
                              False,
                              ['recall-training'],
                              ('experiment',('I',)),
                              ('group',('labelling',)),
                              ('phase',('recall-training',)),
                              ('metric', ('c-Fos+EGFP+ofDAPI+',)))
        #'''
    
    # Retrieve data for statistical tests
    #Choose one of the options below
    
    # one-sample tests
    '''
    data1 = 'data1'
    #'''

    # two-sample tests
    '''
    data1 = 'data1'
    data2 = 'data2'
    #'''

    # one-way anova of ensemble overlap in labeling experiments
    '''
    data1 = 'training'
    data2 = 'neutral'
    data3 = 'home'
    #'''

    # friedman test of simulation results
    #data1 = 'square'
    #data2 = 'circle'
    #data3 = 'pentagon'
    #data4 = 'hexagon'


    # Statistical test of simulation recall metrics
    '''
    filename = 'metrics-all.csv'
    datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
    
    datatester_sim.store_data('data1',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (0,)),
                              ('phase', ('test',)),
                              ('base_prefix', ('rft-0',)),
                              ('pattern',('0',)),
                              ('metric',('tpr',)))

    datatester_sim.store_data('data2',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (0,)),
                              ('phase', ('test',)),
                              ('base_prefix', ('rft-1',)),
                              ('pattern',('0',)),
                              ('metric',('tpr',)))

    datatester_sim.store_data('data3',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (0,)),
                              ('phase', ('test',)),
                              ('base_prefix', ('rft-2',)),
                              ('pattern',('0',)),
                              ('metric',('tpr',)))

    datatester_sim.store_data('data4',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (0,)),
                              ('phase', ('test',)),
                              ('base_prefix', ('rft-3',)),
                              ('pattern',('0',)),
                              ('metric',('tpr',)))
    #'''


    # Statistical test of simulation recall discrimination
    '''
    filename = 'metrics-all.csv'
    datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
    
    datatester_sim.store_data('data1',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (86400,)),
                              ('phase', ('test',)),
                              ('base_prefix', ('rft-0-rft-3',)),
                              ('pattern',(0,)),
                              ('metric',('tpr',)))
    #'''
    
    
    # Statistical test of simulation reactivation rates
    '''
    filename = 'metrics-all.csv'
    datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
    
    datatester_sim.store_data('data1',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (86400,)),
                              ('phase', ('test-learn',)),
                              ('base_prefix',('rft-0-rfl',)),
                              ('metric',('fraction_activated',)))
    
    datatester_sim.store_data('data2',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (86400,)),
                              ('phase', ('test-learn',)),
                              ('base_prefix',('rft-1-rfl',)),
                              ('metric',('fraction_activated',)))
    #'''

    
    # Statistical test of simulation firing rates
    '''
    filename = 'metrics-all.csv'
    datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
    
    datatester_sim.store_data('data1',
                              'score',
                              ('region', ('hpc',)),
                              ('neuron_type',('exc',)),
                              ('learn_time', (300,)),
                              ('consolidation_time', (0,)),
                              ('phase', ('test',)),
                              ('base_prefix',('rft-0',)),
                              ('metric',('avg_ca_rate',)))
    #'''

    # Statistical test of simulation recall in control vs. manipulation
    '''
    filename = 'metrics-compare.csv'
    datatester_sim = DataTester(datadir=datadir_sim, filename=filename)
    
    datatester_sim.store_data('data1',
                              'recall',
                              ('group', ('control',)))

    datatester_sim.store_data('data2',
                              'recall',
                              ('group', ('block-ltp',)))
    #'''

    
    # Statistical test of experimental data
    # Paired-samples test of freezing behavior
    # Experiment A
    '''
    hours_post_cfc = 1
    datatester.store_data(data1,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('recall-training',)),
                          ('metric', ('freezing',)),
                          ('group', ('behavior',)),
                          ('hours_post_cfc', (hours_post_cfc,)))

    datatester.store_data(data2,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('recall-novel',)),
                          ('metric', ('freezing',)),
                          ('group', ('behavior',)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''
    # Experiment B / C / D
    '''
    experiment = 'B' # 'B'/ 'C' / 'D'
    group = 'cck-inhibition' # 'control' / 'cck-inhibition' / 'pv-inhibition'
    hours_post_cfc = 5 # 5 / 12 / 24
    datatester.store_data(data1,
                          'score',
                          ('experiment',(experiment,)),
                          ('phase',('recall-training',)),
                          ('metric', ('freezing',)),
                          ('group', (group,)),
                          ('hours_post_cfc', (hours_post_cfc,)))

    datatester.store_data(data2,
                          'score',
                          ('experiment',(experiment,)),
                          ('phase',('recall-novel',)),
                          ('metric', ('freezing',)),
                          ('group', (group,)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''

    # One-sample test of discrimination index of freezing behavior
    # Experiment A
    '''
    hours_post_cfc = 1
    datatester.store_data(data1,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('recall-discrimination',)),
                          ('metric', ('freezing',)),
                          ('group', ('behavior',)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''
    # Experiment B / C / D
    '''
    experiment = 'B'  # 'B'/ 'C' / 'D'
    group = 'cck-inhibition' # 'control' / 'cck-inhibition' / 'pv-inhibition'
    hours_post_cfc = 5 # 5 / 12 / 24
    datatester.store_data(data1,
                          'score',
                          ('experiment',(experiment,)),
                          ('phase',('recall-discrimination',)),
                          ('metric', ('freezing',)),
                          ('group', (group,)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''

    # Experiment A: ANOVA of ensemble overlap
    '''
    metric = 'c-Fos+EGFP+ofEGFP+'
    #metric = 'c-Fos+EGFP+ofc-Fos+'
    #metric = 'c-Fos+EGFP+ofDAPI+'

    hours_post_cfc = 5 # 5 / 12 / 24
    datatester.store_data(data1,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('recall-training',)),
                          ('metric', (metric,)),
                          ('group', ('labelling',)),
                          ('hours_post_cfc', (hours_post_cfc,)))

    datatester.store_data(data2,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('recall-novel',)),
                          ('metric', (metric,)),
                          ('group', ('labelling',)),
                          ('hours_post_cfc', (hours_post_cfc,)))

    datatester.store_data(data3,
                          'score',
                          ('experiment',('A',)),
                          ('phase',('home-cage',)),
                          ('metric', (metric,)),
                          ('group', ('labelling',)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''

    # Experiment E: Paired t-test of normalized oIPSCs
    '''
    group1 = 'hM4Di-mCh-EGFP+' # 'mCh-EGFP+' / 'hM4Di-mCh-EGFP+'
    group2 = 'hM4Di-mCh-EGFP-' # 'mCh-EGFP-' / 'hM4Di-mCh-EGFP-'
    
    datatester.store_data(data1,
                          'score',
                          ('experiment',('E',)),
                          ('group', (group1,)))

    datatester.store_data(data2,
                          'score',
                          ('experiment',('E',)),
                          ('group', (group2,)))
    #'''

    # Cal-Light activation in neutral context
    '''
    hours_post_cfc = 5 # 5 / 12 / 24
    datatester.store_data(data1,
                          'score',
                          ('experiment',('F',)),
                          ('group', ('mCh',)),
                          ('hours_post_cfc', (hours_post_cfc,)))

    datatester.store_data(data2,
                          'score',
                          ('experiment',('F',)),
                          ('group', ('Chrimson-mCh',)),
                          ('hours_post_cfc', (hours_post_cfc,)))
    #'''
    
    
    # Shapiro-Wilk test for normality
    #'''
    print('\n\n')
    if 'data1' in locals():
        shapiro1 = datatester.shapiro(data1)
        print(data1, 'shapiro')
        print(shapiro1)

    if 'data2' in locals():
        shapiro2 = datatester.shapiro(data2)
        print(data2, 'shapiro')
        print(shapiro2)

    if 'data3' in locals():
        shapiro3 = datatester.shapiro(data3)
        print(data3, 'shapiro')
        print(shapiro3)

    if 'data4' in locals():
        shapiro4 = datatester.shapiro(data4)
        print(data4, 'shapiro')
        print(shapiro4)
    #'''


    # D’Agostino and Pearson’s test for normality
    #'''
    print('\n\n')
    if 'data1' in locals():
        normaltest1 = datatester.normaltest(data1)
        print(data1, "d'agostino and pearson")
        print(normaltest1)

    if 'data2' in locals():
        normaltest2 = datatester.normaltest(data2)
        print(data2, "d'agostino and pearson")
        print(normaltest2)

    if 'data3' in locals():
        normaltest3 = datatester.normaltest(data3)
        print(data3, "d'agostino and pearson")
        print(normaltest3)

    if 'data4' in locals():
        normaltest4 = datatester.normaltest(data4)
        print(data4, "d'agostino and pearson")
        print(normaltest4)

    #'''


    # Levene's test and Bartlett's test for equality of variances
    #'''
    if ('data1' in locals()) and ('data2' in locals()):
        if 'data3' in locals():
            if 'data4' in locals():
                levene = datatester.levene_4(data1, data2, data3, data4)
                bartlett = datatester.bartlett_4(data1, data2, data3, data4)
            else:
                levene = datatester.levene_3(data1, data2, data3)
                bartlett = datatester.bartlett_3(data1, data2, data3)
        else:
            levene = datatester.levene_2(data1, data2)
            bartlett = datatester.bartlett_2(data1, data2)
        print('\n\nlevene')
        print(levene)
        print('\n\nbartlett')
        print(bartlett)
    #'''


    # Statistical tests of a single sample
    '''
    # One-sample t-test for equality of means: one independent sample and a given population mean
    if 'data2' not in locals():
        one_samp_t = datatester.one_samp_t('data1', 0)
        #one_samp_t = datatester.one_samp_t('data1', 0, alternative='two-sided')
        print('\n\none_samp t-test')
        print(one_samp_t)
    
    # One-sample Wilcoxon signed-rank test: differences between samples
    if 'data2' not in locals():
        #one_samp_wilcoxon = datatester.one_samp_wilcoxon('data1')
        one_samp_wilcoxon = datatester.one_samp_wilcoxon('data1', alternative='two-sided')
        print('\n\none_samp wilcoxon')
        print(one_samp_wilcoxon)
    #'''

    
    # Statistical tests of independent samples
    '''
    # Unpaired t-test for equality of means: independent samples with the same variance
    if 'data2' in locals():
        unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=True)
        #unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=True, alternative='two-sided')
        print('\n\nunpaired t-test')
        print(unpaired_t)

    # Welch's unpaired t-test: independent samples with different variances
    if 'data2' in locals():
        welch_unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=False)
        #welch_unpaired_t = datatester.unpaired_t('data1', 'data2', equal_var=False, alternative='two-sided')
        print('\n\nwelch_unpaired t-test')
        print(welch_unpaired_t)

    # Mann-Whitney U test: independent samples whose underlying distributions are not normal
    if 'data2' in locals():
        mannwhitneyu = datatester.mannwhitneyu('data1', 'data2', alternative='two-sided')
        print('\n\nmannwhitneyu')
        print(mannwhitneyu)
    #'''

    
    # Statistical tests of related samples
    '''
    # Paired t-test for equality of means: related samples with the same variance
    if 'data2' in locals():
        paired_t = datatester.paired_t('data1', 'data2')
        #paired_t = datatester.paired_t('data1', 'data2', alternative='two-sided')
        print('\n\npaired t-test')
        print(paired_t)
    
    # Wilcoxon signed-rank test: paired samples whose underlying distributions are not normal
    if 'data2' in locals():
        wilcoxon = datatester.wilcoxon(data1, data2, alternative='two-sided')
        print('\n\nwilcoxon t-test between data1 and data2')
        print(wilcoxon)

    # Friedman test: paired samples whose underlying distributions are not normal
    if 'data2' in locals() and 'data3' in locals() and 'data4' in locals():
        friedman = datatester.friedman(data1, data2, data3, data4)
        print('\n\nfriedman t-test')
        print(friedman)
    #'''


    # One-way ANOVA of ensemble overlap data
    '''
    print('\n\n')
    print('one-way ANOVA')
    fvalue, pvalue = datatester.one_way_anova(data1, data2, data3)
    print('F =', fvalue)
    print('p =', pvalue)

    print('\n\n')
    print('one-way ANOVA for a fitted linear model')
    datatester.build_ols_model(data1, data2, data3)
    anova_table = datatester.one_way_anova_lm()
    print(anova_table)

    print('\n\n')
    print('Tukey HSD post hoc test')
    tukey_summary = datatester.tukey_hsd()
    print(tukey_summary)

    print('\n\n')
    print('residual shapiro')
    shapiro = datatester.residual_shapiro()
    print(shapiro)
    #'''

    # Experiment E: Two-way repeated-measures ANOVA
    '''
    dfE = datatester.select_rows(('experiment',('E',)), ('metric',('oipsc',)))
    print('dfE',type(dfE))
    print(dfE)
    res = pg.rm_anova(dv='score',
                      within=['phase', 'group'],
                      subject='trial',
                      data=dfE,
                      detailed=True)
    print('Two-way repeated measures ANOVA')
    print(res)
    res.to_csv(os.path.join(datadir, 'two-way-anova.csv'))
    #'''
    
    
if __name__ == "__main__":
 main(sys.argv[1:])
