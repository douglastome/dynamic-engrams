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

from analyzer import Analyzer
from results import Results
from weights import Weights
from helper import *

class Parser:
 def __init__(self, run, has_inh_analysis, has_metrics, has_plots, has_neuron_vi_plots, has_cell_assemb_plots, has_activity_stats, has_activity_plots, has_spike_raster_stats, has_spike_stats_plots, has_metrics_plots, has_rates_across_stim_plots, has_weight_plots, has_rf_plots, conf_int, num_mpi_ranks, integration_time_step, rundir, random_trials, brain_areas, prefixes_thl, prefixes_ctx, prefixes_hpc, file_prefix_burn, file_prefix_learn, file_prefix_learn_bkw, file_prefix_consolidation, file_prefix_probe, file_prefix_cue, file_prefix_hm, file_prefix_burn_hm, file_prefix_learn_hm, file_prefix_consolidation_hm, file_prefix_probe_hm, file_prefix_cue_hm, phase_burn, phase_learn, phase_consolidation, phase_probe, phase_cue, nb_stim_learn, nb_stim_probe, nb_stim_cue, nb_exc_neurons_thl, exc_inh_thl, nb_exc_neurons_ctx, exc_inh_ctx, nb_exc_neurons_hpc, exc_inh_hpc, exc_inh2_hpc, nb_neurons_stim_replay, nb_neuron_cons_stim_replay, nb_patterns, nb_test_cue_sim, simtime_burn, simtimes_learn, simtime_learn_bkw, simtimes_consolidation, simtime_probe, simtime_cue, range_burn, range_learn, range_consolidation, range_probe, range_cue, zoom_range, nb_plot_neurons, colors, bin_size, has_stim_hpc, has_stim_ctx, has_stim_thl, has_thl_ctx, has_ctx_thl, has_thl_hpc, has_hpc_thl, has_ctx_hpc, has_hpc_ctx, has_hpc_rep, has_rec_ee, has_inh2, ids_cell_assemb, cell_assemb_method, exc_min_rate, inh_min_rate, inh2_min_rate, min_weight, stim_pat_learn, stim_pat_probe, stim_pat_cue, exc_record_rank_thl, inh_record_rank_thl, exc_ampa_nmda_ratio_thl, inh_ampa_nmda_ratio_thl, exc_record_rank_ctx, inh_record_rank_ctx, exc_ampa_nmda_ratio_ctx, inh_ampa_nmda_ratio_ctx, exc_record_rank_hpc, inh_record_rank_hpc, inh2_record_rank_hpc, exc_ampa_nmda_ratio_hpc, inh_ampa_nmda_ratio_hpc, inh2_ampa_nmda_ratio_hpc, u_rest, u_exc, u_inh):

  self.run = run
  self.has_inh_analysis = has_inh_analysis
  self.has_metrics = has_metrics
  self.has_plots = has_plots
  self.has_neuron_vi_plots = has_neuron_vi_plots
  self.has_cell_assemb_plots = has_cell_assemb_plots
  self.has_activity_stats = has_activity_stats
  self.has_activity_plots = has_activity_plots
  self.has_spike_raster_stats = has_spike_raster_stats
  self.has_spike_stats_plots = has_spike_stats_plots
  self.has_metrics_plots = has_metrics_plots
  self.has_rates_across_stim_plots = has_rates_across_stim_plots
  self.has_weight_plots = has_weight_plots
  self.has_rf_plots = has_rf_plots
  self.conf_int = conf_int
  self.num_mpi_ranks = num_mpi_ranks
  self.integration_time_step = integration_time_step
  self.rundir = rundir
  self.random_trials = random_trials
  self.brain_areas = brain_areas
  self.prefixes_thl = prefixes_thl
  self.prefixes_ctx = prefixes_ctx
  self.prefixes_hpc = prefixes_hpc
  self.file_prefix_burn = file_prefix_burn
  self.file_prefix_learn = file_prefix_learn
  self.file_prefix_learn_bkw = file_prefix_learn_bkw
  self.file_prefix_consolidation = file_prefix_consolidation
  self.file_prefix_probe = file_prefix_probe
  self.file_prefix_cue = file_prefix_cue
  self.file_prefix_hm = file_prefix_hm
  self.file_prefix_burn_hm = file_prefix_burn_hm
  self.file_prefix_learn_hm = file_prefix_learn_hm
  self.file_prefix_consolidation_hm = file_prefix_consolidation_hm
  self.file_prefix_probe_hm = file_prefix_probe_hm
  self.file_prefix_cue_hm = file_prefix_cue_hm
  self.phase_burn = phase_burn  
  self.phase_learn = phase_learn
  self.phase_consolidation = phase_consolidation
  self.phase_probe = phase_probe
  self.phase_cue = phase_cue
  self.nb_stim_learn = nb_stim_learn
  self.nb_stim_probe = nb_stim_probe
  self.nb_stim_cue = nb_stim_cue
  self.nb_exc_neurons_thl = nb_exc_neurons_thl
  self.nb_inh_neurons_thl = int(self.nb_exc_neurons_thl / exc_inh_thl)
  self.nb_exc_neurons_ctx = nb_exc_neurons_ctx
  self.nb_inh_neurons_ctx = int(self.nb_exc_neurons_ctx / exc_inh_ctx)
  self.nb_exc_neurons_hpc = nb_exc_neurons_hpc
  self.nb_inh_neurons_hpc = int(self.nb_exc_neurons_hpc / exc_inh_hpc)
  self.nb_inh2_neurons_hpc = int(self.nb_exc_neurons_hpc / exc_inh2_hpc)
  self.nb_neurons_stim_replay = nb_neurons_stim_replay
  self.nb_neuron_cons_stim_replay = nb_neuron_cons_stim_replay
  self.nb_patterns = nb_patterns
  self.nb_test_cue_sim = nb_test_cue_sim
  self.simtime_burn = simtime_burn
  self.simtimes_learn = simtimes_learn
  self.simtime_learn_bkw = simtime_learn_bkw
  self.simtimes_consolidation = simtimes_consolidation
  self.simtime_probe = simtime_probe
  self.simtime_cue = simtime_cue
  self.range_burn = range_burn
  self.range_learn = range_learn
  self.range_consolidation = range_consolidation
  self.range_probe = range_probe
  self.range_cue = range_cue
  self.zoom_range = zoom_range
  self.nb_plot_neurons = nb_plot_neurons
  self.colors = colors
  self.bin_size = bin_size
  self.has_stim_hpc = has_stim_hpc
  self.has_stim_ctx = has_stim_ctx
  self.has_stim_thl = has_stim_thl
  self.has_thl_ctx = has_thl_ctx
  self.has_ctx_thl = has_ctx_thl
  self.has_thl_hpc = has_thl_hpc
  self.has_hpc_thl = has_hpc_thl
  self.has_ctx_hpc = has_ctx_hpc
  self.has_hpc_ctx = has_hpc_ctx
  self.has_hpc_rep = has_hpc_rep
  self.has_rec_ee = has_rec_ee
  self.has_inh2 = has_inh2
  self.ids_cell_assemb = ids_cell_assemb
  self.cell_assemb_method = cell_assemb_method
  self.exc_min_rate = exc_min_rate
  self.inh_min_rate = inh_min_rate
  self.inh2_min_rate = inh2_min_rate
  self.min_weight = min_weight
  self.stim_pat_learn = stim_pat_learn
  self.stim_pat_probe = stim_pat_probe
  self.stim_pat_cue = stim_pat_cue
  self.exc_record_rank_thl = exc_record_rank_thl
  self.inh_record_rank_thl = inh_record_rank_thl
  self.exc_ampa_nmda_ratio_thl = exc_ampa_nmda_ratio_thl
  self.inh_ampa_nmda_ratio_thl = inh_ampa_nmda_ratio_thl
  self.exc_record_rank_ctx = exc_record_rank_ctx
  self.inh_record_rank_ctx = inh_record_rank_ctx
  self.exc_ampa_nmda_ratio_ctx = exc_ampa_nmda_ratio_ctx
  self.inh_ampa_nmda_ratio_ctx = inh_ampa_nmda_ratio_ctx
  self.exc_record_rank_hpc = exc_record_rank_hpc
  self.inh_record_rank_hpc = inh_record_rank_hpc
  self.inh2_record_rank_hpc = inh2_record_rank_hpc
  self.exc_ampa_nmda_ratio_hpc = exc_ampa_nmda_ratio_hpc
  self.inh_ampa_nmda_ratio_hpc = inh_ampa_nmda_ratio_hpc
  self.inh2_ampa_nmda_ratio_hpc = inh2_ampa_nmda_ratio_hpc
  self.u_rest = u_rest
  self.u_exc = u_exc
  self.u_inh = u_inh

  self.results = Results()
  self.weights = Weights()
  assert len(self.random_trials) > 0,"You must supply at least one trial."
  if len(self.random_trials) == 1:
   self.analyzers = []
   self.analyzers_ids = {}
   self.full_prefixes_thl = []
   self.full_prefixes_ctx = []
   self.full_prefixes_hpc = []
   self.metrics_filename = 'metrics-' + str(self.random_trials[0]) + '.csv'
   self.has_metrics_file = os.path.isfile(os.path.join(self.rundir, self.metrics_filename))
   self.parse()
  else:
   self.metrics_filename = 'metrics-all.csv'
   self.has_metrics_file = os.path.isfile(os.path.join(self.rundir, self.metrics_filename))


 def get_analyzer_id(self, brain_area, full_prefix):
  return brain_area + '-' + full_prefix
 

 def get_analyzer(self, brain_area, full_prefix):
  analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
  if analyzer_id in self.analyzers_ids:
   analyzer_idx = self.analyzers_ids[analyzer_id]
   return self.analyzers[analyzer_idx]
  else:
   return None


 def get_phase_hm(self, phase):
  return phase + '-' + self.file_prefix_hm

 
 def get_phase_bkw(self, phase):
  return phase + '-bkw'

 
 def parse(self):
  if self.run == 1:
   self.parse_run_1()
  elif self.run == 2 or self.run == 3:
   self.parse_run_2_3()
  elif self.run == 4:
   self.parse_run_4()
  elif self.run == 5 or self.run == 6:
   self.parse_run_5_6()
  elif self.run == 7 or self.run == 8:
   self.parse_run_7_8()
  elif self.run == 9:
   self.parse_run_9()
  elif self.run == 10 or self.run == 11:
   self.parse_run_10_11()

   
 def parse_run_1(self):

  file_prefix_assemb = self.file_prefix_learn
  has_rep = False
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    elif brain_area == 'ctx':
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     simtime_assemb = simtime_learn

     for prefix_idx in range(len(prefixes)):
      prefix = prefixes[prefix_idx]
      if brain_area == 'hpc' and self.file_prefix_hm in prefix:
       continue
      full_prefix = prefix
      if prefix == self.file_prefix_burn:
       full_prefix += '-' + str(self.simtime_burn)
       ids_cell_assemb = False
       has_stim_presentation = False
       simtime = self.simtime_burn
       t_stop = self.simtime_burn
       t_start = simtime - self.range_burn
       nb_stim = None
       stim_pat = None
       phase = self.phase_burn       
      else:
       if prefix == self.file_prefix_learn:
        prefix_end = '-' + str(self.simtime_burn)
        full_prefix += prefix_end + '-' + str(simtime_learn)
        ids_cell_assemb = True        
        has_stim_presentation = True
        simtime = simtime_learn
        t_stop = simtime_learn
        t_start = simtime - self.range_learn
        nb_stim = self.nb_stim_learn
        stim_pat = self.stim_pat_learn
        phase = self.phase_learn
       elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
        prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
        full_prefix += prefix_end + '-' + str(self.simtime_cue)
        ids_cell_assemb = False
        has_stim_presentation = True
        simtime = self.simtime_cue
        t_stop = self.simtime_cue
        t_start = simtime - self.range_cue
        nb_stim = self.nb_stim_cue
        stim_pat = self.stim_pat_cue        
        if prefix == self.file_prefix_cue_hm:
         phase = self.get_phase_hm(self.phase_cue)
        else:
         phase = self.phase_cue
       prefix += prefix_end
      
      analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
      if analyzer_id not in self.analyzers_ids:
       full_prefixes.append(full_prefix)

       if has_stim_presentation:
        stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
       else:
        stimtime, stimdata = None, None

       self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

       self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
       print("\n\n\nParsed Analyzer:")
       self.analyzers[-1].print_identifier()

        
 def parse_run_2_3(self):

  #prefixes_ctx = [self.file_prefix_burn, self.file_prefix_learn, self.file_prefix_consolidation, self.file_prefix_probe, self.file_prefix_cue]

  if self.run == 2:
   has_rep = False
  elif self.run == 3:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     file_prefix_assemb = self.file_prefix_probe
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    if brain_area == 'ctx':
     file_prefix_assemb = self.file_prefix_probe
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     file_prefix_assemb = self.file_prefix_probe
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      simtime_assemb = simtime_consolidation
      '''
      if brain_area == 'ctx':
       simtime_assemb = simtime_consolidation
      elif brain_area in ['hpc','thl']:
       simtime_assemb = simtime_learn
      '''
      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_cell_assemb = False
         '''
         if brain_area in ['hpc','thl']:
          ids_cell_assemb = True
         elif brain_area == 'ctx':
          ids_cell_assemb = False
         '''
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         ids_cell_assemb = True
         '''
         if brain_area in ['hpc','thl']:
          ids_cell_assemb = False
         elif brain_area == 'ctx':
          ids_cell_assemb = True
         '''
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation) + '-' + str(self.simtime_probe)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_4(self):

  has_rep = False
  has_rep_presentation = False
  file_prefix_assemb = self.file_prefix_probe
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'ctx':
     other_area = 'hpc'
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = self.has_hpc_ctx
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_area = 'ctx'
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = self.has_ctx_hpc
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:

      simtime_assemb = str(simtime_learn) + '-' + str(simtime_consolidation)

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_learn_bkw:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(self.simtime_learn_bkw)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = self.simtime_learn_bkw
         t_stop = self.simtime_learn_bkw
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.get_phase_bkw(self.phase_learn)
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         ids_cell_assemb = True
         has_stim_presentation = True
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw) + '-' + str(simtime_consolidation) + '-' + str(self.simtime_probe)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end
       full_prefixes.append(full_prefix)

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
        
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_5_6(self):

  #prefixes_ctx = [self.file_prefix_burn, self.file_prefix_learn, self.file_prefix_consolidation, self.file_prefix_probe, self.file_prefix_cue]

  if self.run == 5:
   has_rep = False
  elif self.run == 6:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    elif brain_area == 'ctx':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      simtime_assemb = simtime_learn

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_cell_assemb = True
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         
         t_stop = simtime_consolidation
         t_start = simtime_consolidation - self.range_consolidation

         #t_start = simtime_consolidation - 3600
         #t_stop = t_start + self.range_consolidation

         #t_start = simtime_consolidation - 1800 - 600 - 300 + 300
         #t_stop = t_start + 300
         
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_7_8(self):

  #prefixes_ctx = [self.file_prefix_burn, self.file_prefix_learn, self.file_prefix_consolidation, self.file_prefix_probe, self.file_prefix_cue]

  if self.run == 7:
   has_rep = False
  elif self.run == 8:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     
     file_prefix_assemb = self.file_prefix_probe
     #file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    if brain_area == 'ctx':
     
     file_prefix_assemb = self.file_prefix_probe
     #file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     
     file_prefix_assemb = self.file_prefix_probe
     #file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
     
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      
      simtime_assemb = simtime_consolidation
      #simtime_assemb = simtime_learn
      #if brain_area in ['ctx', 'thl']:
      # simtime_assemb = simtime_consolidation
      #elif brain_area in ['hpc']:
      # simtime_assemb = simtime_learn

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         
         ids_cell_assemb = False
         #ids_cell_assemb = True
         #if brain_area in ['hpc']:
         # ids_cell_assemb = True
         #elif brain_area in ['ctx', 'thl']:
         # ids_cell_assemb = False
         
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         
         ids_cell_assemb = True
         #ids_cell_assemb = False
         #if brain_area in ['hpc']:
         # ids_cell_assemb = False
         #elif brain_area in ['ctx', 'thl']:
         # ids_cell_assemb = True
         
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_9(self):

  file_prefix_assemb = self.file_prefix_learn
  has_rep = False
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    elif brain_area == 'ctx':
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
     
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     simtime_assemb = simtime_learn

     for prefix_idx in range(len(prefixes)):
      prefix = prefixes[prefix_idx]
      if brain_area == 'hpc' and self.file_prefix_hm in prefix:
       continue
      full_prefix = prefix
      if prefix == self.file_prefix_burn:
       base_prefix = prefix
       full_prefix += '-' + str(self.simtime_burn)
       ids_cell_assemb = False
       has_stim_presentation = False
       simtime = self.simtime_burn
       t_stop = self.simtime_burn
       t_start = simtime - self.range_burn
       nb_stim = None
       stim_pat = None
       phase = self.phase_burn       
      else:
       if prefix == self.file_prefix_learn:
        base_prefix = prefix
        prefix_end = '-' + str(self.simtime_burn)
        full_prefix += prefix_end + '-' + str(simtime_learn)
        ids_cell_assemb = True        
        has_stim_presentation = True
        simtime = simtime_learn
        t_stop = simtime_learn
        t_start = simtime - self.range_learn
        nb_stim = self.nb_stim_learn
        stim_pat = self.stim_pat_learn
        phase = self.phase_learn
       elif self.file_prefix_cue in prefix  or self.file_prefix_cue_hm in prefix:
        base_prefix = prefix
        prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
        full_prefix += prefix_end + '-' + str(self.simtime_cue)
        ids_cell_assemb = False
        has_stim_presentation = True
        simtime = self.simtime_cue
        t_stop = self.simtime_cue
        t_start = simtime - self.range_cue
        nb_stim = self.nb_stim_cue
        stim_pat = self.stim_pat_cue        
        if self.file_prefix_cue_hm in prefix:
         phase = self.get_phase_hm(self.phase_cue)
        else:
         phase = self.phase_cue
       prefix += prefix_end
      
      analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
      if analyzer_id not in self.analyzers_ids:
       full_prefixes.append(full_prefix)

       if has_stim_presentation:
        stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
       else:
        stimtime, stimdata = None, None

       self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn))

       self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
       print("\n\n\nParsed Analyzer:")
       self.analyzers[-1].print_identifier()


 def parse_run_10_11(self):

  file_prefix_assemb = self.file_prefix_probe
  file_prefix_encod = self.file_prefix_learn
  base_prefix_encod = self.file_prefix_learn
  phase_encod = self.phase_learn
  has_rep = False
  
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    if brain_area == 'ctx':
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     nb_inh2_neurons = self.nb_inh2_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     inh2_record_rank = self.inh2_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
     inh2_ampa_nmda_ratio = self.inh2_ampa_nmda_ratio_hpc
     
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))
    plot_inh2_neurons = list(np.random.choice(nb_inh2_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     simtime_encod = simtime_learn
     
     for simtime_consolidation in self.simtimes_consolidation:
      simtime_assemb = simtime_consolidation

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        base_prefix = prefix
        full_prefix += '-' + str(self.simtime_burn)
        ids_encod = False
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         base_prefix = prefix
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_encod = True
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         base_prefix = prefix
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_encod = False
         ids_cell_assemb = False
         has_rep_presentation = False
         if self.run == 10:
          has_stim_presentation = False
          nb_stim = None
          stim_pat = None
         elif self.run == 11:
          has_stim_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         base_prefix = prefix
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         ids_encod = False
         ids_cell_assemb = True
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif self.file_prefix_cue in prefix  or self.file_prefix_cue_hm in prefix:
         base_prefix = prefix
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_encod = False
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if self.file_prefix_cue_hm in prefix:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, nb_inh2_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.has_rec_ee, self.has_inh2, self.rundir, trialdir, base_prefix, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, plot_inh2_neurons, exc_record_rank, inh_record_rank, inh2_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, inh2_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.inh2_min_rate, self.bin_size, self.nb_patterns, self.nb_test_cue_sim, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, simtime_consolidation, file_prefix_encod=file_prefix_encod,base_prefix_encod=base_prefix_encod,phase_encod=phase_encod,simtime_encod=simtime_encod,ids_encod=ids_encod))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def save_results(self):
  self.results.build_data_frame()
  self.results.save_data_frame(self.rundir, self.metrics_filename)
  print('\n\n\nResults Data Frame:')
  print(self.results.df)
  print('Length:', len(self.results.df))

  
 def plot_metrics(self):

  if len(self.random_trials) == 1:
   datadir = os.path.join(self.rundir, self.random_trials[0])
  else:
   datadir = self.rundir
  if self.run == 1:
   self.plot_metrics_run_1(datadir)
  elif self.run == 9:
   self.plot_metrics_run_9(datadir)
  elif self.run == 10 or self.run == 11:
   self.plot_metrics_run_10_11(datadir)
  else:
   self.plot_metrics_run(datadir)

   
 def plot_metrics_run_1(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')

  for neuron_type in neuron_types:

   filename = neuron_type + '_neurons-accuracy-phase_all-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)))

   filename = neuron_type + '_neurons-tpr-phase_all-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)))

   filename = neuron_type + '_neurons-fpr-phase_all-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)))

   #'''
   filename = neuron_type + '_neurons-accuracy-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('phase', (self.phase_learn,)))

   filename = neuron_type + '_neurons-tpr-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('phase', (self.phase_learn,)))

   filename = neuron_type + '_neurons-fpr-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('phase', (self.phase_learn,)))

   filename = neuron_type + '_neurons-accuracy-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('phase', (self.phase_cue,)))

   filename = neuron_type + '_neurons-tpr-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('phase', (self.phase_cue,)))

   filename = neuron_type + '_neurons-fpr-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('phase', (self.phase_cue,)))
   #'''

   '''
   filename = neuron_type + '_neurons-accuracy-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)))

   filename = neuron_type + '_neurons-accuracy-phase_tests-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


   filename = neuron_type + '_neurons-tpr-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)))

   filename = neuron_type + '_neurons-tpr-phase_tests-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


   filename = neuron_type + '_neurons-fpr-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)))

   filename = neuron_type + '_neurons-fpr-phase_tests-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
   '''

   '''
   for pat in range(self.nb_patterns):

    filename = neuron_type + '_neurons-accuracy-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)))

    filename = neuron_type + '_neurons-accuracy-phase_tests-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


    filename = neuron_type + '_neurons-tpr-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)))

    filename = neuron_type + '_neurons-tpr-phase_tests-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


    filename = neuron_type + '_neurons-fpr-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)))

    filename = neuron_type + '_neurons-fpr-phase_tests-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
   '''


 def plot_metrics_run_9(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')

  for neuron_type in neuron_types:

   '''
   filename = neuron_type + '_neurons-accuracy-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', self.colors[:self.nb_patterns], ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('phase', (self.phase_learn,)), ('pattern',('0')))
   '''

   filename = neuron_type + '_neurons-tpr-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'recall (%)', self.colors[:self.nb_patterns], ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('phase', (self.phase_learn,)), ('pattern',('0')))

   '''
   filename = neuron_type + '_neurons-fpr-phase_learn-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', self.colors[:self.nb_patterns], ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('phase', (self.phase_learn,)), ('pattern',('0')))
   '''

   '''
   filename = neuron_type + '_neurons-accuracy-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    print(filename)
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', self.colors, ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('phase', (self.phase_cue,)), ('pattern',('0')))
   '''

   filename = neuron_type + '_neurons-tpr-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'recall (%)', self.colors, ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('phase', (self.phase_cue,)), ('pattern',('0')))

   '''
   filename = neuron_type + '_neurons-fpr-phase_test-pat_ind' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.results.plot('learn_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', self.colors, ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('phase', (self.phase_cue,)), ('pattern',('0')))
   '''


 def plot_metrics_run_10_11(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')
   if self.has_inh2:
    neuron_types.append('inh2')

  for area in self.brain_areas:
   
   for neuron_type in neuron_types:

    nb_neurons = getattr(self, 'nb_' + neuron_type + '_neurons_' + area)
    min_rate = getattr(self, neuron_type + '_min_rate')
   
    for simtime_learn in self.simtimes_learn:
     
     suffix = self.file_prefix_learn + '-' + str(simtime_learn) + extension

     '''
     filename = neuron_type + '_neurons-accuracy-phase_probe-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-tpr-phase_probe-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-fpr-phase_probe-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-accuracy-phase_test-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)), ('pattern',('0')))
     '''

     filename = neuron_type + '_neurons-tpr-phase_test-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)), ('pattern',('0')))

     '''
     filename = neuron_type + '_neurons-fpr-phase_test-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-accuracy-phase_probe_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-tpr-phase_probe_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-fpr-phase_probe_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-accuracy-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-tpr-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-fpr-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn,)), ('pattern',('0')))
     '''

     '''
     filename = neuron_type + '_neurons-avg_ca_rate-phase_probe-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall rate (Hz)', self.colors[:self.nb_patterns], 1, None, min_rate, ('neuron_type',(neuron_type,)), ('metric',('avg_ca_rate',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)), ('pattern',('0-0',)))
     '''

     filename = neuron_type + '_neurons-avg_ca_rate-phase_test-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall rate (Hz)', self.colors, 1, None, min_rate, ('neuron_type',(neuron_type,)), ('metric',('avg_ca_rate',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)), ('pattern',('0-0',)))

     '''
     filename = neuron_type + '_neurons-avg_ca_rate-phase_probe_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall rate (Hz)', self.colors[:self.nb_patterns], 1, None, min_rate, ('neuron_type',(neuron_type,)), ('metric',('avg_ca_rate',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe + '-' + self.phase_learn,)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-avg_ca_rate-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Recall rate (Hz)', self.colors, 1, None, min_rate, ('neuron_type',(neuron_type,)), ('metric',('avg_ca_rate',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn,)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-fraction_activated-phase_probe-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ ∩ probing$^+$\n(% of probing$^+$)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)), ('pattern',('0-0',)))
     '''

     filename = neuron_type + '_neurons-fraction_activated-phase_test-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ ∩ probing$^+$\n(% of probing$^+$)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)), ('pattern',('0-0',)))

     '''
     filename = neuron_type + '_neurons-fraction_activated-phase_probe_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ ∩ training$^+$\n(% of training$^+$)', self.colors[:self.nb_patterns], 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe + '-' + self.phase_learn,)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-fraction_activated-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ ∩ training$^+$\n(% of training$^+$)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn,)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-activated_overlap_0-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ ∩ training$^+$\n(% of training$^+$)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('activated_overlap',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn + '-0',)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-activated_overlap_t-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ ∩ training$^+$\n(% of recall$^+$)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('activated_overlap',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn + '-t',)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_neurons-activated_overlap_f-phase_test_encod-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ ∩ training$^+$\n(% of all neurons)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('activated_overlap',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn + '-f',)), ('pattern',('0-0',)))
     '''
     
     '''
     filename = neuron_type + '_neurons-fraction_activated-phase_test_encod_rev-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ training$^+$\n of recall$^+$ (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn + '-reverse',)), ('pattern',('0-0',)))

     filename = neuron_type + '_neurons-fraction_activated-phase_test_encod_all-pat_ind-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'base_prefix', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'recall$^+$ training$^+$\n of all neurons (%)', self.colors, 100, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('fraction_activated',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue + '-' + self.phase_learn + '-all',)), ('pattern',('0-0',)))
     '''

     '''
     filename = neuron_type + '_ensembles-size_neurons-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', '# of neurons', ['black'] + self.colors[:self.nb_patterns], 1, (0, nb_neurons + 0.05), None, ('neuron_type',(neuron_type,)), ('metric',('size_neurons',)), ('learn_time', (simtime_learn,)))
     '''

     filename = neuron_type + '_ensembles-size_fraction-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Fraction of\n neurons (%)', ['white'] + self.colors[:self.nb_patterns], 1, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('size_fraction',)), ('learn_time', (simtime_learn,)))

     '''
     filename = neuron_type + '_ensembles-seq_growth-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'Ensemble growth\n hour-to-hour (%)', ['black'] + self.colors[:self.nb_patterns], 1, None, None, ('neuron_type',(neuron_type,)), ('metric',('seq_growth',)), ('learn_time', (simtime_learn,)))
     '''

     filename = neuron_type + '_ensembles-overlap_0-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ ∩ training$^+$\n(% of training$^+$)', ['white'] + self.colors[:self.nb_patterns], 1, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('overlap_0',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_ensembles-overlap_t-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ ∩ training$^+$\n(% of probing$^+$)', ['white'] + self.colors[:self.nb_patterns], 1, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('overlap_t',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_ensembles-overlap_f-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ ∩ training$^+$\n(% of all neurons)', ['white'] + self.colors[:self.nb_patterns], 1, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('overlap_f',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_ensembles-seq_overlap-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'Consolidation', 'probing$^+$ overlap\n hour-to-hour (%)', ['white'] + self.colors[:self.nb_patterns], 1, (0, 100.05), None, ('neuron_type',(neuron_type,)), ('metric',('seq_overlap',)), ('learn_time', (simtime_learn,)))

    


 def plot_metrics_run(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')

  for neuron_type in neuron_types:
   
   for simtime_learn in self.simtimes_learn:
    suffix = self.file_prefix_learn + '-' + str(simtime_learn) + extension

    filename = neuron_type + '_neurons-accuracy-phase_all-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,self.phase_cue)))

    filename = neuron_type + '_neurons-tpr-phase_all-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,self.phase_cue)))

    filename = neuron_type + '_neurons-fpr-phase_all-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,self.phase_cue)))

    filename = neuron_type + '_neurons-accuracy-phase_probe-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)))

    filename = neuron_type + '_neurons-tpr-phase_probe-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)))

    filename = neuron_type + '_neurons-fpr-phase_probe-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_probe,)))

    filename = neuron_type + '_neurons-accuracy-phase_test-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)))

    filename = neuron_type + '_neurons-tpr-phase_test-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)))

    filename = neuron_type + '_neurons-fpr-phase_test-pat_ind-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'pattern', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,)))

    '''
    filename = neuron_type + '_neurons-accuracy-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-accuracy-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

    filename = neuron_type + '_neurons-tpr-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-tpr-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

    filename = neuron_type + '_neurons-fpr-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-fpr-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
    '''

    '''
    for pat in range(self.nb_patterns):

     filename = neuron_type + '_neurons-accuracy-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-accuracy-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

     filename = neuron_type + '_neurons-tpr-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-tpr-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

     filename = neuron_type + '_neurons-fpr-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-fpr-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
       self.results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
    '''


 def get_times_clusters(self, area, neuron_type, simtimes, simtime, phase):
  self.times = [] # list of simulation time check points
  self.clusters = [] # list of clusters at each simulation check point; each element is a list of clusters
  for time in getattr(self, simtimes):
   for analyzer in self.analyzers:
    if (analyzer.brain_area == area) and (getattr(analyzer, simtime) == time) and (analyzer.phase == phase):
     self.times.append(time)
     self.clusters.append(analyzer.get_clusters(neuron_type))
  #print('len(times)', len(self.times))
  #print('len(clusters)', len(self.clusters))
  #for i in range(len(self.clusters)):
   #print('times[', i, ']', self.times[i])
   #print('len(clusters[', i, '])', len(self.clusters[i]))


 def get_encod_times(self, simtimes_encod):
  self.encod_times = [] # list of encoding-phase simulation time check points
  for time in getattr(self, simtimes_encod):
   self.encod_times.append(time)

 
 def get_encod_clusters(self, area, neuron_type, simtimes, simtime, phase):
  self.encod_clusters = [] # list of encoding-time clusters; each element is a list of clusters
  for time in getattr(self, simtimes):
   for analyzer in self.analyzers:
    if (analyzer.brain_area == area) and (getattr(analyzer, simtime) == time) and (analyzer.phase == phase):
     self.encod_clusters.append(analyzer.get_encod_clusters(neuron_type))
  #print('len(encod_clusters)', len(self.encod_clusters))
  #for i in range(len(self.encod_clusters)):
   #print('len(encod_clusters[', i, '])', len(self.encod_clusters[i]))


 def get_cluster_size_neurons(self):
  # cluster_size_neurons: 0 idx: pat / 1 idx: time-ordered list of cluster size
  self.cluster_size_neurons = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.clusters)):
   clstrs = self.clusters[time_idx]
   for cluster,size in zip(clstrs,self.cluster_size_neurons):
    size.append(len(cluster))
  #print('len(cluster_size_neurons)', len(self.cluster_size_neurons))
  #for i in range(len(self.cluster_size_neurons)):
   #print('len(cluster_size_neurons[', i, '])', len(self.cluster_size_neurons[i]))
  #print('cluster_size_neurons')
  #print(self.cluster_size_neurons)


 def get_encod_cluster_size_neurons(self):
  # encod_cluster_size_neurons: 0 idx: pat / 1 idx: time-ordered list of encoding-time cluster size
  self.encod_cluster_size_neurons = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.encod_clusters)):
   clstrs = self.encod_clusters[time_idx]
   for cluster,size in zip(clstrs,self.encod_cluster_size_neurons):
    size.append(len(cluster))
  #print('len(encod_cluster_size_neurons)', len(self.encod_cluster_size_neurons))
  #for i in range(len(self.encod_cluster_size_neurons)):
  # print('len(encod_cluster_size_neurons[', i, '])', len(self.encod_cluster_size_neurons[i]))
  #print('encod_cluster_size_neurons')
  #print(self.encod_cluster_size_neurons)


 def get_cluster_size_fraction(self, area, neuron_type):
  # cluster_size_fraction: 0 idx -> pat / 1 idx: time-ordered list of cluster size fraction
  nb_neurons = getattr(self, 'nb_' + neuron_type + '_neurons_' + area)
  self.cluster_size_fraction = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.clusters)):
   clstrs = self.clusters[time_idx]
   for cluster,size in zip(clstrs,self.cluster_size_fraction):
    size.append(len(cluster) / float(nb_neurons) * 100)
  #print('len(cluster_size_fraction)', len(self.cluster_size_fraction))
  #for i in range(len(self.cluster_size_fraction)):
  # print('len(cluster_size_fraction[', i, '])', len(self.cluster_size_fraction[i]))
  #print('cluster_size_fraction')
  #print(self.cluster_size_fraction)


 def get_encod_cluster_size_fraction(self, area, neuron_type):
  # encod_cluster_size_fraction: 0 idx -> pat / 1 idx: time-ordered list of encoding-time cluster size fraction
  nb_neurons = getattr(self, 'nb_' + neuron_type + '_neurons_' + area)
  self.encod_cluster_size_fraction = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.encod_clusters)):
   clstrs = self.encod_clusters[time_idx]
   for cluster,size in zip(clstrs,self.encod_cluster_size_fraction):
    size.append(len(cluster) / float(nb_neurons) * 100)
  #print('len(encod_cluster_size_fraction)', len(self.encod_cluster_size_fraction))
  #for i in range(len(self.encod_cluster_size_fraction)):
  # print('len(encod_cluster_size_fraction[', i, '])', len(self.encod_cluster_size_fraction[i]))
  #print('encod_cluster_size_fraction')
  #print(self.encod_cluster_size_fraction)


 '''
 def get_cluster_overlap_growth(self, clusters):
  # overlaps: 0 idx -> pat; 0/1 -> overlap/growth
  overlaps = [[[],[]] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(clusters) - 1):
   clusters1 = clusters[time_idx]
   clusters2 = clusters[time_idx + 1]
   for cluster1,cluster2,overlap in zip(clusters1,clusters2,overlaps):
    overlap[0].append(self.compute_cluster_overlap(cluster1, cluster2))
    overlap[1].append(self.compute_cluster_growth(cluster1, cluster2))
  return overlaps
 '''
 
 def compute_cluster_overlap(self, cluster1, cluster2):
  if len(cluster1) == 0:
   return 0
  return get_size_intersection(cluster1, cluster2) / float(len(cluster1))


 def compute_cluster_growth(self, cluster1, cluster2):
  if len(cluster1) == 0:
   return 0
  return len(cluster2) / float(len(cluster1))


 def get_sequential_cluster_growth(self):
  # sequential_cluster_growth: 0 idx -> pat / 1 idx: time-ordered list of sequential cluster growth
  self.sequential_cluster_growth = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.clusters) - 1):
   clusters1 = self.clusters[time_idx]
   clusters2 = self.clusters[time_idx + 1]
   for cluster1,cluster2,growth in zip(clusters1,clusters2,self.sequential_cluster_growth):
    growth.append(self.compute_cluster_growth(cluster1, cluster2) * 100)


 def get_sequential_cluster_overlap(self):
  # sequential_cluster_overlap: 0 idx -> pat / 1 idx: time-ordered list of sequential cluster overlap
  self.sequential_cluster_overlap = [[] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(self.clusters) - 1):
   clusters1 = self.clusters[time_idx]
   clusters2 = self.clusters[time_idx + 1]
   for cluster1,cluster2,overlap in zip(clusters1,clusters2,self.sequential_cluster_overlap):
    overlap.append(self.compute_cluster_overlap(cluster1, cluster2) * 100)

 
 def get_cluster_overlap_intersection(self, relative_time):
  # cluster_overlap: 0 idx -> pat / 1 idx: time-ordered list of cluster overlap wrt relative_time
  cluster_overlap = [[] for _ in range(self.nb_patterns + 1)]
  cluster_intersection = [[] for _ in range(self.nb_patterns + 1)]

  #print('cluster_intersection', type(cluster_intersection), len(cluster_intersection))
  #for c,i  in enumerate(cluster_intersection):
  # print(c, type(i), len(i))
   
  if type(self.encod_clusters) != type(None):
   clstrs0 = self.encod_clusters[0]
  else:
   clstrs0 = self.clusters[0]
  for time_idx in range(len(self.clusters)):
   clstrs = self.clusters[time_idx]
   for cluster0,cluster,overlap,intersection in zip(clstrs0,clstrs,cluster_overlap,cluster_intersection):
    intersection.append(cluster)
    
    #print('cluster_intersection', type(cluster_intersection), len(cluster_intersection))
    #for c,i  in enumerate(cluster_intersection):
    # print(c, type(i), len(i))
    # for cl in i:
    #  print('len(cluster)', len(cl))
     
    if relative_time == '0':
     overlap.append(self.compute_cluster_overlap(cluster0, cluster) * 100)
    elif relative_time == 't':
     overlap.append(self.compute_cluster_overlap(cluster, cluster0) * 100)
    elif relative_time == 'f':
     overlap.append(get_size_intersection(cluster, cluster0) / float(self.nb_neurons) * 100)

  clt_inter = []
  #print('Computation of intersection')
  for pat,clusters in enumerate(cluster_intersection):
   #print('Start cluster')
   inter = set(clstrs0[pat])
   #print('len(inter)', len(inter))
   #print('Intersection')
   for cluster in clusters:
    inter = inter & set(cluster)
    #print('len(inter)', len(inter))
   clt_inter.append(len(inter) / float(len(clstrs0[pat])) * 100)

  
  #for cluster0,intersection in zip(clstrs0,cluster_intersection):
  # intersection = len(intersection) / float(len(cluster0)) * 100
   
  #print('cluster_overlap relative_time:', relative_time)
  #print('len(cluster_overlap)', len(cluster_overlap))
  #for i in range(len(cluster_overlap)):
  # print('len(cluster_overlap[', i, '])', len(cluster_overlap[i]))
  #print('cluster_overlap')
  #print(cluster_overlap)
  
  #print('clt_inter', type(clt_inter), len(clt_inter))
  #for c,i  in enumerate(clt_inter):
  # print(c, type(i), i)
  #sys.exit(0)
  return cluster_overlap,clt_inter


 def plot_cluster_size_neurons(self, times, sizes, xlabel, area, neuron_type, nb_neurons, datadir, filename):
  
  plt.rcParams.update({'font.size': 7})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
  #fig = plt.figure()
  fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
  #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

  for cluster in range(self.nb_patterns + 1):
   if cluster == 0:
    plt.plot(times, sizes[cluster], color='black')
   else:
    plt.plot(times, sizes[cluster], color=self.colors[cluster - 1])

  if type(self.encod_cluster_size_neurons) != type(None):
   for encod_cluster in range(self.nb_patterns + 1):
    if encod_cluster == 0:
     plt.axhline(y=self.encod_cluster_size_neurons[encod_cluster][0], color='black', linestyle='--')
    else:
     plt.axhline(y=self.encod_cluster_size_neurons[encod_cluster][0], color=self.colors[encod_cluster - 1], linestyle='--')
  
  sns.despine()
  plt.xlabel(xlabel)
  plt.ylabel('# of neurons')
  plt.ylim((0,nb_neurons + 0.05))
  #plt.xlim((-0.05, 12.05))
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s cluster size neurons"%(area, neuron_type))
  plt.close(fig)


 def plot_cluster_size_fraction(self, times, sizes, xlabel, area, neuron_type, nb_neurons, datadir, filename):
  
  plt.rcParams.update({'font.size': 7})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
  #fig = plt.figure()
  fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
  #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

  for cluster in range(self.nb_patterns + 1):
   if cluster == 0:
    plt.plot(times, sizes[cluster], color='black')
   else:
    plt.plot(times, sizes[cluster], color=self.colors[cluster - 1])

  if type(self.encod_cluster_size_fraction) != type(None):
   for encod_cluster in range(self.nb_patterns + 1):
    if encod_cluster == 0:
     plt.axhline(y=self.encod_cluster_size_fraction[encod_cluster][0], color='black', linestyle='--')
    else:
     plt.axhline(y=self.encod_cluster_size_fraction[encod_cluster][0], color=self.colors[encod_cluster - 1], linestyle='--')
  
  sns.despine()
  plt.xlabel(xlabel)
  plt.ylabel('fraction of neurons (%)')
  plt.ylim((0,100.05))
  #plt.xlim((-0.05, 12.05))
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s cluster size fraction"%(area, neuron_type))
  plt.close(fig)


 def plot_sequential_cluster_growth(self, times, growth, xlabel, area, neuron_type, datadir, filename):

  plt.rcParams.update({'font.size': 7})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
  #fig = plt.figure()
  fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
  #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
  for cluster in range(self.nb_patterns + 1):
   if cluster == 0:
    plt.plot(times, growth[cluster], color='black')
   else:
    plt.plot(times, growth[cluster], color=self.colors[cluster - 1])
  
  sns.despine()
  plt.xlabel(xlabel)
  plt.ylabel('ensemble growth\n hour-to-hour (%)')
  #plt.ylim((0,100.05))
  #plt.xlim((-0.05, 12.05))
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s sequential cluster growth"%(area, neuron_type))
  plt.close(fig)


 def plot_cluster_overlap_rt(self, times, overlap, intersection, relative_time, xlabel, area, neuron_type, nb_neurons, datadir, filename):
  
  plt.rcParams.update({'font.size': 7})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
  #fig = plt.figure()
  fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
  #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)

  for cluster in range(self.nb_patterns + 1):
   if cluster == 0:
    plt.plot(times, overlap[cluster], color='black')
    #plt.hlines(intersection[cluster], times[0], times[-1], color='black', linestyle='dashed')
   else:
    plt.plot(times, overlap[cluster], color=self.colors[cluster - 1])
    #plt.hlines(intersection[cluster], times[0], times[-1], color=self.colors[cluster - 1], linestyle='dashed')
  
  sns.despine()
  plt.xlabel(xlabel)
  if relative_time == '0':
   plt.ylabel('training$^+$ probing$^+$\n of training$^+$ (%)')
  elif relative_time == 't':
   plt.ylabel('training$^+$ probing$^+$\n of probing$^+$ (%)')
  elif relative_time == 'f':
   plt.ylabel('training$^+$ probing$^+$\n of all neurons (%)') 
  plt.ylim((0,100.05))
  #plt.xlim((-0.05, 12.05))
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s cluster overlap simtime %s"%(area, neuron_type, relative_time))
  plt.close(fig)


 def plot_sequential_cluster_overlap(self, times, overlap, intersection, xlabel, area, neuron_type, datadir, filename):

  plt.rcParams.update({'font.size': 7})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})
  
  #fig = plt.figure()
  fig = plt.figure(figsize=(2.24, 1.68), dpi=300)
  #fig = plt.figure(figsize=(1.68, 1.26), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
  for cluster in range(self.nb_patterns + 1):
   if cluster == 0:
    plt.plot(times, overlap[cluster], color='black')
    plt.hlines(intersection[cluster], times[0], times[-1], color='black', linestyle='dashed')
   else:
    plt.plot(times, overlap[cluster], color=self.colors[cluster - 1])
    plt.hlines(intersection[cluster], times[0], times[-1], color=self.colors[cluster - 1], linestyle='dashed')
  
  sns.despine()
  plt.xlabel(xlabel)
  plt.ylabel('probing$^+$ overlap\n hour-to-hour (%)')
  plt.ylim((0,100.05))
  #plt.xlim((-0.05, 12.05))
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s hour-to-hour cluster overlap"%(area, neuron_type))
  plt.close(fig)


 def analyze_clusters(self, area, neuron_type):
  has_multiple_clusters = True
  if self.run in [1,9]:
   simtimes = 'simtimes_learn'
   simtime = 'simtime_learn'
   phase = self.phase_learn
   base_prefix = self.file_prefix_learn
   xlabel = 'training'
   has_encod = False
  elif self.run in [2,3,7,8,10,11]:
   simtimes = 'simtimes_consolidation'
   simtime = 'simtime_consolidation'
   phase = self.phase_probe
   base_prefix = self.file_prefix_probe
   xlabel = 'consolidation'
   has_encod = True
   simtimes_encod = 'simtimes_learn'
   simtime_encod = 'simtime_learn'
   phase_encod = self.phase_learn
   base_prefix_encod = self.file_prefix_learn
   xlabel_encod = 'training'
  else:
   has_multiple_clusters = False

  if has_multiple_clusters:
   extension = '.svg'
   trial = self.random_trials[0]
   datadir = os.path.join(self.rundir, trial)
   
   size_neurons_filename = area + '-' + neuron_type + '-cluster-size-neurons-' + phase + extension
   size_fraction_filename = area + '-' + neuron_type + '-cluster-size-fraction-' + phase + extension
   size_growth_sequential_filename = area + '-' + neuron_type + '-cluster-size-growth-sequential-' + phase + extension
   
   overlap_simtime_0_filename = area + '-' + neuron_type + '-cluster-overlap-simtime_0-' + phase + extension
   overlap_simtime_t_filename = area + '-' + neuron_type + '-cluster-overlap-simtime_t-' + phase + extension
   overlap_fraction_filename = area + '-' + neuron_type + '-cluster-overlap-fraction-' + phase + extension
   overlap_sequential_filename = area + '-' + neuron_type + '-cluster-overlap-sequential-' + phase + extension

   has_cluster_plots = True
   for filename in [size_neurons_filename, size_fraction_filename, size_growth_sequential_filename, overlap_simtime_0_filename, overlap_simtime_t_filename, overlap_fraction_filename, overlap_sequential_filename]:
    if not os.path.isfile(os.path.join(datadir, filename)):
     has_cluster_plots = False
   
   if (self.has_plots and self.has_cell_assemb_plots and (not has_cluster_plots)) or (self.has_metrics and (not self.has_metrics_file)):
                                                                        
    print ("Analyzing %s %s cluster dynamics..."%(area, neuron_type))

    self.nb_neurons = getattr(self, 'nb_' + neuron_type + '_neurons_' + area)
    
    self.get_times_clusters(area, neuron_type, simtimes, simtime, phase)
                                                       
    self.get_cluster_size_neurons()

    self.get_cluster_size_fraction(area, neuron_type)

    self.get_sequential_cluster_growth()

    if has_encod:
     self.get_encod_times(simtimes_encod)
     self.get_encod_clusters(area, neuron_type, simtimes_encod, simtime_encod, phase_encod)
     self.get_encod_cluster_size_neurons()
     self.get_encod_cluster_size_fraction(area, neuron_type)
    else:
     self.encod_times = None
     self.encod_clusters = None
     self.encod_cluster_size_neurons = None
     self.encod_cluster_size_fraction = None
    
    self.overlap_0,self.intersection_0 = self.get_cluster_overlap_intersection('0')

    self.overlap_t,self.intersection_t = self.get_cluster_overlap_intersection('t')

    self.overlap_f,self.intersection_f = self.get_cluster_overlap_intersection('f')

    self.get_sequential_cluster_overlap()

   if (self.has_plots and self.has_cell_assemb_plots and (not has_cluster_plots)):

    plot_times = np.array(self.times) / 3600.0

    plot_xlabel = xlabel + ' time (h)'

    nb_neurons = getattr(self, 'nb_' + neuron_type + '_neurons_' + area)

    '''
    self.plot_cluster_size_neurons(plot_times, self.cluster_size_neurons, plot_xlabel, area, neuron_type, nb_neurons, datadir, size_neurons_filename)
    '''

    self.plot_cluster_size_fraction(plot_times, self.cluster_size_fraction, plot_xlabel, area, neuron_type, nb_neurons, datadir, size_fraction_filename)

    '''
    self.plot_sequential_cluster_growth(plot_times[1:], self.sequential_cluster_growth, plot_xlabel, area, neuron_type, datadir, size_growth_sequential_filename)
    '''

    self.plot_cluster_overlap_rt(plot_times, self.overlap_0, self.intersection_0, '0', plot_xlabel, area, neuron_type, nb_neurons, datadir, overlap_simtime_0_filename)

    self.plot_cluster_overlap_rt(plot_times, self.overlap_t, self.intersection_t, 't', plot_xlabel, area, neuron_type, nb_neurons, datadir, overlap_simtime_t_filename)

    self.plot_cluster_overlap_rt(plot_times, self.overlap_f, self.intersection_f, 'f', plot_xlabel, area, neuron_type, nb_neurons, datadir, overlap_fraction_filename)
                                                                        
    self.plot_sequential_cluster_overlap(plot_times[1:], self.sequential_cluster_overlap, self.intersection_0, plot_xlabel, area, neuron_type, datadir, overlap_sequential_filename)

   if (self.has_metrics and (not self.has_metrics_file)):

    for cluster in range(self.nb_patterns + 1):
     pattern = cluster - 1
     
     for time,size_neurons,size_fraction,overlap0,overlapt,overlapf in zip(self.times,self.cluster_size_neurons[cluster],self.cluster_size_fraction[cluster],self.overlap_0[cluster],self.overlap_t[cluster],self.overlap_f[cluster]):
      if has_encod:
       simtime_learn = self.encod_times[0]
       simtime_consolidation = time
      else:
       simtime_learn = time
       simtime_consolidation = 0
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'size_neurons', size_neurons)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'size_fraction', size_fraction)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'overlap_0', overlap0)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'overlap_t', overlapt)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'overlap_f', overlapf)
      if has_encod:
       self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase_encod, base_prefix_encod, pattern, 'size_neurons', self.encod_cluster_size_neurons[cluster][0])
       self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase_encod, base_prefix_encod, pattern, 'size_fraction', self.encod_cluster_size_fraction[cluster][0])
       #self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase_encod, base_prefix_encod, pattern, 'overlap_0', self.intersection_0[cluster])
       #self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase_encod, base_prefix_encod, pattern, 'overlap_t', self.intersection_t[cluster])
      
     for time,seq_growth,seq_overlap in zip(self.times[1:],self.sequential_cluster_growth[cluster],self.sequential_cluster_overlap[cluster]):
      if has_encod:
       simtime_learn = self.encod_times[0]
       simtime_consolidation = time
      else:
       simtime_learn = time
       simtime_consolidation = 0
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'seq_growth', seq_growth)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, 'seq_overlap', seq_overlap)
      self.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase_encod, base_prefix_encod, pattern, 'seq_overlap', self.intersection_0[cluster])


 def add_result(self, trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, metric, score):
  self.results.add_result(trial, area, neuron_type, simtime_learn, simtime_consolidation, phase, base_prefix, pattern, metric, score)

  
 def analyze_dynamics(self):
  print ("\n\nAnalyzing dynamics...")
  for area in self.brain_areas:
   self.analyze_clusters(area, 'exc')
   if self.has_inh_analysis:
    self.analyze_clusters(area, 'inh')
    if self.has_inh2:
     self.analyze_clusters(area, 'inh2')


 def plot_area_area_weights(self):
  for analyzer in self.analyzers:
   if (analyzer.simtime > 0):
    print ("\nRunning Parser Cross-Area Weight Plots...")
    analyzer.print_identifier()
    for other_area in analyzer.other_areas:
     other_area_ana = self.get_analyzer(other_area, analyzer.full_prefix)
     #if (self.file_prefix_hm not in analyzer.full_prefix):
     #if (self.file_prefix_hm in chosen_prefix) and (self.brain_area == 'hpc' or other_area == 'hpc'):
     # continue
     if type(other_area_ana) != type(None):
      prefixes = [analyzer.full_prefix]
      if analyzer.phase == 'burn-in':
       prefixes.append(analyzer.prefix)
      for prefix in prefixes:
       print(prefix + ':')
       analyzer.plot_weights(prefix,
                             self.weights,
                             has_rec_plots=False,
                             has_stim_brain_area_plots=False,
                             has_area_area_plots=True,
                             other_area=other_area,
                             other_area_ana=other_area_ana)


 def plot_incoming_weights(self):
  for analyzer in self.analyzers:
   if analyzer.simtime > 0:
    print ("\nRunning Parser Incoming Weights Plots...")
    analyzer.print_identifier()

    other_area_anas = []
    for other_area in analyzer.other_areas:
     other_area_anas.append(self.get_analyzer(other_area, analyzer.full_prefix))
      
    #if self.file_prefix_hm not in analyzer.full_prefix:
    # other_area_ana = self.get_analyzer(analyzer.other_areas, analyzer.full_prefix)
    #else:
    # other_area_ana = None
    
    prefixes = [analyzer.full_prefix]
    if analyzer.phase == 'burn-in':
     prefixes.append(analyzer.prefix)
    for prefix in prefixes:
     print(prefix + ':')
     analyzer.plot_incoming_weights(prefix, other_area_anas)


 def plot_weights(self):
  print ("\n\nRunning Parser Weight Plots...")
  self.plot_area_area_weights()
  #self.plot_incoming_weights()


 def merge_metrics(self):
  print ('Merging individual trial metrics...')
  frames = []
  for metrics in ['metrics-'+ str(i) + '.csv' for i in self.random_trials]:
   print (metrics)
   #frames.append(pd.read_csv(os.path.join(self.rundir, metrics)))
   frames.append(pd.read_csv(os.path.join(self.rundir, metrics), low_memory=False))
  metrics_all = pd.concat(frames, ignore_index=True)
  metrics_all.to_csv(os.path.join(self.rundir, self.metrics_filename))


 def analyze(self):
  '''
  List of saved files:
  - metrics (*.metrics)
  '''

  if len(self.random_trials) == 1:
   for analyzer in self.analyzers:
    analyzer.analyze(self.results, self.weights)
   self.analyze_dynamics()
   if (self.has_plots and self.has_weight_plots):
    self.plot_weights()
   
  if self.has_metrics:
   if self.has_metrics_file:
    self.results.load_df(self.rundir, self.metrics_filename)
   else:
    if len(self.random_trials) == 1:
     self.save_results()
    else:
     self.merge_metrics()
    self.results.load_df(self.rundir, self.metrics_filename)

  if self.has_metrics and self.has_metrics_plots:
   self.plot_metrics()
