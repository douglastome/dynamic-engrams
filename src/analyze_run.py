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

#!/usr/bin/env python
# coding: utf-8

import sys
import os
import getopt

#from random import sample
#from scipy.sparse import *
#from copy import deepcopy

from parser import Parser
from helper import *
    
def main(argv):
 # Simulation and Analysis Parameters
 run = 1
 has_inh_analysis = False
 has_metrics = False
 has_plots = False
 has_neuron_vi_plots = False
 has_cell_assemb_plots = False
 has_activity_stats = False
 has_activity_plots = False
 has_spike_raster_stats = False
 has_spike_stats_plots = False
 has_metrics_plots = False
 has_rates_across_stim_plots = False
 has_weight_plots = False
 has_rf_plots = False
 conf_int = 68
 num_mpi_ranks = 16
 integration_time_step = 0.0001
 rundir = os.path.expanduser("~")
 random_trials = []
 brain_areas = []
 prefixes_thl = []
 prefixes_ctx = []
 prefixes_hpc = []
 file_prefix_burn = 'rfb'
 file_prefix_learn = 'rfl'
 file_prefix_learn_bkw = 'rfl-bkw'
 file_prefix_consolidation = 'rfs'
 file_prefix_probe ='rfp'
 file_prefix_cue = 'rfc'
 file_prefix_hm = 'hm'
 file_prefix_burn_hm = 'rfb-hm'
 file_prefix_learn_hm = 'rfl-hm'
 file_prefix_consolidation_hm = 'rfs-hm'
 file_prefix_probe_hm ='rfp-hm'
 file_prefix_cue_hm = 'rfc-hm'
 phase_burn = 'burn-in'
 phase_learn = 'learn'
 phase_consolidation = 'consolidation'
 phase_probe = 'probe'
 phase_cue = 'test'
 nb_stim_learn = 4
 nb_stim_probe = 4
 nb_stim_cue = 16
 nb_exc_neurons_thl = 4096
 exc_inh_thl = 4
 nb_exc_neurons_ctx = 4096
 exc_inh_ctx = 4
 nb_exc_neurons_hpc = 4096
 exc_inh_hpc = 8
 exc_inh2_hpc = 8
 nb_neurons_stim_replay = 1
 nb_neuron_cons_stim_replay = 0.01
 nb_patterns = 4
 nb_test_cue_sim = 1
 simtime_burn = 120
 simtimes_learn = []
 simtime_learn_bkw = 1800
 simtimes_consolidation = []
 simtime_probe = 600
 simtime_cue = 300
 range_burn = 120
 range_learn = 600
 range_consolidation = 600
 range_probe = 600
 range_cue = 300
 zoom_range = []
 nb_plot_neurons = 256
 colors = []
 bin_size = 10e-3
 has_stim_hpc = False
 has_stim_ctx = False
 has_stim_thl = True
 has_thl_ctx = False
 has_ctx_thl = False
 has_thl_hpc = False
 has_hpc_thl = False
 has_ctx_hpc = False
 has_hpc_ctx = False
 has_hpc_rep = False
 has_rec_ee = True
 has_inh2 = False
 ids_cell_assemb = True
 cell_assemb_method = 'rate'
 exc_min_rate = 10
 inh_min_rate = 10
 inh2_min_rate = 10
 min_weight = 0.5
 stim_pat_learn = {}
 stim_pat_probe = {}
 stim_pat_cue = {}
 exc_record_rank_thl = 1
 inh_record_rank_thl = 1
 exc_ampa_nmda_ratio_thl = 0.2
 inh_ampa_nmda_ratio_thl = 0.3
 exc_record_rank_ctx = 1
 inh_record_rank_ctx = 1
 exc_ampa_nmda_ratio_ctx = 0.2
 inh_ampa_nmda_ratio_ctx = 0.3
 exc_record_rank_hpc = 1
 inh_record_rank_hpc = 1
 inh2_record_rank_hpc = 1
 exc_ampa_nmda_ratio_hpc = 0.2
 inh_ampa_nmda_ratio_hpc = 0.3
 inh2_ampa_nmda_ratio_hpc = 0.3
 u_rest = -0.07
 u_exc = 0
 u_inh = -0.08

 help_msg = 'analyze_run.py --run <run> --has_inh_analysis <has_inh_analysis> --has_metrics  <has_metrics> --has_plots <has_plots> --has_neuron_vi_plots <has_neuron_vi_plots> --has_cell_assemb_plots <has_cell_assemb_plots> --has_activity_stats <has_activity_stats> --has_activity_plots <has_activity_plots> --has_spike_raster_stats <has_spike_raster_stats> --has_spike_stats_plots <has_spike_stats_plots> --has_metrics_plots <has_metrics_plots> --has_rates_across_stim_plots <has_rates_across_stim_plots> --has_weight_plots <has_weight_plots> --has_rf_plots <has_rf_plots> --conf_int <conf_int> --n_ranks <num_mpi_ranks> --time_step <integration_time_step> --rundir <rundir> --trials <random_trials> --brain_areas <brain_areas> --prefixes_thl <prefixes_thl> --prefixes_ctx <prefixes_ctx> --prefixes_hpc <prefixes_hpc> --file_prefix_burn <file_prefix_burn> --file_prefix_learn <file_prefix_learn> --file_prefix_learn_bkw <file_prefix_learn_bkw> --file_prefix_consolidation <file_prefix_consolidation> --file_prefix_probe <file_prefix_probe> --file_prefix_cue <file_prefix_cue> --file_prefix_hm --file_prefix_burn_hm <file_prefix_burn_hm> --file_prefix_learn_hm <file_prefix_learn_hm> --file_prefix_consolidation_hm <file_prefix_consolidation_hm> --file_prefix_probe_hm <file_prefix_probe_hm> --file_prefix_cue_hm <file_prefix_cue_hm> --phase_burn <phase_burn> --phase_learn <phase_learn> --phase_consolidation <phase_consolidation> --phase_probe <phase_probe> --phase_cue <phase_cue> --n_stim_learn <nb_stim_learn> --n_stim_probe <nb_stim_probe> --n_stim_cue <nb_stim_cue> --exc_size_thl <nb_exc_neurons_thl> --exc_inh_thl <exc_inh_thl> --exc_size_ctx <nb_exc_neurons_ctx> --exc_inh_ctx <exc_inh_ctx> --exc_size_hpc <nb_exc_neurons_hpc> --exc_inh_hpc <exc_inh_hpc> --exc_inh2_hpc <exc_inh2_hpc> --stim_replay <nb_neurons_stim_replay> --cons_stim_replay <nb_neuron_cons_stim_replay> --n_patterns <nb_patterns> --n_test_cue_sim <nb_test_cue_simulations> --simtime_burn <simtime_burn> --simtimes_learn <simtimes_learn> --simtime_learn_bkw <simtime_learn_bkw> --simtimes_consolidation <simtimes_consolidation> --simtime_probe <simtime_probe> --simtime_cue <simtime_cue> --range_burn <range_burn> --range_learn <range_learn> --range_consolidation <range_consolidation> --range_probe <range_probe> --range_cue <range_cue> --zoom_range <zoom_range> --n_plot_neurons <nb_plot_neurons> --colors <colors> --bin <bin_size> --has_stim_hpc <has_stim_hpc> --has_stim_ctx <has_stim_ctx> --has_stim_thl <has_stim_thl> --has_thl_ctx <has_thl_ctx> --has_ctx_thl <has_ctx_thl> --has_thl_hpc <has_thl_hpc> --has_hpc_thl <has_hpc_thl> --has_ctx_hpc <has_ctx_hpc> --has_hpc_ctx <has_hpc_ctx> --has_hpc_rep <has_hpc_rep> --has_rec_ee <has_rec_ee> --has_inh2 <has_inh2> --ids_cell_assemb <ids_cell_assemb> --cell_assemb_method <cell_assemb_method> --exc_min_rate <exc_min_rate> --inh_min_rate <inh_min_rate> --inh2_min_rate <inh2_min_rate> --min_weight <min_weight> --stim_pat_learn <stim_pat_learn> --stim_pat_probe <stim_pat_probe> --stim_pat_cue <stim_pat_cue> --exc_record_rank_thl <exc_record_rank_thl> --inh_record_rank_thl <inh_record_rank_thl> --exc_ampa_nmda_ratio_thl <exc_ampa_nmda_ratio_thl> --inh_ampa_nmda_ratio_thl <inh_ampa_nmda_ratio_thl> --exc_record_rank_ctx <exc_record_rank_ctx> --inh_record_rank_ctx <inh_record_rank_ctx> --exc_ampa_nmda_ratio_ctx <exc_ampa_nmda_ratio_ctx> --inh_ampa_nmda_ratio_ctx <inh_ampa_nmda_ratio_ctx> --exc_record_rank_hpc <exc_record_rank_hpc> --inh_record_rank_hpc <inh_record_rank_hpc> --inh2_record_rank_hpc <inh2_record_rank_hpc> --exc_ampa_nmda_ratio_hpc <exc_ampa_nmda_ratio_hpc> --inh_ampa_nmda_ratio_hpc <inh_ampa_nmda_ratio_hpc> --inh2_ampa_nmda_ratio_hpc <inh2_ampa_nmda_ratio_hpc> --u_rest <u_rest> --u_exc <u_exc> --u_inh <u_inh>'
 
 # Parse command-line arguments
 try:
     opts, args = getopt.getopt(argv,"",["script","run=","has_inh_analysis=","has_metrics=","has_plots=","has_neuron_vi_plots=","has_cell_assemb_plots=","has_activity_stats=","has_activity_plots=","has_spike_raster_stats=","has_spike_stats_plots=","has_metrics_plots=","has_rates_across_stim_plots=","has_weight_plots=","has_rf_plots=","conf_int=","n_ranks=","time_step=","rundir=","trials=","brain_areas=","prefixes_thl=","prefixes_ctx=","prefixes_hpc=","file_prefix_burn=","file_prefix_learn=","file_prefix_learn_bkw=","file_prefix_consolidation=","file_prefix_probe=","file_prefix_cue=","file_prefix_hm=","file_prefix_burn_hm=","file_prefix_learn_hm=","file_prefix_consolidation_hm=","file_prefix_probe_hm=","file_prefix_cue_hm=","phase_burn=","phase_learn=","phase_consolidation=","phase_probe=","phase_cue=","n_stim_learn=","n_stim_probe=","n_stim_cue=","exc_size_thl=","exc_inh_thl=","exc_size_ctx=","exc_inh_ctx=","exc_size_hpc=","exc_inh_hpc=","exc_inh2_hpc=","stim_replay=","cons_stim_replay=","n_patterns=","n_test_cue_sim=","simtime_burn=","simtimes_learn=","simtime_learn_bkw=","simtimes_consolidation=","simtime_probe=","simtime_cue=","range_burn=","range_learn=","range_consolidation=","range_probe=","range_cue=","zoom_range=","n_plot_neurons=","colors=","bin=","has_stim_hpc=","has_stim_ctx=","has_stim_thl=","has_thl_ctx=","has_ctx_thl=","has_thl_hpc=","has_hpc_thl=","has_ctx_hpc=","has_hpc_ctx=","has_hpc_rep=","has_rec_ee=","has_inh2=","ids_cell_assemb=","cell_assemb_method=","exc_min_rate=","inh_min_rate=","inh2_min_rate=","min_weight=","stim_pat_learn=","stim_pat_probe=","stim_pat_cue=","exc_record_rank_thl=","inh_record_rank_thl=","exc_ampa_nmda_ratio_thl=","inh_ampa_nmda_ratio_thl=","exc_record_rank_ctx=","inh_record_rank_ctx=","exc_ampa_nmda_ratio_ctx=","inh_ampa_nmda_ratio_ctx=","exc_record_rank_hpc=","inh_record_rank_hpc=","inh2_record_rank_hpc=","exc_ampa_nmda_ratio_hpc=","inh_ampa_nmda_ratio_hpc=","inh2_ampa_nmda_ratio_hpc=","u_rest=","u_exc=","u_inh="])
 except getopt.GetoptError:
     print (help_msg)
     sys.exit(2)
 #print ('number of command-line arguments:', len(opts))
 for opt, arg in opts:
     #print ('opt, arg: ', opt, ', ', arg) 
     if opt == '--script':
         print (help_msg)
         sys.exit()
     elif opt == "--run":
         run = int(arg)
     elif opt == "--has_inh_analysis":
         has_inh_analysis = arg.lower() == 'true'
     elif opt == "--has_metrics":
         has_metrics = arg.lower() == 'true'
     elif opt == "--has_plots":
         has_plots = arg.lower() == 'true'
     elif opt == "--has_neuron_vi_plots":
         has_neuron_vi_plots = arg.lower() == 'true'
     elif opt == "--has_cell_assemb_plots":
         has_cell_assemb_plots = arg.lower() == 'true'
     elif opt == "--has_activity_stats":
         has_activity_stats = arg.lower() == 'true'
     elif opt == "--has_activity_plots":
         has_activity_plots = arg.lower() == 'true'
     elif opt == "--has_spike_raster_stats":
         has_spike_raster_stats = arg.lower() == 'true'
     elif opt == "--has_spike_stats_plots":
         has_spike_stats_plots = arg.lower() == 'true'
     elif opt == "--has_metrics_plots":
         has_metrics_plots = arg.lower() == 'true'
     elif opt == "--has_rates_across_stim_plots":
         has_rates_across_stim_plots = arg.lower() == 'true'
     elif opt == "--has_weight_plots":
         has_weight_plots = arg.lower() == 'true'
     elif opt == "--has_rf_plots":
         has_rf_plots = arg.lower() == 'true'
     elif opt == "--conf_int":
         conf_int = int(arg)
     elif opt == "--n_ranks":
         num_mpi_ranks = int(arg)
     elif opt == "--time_step":
         integration_time_step = float(arg)
     elif opt == "--rundir":
         rundir = os.path.expanduser(arg)
     elif opt == "--trials":
      random_trials = arg.split(',')
     elif opt == "--brain_areas":
      brain_areas = arg[1:-1].split(',')
     elif opt == "--prefixes_thl":
         prefixes_thl = arg[1:-1].split(',')
     elif opt == "--prefixes_ctx":
         prefixes_ctx = arg[1:-1].split(',')
     elif opt == "--prefixes_hpc":
         prefixes_hpc = arg[1:-1].split(',')
     elif opt == "--file_prefix_burn":
      file_prefix_burn = arg
     elif opt == "--file_prefix_learn":
      file_prefix_learn = arg
     elif opt == "--file_prefix_learn_bkw":
      file_prefix_learn_bkw = arg
     elif opt == "--file_prefix_consolidation":
      file_prefix_consolidation = arg
     elif opt == "--file_prefix_probe":
      file_prefix_probe = arg
     elif opt == "--file_prefix_cue":
      file_prefix_cue = arg
     elif opt == "--file_prefix_hm":
      file_prefix_hm = arg
     elif opt == "--file_prefix_burn_hm":
      file_prefix_burn_hm = arg
     elif opt == "--file_prefix_learn_hm":
      file_prefix_learn_hm = arg
     elif opt == "--file_prefix_consolidation_hm":
      file_prefix_consolidation_hm = arg
     elif opt == "--file_prefix_probe_hm":
      file_prefix_probe_hm = arg
     elif opt == "--file_prefix_cue_hm":
      file_prefix_cue_hm = arg
     elif opt == "--phase_burn":
         phase_burn = arg
     elif opt == "--phase_learn":
         phase_learn = arg
     elif opt == "--phase_consolidation":
         phase_consolidation = arg
     elif opt == "--phase_probe":
         phase_probe = arg
     elif opt == "--phase_cue":
         phase_cue = arg
     elif opt == "--n_stim_learn":
         nb_stim_learn = int(arg)
     elif opt == "--n_stim_probe":
         nb_stim_probe = int(arg)
     elif opt == "--n_stim_cue":
         nb_stim_cue = int(arg)
     elif opt == "--exc_size_thl":
         nb_exc_neurons_thl = int(arg)
     elif opt == "--exc_inh_thl":
         exc_inh_thl = float(arg)
     elif opt == "--exc_size_ctx":
         nb_exc_neurons_ctx = int(arg)
     elif opt == "--exc_inh_ctx":
         exc_inh_ctx = float(arg)
     elif opt == "--exc_size_hpc":
         nb_exc_neurons_hpc = int(arg)
     elif opt == "--exc_inh_hpc":
         exc_inh_hpc = float(arg)
     elif opt == "--exc_inh2_hpc":
         exc_inh2_hpc = float(arg)
     elif opt == "--stim_replay":
         nb_neurons_stim_replay  = int(arg)
     elif opt == "--cons_stim_replay":
         nb_neuron_cons_stim_replay = float(arg)
     elif opt == "--n_patterns":
         nb_patterns = int(arg)
     elif opt == "--n_test_cue_sim":
         nb_test_cue_sim = int(arg)
     elif opt == "--simtime_burn":
         simtime_burn = int(arg)
     elif opt == "--simtimes_learn":
         simtimes_learn = [int(t) for t in arg.split(',')]
     elif opt == "--simtime_learn_bkw":
         simtime_learn_bkw = int(arg)
     elif opt == "--simtimes_consolidation":
         simtimes_consolidation = [int(t) for t in arg.split(',')]
     elif opt == "--simtime_probe":
         simtime_probe = int(arg)
     elif opt == "--simtime_cue":
         simtime_cue = int(arg)
     elif opt == "--range_burn":
         range_burn = int(arg)
     elif opt == "--range_learn":
         range_learn = int(arg)
     elif opt == "--range_consolidation":
         range_consolidation = int(arg)
     elif opt == "--range_probe":
         range_probe = int(arg)
     elif opt == "--range_cue":
         range_cue = int(arg)
     elif opt == "--zoom_range":
         zoom_range = float(arg)
     elif opt == "--n_plot_neurons":
         nb_plot_neurons = int(arg)
     elif opt == "--colors":
      #colors = arg[1:-1].split(',')
      colors = [color for color in arg[1:-1].split(',')]
      '''
      colors = []
      for rgb in arg[2:-2].split('),('):
       r,g,b = rgb.split(',')
       colors.append((float(r), float(g), float(b)))
      '''
     elif opt == "--bin":
         bin_size = float(arg)
     elif opt == "--has_stim_hpc":
         has_stim_hpc = arg.lower() == 'true'
     elif opt == "--has_stim_ctx":
         has_stim_ctx = arg.lower() == 'true'
     elif opt == "--has_stim_thl":
         has_stim_thl = arg.lower() == 'true'
     elif opt == "--has_thl_ctx":
         has_thl_ctx = arg.lower() == 'true'
     elif opt == "--has_ctx_thl":
         has_ctx_thl = arg.lower() == 'true'
     elif opt == "--has_thl_hpc":
         has_thl_hpc = arg.lower() == 'true'
     elif opt == "--has_hpc_thl":
         has_hpc_thl = arg.lower() == 'true'
     elif opt == "--has_ctx_hpc":
         has_ctx_hpc = arg.lower() == 'true'
     elif opt == "--has_hpc_ctx":
         has_hpc_ctx = arg.lower() == 'true'
     elif opt == "--has_hpc_rep":
         has_hpc_rep = arg.lower() == 'true'
     elif opt == "--has_rec_ee":
         has_rec_ee = arg.lower() == 'true'
     elif opt == "--has_inh2":
         has_inh2 = arg.lower() == 'true'
     elif opt == "--ids_cell_assemb":
         ids_cell_assemb = arg.lower() == 'true'
     elif opt == "--cell_assemb_method":
         cell_assemb_method = arg
     elif opt == "--exc_min_rate":
         exc_min_rate = float(arg)
     elif opt == "--inh_min_rate":
         inh_min_rate = float(arg)
     elif opt == "--inh2_min_rate":
         inh2_min_rate = float(arg)
     elif opt == "--min_weight":
         min_weight = float(arg)
     elif opt == "--stim_pat_learn":
         stim_pat_learn = parse_stim_pat(arg)
     elif opt == "--stim_pat_probe":
         stim_pat_probe = parse_stim_pat(arg)
     elif opt == "--stim_pat_cue":
         stim_pat_cue = parse_stim_pat(arg)
     elif opt == "--exc_record_rank_thl":
         exc_record_rank_thl = int(arg)
     elif opt == "--inh_record_rank_thl":
         inh_record_rank_thl = int(arg)
     elif opt == "--exc_ampa_nmda_ratio_thl":
         exc_ampa_nmda_ratio_thl = float(arg)
     elif opt == "--inh_ampa_nmda_ratio_thl":
         inh_ampa_nmda_ratio_thl = float(arg)
     elif opt == "--exc_record_rank_ctx":
         exc_record_rank_ctx = int(arg)
     elif opt == "--inh_record_rank_ctx":
         inh_record_rank_ctx = int(arg)
     elif opt == "--exc_ampa_nmda_ratio_ctx":
         exc_ampa_nmda_ratio_ctx = float(arg)
     elif opt == "--inh_ampa_nmda_ratio_ctx":
         inh_ampa_nmda_ratio_ctx = float(arg)
     elif opt == "--exc_record_rank_hpc":
         exc_record_rank_hpc = int(arg)
     elif opt == "--inh_record_rank_hpc":
         inh_record_rank_hpc = int(arg)
     elif opt == "--inh2_record_rank_hpc":
         inh2_record_rank_hpc = int(arg)
     elif opt == "--exc_ampa_nmda_ratio_hpc":
         exc_ampa_nmda_ratio_hpc = float(arg)
     elif opt == "--inh_ampa_nmda_ratio_hpc":
         inh_ampa_nmda_ratio_hpc = float(arg)
     elif opt == "--inh2_ampa_nmda_ratio_hpc":
         inh2_ampa_nmda_ratio_hpc = float(arg)
     elif opt == "--u_rest":
         u_rest = float(arg)
     elif opt == "--u_exc":
         u_exc = float(arg)
     elif opt == "--u_inh":
         u_inh = float(arg)
         
 # Print command-line arguments
 print ('run', run, type(run))
 print ('has_inh_analysis', has_inh_analysis, type(has_inh_analysis))
 print ('has_metrics', has_metrics, type(has_metrics))
 print ('has_plots', has_plots, type(has_plots))
 print ('has_neuron_vi_plots', has_neuron_vi_plots, type(has_neuron_vi_plots))
 print ('has_cell_assemb_plots', has_cell_assemb_plots, type(has_cell_assemb_plots))
 print ('has_activity_stats', has_activity_stats, type(has_activity_stats))
 print ('has_activity_plots', has_activity_plots, type(has_activity_plots))
 print ('has_spike_raster_stats', has_spike_raster_stats, type(has_spike_raster_stats))
 print ('has_spike_stats_plots', has_spike_stats_plots, type(has_spike_stats_plots))
 print ('has_metrics_plots', has_metrics_plots, type(has_metrics_plots))
 print ('has_rates_across_stim_plots', has_rates_across_stim_plots, type(has_rates_across_stim_plots))
 print ('has_weight_plots', has_weight_plots, type(has_weight_plots))
 print ('has_rf_plots', has_rf_plots, type(has_rf_plots))
 print ('conf_int', conf_int, type(conf_int))
 print ('num_mpi_ranks =', num_mpi_ranks, type(num_mpi_ranks))
 print ('integration_time_step =', integration_time_step, type(integration_time_step))
 print ('rundir =', rundir, type(rundir))
 print ('random_trials =', random_trials, type(random_trials))
 print ('brain_areas =', brain_areas, type(brain_areas))
 print ('prefixes_thl =', prefixes_thl, type(prefixes_thl))
 print ('prefixes_ctx =', prefixes_ctx, type(prefixes_ctx))
 print ('prefixes_hpc =', prefixes_hpc, type(prefixes_hpc))
 print ('file_prefix_burn =', file_prefix_burn, type(file_prefix_burn))
 print ('file_prefix_learn =', file_prefix_learn, type(file_prefix_learn))
 print ('file_prefix_learn_bkw =', file_prefix_learn_bkw, type(file_prefix_learn_bkw))
 print ('file_prefix_consolidation =', file_prefix_consolidation, type(file_prefix_consolidation))
 print ('file_prefix_probe =', file_prefix_probe, type(file_prefix_probe))
 print ('file_prefix_cue =', file_prefix_cue, type(file_prefix_cue))
 print ('file_prefix_hm =', file_prefix_hm, type(file_prefix_hm))
 print ('file_prefix_burn_hm =', file_prefix_burn_hm, type(file_prefix_burn_hm))
 print ('file_prefix_learn_hm =', file_prefix_learn_hm, type(file_prefix_learn_hm))
 print ('file_prefix_consolidation_hm =', file_prefix_consolidation_hm, type(file_prefix_consolidation_hm))
 print ('file_prefix_probe_hm =', file_prefix_probe_hm, type(file_prefix_probe_hm))
 print ('file_prefix_cue_hm =', file_prefix_cue_hm, type(file_prefix_cue_hm))
 print ('phase_burn', phase_burn, type(phase_burn))
 print ('phase_learn', phase_learn, type(phase_learn))
 print ('phase_consolidation', phase_consolidation, type(phase_consolidation))
 print ('phase_probe', phase_probe, type(phase_probe))
 print ('phase_cue', phase_cue, type(phase_cue))
 print ('nb_stim_learn =', nb_stim_learn, type(nb_stim_learn))
 print ('nb_stim_probe =', nb_stim_probe, type(nb_stim_probe))
 print ('nb_stim_cue =', nb_stim_cue, type(nb_stim_cue))
 print ('nb_exc_neurons_thl =', nb_exc_neurons_thl, type(nb_exc_neurons_thl))
 print ('exc_inh_thl = ', exc_inh_thl, type(exc_inh_thl))
 print ('nb_exc_neurons_ctx =', nb_exc_neurons_ctx, type(nb_exc_neurons_ctx))
 print ('exc_inh_ctx = ', exc_inh_ctx, type(exc_inh_ctx))
 print ('nb_exc_neurons_hpc =', nb_exc_neurons_hpc, type(nb_exc_neurons_hpc))
 print ('exc_inh_hpc = ', exc_inh_hpc, type(exc_inh_hpc))
 print ('exc_inh2_hpc = ', exc_inh2_hpc, type(exc_inh2_hpc))
 print ('nb_neurons_stim_replay =', nb_neurons_stim_replay, type(nb_neurons_stim_replay))
 print ('nb_neuron_cons_stim_replay =', nb_neuron_cons_stim_replay, type(nb_neuron_cons_stim_replay))
 print ('nb_patterns =', nb_patterns, type(nb_patterns))
 print ('nb_test_cue_sim =', nb_test_cue_sim, type(nb_test_cue_sim))
 print ('simtime_burn', simtime_burn, type(simtime_burn))
 print ('simtimes_learn =', simtimes_learn, type(simtimes_learn))
 print ('simtime_learn_bkw =', simtime_learn_bkw, type(simtime_learn_bkw))
 print ('simtimes_consolidation =', simtimes_consolidation, type(simtimes_consolidation))
 print ('simtime_probe =', simtime_probe, type(simtime_probe))
 print ('simtime_cue =', simtime_cue, type(simtime_cue))
 print ('range_burn =', range_burn, type(range_burn))
 print ('range_learn =', range_learn, type(range_learn))
 print ('range_consolidation =', range_consolidation, type(range_consolidation))
 print ('range_probe =', range_probe, type(range_probe))
 print ('range_cue =', range_cue, type(range_cue))
 print ('zoom_range =', zoom_range, type(zoom_range))
 print ('nb_plot_neurons =', nb_plot_neurons, type(nb_plot_neurons))
 print ('colors =', colors, type(colors))
 print ('bin_size =', bin_size, type(bin_size))
 print ('has_stim_hpc =', has_stim_hpc, type(has_stim_hpc))
 print ('has_stim_ctx =', has_stim_ctx, type(has_stim_ctx))
 print ('has_stim_thl =', has_stim_thl, type(has_stim_thl))
 print ('has_thl_ctx', has_thl_ctx, type(has_thl_ctx))
 print ('has_ctx_thl', has_ctx_thl, type(has_ctx_thl))
 print ('has_thl_hpc', has_thl_hpc, type(has_thl_hpc))
 print ('has_hpc_thl', has_hpc_thl, type(has_hpc_thl))
 print ('has_ctx_hpc', has_ctx_hpc, type(has_ctx_hpc))
 print ('has_hpc_ctx', has_hpc_ctx, type(has_hpc_ctx))
 print ('has_hpc_rep', has_hpc_rep, type(has_hpc_rep))
 print ('has_rec_ee', has_rec_ee, type(has_rec_ee))
 print ('has_inh2', has_inh2, type(has_inh2))
 print ('ids_cell_assemb =', ids_cell_assemb, type(ids_cell_assemb))
 print ('cell_assemb_method =', cell_assemb_method, type(cell_assemb_method))
 print ('exc_min_rate =', exc_min_rate, type(exc_min_rate))
 print ('inh_min_rate =', inh_min_rate, type(inh_min_rate))
 print ('inh2_min_rate =', inh2_min_rate, type(inh2_min_rate))
 print ('min_weight =', min_weight, type(min_weight))
 print ('stim_pat_learn =', stim_pat_learn, type(stim_pat_learn))
 print ('stim_pat_probe =', stim_pat_probe, type(stim_pat_probe))
 print ('stim_pat_cue =', stim_pat_cue, type(stim_pat_cue))
 print ('exc_record_rank_thl =', exc_record_rank_thl, type(exc_record_rank_thl))
 print ('inh_record_rank_thl =', inh_record_rank_thl, type(inh_record_rank_thl))
 print ('exc_ampa_nmda_ratio_thl =', exc_ampa_nmda_ratio_thl, type(exc_ampa_nmda_ratio_thl))
 print ('inh_ampa_nmda_ratio_thl =', inh_ampa_nmda_ratio_thl, type(inh_ampa_nmda_ratio_thl))
 print ('exc_record_rank_ctx =', exc_record_rank_ctx, type(exc_record_rank_ctx))
 print ('inh_record_rank_ctx =', inh_record_rank_ctx, type(inh_record_rank_ctx))
 print ('exc_ampa_nmda_ratio_ctx =', exc_ampa_nmda_ratio_ctx, type(exc_ampa_nmda_ratio_ctx))
 print ('inh_ampa_nmda_ratio_ctx =', inh_ampa_nmda_ratio_ctx, type(inh_ampa_nmda_ratio_ctx))
 print ('exc_record_rank_hpc =', exc_record_rank_hpc, type(exc_record_rank_hpc))
 print ('inh_record_rank_hpc =', inh_record_rank_hpc, type(inh_record_rank_hpc))
 print ('inh2_record_rank_hpc =', inh2_record_rank_hpc, type(inh2_record_rank_hpc))
 print ('exc_ampa_nmda_ratio_hpc =', exc_ampa_nmda_ratio_hpc, type(exc_ampa_nmda_ratio_hpc))
 print ('inh_ampa_nmda_ratio_hpc =', inh_ampa_nmda_ratio_hpc, type(inh_ampa_nmda_ratio_hpc))
 print ('inh2_ampa_nmda_ratio_hpc =', inh2_ampa_nmda_ratio_hpc, type(inh2_ampa_nmda_ratio_hpc))
 print ('u_rest =', u_rest, type(u_rest))
 print ('u_exc =', u_exc, type(u_exc))
 print ('u_inh =', u_inh, type(u_inh))
 
 parser = Parser(run, has_inh_analysis, has_metrics, has_plots, has_neuron_vi_plots, has_cell_assemb_plots, has_activity_stats, has_activity_plots, has_spike_raster_stats, has_spike_stats_plots, has_metrics_plots, has_rates_across_stim_plots, has_weight_plots, has_rf_plots, conf_int, num_mpi_ranks, integration_time_step, rundir, random_trials, brain_areas, prefixes_thl, prefixes_ctx, prefixes_hpc, file_prefix_burn, file_prefix_learn, file_prefix_learn_bkw, file_prefix_consolidation, file_prefix_probe, file_prefix_cue, file_prefix_hm, file_prefix_burn_hm, file_prefix_learn_hm, file_prefix_consolidation_hm, file_prefix_probe_hm, file_prefix_cue_hm, phase_burn, phase_learn, phase_consolidation, phase_probe, phase_cue, nb_stim_learn, nb_stim_probe, nb_stim_cue, nb_exc_neurons_thl, exc_inh_thl, nb_exc_neurons_ctx, exc_inh_ctx, nb_exc_neurons_hpc, exc_inh_hpc, exc_inh2_hpc, nb_neurons_stim_replay, nb_neuron_cons_stim_replay, nb_patterns, nb_test_cue_sim, simtime_burn, simtimes_learn, simtime_learn_bkw, simtimes_consolidation, simtime_probe, simtime_cue, range_burn, range_learn, range_consolidation, range_probe, range_cue, zoom_range, nb_plot_neurons, colors, bin_size, has_stim_hpc, has_stim_ctx, has_stim_thl, has_thl_ctx, has_ctx_thl, has_thl_hpc, has_hpc_thl, has_ctx_hpc, has_hpc_ctx, has_hpc_rep, has_rec_ee, has_inh2, ids_cell_assemb, cell_assemb_method, exc_min_rate, inh_min_rate, inh2_min_rate, min_weight, stim_pat_learn, stim_pat_probe, stim_pat_cue, exc_record_rank_thl, inh_record_rank_thl, exc_ampa_nmda_ratio_thl, inh_ampa_nmda_ratio_thl, exc_record_rank_ctx, inh_record_rank_ctx, exc_ampa_nmda_ratio_ctx, inh_ampa_nmda_ratio_ctx, exc_record_rank_hpc, inh_record_rank_hpc, inh2_record_rank_hpc, exc_ampa_nmda_ratio_hpc, inh_ampa_nmda_ratio_hpc, inh2_ampa_nmda_ratio_hpc, u_rest, u_exc, u_inh)

 parser.analyze()

 
if __name__ == "__main__":
 main(sys.argv[1:])
 
