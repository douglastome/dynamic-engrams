/* 
* Source code modified by Douglas Feitosa Tomé in 2023
*
* The modified code is released under the the GNU General Public License
* as published by the Free Software Foundation, either version 3 
* of the License, or (at your option) any later version.
*
* Copyright 2015 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "auryn.h"
#include "SparseBConnection.h"
#include "STPBConnection.h"
#include "P11BConnection.h"
#include "GlobalPFBConnection.h"
#include "GlobalPF2BConnection.h"
#include "AIF2IGroup.h"

#include <string>

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

// comment out if there is no thalamus
//#define HAS_THL

// comment out if there is no cortex
//#define HAS_CTX

// comment out if there is no hippocampus
#define HAS_HPC

// comment out if there is no replay group
#define HAS_REPLAY

// comment out if there is are no background groups
#define HAS_BACKGROUND

// comment out if there are no cross-region excitatory connections onto inh neurons
//#define HAS_CROSS_REGION_EI_CON

// comment out if there are no recurrent E->E connections in a given region
#define HAS_REC_EE

// comment out if there is no second group of inhibitory neurons in a given region
//#define HAS_INH2

// if HAS_THL, choose 1 option and comment out the remaining ones
// else comment all out
//#define CON_STIM_THL_NONE
//#define CON_STIM_THL_SPARSEB
//#define CON_STIM_THL_STPB
//#define CON_STIM_THL_P11B

// if HAS_CTX, choose 1 option and comment out the remaining ones
// else comment all out
//#define CON_STIM_CTX_NONE
//#define CON_STIM_CTX_SPARSEB
//#define CON_STIM_CTX_STPB
//#define CON_STIM_CTX_P11B

// if HAS_HPC, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_STIM_HPC_NONE 
//#define CON_STIM_HPC_SPARSEB
//#define CON_STIM_HPC_STPB
#define CON_STIM_HPC_P11B

// if HAS_THL and HAS_CTX, choose 1 option and comment out the remaining ones
// else, comment out all
//#define CON_THL_CTX_NONE 
//#define CON_THL_CTX_SPARSEB
//#define CON_THL_CTX_STPB
//#define CON_THL_CTX_P11B

// if HAS_CTX and HAS_THL, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_CTX_THL_NONE 
//#define CON_CTX_THL_SPARSEB
//#define CON_CTX_THL_STPB
//#define CON_CTX_THL_P11B

// if HAS_THL and HAS_HPC, choose 1 option and comment out the remaining ones
// else, comment out all
//#define CON_THL_HPC_NONE 
//#define CON_THL_HPC_SPARSEB
//#define CON_THL_HPC_STPB
//#define CON_THL_HPC_P11B

// if HAS_HPC and HAS_THL, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_HPC_THL_NONE 
//#define CON_HPC_THL_SPARSEB
//#define CON_HPC_THL_STPB
//#define CON_HPC_THL_P11B

// if HAS_CTX and HAS_HPC, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_CTX_HPC_NONE 
//#define CON_CTX_HPC_SPARSEB
//#define CON_CTX_HPC_STPB
//#define CON_CTX_HPC_P11B

// if HAS_HPC and HAS_CTX, choose 1 option and comment out the remaining ones
// else, comment out all
//#define CON_HPC_CTX_NONE 
//#define CON_HPC_CTX_SPARSEB
//#define CON_HPC_CTX_STPB
//#define CON_HPC_CTX_P11B

// if HAS_REPLAY and HAS_THL, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_REPLAY_THL_NONE
//#define CON_REPLAY_THL_SPARSEB

// if HAS_REPLAY and HAS_CTX, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_REPLAY_CTX_NONE
//#define CON_REPLAY_CTX_SPARSEB

// if HAS_REPLAY and HAS_HPC, then choose 1 option and comment out the remaining ones
// else comment out all
//#define CON_REPLAY_HPC_NONE
#define CON_REPLAY_HPC_SPARSEB


int main(int ac, char* av[]) 
{
  // simulation
  string binary = "sim_rc_p11";
  string out_dir = "/data/";
  string file_prefix = "rc";
  string file_prefix_hm = "hm";
  string load_file = "";
  std::vector<double> simtimes;

  // buffers
  char strbuf [255];
  string msg;

  // simulation flags
  bool save = false;
  bool save_without_hpc = false;
  bool load_without_hpc = false;
  bool chain = false;
  bool prime = false;

  // initialization flags
  bool noisy_initial_weights = false;
  bool consolidate_initial_weights = false;

  // plasticity flags
  bool consolidation = true;
  bool isp_active = true;

  bool inh_input = false;  //NOT USED
  bool quiet = false;  //NOT USED

  // stim flags
  bool stim_spike_mon = false;

  // blocking flags
  bool block_local = false;
  bool block_cross_region = false;

  // thl flags
  bool weight_mon_ee_stim_thl = false;
  bool weightstat_mon_ee_stim_thl = false;
  bool weightpat_mon_ee_stim_thl = false;
  bool exc_spike_mon_thl = false;
  bool exc_prate_mon_thl = false;
  bool exc_pattern_mon_thl = false;
  bool exc_voltage_mon_thl = false;
  bool exc_g_ampa_mon_thl = false;
  bool exc_g_nmda_mon_thl = false;
  bool exc_g_gaba_mon_thl = false;
  bool exc_g_adapt1_mon_thl = false;
  bool exc_g_adapt2_mon_thl = false;
  bool exc_thr_mon_thl = false;
  bool exc_ratechk_thl = false;
  bool inh_spike_mon_thl = false;
  bool inh_prate_mon_thl = false;
  bool inh_voltage_mon_thl = false;
  bool weight_mon_ee_thl = false;
  bool weightstat_mon_ee_thl = false;
  bool weightpat_mon_ee_thl = false;
  bool ee_hom_mon_thl = false;
  bool ei_weight_mon_thl = false;
  bool ie_weight_mon_thl = false;
  bool ie_weightstat_mon_thl = false;
  bool ii_weight_mon_thl = false;

  // ctx flags
  bool weight_mon_ee_stim_ctx = false;
  bool weightstat_mon_ee_stim_ctx = false;
  bool weightpat_mon_ee_stim_ctx = false;
  bool exc_spike_mon_ctx = false;
  bool exc_prate_mon_ctx = false;
  bool exc_pattern_mon_ctx = false;
  bool exc_voltage_mon_ctx = false;
  bool exc_g_ampa_mon_ctx = false;
  bool exc_g_nmda_mon_ctx = false;
  bool exc_g_gaba_mon_ctx = false;
  bool exc_g_adapt1_mon_ctx = false;
  bool exc_g_adapt2_mon_ctx = false;
  bool exc_thr_mon_ctx = false;
  bool exc_ratechk_ctx = false;
  bool inh_spike_mon_ctx = false;
  bool inh_prate_mon_ctx = false;
  bool inh_voltage_mon_ctx = false;
  bool weight_mon_ee_ctx = false;
  bool weightstat_mon_ee_ctx = false;
  bool weightpat_mon_ee_ctx = false;
  bool ee_hom_mon_ctx = false;
  bool ei_weight_mon_ctx = false;
  bool ie_weight_mon_ctx = false;
  bool ie_weightstat_mon_ctx = false;
  bool ii_weight_mon_ctx = false;

  // hpc flags
  bool weight_mon_ee_stim_hpc = false;
  bool weightstat_mon_ee_stim_hpc = false;
  bool weightpat_mon_ee_stim_hpc = false;
  bool exc_spike_mon_hpc = false;
  bool exc_prate_mon_hpc = false;
  bool exc_pattern_mon_hpc = false;
  bool exc_voltage_mon_hpc = false;
  bool exc_g_ampa_mon_hpc = false;
  bool exc_g_nmda_mon_hpc = false;
  bool exc_g_gaba_mon_hpc = false;
  bool exc_g_adapt1_mon_hpc = false;
  bool exc_g_adapt2_mon_hpc = false;
  bool exc_thr_mon_hpc = false;
  bool exc_ratechk_hpc = false;
  bool inh_spike_mon_hpc = false;
  bool inh_prate_mon_hpc = false;
  bool inh_voltage_mon_hpc = false;
  bool weight_mon_ee_hpc = false;
  bool weightstat_mon_ee_hpc = false;
  bool weightpat_mon_ee_hpc = false;
  bool ee_hom_mon_hpc = false;
  bool ei_weight_mon_hpc = false;
  bool ie_weight_mon_hpc = false;
  bool ie_weightstat_mon_hpc = false;
  bool ii_weight_mon_hpc = false;

  // background flags
  bool bg_spike_mon = false;
  
  // replay flags
  bool rep_spike_mon = false;

  // monitors
  NeuronID record_neuron_exc_ctx = 1;
  NeuronID record_neuron_inh_ctx = 1;
  NeuronID record_neuron_exc_hpc = 1;
  NeuronID record_neuron_inh_hpc = 1;
  NeuronID record_neuron_exc_thl = 1;
  NeuronID record_neuron_inh_thl = 1;
  string monfile_ctx = "";
  string premonfile_ctx = "";
  string monfile_hpc = "";
  string premonfile_hpc = "";
  string monfile_thl = "";
  string premonfile_thl = "";

  // pseudo-random number generation
  int master_seed = 42;
  NeuronID stim_seed = 1; // 1 leads to not seeding the stimulus group
  NeuronID rep_seed = 1; // 1 leads to not seeding the replay group
  NeuronID bg_thl_seed = 1; // 1 leads to not seeding the thl background group
  NeuronID bg_ctx_seed = 1; // 1 leads to not seeding the ctx background group
  NeuronID bg_hpc_seed = 1; // 1 leads to not seeding the hpc background group

  // stimulus
  double ontime = 1.0;
  double offtime = 5.0;
  double scale = 1000.0;
  double bgrate = 5.0;
  int preferred = -1;
  string stimfile = "";
  string recfile_stim_ctx = "";
  AurynWeight xi_stim_ctx = -10.0;
  string recfile_stim_hpc = "";
  AurynWeight xi_stim_hpc = -10.0;
  string recfile_stim_thl = "";
  AurynWeight xi_stim_thl = -10.0;
  double prime_ontime = 1.0;
  double prime_offtime = 0.2;
  double prime_duration = 200.0;

  // background
  string bgfile_thl = "";
  double bgrate_thl = 0.0;
  double sparseness_bg_thl = 0.10;
  double w_bg_thl = 1.0;
  double w_bg_ei_thl = 1.0;
  string recfile_bg_thl = "";
  AurynWeight xi_bg_thl = -10.0;
  string recfile_ei_bg_thl = "";
  AurynWeight xi_ei_bg_thl = -10.0;

  string bgfile_ctx = "";
  double bgrate_ctx = 0.0;
  double sparseness_bg_ctx = 0.10;
  double w_bg_ctx = 1.0;
  double w_bg_ei_ctx = 1.0;
  string recfile_bg_ctx = "";
  AurynWeight xi_bg_ctx = -10.0;
  string recfile_ei_bg_ctx = "";
  AurynWeight xi_ei_bg_ctx = -10.0;

  string bgfile_hpc = "";
  double bgrate_hpc = 0.0;
  double sparseness_bg_hpc = 0.10;
  double w_bg_hpc = 1.0;
  double w_bg_ei_hpc = 1.0;
  string recfile_bg_hpc = "";
  AurynWeight xi_bg_hpc = -10.0;
  string recfile_ei_bg_hpc = "";
  AurynWeight xi_ei_bg_hpc = -10.0;

  // replay
  NeuronID size_rep = 4096;
  string repfile = "";
  double bgrate_rep = 0.0;

  double sparseness_rep_thl = 0.10;
  double w_rep_thl = 0.05;
  string recfile_rep_thl = "";
  AurynWeight xi_rep_thl = -10.0;
  string recfile_ei_rep_thl = "";
  AurynWeight xi_ei_rep_thl = -10.0;

  double sparseness_rep_ctx = 0.10;
  double w_rep_ctx = 0.05;
  string recfile_rep_ctx = "";
  AurynWeight xi_rep_ctx = -10.0;
  string recfile_ei_rep_ctx = "";
  AurynWeight xi_ei_rep_ctx = -10.0;

  double sparseness_rep_hpc = 0.10;
  double w_rep_hpc = 0.05;
  string recfile_rep_hpc = "";
  AurynWeight xi_rep_hpc = -10.0;
  string recfile_ei_rep_hpc = "";
  AurynWeight xi_ei_rep_hpc = -10.0;

  // thl network
  NeuronID exc_size_thl = 4096;
  double eta_thl = 1e-3;
  double eta_stim_thl = 1e-3;
  double eta_exc_inh_thl = 50;
  int exc_inh_thl = 4;
  double alpha_thl = 4;
  double kappa_thl = 10;
  double tauf_ee_thl = 0.6;
  double taud_ee_thl = 0.15;
  double tauh_ee_thl = 600.0;
  double tauc_ee_thl = 1200.0;
  double ujump_ee_thl = 0.2;
  double tauf_ee_stim_thl = 0.6;
  double taud_ee_stim_thl = 0.15;
  double tauh_ee_stim_thl = 600.0;
  double tauc_ee_stim_thl = 1200.0;
  double ujump_ee_stim_thl = 0.2;
  double beta_thl = 50.0;
  double beta_stim_thl = 50.0;
  double delta_thl = 2.0e-2;
  double weight_a_thl = 0.0;
  double weight_c_thl = 0.5;
  double adapt1_thl = 0.1;
  double adapt2_thl = 0.0;
  double pot_strength_thl = 0.1;
  double wmax_exc_thl = 5.0;
  double wmin_exc_thl = 0.0;
  double wmax_inh_thl = 5.0;
  double wmin_inh_thl = 0.0;
  AurynWeight wee_thl = 0.1;
  AurynWeight wei_thl = 0.2;
  AurynWeight wie_thl = 0.2;
  AurynWeight wii_thl = 0.2;
  double wext_thl = 0.05;
  double wext_ei_thl = 0.05;
  double sparseness_int_ee_thl = 0.05;
  double sparseness_int_ei_thl = 0.05;
  double sparseness_int_ii_thl = 0.05;
  double sparseness_int_ie_thl = 0.05;
  double sparseness_ext_thl = 0.10;
  double tauf_ei_thl = 0.6;
  double taud_ei_thl = 200e-3;
  double ujump_ei_thl = 0.2;
  double tauh_ie_thl = 10.0;
  double taud_ei_stim_thl = 200e-3;
  double tauf_ei_stim_thl = 0.6;
  double ujump_ei_stim_thl = 0.2;
  double tau_ampa_e_thl = 5e-3;
  double tau_gaba_e_thl = 10e-3;
  double tau_nmda_e_thl = 100e-3;
  double ampa_nmda_e_thl = 0.2;
  double tau_ampa_i_thl = 5e-3;
  double tau_gaba_i_thl = 10e-3;
  double tau_nmda_i_thl = 100e-3;
  double ampa_nmda_i_thl = 0.3;
  string prefile_thl = "";
  AurynWeight chi_thl = 1.0;

  // ctx network
  NeuronID exc_size_ctx = 4096;
  double eta_ctx = 1e-3;
  double eta_stim_ctx = 1e-3;
  double eta_exc_inh_ctx = 50;
  int exc_inh_ctx = 4;
  double alpha_ctx = 4;
  double kappa_ctx = 10;
  double tauf_ee_ctx = 0.6;
  double taud_ee_ctx = 0.15;
  double tauh_ee_ctx = 600.0;
  double tauc_ee_ctx = 1200.0;
  double ujump_ee_ctx = 0.2;
  double tauf_ee_stim_ctx = 0.6;
  double taud_ee_stim_ctx = 0.15;
  double tauh_ee_stim_ctx = 600.0;
  double tauc_ee_stim_ctx = 1200.0;
  double ujump_ee_stim_ctx = 0.2;
  double beta_ctx = 50.0;
  double beta_stim_ctx = 50.0;
  double delta_ctx = 2.0e-2;
  double weight_a_ctx = 0.0;
  double weight_c_ctx = 0.5;
  double adapt1_ctx = 0.1;
  double adapt2_ctx = 0.0;
  double pot_strength_ctx = 0.1;
  double wmax_exc_ctx = 5.0;
  double wmin_exc_ctx = 0.0;
  double wmax_inh_ctx = 5.0;
  double wmin_inh_ctx = 0.0;
  AurynWeight wee_ctx = 0.1;
  AurynWeight wei_ctx = 0.2;
  AurynWeight wie_ctx = 0.2;
  AurynWeight wii_ctx = 0.2;
  double wext_ctx = 0.05;
  double wext_ei_ctx = 0.05;
  double sparseness_int_ee_ctx = 0.05;
  double sparseness_int_ei_ctx = 0.05;
  double sparseness_int_ii_ctx = 0.05;
  double sparseness_int_ie_ctx = 0.05;
  double sparseness_ext_ctx = 0.10;
  double tauf_ei_ctx = 0.6;
  double taud_ei_ctx = 200e-3;
  double ujump_ei_ctx = 0.2;
  double tauh_ie_ctx = 10.0;
  double taud_ei_stim_ctx = 200e-3;
  double tauf_ei_stim_ctx = 0.6;
  double ujump_ei_stim_ctx = 0.2;
  double tau_ampa_e_ctx = 5e-3;
  double tau_gaba_e_ctx = 10e-3;
  double tau_nmda_e_ctx = 100e-3;
  double ampa_nmda_e_ctx = 0.2;
  double tau_ampa_i_ctx = 5e-3;
  double tau_gaba_i_ctx = 10e-3;
  double tau_nmda_i_ctx = 100e-3;
  double ampa_nmda_i_ctx = 0.3;
  string prefile_ctx = "";
  AurynWeight chi_ctx = 1.0;

  // hpc network
  NeuronID exc_size_hpc = 4096;
  double eta_hpc = 1e-3;
  double eta_stim_hpc = 1e-3;
  double eta_exc_inh_hpc = 50;
  int exc_inh_hpc = 4;
  double alpha_hpc = 4;
  double kappa_hpc = 10;
  double tauf_ee_hpc = 0.6;
  double taud_ee_hpc = 0.15;
  double tauh_ee_hpc = 600.0;
  double tauc_ee_hpc = 1200.0;
  double ujump_ee_hpc = 0.2;
  double tauf_ee_stim_hpc = 0.6;
  double taud_ee_stim_hpc = 0.15;
  double tauh_ee_stim_hpc = 600.0;
  double tauc_ee_stim_hpc = 1200.0;
  double ujump_ee_stim_hpc = 0.2;
  double beta_hpc = 50.0;
  double beta_stim_hpc = 50.0;
  double delta_hpc = 2.0e-2;
  double weight_a_hpc = 0.0;
  double weight_c_hpc = 0.5;
  double adapt1_hpc = 0.1;
  double adapt2_hpc = 0.0;
  double pot_strength_hpc = 0.1;
  double wmax_exc_hpc = 5.0;
  double wmin_exc_hpc = 0.0;
  double wmax_inh_hpc = 5.0;
  double wmin_inh_hpc = 0.0;
  AurynWeight wee_hpc = 0.1;
  AurynWeight wei_hpc = 0.2;
  AurynWeight wie_hpc = 0.2;
  AurynWeight wii_hpc = 0.2;
  double wext_hpc = 0.05;
  double wext_ei_hpc = 0.05;
  double sparseness_int_ee_hpc = 0.05;
  double sparseness_int_ei_hpc = 0.05;
  double sparseness_int_ii_hpc = 0.05;
  double sparseness_int_ie_hpc = 0.05;
  double sparseness_ext_hpc = 0.10;
  double tauf_ei_hpc = 0.6;
  double taud_ei_hpc = 200e-3;
  double ujump_ei_hpc = 0.2;
  double tauh_ie_hpc = 10.0;
  double taud_ei_stim_hpc = 200e-3;
  double tauf_ei_stim_hpc = 0.6;
  double ujump_ei_stim_hpc = 0.2;
  double tau_ampa_e_hpc = 5e-3;
  double tau_gaba_e_hpc = 10e-3;
  double tau_nmda_e_hpc = 100e-3;
  double ampa_nmda_e_hpc = 0.2;
  double tau_ampa_i_hpc = 5e-3;
  double tau_gaba_i_hpc = 10e-3;
  double tau_nmda_i_hpc = 100e-3;
  double ampa_nmda_i_hpc = 0.3;
  string prefile_hpc = "";
  AurynWeight chi_hpc = 1.0;
  // hpc inh2
  double eta_exc_inh2_hpc = 50;
  int exc_inh2_hpc = 4;
  double alpha2_hpc = 4;
  double tau_ampa_i2_hpc = 5e-3;
  double tau_gaba_i2_hpc = 10e-3;
  double tau_nmda_i2_hpc = 100e-3;
  double ampa_nmda_i2_hpc = 0.3;
  AurynWeight wei2_hpc = 0.2;
  AurynWeight wi2e_hpc = 0.2;
  AurynWeight wi2i2_hpc = 0.2;
  AurynWeight wii2_hpc = 0.2;
  AurynWeight wi2i_hpc = 0.2;
  double sparseness_int_ei2_hpc = 0.05;
  double sparseness_int_i2e_hpc = 0.05;
  double sparseness_int_i2i2_hpc = 0.05;
  double sparseness_int_ii2_hpc = 0.05;
  double sparseness_int_i2i_hpc = 0.05;
  double tauf_ei2_hpc = 0.6;
  double taud_ei2_hpc = 200e-3;
  double ujump_ei2_hpc = 0.2;
  double tauh_i2e_hpc = 10.0;
  double wmax_inh2_hpc = 5.0;
  double wmin_inh2_hpc = 0.0;
  NeuronID record_neuron_inh2_hpc = 1;
  string block_inh2_hpc = "";
  
  // thl->ctx connection
  // used in case thl->ctx is: SPARSEB, STPB, P11B
  AurynWeight wee_thl_ctx = 0.1;
  AurynWeight wei_thl_ctx = 0.1;
  double sparseness_thl_ctx = 0.10;
  string recfile_thl_ctx = "";
  AurynWeight xi_thl_ctx = -10.0;
  // used in case thl->ctx is: P11B
  double eta_thl_ctx = 1e-3;
  double kappa_thl_ctx = 10;
  double delta_thl_ctx = 2.0e-2;
  double tauf_thl_ctx = 0.6;
  double taud_thl_ctx = 0.15;
  double tauh_thl_ctx = 600.0;
  double tauc_thl_ctx = 1200.0;
  double ujump_thl_ctx = 0.2;
  double beta_thl_ctx = 50.0;
  double weight_a_thl_ctx = 0.0;
  double weight_c_thl_ctx = 0.5;
  double pot_strength_thl_ctx = 0.1;
  double wmax_thl_ctx = 5.0;
  double wmin_thl_ctx = 0.0;

  // ctx->thl connection
  // used in case ctx->thl is: SPARSEB, STPB, P11B
  AurynWeight wee_ctx_thl = 0.1;
  AurynWeight wei_ctx_thl = 0.1;
  double sparseness_ctx_thl = 0.10;
  string recfile_ctx_thl = "";
  AurynWeight xi_ctx_thl = -10.0;
  // used in case ctx->thl is: P11B
  double eta_ctx_thl = 1e-3;
  double kappa_ctx_thl = 10;
  double tauf_ctx_thl = 0.6;
  double taud_ctx_thl = 0.15;
  double tauh_ctx_thl = 600.0;
  double tauc_ctx_thl = 1200.0;
  double ujump_ctx_thl = 0.2;
  double beta_ctx_thl = 50.0;
  double delta_ctx_thl = 2.0e-2;
  double weight_a_ctx_thl = 0.0;
  double weight_c_ctx_thl = 0.5;
  double pot_strength_ctx_thl = 0.1;
  double wmax_ctx_thl = 5.0;
  double wmin_ctx_thl = 0.0;

  // thl->hpc connection
  // used in case thl->hpc is: SPARSEB, STPB, P11B
  AurynWeight wee_thl_hpc = 0.1;
  AurynWeight wei_thl_hpc = 0.1;
  double sparseness_thl_hpc = 0.10;
  string recfile_thl_hpc = "";
  AurynWeight xi_thl_hpc = -10.0;
  // used in case thl->hpc is: P11B
  double eta_thl_hpc = 1e-3;
  double kappa_thl_hpc = 10;
  double delta_thl_hpc = 2.0e-2;
  double tauf_thl_hpc = 0.6;
  double taud_thl_hpc = 0.15;
  double tauh_thl_hpc = 600.0;
  double tauc_thl_hpc = 1200.0;
  double ujump_thl_hpc = 0.2;
  double beta_thl_hpc = 50.0;
  double weight_a_thl_hpc = 0.0;
  double weight_c_thl_hpc = 0.5;
  double pot_strength_thl_hpc = 0.1;
  double wmax_thl_hpc = 5.0;
  double wmin_thl_hpc = 0.0;

  // hpc->thl connection
  // used in case hpc->thl is: SPARSEB, STPB, P11B
  AurynWeight wee_hpc_thl = 0.1;
  AurynWeight wei_hpc_thl = 0.1;
  double sparseness_hpc_thl = 0.10;
  string recfile_hpc_thl = "";
  AurynWeight xi_hpc_thl = -10.0;
  // used in case hpc->thl is: P11B
  double eta_hpc_thl = 1e-3;
  double kappa_hpc_thl = 10;
  double tauf_hpc_thl = 0.6;
  double taud_hpc_thl = 0.15;
  double tauh_hpc_thl = 600.0;
  double tauc_hpc_thl = 1200.0;
  double ujump_hpc_thl = 0.2;
  double beta_hpc_thl = 50.0;
  double delta_hpc_thl = 2.0e-2;
  double weight_a_hpc_thl = 0.0;
  double weight_c_hpc_thl = 0.5;
  double pot_strength_hpc_thl = 0.1;
  double wmax_hpc_thl = 5.0;
  double wmin_hpc_thl = 0.0;

  // ctx->hpc connection
  // used in case ctx->hpc is: SPARSEB, STPB, P11B
  AurynWeight wee_ctx_hpc = 0.1;
  AurynWeight wei_ctx_hpc = 0.1;
  double sparseness_ctx_hpc = 0.10;
  string recfile_ctx_hpc = "";
  AurynWeight xi_ctx_hpc = -10.0;
  // used in case ctx->hpc is: P11B
  double eta_ctx_hpc = 1e-3;
  double kappa_ctx_hpc = 10;
  double tauf_ctx_hpc = 0.6;
  double taud_ctx_hpc = 0.15;
  double tauh_ctx_hpc = 600.0;
  double tauc_ctx_hpc = 1200.0;
  double ujump_ctx_hpc = 0.2;
  double beta_ctx_hpc = 50.0;
  double delta_ctx_hpc = 2.0e-2;
  double weight_a_ctx_hpc = 0.0;
  double weight_c_ctx_hpc = 0.5;
  double pot_strength_ctx_hpc = 0.1;
  double wmax_ctx_hpc = 5.0;
  double wmin_ctx_hpc = 0.0;

  // hpc->ctx connection
  // used in case hpc->ctx is: SPARSEB, STPB, P11B
  AurynWeight wee_hpc_ctx = 0.1;
  AurynWeight wei_hpc_ctx = 0.1;
  double sparseness_hpc_ctx = 0.10;
  string recfile_hpc_ctx = "";
  AurynWeight xi_hpc_ctx = -10.0;
  // used in case hpc->ctx is: P11B
  double eta_hpc_ctx = 1e-3;
  double kappa_hpc_ctx = 10;
  double delta_hpc_ctx = 2.0e-2;
  double tauf_hpc_ctx = 0.6;
  double taud_hpc_ctx = 0.15;
  double tauh_hpc_ctx = 600.0;
  double tauc_hpc_ctx = 1200.0;
  double ujump_hpc_ctx = 0.2;
  double beta_hpc_ctx = 50.0;
  double weight_a_hpc_ctx = 0.0;
  double weight_c_hpc_ctx = 0.5;
  double pot_strength_hpc_ctx = 0.1;
  double wmax_hpc_ctx = 5.0;
  double wmin_hpc_ctx = 0.0;

  // block neurons
  string block_stim = "";
  string block_rep = "";
  string block_bg_thl = "";
  string block_bg_ctx = "";
  string block_bg_hpc = "";
  string block_exc_thl = "";
  string block_inh_thl = "";
  string block_exc_ctx = "";
  string block_inh_ctx = "";
  string block_exc_hpc = "";
  string block_inh_hpc = "";

  // inhibit neurons
  string inhibit_exc_hpc = "";

  int errcode = 0;

  try {

    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("binary", po::value<string>(), "binary executable name")
      ("out_dir", po::value<string>(), "output dir")
      ("file_prefix", po::value<string>(), "set network state file prefix")
      ("file_prefix_hm", po::value<string>(), "set network state file prefix without hpc")
      ("load_file", po::value<string>(), "file to load weight matrix")
      ("simtimes", po::value< std::vector<double> >()->multitoken()->composing(), "simulation times")
      ("save", "save network state at end of sim")
      ("save_without_hpc", "save network state excluding hpc at the end of sim")
      ("load_without_hpc", "exclude hpc when loading network state")
      ("chain", "chain mode for pattern loader")
      ("prime", "prime network with a burn-in phase")
      ("noconsolidation", "switches off consolidation")
      ("noisp", "switches off isp")
      ("inhinput", "switches external input to inh on")
      ("noisyweights", "switches noisy initial weights on")
      ("consolidateweights", "initialize weights as consolidated")
      ("quiet", "quiet mode")
      ("stim_spike_mon", "monitor spikes of stimulus group")
      ("block_local", "block local outgoing synapses when blocking neurons")
      ("block_cross_region", "block cross-region outgoing synapses when blocking neurons")
      ("weight_mon_ee_stim_thl", "monitor weights of stim->thl EE connection")
      ("weightstat_mon_ee_stim_thl", "monitor weight stats of stim->thl EE connection")
      ("weightpat_mon_ee_stim_thl", "monitor weight pattern of stim->thl EE connection")
      ("exc_spike_mon_thl", "monitor spikes of excitatory neurons in thl")
      ("exc_prate_mon_thl", "monitor population rate of excitatory neurons in thl")
      ("exc_pattern_mon_thl", "monitor pattern of excitatory neurons in thl")
      ("exc_voltage_mon_thl", "monitor voltage of thl excitatory neuron record_neuron_exc_thl")
      ("exc_g_ampa_mon_thl", "monitor g_ampa of thl excitatory neuron record_neuron_exc_thl")
      ("exc_g_nmda_mon_thl", "monitor g_nmda of thl excitatory neuron record_neuron_exc_thl")
      ("exc_g_gaba_mon_thl", "monitor g_gaba of thl excitatory neuron record_neuron_exc_thl")
      ("exc_g_adapt1_mon_thl", "monitor g_adapt1 of thl excitatory neuron record_neuron_exc_thl")
      ("exc_g_adapt2_mon_thl", "monitor g_adapt2 of thl excitatory neuron record_neuron_exc_thl")
      ("exc_thr_mon_thl", "monitor threshold of thl excitatory neuron record_neuron_exc_thl")
      ("exc_ratechk_thl", "check firing rate of excitatory neurons in thl")
      ("inh_spike_mon_thl", "monitor spikes of inhibitory neurons in thl")
      ("inh_prate_mon_thl", "monitor population rate of inhibitory neurons in thl")
      ("inh_voltage_mon_thl", "monitor voltage of thl inhibitory neuron record_neuron_inh_thl")
      ("weight_mon_ee_thl", "monitor weight of recurrent excitatory connections in thl")
      ("weightstat_mon_ee_thl", "monitor weight stats of recurrent excitatory connections in thl")
      ("weightpat_mon_ee_thl", "monitor weight pattern of recurrent excitatory connections in thl")
      ("ee_hom_mon_thl", "monitor state of recurrent excitatory connections in thl")
      ("ei_weight_mon_thl", "monitor weight of excitatory->inhibitory connections in thl")
      ("ie_weight_mon_thl", "monitor weight of inhibitory->excitatory connections in thl")
      ("ie_weightstat_mon_thl", "monitor weight stats of inh->exc connections in thl")
      ("ii_weight_mon_thl", "monitor weight of recurrent inhibitory connections in thl")
      ("weight_mon_ee_stim_ctx", "monitor weights of stim->ctx EE connection")
      ("weightstat_mon_ee_stim_ctx", "monitor weight stats of stim->ctx EE connection")
      ("weightpat_mon_ee_stim_ctx", "monitor weight pattern of stim->ctx EE connection")
      ("exc_spike_mon_ctx", "monitor spikes of excitatory neurons in ctx")
      ("exc_prate_mon_ctx", "monitor population rate of excitatory neurons in ctx")
      ("exc_pattern_mon_ctx", "monitor pattern of excitatory neurons in ctx")
      ("exc_voltage_mon_ctx", "monitor voltage of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_g_ampa_mon_ctx", "monitor g_ampa of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_g_nmda_mon_ctx", "monitor g_nmda of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_g_gaba_mon_ctx", "monitor g_gaba of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_g_adapt1_mon_ctx", "monitor g_adapt1 of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_g_adapt2_mon_ctx", "monitor g_adapt2 of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_thr_mon_ctx", "monitor threshold of ctx excitatory neuron record_neuron_exc_ctx")
      ("exc_ratechk_ctx", "check firing rate of excitatory neurons in ctx")
      ("inh_spike_mon_ctx", "monitor spikes of inhibitory neurons in ctx")
      ("inh_prate_mon_ctx", "monitor population rate of inhibitory neurons in ctx")
      ("inh_voltage_mon_ctx", "monitor voltage of ctx inhibitory neuron record_neuron_inh_ctx")
      ("weight_mon_ee_ctx", "monitor weight of recurrent excitatory connections in ctx")
      ("weightstat_mon_ee_ctx", "monitor weight stats of recurrent excitatory connections in ctx")
      ("weightpat_mon_ee_ctx", "monitor weight pattern of recurrent excitatory connections in ctx")
      ("ee_hom_mon_ctx", "monitor state of recurrent excitatory connections in ctx")
      ("ei_weight_mon_ctx", "monitor weight of excitatory->inhibitory connections in ctx")
      ("ie_weight_mon_ctx", "monitor weight of inhibitory->excitatory connections in ctx")
      ("ie_weightstat_mon_ctx", "monitor weight stats of inh->exc connections in ctx")
      ("ii_weight_mon_ctx", "monitor weight of recurrent inhibitory connections in ctx")
      ("weight_mon_ee_stim_hpc", "monitor weights of stim->ctx EE connection")
      ("weightstat_mon_ee_stim_hpc", "monitor weight stats of stim->ctx EE connection")
      ("weightpat_mon_ee_stim_hpc", "monitor weight pattern of stim->ctx EE connection")
      ("exc_spike_mon_hpc", "monitor spikes of excitatory neurons in hpc")
      ("exc_prate_mon_hpc", "monitor population rate of excitatory neurons in hpc")
      ("exc_pattern_mon_hpc", "monitor pattern of excitatory neurons in hpc")
      ("exc_voltage_mon_hpc", "monitor voltage of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_g_ampa_mon_hpc", "monitor g_ampa of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_g_nmda_mon_hpc", "monitor g_nmda of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_g_gaba_mon_hpc", "monitor g_gaba of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_g_adapt1_mon_hpc", "monitor g_adapt1 of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_g_adapt2_mon_hpc", "monitor g_adapt2 of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_thr_mon_hpc", "monitor threshold of hpc excitatory neuron record_neuron_exc_hpc")
      ("exc_ratechk_hpc", "check firing rate of excitatory neurons in hpc")
      ("inh_spike_mon_hpc", "monitor spikes of inhibitory neurons in hpc")
      ("inh_prate_mon_hpc", "monitor population rate of inhibitory neurons in hpc")
      ("inh_voltage_mon_hpc", "monitor voltage of hpc inhibitory neuron record_neuron_inh_hpc")
      ("weight_mon_ee_hpc", "monitor weight of recurrent excitatory connections in hpc")
      ("weightstat_mon_ee_hpc", "monitor weight stats of recurrent excitatory connections in hpc")
      ("weightpat_mon_ee_hpc", "monitor weight pattern of recurrent excitatory connections in hpc")
      ("ee_hom_mon_hpc", "monitor state of recurrent excitatory connections in hpc")
      ("ei_weight_mon_hpc", "monitor weight of excitatory->inhibitory connections in hpc")
      ("ie_weight_mon_hpc", "monitor weight of inhibitory->excitatory connections in hpc")
      ("ie_weightstat_mon_hpc", "monitor weight stats of inh->exc connections in hpc")
      ("ii_weight_mon_hpc", "monitor weight of recurrent inhibitory connections in hpc")
      ("stim_seed", po::value<int>(), "stimulus random seed ")
      ("master_seed", po::value<int>(), "master random seed ")
      ("rep_seed", po::value<int>(), "replay random seed ")
      ("bg_thl_seed", po::value<int>(), "thl background input random seed ")
      ("bg_ctx_seed", po::value<int>(), "ctx background input random seed ")
      ("bg_hpc_seed", po::value<int>(), "hpc background input random seed ")
      ("bg_spike_mon", "monitor spikes of background input groups")
      ("bgfile_thl", po::value<string>(), "thl background input pattern file")
      ("w_bg_thl", po::value<double>(), "weight for background->thl")
      ("w_bg_ei_thl", po::value<double>(), "weight for background->thl")
      ("sparseness_bg_thl", po::value<double>(), "sparseness of background->thl")
      ("recfile_bg_thl", po::value<string>(), "background->thl receptive field file")
      ("xi_bg_thl", po::value<double>(), "scaling factor for background->thl receptive field")
      ("recfile_ei_bg_thl", po::value<string>(), "background->thl receptive field file")
      ("xi_ei_bg_thl", po::value<double>(), "scaling factor for background->thl receptive field")
      ("bgfile_ctx", po::value<string>(), "ctx background input pattern file")
      ("w_bg_ctx", po::value<double>(), "weight for background->ctx")
      ("w_bg_ei_ctx", po::value<double>(), "weight for background->ctx")
      ("sparseness_bg_ctx", po::value<double>(), "sparseness of background->ctx")
      ("recfile_bg_ctx", po::value<string>(), "background->ctx receptive field file")
      ("xi_bg_ctx", po::value<double>(), "scaling factor for background->ctx receptive field")
      ("recfile_ei_bg_ctx", po::value<string>(), "background->ctx receptive field file")
      ("xi_ei_bg_ctx", po::value<double>(), "scaling factor for background->ctx receptive field")
      ("bgfile_hpc", po::value<string>(), "hpc background input pattern file")
      ("w_bg_hpc", po::value<double>(), "weight for background->hpc")
      ("w_bg_ei_hpc", po::value<double>(), "weight for background->hpc")
      ("sparseness_bg_hpc", po::value<double>(), "sparseness of background->hpc")
      ("recfile_bg_hpc", po::value<string>(), "background->hpc receptive field file")
      ("xi_bg_hpc", po::value<double>(), "scaling factor for background->hpc receptive field")
      ("recfile_ei_bg_hpc", po::value<string>(), "background->hpc receptive field file")
      ("xi_ei_bg_hpc", po::value<double>(), "scaling factor for background->hpc receptive field")
      ("size_rep", po::value<int>(), "number of Poisson neurons in replay")
      ("repfile", po::value<string>(), "replay pattern file")
      ("rep_spike_mon", "monitor spikes of replay group")
      ("w_rep_thl", po::value<double>(), "weight for replay->thl")
      ("sparseness_rep_thl", po::value<double>(), "sparseness of replay->thl")
      ("recfile_rep_thl", po::value<string>(), "replay->thl receptive field file")
      ("xi_rep_thl", po::value<double>(), "scaling factor for rep->thl receptive field")
      ("recfile_ei_rep_thl", po::value<string>(), "replay->thl receptive field file")
      ("xi_ei_rep_thl", po::value<double>(), "scaling factor for rep->thl receptive field")
      ("w_rep_ctx", po::value<double>(), "weight for replay->ctx")
      ("sparseness_rep_ctx", po::value<double>(), "sparseness of replay->ctx")
      ("recfile_rep_ctx", po::value<string>(), "replay->ctx receptive field file")
      ("xi_rep_ctx", po::value<double>(), "scaling factor for rep->ctx receptive field")
      ("recfile_ei_rep_ctx", po::value<string>(), "replay->ctx receptive field file")
      ("xi_ei_rep_ctx", po::value<double>(), "scaling factor for rep->ctx receptive field")
      ("w_rep_hpc", po::value<double>(), "weight for replay->hpc")
      ("sparseness_rep_hpc", po::value<double>(), "sparseness of replay->hpc")
      ("recfile_rep_hpc", po::value<string>(), "replay->hpc receptive field file")
      ("xi_rep_hpc", po::value<double>(), "scaling factor for rep->hpc receptive field")
      ("recfile_ei_rep_hpc", po::value<string>(), "replay->hpc receptive field file")
      ("xi_ei_rep_hpc", po::value<double>(), "scaling factor for rep->hpc receptive field")
      ("bgrate_rep", po::value<double>(), "background rate of replay")
      ("bgrate_thl", po::value<double>(), "rate of background input to thl")
      ("bgrate_ctx", po::value<double>(), "rate of background input to ctx")
      ("bgrate_hpc", po::value<double>(), "rate of background input to hpc")
      ("exc_size_thl", po::value<int>(), "number of excitatory neurons in thl")
      ("eta_thl", po::value<double>(), "EE learning rate in thl (<0 values turn off stdp if P11B)")
      ("eta_stim_thl", po::value<double>(), "learning rate in stim->thl (<0 values turn off stdp if P11B)")
      ("eta_exc_inh_thl", po::value<double>(), "ratio of EE and IE learning rates in thl")
      ("exc_inh_thl", po::value<int>(), "excitatory to inhibitory ratio in thl")
      ("alpha_thl", po::value<double>(), "target exc rate for inh neurons in thl")
      ("kappa_thl", po::value<double>(), "hom parameter in thl")
      ("tauf_ee_thl", po::value<double>(), "time constant of synaptic facilitation in thl E->E")
      ("taud_ee_thl", po::value<double>(), "time constant of synaptic depression in thl E->E")
      ("tauh_ee_thl", po::value<double>(), "time constant of homeostasis in thl E->E")
      ("tauc_ee_thl", po::value<double>(), "time constant of consolidation in thl E->E")
      ("ujump_ee_thl", po::value<double>(), "u jump STP constant in thl E->E")
      ("tauf_ee_stim_thl", po::value<double>(), "time constant of synaptic facilitation in stim->thl")
      ("taud_ee_stim_thl", po::value<double>(), "time constant of synaptic depression in stim->thl")
      ("tauh_ee_stim_thl", po::value<double>(), "time constant of homeostasis in stim->thl")
      ("tauc_ee_stim_thl", po::value<double>(), "time constant of consolidation in stim->thl")
      ("ujump_ee_stim_thl", po::value<double>(), "u jump STP constant in stim->thl")
      ("beta_thl", po::value<double>(), "heterosynaptic plasticity strength parameter in thl ee")
      ("beta_stim_thl", po::value<double>(), "heterosynaptic plasticity strength parameter in stim->thl")
      ("delta_thl", po::value<double>(), "transmitter triggered plasticity strength in thl")
      ("weight_a_thl", po::value<double>(), "weight_a_thl")
      ("weight_c_thl", po::value<double>(), "weight_c_thl")
      ("adapt1_thl", po::value<double>(), "adaptation 1 jump size for long time constant in thl")
      ("adapt2_thl", po::value<double>(), "adaptation 2 jump size for long time constant in thl")
      ("pot_strength_thl", po::value<double>(), "potential strength parameter in thl")
      ("wmax_exc_thl", po::value<double>(), "wmax_exc_thl")
      ("wmin_exc_thl", po::value<double>(), "wmin_exc_thl")
      ("wmax_inh_thl", po::value<double>(), "wmax_inh_thl")
      ("wmin_inh_thl", po::value<double>(), "wmin_inh_thl")
      ("wee_thl", po::value<double>(), "recurrent weight (wee_thl)")
      ("wei_thl", po::value<double>(), "recurrent weight (wei_thl)")
      ("wie_thl", po::value<double>(), "recurrent weight (wie_thl)")
      ("wii_thl", po::value<double>(), "recurrent weight (wii_thl)")
      ("wext_thl", po::value<double>(), "stim->thl excitatory weight onto excitatory neurons")
      ("wext_ei_thl", po::value<double>(), "stim->thl excitatory weight onto inhibitory neurons")
      ("sparseness_int_ee_thl", po::value<double>(), "internal ee sparseness in thl")
      ("sparseness_int_ei_thl", po::value<double>(), "internal ei sparseness in thl")
      ("sparseness_int_ii_thl", po::value<double>(), "internal ii sparseness in thl")
      ("sparseness_int_ie_thl", po::value<double>(), "internal ie sparseness in thl")
      ("sparseness_ext_thl", po::value<double>(), "stim->thl sparseness")
      ("tauf_ei_thl", po::value<double>(), "time constant of synaptic facilitation in thl E->I")
      ("taud_ei_thl", po::value<double>(), "time constant of synaptic depression in thl E->I")
      ("ujump_ei_thl", po::value<double>(), "u jump STP constant in thl E->I")
      ("tauh_ie_thl", po::value<double>(), "time constant of homeostasis in thl I->E")
      ("taud_ei_stim_thl", po::value<double>(), "time constant of synaptic depression in thl S->I")
      ("tauf_ei_stim_thl", po::value<double>(), "time constant of synaptic facilitation in thl S->I")
      ("ujump_ei_stim_thl", po::value<double>(), "u jump STP constant in thl S->I")
      ("tau_ampa_e_thl", po::value<double>(), "AMPA time constant of excitatory neurons in thl")
      ("tau_gaba_e_thl", po::value<double>(), "GABA time constant of excitatory neurons in thl")
      ("tau_nmda_e_thl", po::value<double>(), "NMDA time constant of excitatory neurons in thl")
      ("ampa_nmda_e_thl", po::value<double>(), "AMPA/NMDA ratio for excitatory neurons in thl")
      ("tau_ampa_i_thl", po::value<double>(), "AMPA time constant of inhibitory neurons in thl")
      ("tau_gaba_i_thl", po::value<double>(), "GABA time constant of inhibitory neurons in thl")
      ("tau_nmda_i_thl", po::value<double>(), "NMDA time constant of inhibitory neurons in thl")
      ("ampa_nmda_i_thl", po::value<double>(), "AMPA/NMDA ratio for inhibitory neurons in thl")
      ("prefile_thl", po::value<string>(), "thl preload pattern file")
      ("chi_thl", po::value<double>(), "thl chi factor - pattern preload strength")
      ("exc_size_ctx", po::value<int>(), "number of excitatory neurons in ctx")
      ("eta_ctx", po::value<double>(), "EE learning rate in ctx (<0 values turn off stdp if P11B)")
      ("eta_stim_ctx", po::value<double>(), "learning rate in stim->ctx (<0 values turn off stdp if P11B)")
      ("eta_exc_inh_ctx", po::value<double>(), "ratio of EE and IE learning rates in ctx")
      ("exc_inh_ctx", po::value<int>(), "excitatory to inhibitory ratio in ctx")
      ("alpha_ctx", po::value<double>(), "target exc rate for inh neurons in ctx")
      ("kappa_ctx", po::value<double>(), "hom parameter in ctx")
      ("tauf_ee_ctx", po::value<double>(), "time constant of synaptic facilitation in ctx E->E")
      ("taud_ee_ctx", po::value<double>(), "time constant of synaptic depression in ctx E->E")
      ("tauh_ee_ctx", po::value<double>(), "time constant of homeostasis in ctx E->E")
      ("tauc_ee_ctx", po::value<double>(), "time constant of consolidation in ctx E->E")
      ("ujump_ee_ctx", po::value<double>(), "u jump STP constant in ctx E->E")
      ("tauf_ee_stim_ctx", po::value<double>(), "time constant of synaptic facilitation in stim->ctx")
      ("taud_ee_stim_ctx", po::value<double>(), "time constant of synaptic depression in stim->ctx")
      ("tauh_ee_stim_ctx", po::value<double>(), "time constant of homeostasis in stim->ctx")
      ("tauc_ee_stim_ctx", po::value<double>(), "time constant of consolidation in stim->ctx")
      ("ujump_ee_stim_ctx", po::value<double>(), "u jump STP constant in stim->ctx")
      ("beta_ctx", po::value<double>(), "heterosynaptic plasticity strength parameter in ctx ee")
      ("beta_stim_ctx", po::value<double>(), "heterosynaptic plasticity strength parameter in stim->ctx")
      ("delta_ctx", po::value<double>(), "transmitter triggered plasticity strength in ctx")
      ("weight_a_ctx", po::value<double>(), "weight_a_ctx")
      ("weight_c_ctx", po::value<double>(), "weight_c_ctx")
      ("adapt1_ctx", po::value<double>(), "adaptation 1 jump size for long time constant in ctx")
      ("adapt2_ctx", po::value<double>(), "adaptation 2 jump size for long time constant in ctx")
      ("pot_strength_ctx", po::value<double>(), "potential strength parameter in ctx")
      ("wmax_exc_ctx", po::value<double>(), "wmax_exc_ctx")
      ("wmin_exc_ctx", po::value<double>(), "wmin_exc_ctx")
      ("wmax_inh_ctx", po::value<double>(), "wmax_inh_ctx")
      ("wmin_inh_ctx", po::value<double>(), "wmin_inh_ctx")
      ("wee_ctx", po::value<double>(), "recurrent weight (wee_ctx)")
      ("wei_ctx", po::value<double>(), "recurrent weight (wei_ctx)")
      ("wie_ctx", po::value<double>(), "recurrent weight (wie_ctx)")
      ("wii_ctx", po::value<double>(), "recurrent weight (wii_ctx)")
      ("wext_ctx", po::value<double>(), "stim->ctx excitatory weight onton excitatory neurons")
      ("wext_ei_ctx", po::value<double>(), "stim->ctx excitatory weight onto inhibitory neurons")
      ("sparseness_int_ee_ctx", po::value<double>(), "internal ee sparseness in ctx")
      ("sparseness_int_ei_ctx", po::value<double>(), "internal ei sparseness in ctx")
      ("sparseness_int_ii_ctx", po::value<double>(), "internal ii sparseness in ctx")
      ("sparseness_int_ie_ctx", po::value<double>(), "internal ie sparseness in ctx")
      ("sparseness_ext_ctx", po::value<double>(), "stim->ctx sparseness")
      ("tauf_ei_ctx", po::value<double>(), "time constant of synaptic facilitation in ctx E->I")
      ("taud_ei_ctx", po::value<double>(), "time constant of synaptic depression in ctx E->I")
      ("ujump_ei_ctx", po::value<double>(), "u jump STP constant in ctx E->I")
      ("tauh_ie_ctx", po::value<double>(), "time constant of homeostasis in ctx I->E")
      ("taud_ei_stim_ctx", po::value<double>(), "time constant of synaptic depression in ctx S->I")
      ("tauf_ei_stim_ctx", po::value<double>(), "time constant of synaptic facilitation in ctx S->I")
      ("ujump_ei_stim_ctx", po::value<double>(), "u jump STP constant in ctx S->I")
      ("tau_ampa_e_ctx", po::value<double>(), "AMPA time constant of excitatory neurons in ctx")
      ("tau_gaba_e_ctx", po::value<double>(), "GABA time constant of excitatory neurons in ctx")
      ("tau_nmda_e_ctx", po::value<double>(), "NMDA time constant of excitatory neurons in ctx")
      ("ampa_nmda_e_ctx", po::value<double>(), "AMPA/NMDA ratio for excitatory neurons in ctx")
      ("tau_ampa_i_ctx", po::value<double>(), "AMPA time constant of inhibitory neurons in ctx")
      ("tau_gaba_i_ctx", po::value<double>(), "GABA time constant of inhibitory neurons in ctx")
      ("tau_nmda_i_ctx", po::value<double>(), "NMDA time constant of inhibitory neurons in ctx")
      ("ampa_nmda_i_ctx", po::value<double>(), "AMPA/NMDA ratio for inhibitory neurons in ctx")
      ("prefile_ctx", po::value<string>(), "ctx preload pattern file")
      ("chi_ctx", po::value<double>(), "ctx chi factor - pattern preload strength")
      ("exc_size_hpc", po::value<int>(), "number of excitatory neurons in hpc")
      ("eta_hpc", po::value<double>(), "learning rate in hpc (<0 values turn off stdp if P11B)")
      ("eta_stim_hpc", po::value<double>(), "learning rate in stim->hpc (<0 values turn off stdp if P11B)")
      ("eta_exc_inh_hpc", po::value<double>(), "ratio of EE and IE learning rates in hpc")
      ("exc_inh_hpc", po::value<int>(), "excitatory to inhibitory ratio in hpc")
      ("alpha_hpc", po::value<double>(), "target exc rate for inh neurons in hpc")
      ("kappa_hpc", po::value<double>(), "hom parameter in hpc")
      ("tauf_ee_hpc", po::value<double>(), "time constant of synaptic facilitation in hpc E->E")
      ("taud_ee_hpc", po::value<double>(), "time constant of synaptic depression in hpc E->E")
      ("tauh_ee_hpc", po::value<double>(), "time constant of homeostasis in hpc E->E")
      ("tauc_ee_hpc", po::value<double>(), "time constant of consolidation in hpc E->E")
      ("ujump_ee_hpc", po::value<double>(), "u jump STP constant in hpc E->E")
      ("tauf_ee_stim_hpc", po::value<double>(), "time constant of synaptic facilitation in stim->hpc")
      ("taud_ee_stim_hpc", po::value<double>(), "time constant of synaptic depression in stim->hpc")
      ("tauh_ee_stim_hpc", po::value<double>(), "time constant of homeostasis in stim->hpc")
      ("tauc_ee_stim_hpc", po::value<double>(), "time constant of consolidation in stim->hpc")
      ("ujump_ee_stim_hpc", po::value<double>(), "u jump STP constant in stim->hpc")
      ("beta_hpc", po::value<double>(), "heterosynaptic plasticity strength parameter in hpc")
      ("beta_stim_hpc", po::value<double>(), "heterosynaptic plasticity strength parameter in stim->hpc")
      ("delta_hpc", po::value<double>(), "transmitter triggered plasticity strength in hpc")
      ("weight_a_hpc", po::value<double>(), "weight_a_hpc")
      ("weight_c_hpc", po::value<double>(), "weight_c_hpc")
      ("adapt1_hpc", po::value<double>(), "adaptation 1 jump size for long time constant in hpc")
      ("adapt2_hpc", po::value<double>(), "adaptation 2 jump size for long time constant in hpc")
      ("pot_strength_hpc", po::value<double>(), "potential strength parameter in hpc")
      ("wmax_exc_hpc", po::value<double>(), "wmax_exc_hpc")
      ("wmin_exc_hpc", po::value<double>(), "wmin_exc_hpc")
      ("wmax_inh_hpc", po::value<double>(), "wmax_inh_hpc")
      ("wmin_inh_hpc", po::value<double>(), "wmin_inh_hpc")
      ("wee_hpc", po::value<double>(), "recurrent weight (wee_hpc)")
      ("wei_hpc", po::value<double>(), "recurrent weight (wei_hpc)")
      ("wie_hpc", po::value<double>(), "recurrent weight (wie_hpc)")
      ("wii_hpc", po::value<double>(), "recurrent weight (wii_hpc)")
      ("wext_hpc", po::value<double>(), "stim->hpc excitatory weight onto excitatory neurons")
      ("wext_ei_hpc", po::value<double>(), "stim->hpc excitatory weight onto inhibitory neurons")
      ("sparseness_int_ee_hpc", po::value<double>(), "internal ee sparseness in hpc")
      ("sparseness_int_ei_hpc", po::value<double>(), "internal ei sparseness in hpc")
      ("sparseness_int_ii_hpc", po::value<double>(), "internal ii sparseness in hpc")
      ("sparseness_int_ie_hpc", po::value<double>(), "internal ie sparseness in hpc")
      ("sparseness_ext_hpc", po::value<double>(), "stim->hpc sparseness")
      ("tauf_ei_hpc", po::value<double>(), "time constant of synaptic facilitation in hpc E->I")
      ("taud_ei_hpc", po::value<double>(), "time constant of synaptic depression in hpc E->I")
      ("ujump_ei_hpc", po::value<double>(), "u jump STP constant in hpc E->I")
      ("tauh_ie_hpc", po::value<double>(), "time constant of homeostasis in hpc I->E")
      ("taud_ei_stim_hpc", po::value<double>(), "time constant of synaptic depression in hpc S->I")
      ("tauf_ei_stim_hpc", po::value<double>(), "time constant of synaptic facilitation in hpc S->I")
      ("ujump_ei_stim_hpc", po::value<double>(), "u jump STP constant in hpc S->I")
      ("tau_ampa_e_hpc", po::value<double>(), "AMPA time constant of excitatory neurons in hpc")
      ("tau_gaba_e_hpc", po::value<double>(), "GABA time constant of excitatory neurons in hpc")
      ("tau_nmda_e_hpc", po::value<double>(), "NMDA time constant of excitatory neurons in hpc")
      ("ampa_nmda_e_hpc", po::value<double>(), "AMPA/NMDA ratio for excitatory neurons in hpc")
      ("tau_ampa_i_hpc", po::value<double>(), "AMPA time constant of inhibitory neurons in hpc")
      ("tau_gaba_i_hpc", po::value<double>(), "GABA time constant of inhibitory neurons in hpc")
      ("tau_nmda_i_hpc", po::value<double>(), "NMDA time constant of inhibitory neurons in hpc")
      ("ampa_nmda_i_hpc", po::value<double>(), "AMPA/NMDA ratio for inhibitory neurons in hpc")
      ("prefile_hpc", po::value<string>(), "hpc preload pattern file")
      ("chi_hpc", po::value<double>(), "hpc chi factor - pattern preload strength")
      ("eta_exc_inh2_hpc", po::value<double>(), "ratio of EE and I2E learning rates in hpc")
      ("exc_inh2_hpc", po::value<int>(), "excitatory to inhibitory2 ratio in hpc")
      ("alpha2_hpc", po::value<double>(), "target exc rate for inh2 neurons in hpc")
      ("tau_ampa_i2_hpc", po::value<double>(), "AMPA time constant of inhibitory2 neurons in hpc")
      ("tau_gaba_i2_hpc", po::value<double>(), "GABA time constant of inhibitory2 neurons in hpc")
      ("tau_nmda_i2_hpc", po::value<double>(), "NMDA time constant of inhibitory2 neurons in hpc")
      ("ampa_nmda_i2_hpc", po::value<double>(), "AMPA/NMDA ratio for inhibitory2 neurons in hpc")
      ("wei2_hpc", po::value<double>(), "recurrent weight (wei2_hpc)")
      ("wi2e_hpc", po::value<double>(), "recurrent weight (wi2e_hpc)")
      ("wi2i2_hpc", po::value<double>(), "recurrent weight (wi2i2_hpc)")
      ("wii2_hpc", po::value<double>(), "recurrent weight (wii2_hpc)")
      ("wi2i_hpc", po::value<double>(), "recurrent weight (wi2i_hpc)")
      ("sparseness_int_ei2_hpc", po::value<double>(), "internal ei2 sparseness in hpc")
      ("sparseness_int_i2e_hpc", po::value<double>(), "internal i2e sparseness in hpc")
      ("sparseness_int_i2i2_hpc", po::value<double>(), "internal i2i2 sparseness in hpc")
      ("sparseness_int_ii2_hpc", po::value<double>(), "internal ii2 sparseness in hpc")
      ("sparseness_int_i2i_hpc", po::value<double>(), "internal i2i sparseness in hpc")
      ("tauf_ei2_hpc", po::value<double>(), "time constant of synaptic facilitation in hpc E->I2")
      ("taud_ei2_hpc", po::value<double>(), "time constant of synaptic depression in hpc E->I2")
      ("ujump_ei2_hpc", po::value<double>(), "u jump STP constant in hpc E->I2")
      ("tauh_i2e_hpc", po::value<double>(), "time constant of homeostasis in hpc I2->E")
      ("wmax_inh2_hpc", po::value<double>(), "wmax_inh2_hpc")
      ("wmin_inh2_hpc", po::value<double>(), "wmin_inh2_hpc")
      ("record_neuron_inh2_hpc", po::value<int>(), "id of hpc inh2 neuron to be recorded")
      ("block_inh2_hpc", po::value<string>(), "file with hpc inh2 neurons to be blocked")
      ("wee_thl_ctx", po::value<double>(), "weight for thl->ctx")
      ("wei_thl_ctx", po::value<double>(), "weight for thl->ctx")
      ("sparseness_thl_ctx", po::value<double>(), "sparseness of thl->ctx")
      ("recfile_thl_ctx", po::value<string>(), "thl->ctx receptive field file")
      ("xi_thl_ctx", po::value<double>(), "scaling factor for thl->ctx receptive field")
      ("eta_thl_ctx", po::value<double>(), "learning rate in thl->ctx")
      ("kappa_thl_ctx", po::value<double>(), "hom parameter in thl->ctx")
      ("tauf_thl_ctx", po::value<double>(), "time constant of synaptic facilitation in thl->ctx")
      ("taud_thl_ctx", po::value<double>(), "time constant of synaptic depression in thl->ctx")
      ("tauh_thl_ctx", po::value<double>(), "time constant of homeostasis in thl->ctx")
      ("tauc_thl_ctx", po::value<double>(), "time constant of consolidation in thl->ctx")
      ("ujump_thl_ctx", po::value<double>(), "u jump STP constant in thl->ctx")
      ("beta_thl_ctx", po::value<double>(), "heterosynaptic plasticity strength parameter in thl->ctx")
      ("delta_thl_ctx", po::value<double>(), "transmitter triggered plasticity strength in thl->ctx")
      ("weight_a_thl_ctx", po::value<double>(), "weight_a_thl_ctx")
      ("weight_c_thl_ctx", po::value<double>(), "weight_c_thl_ctx")
      ("pot_strength_thl_ctx", po::value<double>(), "potential strength parameter in thl->ctx")
      ("wmax_thl_ctx", po::value<double>(), "wmax_thl_ctx")
      ("wmin_thl_ctx", po::value<double>(), "wmin_thl_ctx")
      ("wee_ctx_thl", po::value<double>(), "weight for ctx->thl")
      ("wei_ctx_thl", po::value<double>(), "weight for ctx->thl")
      ("sparseness_ctx_thl", po::value<double>(), "sparseness of ctx->thl")
      ("recfile_ctx_thl", po::value<string>(), "ctx->thl receptive field file")
      ("xi_ctx_thl", po::value<double>(), "scaling factor for ctx->thl receptive field")
      ("eta_ctx_thl", po::value<double>(), "learning rate in ctx->thl")
      ("kappa_ctx_thl", po::value<double>(), "hom parameter in ctx->thl")
      ("tauf_ctx_thl", po::value<double>(), "time constant of synaptic facilitation in ctx->thl")
      ("taud_ctx_thl", po::value<double>(), "time constant of synaptic depression in ctx->thl")
      ("tauh_ctx_thl", po::value<double>(), "time constant of homeostasis in ctx->thl")
      ("tauc_ctx_thl", po::value<double>(), "time constant of consolidation in ctx->thl")
      ("ujump_ctx_thl", po::value<double>(), "u jump STP constant in ctx->thl")
      ("beta_ctx_thl", po::value<double>(), "heterosynaptic plasticity strength parameter in ctx->thl")
      ("delta_ctx_thl", po::value<double>(), "transmitter triggered plasticity strength in ctx->thl")
      ("weight_a_ctx_thl", po::value<double>(), "weight_a_ctx_thl")
      ("weight_c_ctx_thl", po::value<double>(), "weight_c_ctx_thl")
      ("pot_strength_ctx_thl", po::value<double>(), "potential strength parameter in ctx->thl")
      ("wmax_ctx_thl", po::value<double>(), "wmax_ctx_thl")
      ("wmin_ctx_thl", po::value<double>(), "wmin_ctx_thl")
      ("wee_thl_hpc", po::value<double>(), "weight for thl->hpc")
      ("wei_thl_hpc", po::value<double>(), "weight for thl->hpc")
      ("sparseness_thl_hpc", po::value<double>(), "sparseness of thl->hpc")
      ("recfile_thl_hpc", po::value<string>(), "thl->hpc receptive field file")
      ("xi_thl_hpc", po::value<double>(), "scaling factor for thl->hpc receptive field")
      ("eta_thl_hpc", po::value<double>(), "learning rate in thl->hpc")
      ("kappa_thl_hpc", po::value<double>(), "hom parameter in thl->hpc")
      ("tauf_thl_hpc", po::value<double>(), "time constant of synaptic facilitation in thl->hpc")
      ("taud_thl_hpc", po::value<double>(), "time constant of synaptic depression in thl->hpc")
      ("tauh_thl_hpc", po::value<double>(), "time constant of homeostasis in thl->hpc")
      ("tauc_thl_hpc", po::value<double>(), "time constant of consolidation in thl->hpc")
      ("ujump_thl_hpc", po::value<double>(), "u jump STP constant in thl->hpc")
      ("beta_thl_hpc", po::value<double>(), "heterosynaptic plasticity strength parameter in thl->hpc")
      ("delta_thl_hpc", po::value<double>(), "transmitter triggered plasticity strength in thl->hpc")
      ("weight_a_thl_hpc", po::value<double>(), "weight_a_thl_hpc")
      ("weight_c_thl_hpc", po::value<double>(), "weight_c_thl_hpc")
      ("pot_strength_thl_hpc", po::value<double>(), "potential strength parameter in thl->hpc")
      ("wmax_thl_hpc", po::value<double>(), "wmax_thl_hpc")
      ("wmin_thl_hpc", po::value<double>(), "wmin_thl_hpc")
      ("wee_hpc_thl", po::value<double>(), "weight for hpc->thl")
      ("wei_hpc_thl", po::value<double>(), "weight for hpc->thl")
      ("sparseness_hpc_thl", po::value<double>(), "sparseness of hpc->thl")
      ("recfile_hpc_thl", po::value<string>(), "hpc->thl receptive field file")
      ("xi_hpc_thl", po::value<double>(), "scaling factor for hpc->thl receptive field")
      ("eta_hpc_thl", po::value<double>(), "learning rate in hpc->thl")
      ("kappa_hpc_thl", po::value<double>(), "hom parameter in hpc->thl")
      ("tauf_hpc_thl", po::value<double>(), "time constant of synaptic facilitation in hpc->thl")
      ("taud_hpc_thl", po::value<double>(), "time constant of synaptic depression in hpc->thl")
      ("tauh_hpc_thl", po::value<double>(), "time constant of homeostasis in hpc->thl")
      ("tauc_hpc_thl", po::value<double>(), "time constant of consolidation in hpc->thl")
      ("ujump_hpc_thl", po::value<double>(), "u jump STP constant in hpc->thl")
      ("beta_hpc_thl", po::value<double>(), "heterosynaptic plasticity strength parameter in hpc->thl")
      ("delta_hpc_thl", po::value<double>(), "transmitter triggered plasticity strength in hpc->thl")
      ("weight_a_hpc_thl", po::value<double>(), "weight_a_hpc_thl")
      ("weight_c_hpc_thl", po::value<double>(), "weight_c_hpc_thl")
      ("pot_strength_hpc_thl", po::value<double>(), "potential strength parameter in hpc->thl")
      ("wmax_hpc_thl", po::value<double>(), "wmax_hpc_thl")
      ("wmin_hpc_thl", po::value<double>(), "wmin_hpc_thl")
      ("wee_ctx_hpc", po::value<double>(), "weight for ctx->hpc")
      ("wei_ctx_hpc", po::value<double>(), "weight for ctx->hpc")
      ("sparseness_ctx_hpc", po::value<double>(), "sparseness of ctx->hpc")
      ("recfile_ctx_hpc", po::value<string>(), "ctx->hpc receptive field file")
      ("xi_ctx_hpc", po::value<double>(), "scaling factor for ctx->hpc receptive field")
      ("eta_ctx_hpc", po::value<double>(), "learning rate in ctx->hpc")
      ("kappa_ctx_hpc", po::value<double>(), "hom parameter in ctx->hpc")
      ("tauf_ctx_hpc", po::value<double>(), "time constant of synaptic facilitation in ctx->hpc")
      ("taud_ctx_hpc", po::value<double>(), "time constant of synaptic depression in ctx->hpc")
      ("tauh_ctx_hpc", po::value<double>(), "time constant of homeostasis in ctx->hpc")
      ("tauc_ctx_hpc", po::value<double>(), "time constant of consolidation in ctx->hpc")
      ("ujump_ctx_hpc", po::value<double>(), "u jump STP constant in ctx->hpc")
      ("beta_ctx_hpc", po::value<double>(), "heterosynaptic plasticity strength parameter in ctx->hpc")
      ("delta_ctx_hpc", po::value<double>(), "transmitter triggered plasticity strength in ctx->hpc")
      ("weight_a_ctx_hpc", po::value<double>(), "weight_a_ctx_hpc")
      ("weight_c_ctx_hpc", po::value<double>(), "weight_c_ctx_hpc")
      ("pot_strength_ctx_hpc", po::value<double>(), "potential strength parameter in ctx->hpc")
      ("wmax_ctx_hpc", po::value<double>(), "wmax_ctx_hpc")
      ("wmin_ctx_hpc", po::value<double>(), "wmin_ctx_hpc")
      ("wee_hpc_ctx", po::value<double>(), "weight for hpc->ctx")
      ("wei_hpc_ctx", po::value<double>(), "weight for hpc->ctx")
      ("sparseness_hpc_ctx", po::value<double>(), "sparseness of hpc->ctx")
      ("recfile_hpc_ctx", po::value<string>(), "hpc->ctx receptive field file")
      ("xi_hpc_ctx", po::value<double>(), "scaling factor for hpc->ctx receptive field")
      ("eta_hpc_ctx", po::value<double>(), "learning rate in hpc->ctx")
      ("kappa_hpc_ctx", po::value<double>(), "hom parameter in hpc->ctx")
      ("tauf_hpc_ctx", po::value<double>(), "time constant of synaptic facilitation in hpc->ctx")
      ("taud_hpc_ctx", po::value<double>(), "time constant of synaptic depression in hpc->ctx")
      ("tauh_hpc_ctx", po::value<double>(), "time constant of homeostasis in hpc->ctx")
      ("tauc_hpc_ctx", po::value<double>(), "time constant of consolidation in hpc->ctx")
      ("ujump_hpc_ctx", po::value<double>(), "u jump STP constant in hpc->ctx")
      ("beta_hpc_ctx", po::value<double>(), "heterosynaptic plasticity strength parameter in hpc->ctx")
      ("delta_hpc_ctx", po::value<double>(), "transmitter triggered plasticity strength in hpc->ctx")
      ("weight_a_hpc_ctx", po::value<double>(), "weight_a_hpc_ctx")
      ("weight_c_hpc_ctx", po::value<double>(), "weight_c_hpc_ctx")
      ("pot_strength_hpc_ctx", po::value<double>(), "potential strength parameter in hpc->ctx")
      ("wmax_hpc_ctx", po::value<double>(), "wmax_hpc_ctx")
      ("wmin_hpc_ctx", po::value<double>(), "wmin_hpc_ctx")
      ("ontime", po::value<double>(), "stimulus average on time")
      ("offtime", po::value<double>(), "stimulus average off time")
      ("scale", po::value<double>(), "stimulus strength")
      ("bgrate", po::value<double>(), "background rate of input")
      ("preferred", po::value<int>(), "num of preferred stim")
      ("stimfile", po::value<string>(), "stimulus pattern file")
      ("recfile_stim_ctx", po::value<string>(), "stimulus->ctx receptive field file")
      ("xi_stim_ctx", po::value<double>(), "scaling factor for Stim->Ctx receptive field")
      ("recfile_stim_hpc", po::value<string>(), "stimulus->hpc receptive field file")
      ("xi_stim_hpc", po::value<double>(), "scaling factor for Stim->HPC receptive field")
      ("recfile_stim_thl", po::value<string>(), "stimulus->thl receptive field file")
      ("xi_stim_thl", po::value<double>(), "scaling factor for Stim->THL receptive field")
      ("prime_ontime", po::value<double>(), "stimulus average on time during priming")
      ("prime_offtime", po::value<double>(), "stimulus average off time during priming")
      ("prime_duration", po::value<double>(), "priming training period duration")
      ("record_neuron_exc_ctx", po::value<int>(), "id of ctx exc neuron to be recorded")
      ("record_neuron_inh_ctx", po::value<int>(), "id of ctx inh neuron to be recorded")
      ("record_neuron_exc_hpc", po::value<int>(), "id of hpc exc neuron to be recorded")
      ("record_neuron_inh_hpc", po::value<int>(), "id of hpc inh neuron to be recorded")
      ("record_neuron_exc_thl", po::value<int>(), "id of thl exc neuron to be recorded")
      ("record_neuron_inh_thl", po::value<int>(), "id of thl inh neuron to be recorded")
      ("monfile_ctx", po::value<string>(), "ctx monitor file")
      ("premonfile_ctx", po::value<string>(), "ctx presynaptic monitor file")
      ("monfile_hpc", po::value<string>(), "hpc monitor file")
      ("premonfile_hpc", po::value<string>(), "hpc presynaptic monitor file")
      ("monfile_thl", po::value<string>(), "thl monitor file")
      ("premonfile_thl", po::value<string>(), "thl presynaptic monitor file")
      ("block_stim", po::value<string>(), "file with stim neurons to be blocked")
      ("block_rep", po::value<string>(), "file with replay neurons to be blocked")
      ("block_bg_thl", po::value<string>(), "file with thl background neurons to be blocked")
      ("block_bg_ctx", po::value<string>(), "file with ctx background neurons to be blocked")
      ("block_bg_hpc", po::value<string>(), "file with hpc background neurons to be blocked")
      ("block_exc_thl", po::value<string>(), "file with thl exc neurons to be blocked")
      ("block_inh_thl", po::value<string>(), "file with thl inh neurons to be blocked")
      ("block_exc_ctx", po::value<string>(), "file with ctx exc neurons to be blocked")
      ("block_inh_ctx", po::value<string>(), "file with ctx inh neurons to be blocked")
      ("block_exc_hpc", po::value<string>(), "file with hpc exc neurons to be blocked")
      ("block_inh_hpc", po::value<string>(), "file with hpc inh neurons to be blocked")
      ("inhibit_exc_hpc", po::value<string>(), "file with hpc exc neurons to be inhibited")
      ;

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }

    if (vm.count("binary")) {
      std::cout << "binary " 
		<< vm["binary"].as<string>() << ".\n";
      binary = vm["binary"].as<string>();
    }

    if (vm.count("out_dir")) {
      std::cout << "out_dir set to " 
		<< vm["out_dir"].as<string>() << ".\n";
      out_dir = vm["out_dir"].as<string>();
    }

    if (vm.count("file_prefix")) {
      std::cout << "file_prefix set to " 
		<< vm["file_prefix"].as<string>() << ".\n";
      file_prefix = vm["file_prefix"].as<string>();
    }

    if (vm.count("file_prefix_hm")) {
      std::cout << "file_prefix_hm set to " 
		<< vm["file_prefix_hm"].as<string>() << ".\n";
      file_prefix_hm = vm["file_prefix_hm"].as<string>();
    }

    if (vm.count("load_file")) {
      std::cout << "load_file " 
		<< vm["load_file"].as<string>() << ".\n";
      load_file = vm["load_file"].as<string>();
    }

    if (vm.count("simtimes")) {
      std::cout << "simtimes set to ";      
      for (int index=0; index < vm["simtimes"].as< std::vector<double> >().size(); ++index) {
	std::cout << vm["simtimes"].as< std::vector<double> >()[index] << " ";
	simtimes.push_back(vm["simtimes"].as< std::vector<double> >()[index]);
      }
      std::cout<< ".\n";
      
    }

    if (vm.count("save")) {
      save = true;
    }

    if (vm.count("save_without_hpc")) {
      save_without_hpc = true;
    }

    if (vm.count("load_without_hpc")) {
      load_without_hpc = true;
    }

    if (vm.count("chain")) {
      chain = true;
    } 

    if (vm.count("prime")) {
      prime = true;
    } 

    if (vm.count("noconsolidation")) {
      consolidation = false;
    } 

    if (vm.count("noisp")) {
      isp_active = false;
    }

    if (vm.count("inhinput")) {
      inh_input = true;
    }

    if (vm.count("noisyweights")) {
      noisy_initial_weights = true;
    } 

    if (vm.count("consolidateweights")) {
      consolidate_initial_weights = true;
    }

    if (vm.count("quiet")) {
      quiet = true;
    }

    if (vm.count("stim_spike_mon")) {
      stim_spike_mon = true;
    }

    if (vm.count("block_local")) {
      block_local = true;
    }

    if (vm.count("block_cross_region")) {
      block_cross_region = true;
    }

    if (vm.count("weight_mon_ee_stim_thl")) {
      weight_mon_ee_stim_thl = true;
    }

    if (vm.count("weightstat_mon_ee_stim_thl")) {
      weightstat_mon_ee_stim_thl = true;
    }

    if (vm.count("weightpat_mon_ee_stim_thl")) {
      weightpat_mon_ee_stim_thl = true;
    }

    if (vm.count("exc_spike_mon_thl")) {
      exc_spike_mon_thl = true;
    }

    if (vm.count("exc_prate_mon_thl")) {
      exc_prate_mon_thl = true;
    }

    if (vm.count("exc_pattern_mon_thl")) {
      exc_pattern_mon_thl = true;
    }

    if (vm.count("exc_voltage_mon_thl")) {
      exc_voltage_mon_thl = true;
    }

    if (vm.count("exc_g_ampa_mon_thl")) {
      exc_g_ampa_mon_thl = true;
    }

    if (vm.count("exc_g_nmda_mon_thl")) {
      exc_g_nmda_mon_thl = true;
    }

    if (vm.count("exc_g_gaba_mon_thl")) {
      exc_g_gaba_mon_thl = true;
    }

    if (vm.count("exc_g_adapt1_mon_thl")) {
      exc_g_adapt1_mon_thl = true;
    }

    if (vm.count("exc_g_adapt2_mon_thl")) {
      exc_g_adapt2_mon_thl = true;
    }

    if (vm.count("exc_thr_mon_thl")) {
      exc_thr_mon_thl = true;
    }

    if (vm.count("exc_ratechk_thl")) {
      exc_ratechk_thl = true;
    }

    if (vm.count("inh_spike_mon_thl")) {
      inh_spike_mon_thl = true;
    }

    if (vm.count("inh_prate_mon_thl")) {
      inh_prate_mon_thl = true;
    }

    if (vm.count("inh_voltage_mon_thl")) {
      inh_voltage_mon_thl = true;
    }

    if (vm.count("weight_mon_ee_thl")) {
      weight_mon_ee_thl = true;
    }

    if (vm.count("weightstat_mon_ee_thl")) {
      weightstat_mon_ee_thl = true;
    }

    if (vm.count("weightpat_mon_ee_thl")) {
       weightpat_mon_ee_thl= true;
    }

    if (vm.count("ee_hom_mon_thl")) {
      ee_hom_mon_thl = true;
    }

    if (vm.count("ei_weight_mon_thl")) {
      ei_weight_mon_thl = true;
    }

    if (vm.count("ie_weight_mon_thl")) {
      ie_weight_mon_thl = true;
    }

    if (vm.count("ie_weightstat_mon_thl")) {
      ie_weightstat_mon_thl = true;
    }

    if (vm.count("ii_weight_mon_thl")) {
       ii_weight_mon_thl = true;
    }

    if (vm.count("weight_mon_ee_stim_ctx")) {
      weight_mon_ee_stim_ctx = true;
    }

    if (vm.count("weightstat_mon_ee_stim_ctx")) {
      weightstat_mon_ee_stim_ctx = true;
    }

    if (vm.count("weightpat_mon_ee_stim_ctx")) {
      weightpat_mon_ee_stim_ctx = true;
    }

    if (vm.count("exc_spike_mon_ctx")) {
      exc_spike_mon_ctx = true;
    }

    if (vm.count("exc_prate_mon_ctx")) {
      exc_prate_mon_ctx = true;
    }

    if (vm.count("exc_pattern_mon_ctx")) {
      exc_pattern_mon_ctx = true;
    }

    if (vm.count("exc_voltage_mon_ctx")) {
      exc_voltage_mon_ctx = true;
    }

    if (vm.count("exc_g_ampa_mon_ctx")) {
      exc_g_ampa_mon_ctx = true;
    }

    if (vm.count("exc_g_nmda_mon_ctx")) {
      exc_g_nmda_mon_ctx = true;
    }

    if (vm.count("exc_g_gaba_mon_ctx")) {
      exc_g_gaba_mon_ctx = true;
    }

    if (vm.count("exc_g_adapt1_mon_ctx")) {
      exc_g_adapt1_mon_ctx = true;
    }

    if (vm.count("exc_g_adapt2_mon_ctx")) {
      exc_g_adapt2_mon_ctx = true;
    }

    if (vm.count("exc_thr_mon_ctx")) {
      exc_thr_mon_ctx = true;
    }

    if (vm.count("exc_ratechk_ctx")) {
      exc_ratechk_ctx = true;
    }

    if (vm.count("inh_spike_mon_ctx")) {
      inh_spike_mon_ctx = true;
    }

    if (vm.count("inh_prate_mon_ctx")) {
      inh_prate_mon_ctx = true;
    }

    if (vm.count("inh_voltage_mon_ctx")) {
      inh_voltage_mon_ctx = true;
    }

    if (vm.count("weight_mon_ee_ctx")) {
      weight_mon_ee_ctx = true;
    }

    if (vm.count("weightstat_mon_ee_ctx")) {
      weightstat_mon_ee_ctx = true;
    }

    if (vm.count("weightpat_mon_ee_ctx")) {
       weightpat_mon_ee_ctx= true;
    }

    if (vm.count("ee_hom_mon_ctx")) {
      ee_hom_mon_ctx = true;
    }

    if (vm.count("ei_weight_mon_ctx")) {
      ei_weight_mon_ctx = true;
    }

    if (vm.count("ie_weight_mon_ctx")) {
      ie_weight_mon_ctx = true;
    }

    if (vm.count("ie_weightstat_mon_ctx")) {
      ie_weightstat_mon_ctx = true;
    }

    if (vm.count("ii_weight_mon_ctx")) {
       ii_weight_mon_ctx = true;
    }

    if (vm.count("weight_mon_ee_stim_hpc")) {
      weight_mon_ee_stim_hpc = true;
    }

    if (vm.count("weightstat_mon_ee_stim_hpc")) {
      weightstat_mon_ee_stim_hpc = true;
    }

    if (vm.count("weightpat_mon_ee_stim_hpc")) {
      weightpat_mon_ee_stim_hpc = true;
    }
    
    if (vm.count("exc_spike_mon_hpc")) {
      exc_spike_mon_hpc = true;
    }

    if (vm.count("exc_prate_mon_hpc")) {
      exc_prate_mon_hpc = true;
    }

    if (vm.count("exc_pattern_mon_hpc")) {
      exc_pattern_mon_hpc = true;
    }

    if (vm.count("exc_voltage_mon_hpc")) {
      exc_voltage_mon_hpc = true;
    }

    if (vm.count("exc_g_ampa_mon_hpc")) {
      exc_g_ampa_mon_hpc = true;
    }

    if (vm.count("exc_g_nmda_mon_hpc")) {
      exc_g_nmda_mon_hpc = true;
    }

    if (vm.count("exc_g_gaba_mon_hpc")) {
      exc_g_gaba_mon_hpc = true;
    }

    if (vm.count("exc_g_adapt1_mon_hpc")) {
      exc_g_adapt1_mon_hpc = true;
    }

    if (vm.count("exc_g_adapt2_mon_hpc")) {
      exc_g_adapt2_mon_hpc = true;
    }

    if (vm.count("exc_thr_mon_hpc")) {
      exc_thr_mon_hpc = true;
    }

    if (vm.count("exc_ratechk_hpc")) {
      exc_ratechk_hpc = true;
    }

    if (vm.count("inh_spike_mon_hpc")) {
      inh_spike_mon_hpc = true;
    }

    if (vm.count("inh_prate_mon_hpc")) {
      inh_prate_mon_hpc = true;
    }

    if (vm.count("inh_voltage_mon_hpc")) {
      inh_voltage_mon_hpc = true;
    }

    if (vm.count("weight_mon_ee_hpc")) {
      weight_mon_ee_hpc = true;
    }

    if (vm.count("weightstat_mon_ee_hpc")) {
      weightstat_mon_ee_hpc = true;
    }

    if (vm.count("weightpat_mon_ee_hpc")) {
       weightpat_mon_ee_hpc= true;
    }

    if (vm.count("ee_hom_mon_hpc")) {
      ee_hom_mon_hpc = true;
    }

    if (vm.count("ei_weight_mon_hpc")) {
      ei_weight_mon_hpc = true;
    }

    if (vm.count("ie_weight_mon_hpc")) {
      ie_weight_mon_hpc = true;
    }

    if (vm.count("ie_weightstat_mon_hpc")) {
      ie_weightstat_mon_hpc = true;
    }

    if (vm.count("ii_weight_mon_hpc")) {
       ii_weight_mon_hpc = true;
    }

    if (vm.count("stim_seed")) {
      std::cout << "stim_seed set to " 
		<< vm["stim_seed"].as<int>() << ".\n";
      stim_seed = vm["stim_seed"].as<int>();
    }

    if (vm.count("master_seed")) {
      std::cout << "master_seed set to " 
		<< vm["master_seed"].as<int>() << ".\n";
      master_seed = vm["master_seed"].as<int>();
    }

    if (vm.count("rep_seed")) {
      std::cout << "rep_seed set to " 
		<< vm["rep_seed"].as<int>() << ".\n";
      rep_seed = vm["rep_seed"].as<int>();
    }

    if (vm.count("bg_thl_seed")) {
      std::cout << "bg_thl_seed set to " 
		<< vm["bg_thl_seed"].as<int>() << ".\n";
      bg_thl_seed = vm["bg_thl_seed"].as<int>();
    }

    if (vm.count("bg_ctx_seed")) {
      std::cout << "bg_ctx_seed set to " 
		<< vm["bg_ctx_seed"].as<int>() << ".\n";
      bg_ctx_seed = vm["bg_ctx_seed"].as<int>();
    }

    if (vm.count("bg_hpc_seed")) {
      std::cout << "bg_hpc_seed set to " 
		<< vm["bg_hpc_seed"].as<int>() << ".\n";
      bg_hpc_seed = vm["bg_hpc_seed"].as<int>();
    }

    if (vm.count("bg_spike_mon")) {
      bg_spike_mon = true;
    }

    if (vm.count("bgfile_thl")) {
      std::cout << "bgfile_thl set to " 
		<< vm["bgfile_thl"].as<string>() << ", ";
      bgfile_thl = vm["bgfile_thl"].as<string>();
    }

    if (vm.count("w_bg_thl")) {
      std::cout << "w_bg_thl set to " 
		<< vm["w_bg_thl"].as<double>() << ".\n";
      w_bg_thl = vm["w_bg_thl"].as<double>();
    }

    if (vm.count("w_bg_ei_thl")) {
      std::cout << "w_bg_ei_thl set to " 
		<< vm["w_bg_ei_thl"].as<double>() << ".\n";
      w_bg_ei_thl = vm["w_bg_ei_thl"].as<double>();
    }

    if (vm.count("sparseness_bg_thl")) {
      std::cout << "sparseness_bg_thl set to " 
		<< vm["sparseness_bg_thl"].as<double>() << ".\n";
      sparseness_bg_thl = vm["sparseness_bg_thl"].as<double>();
    }

    if (vm.count("recfile_bg_thl")) {
      std::cout << "recfile_bg_thl set to " 
		<< vm["recfile_bg_thl"].as<string>() << ", ";
      recfile_bg_thl = vm["recfile_bg_thl"].as<string>();
    }

    if (vm.count("xi_bg_thl")) {
      std::cout << "xi_bg_thl set to " 
		<< vm["xi_bg_thl"].as<double>() << ".\n";
      xi_bg_thl = vm["xi_bg_thl"].as<double>();
    }

    if (vm.count("recfile_ei_bg_thl")) {
      std::cout << "recfile_ei_bg_thl set to " 
		<< vm["recfile_ei_bg_thl"].as<string>() << ", ";
      recfile_ei_bg_thl = vm["recfile_ei_bg_thl"].as<string>();
    }

    if (vm.count("xi_ei_bg_thl")) {
      std::cout << "xi_ei_bg_thl set to " 
		<< vm["xi_ei_bg_thl"].as<double>() << ".\n";
      xi_ei_bg_thl = vm["xi_ei_bg_thl"].as<double>();
    }

    if (vm.count("bgfile_ctx")) {
      std::cout << "bgfile_ctx set to " 
		<< vm["bgfile_ctx"].as<string>() << ", ";
      bgfile_ctx = vm["bgfile_ctx"].as<string>();
    }

    if (vm.count("w_bg_ctx")) {
      std::cout << "w_bg_ctx set to " 
		<< vm["w_bg_ctx"].as<double>() << ".\n";
      w_bg_ctx = vm["w_bg_ctx"].as<double>();
    }

    if (vm.count("w_bg_ei_ctx")) {
      std::cout << "w_bg_ei_ctx set to " 
		<< vm["w_bg_ei_ctx"].as<double>() << ".\n";
      w_bg_ei_ctx = vm["w_bg_ei_ctx"].as<double>();
    }

    if (vm.count("sparseness_bg_ctx")) {
      std::cout << "sparseness_bg_ctx set to " 
		<< vm["sparseness_bg_ctx"].as<double>() << ".\n";
      sparseness_bg_ctx = vm["sparseness_bg_ctx"].as<double>();
    }

    if (vm.count("recfile_bg_ctx")) {
      std::cout << "recfile_bg_ctx set to " 
		<< vm["recfile_bg_ctx"].as<string>() << ", ";
      recfile_bg_ctx = vm["recfile_bg_ctx"].as<string>();
    }

    if (vm.count("xi_bg_ctx")) {
      std::cout << "xi_bg_ctx set to " 
		<< vm["xi_bg_ctx"].as<double>() << ".\n";
      xi_bg_ctx = vm["xi_bg_ctx"].as<double>();
    }

    if (vm.count("recfile_ei_bg_ctx")) {
      std::cout << "recfile_ei_bg_ctx set to " 
		<< vm["recfile_ei_bg_ctx"].as<string>() << ", ";
      recfile_ei_bg_ctx = vm["recfile_ei_bg_ctx"].as<string>();
    }

    if (vm.count("xi_ei_bg_ctx")) {
      std::cout << "xi_ei_bg_ctx set to " 
		<< vm["xi_ei_bg_ctx"].as<double>() << ".\n";
      xi_ei_bg_ctx = vm["xi_ei_bg_ctx"].as<double>();
    }

    if (vm.count("bgfile_hpc")) {
      std::cout << "bgfile_hpc set to " 
		<< vm["bgfile_hpc"].as<string>() << ", ";
      bgfile_hpc = vm["bgfile_hpc"].as<string>();
    }

    if (vm.count("w_bg_hpc")) {
      std::cout << "w_bg_hpc set to " 
		<< vm["w_bg_hpc"].as<double>() << ".\n";
      w_bg_hpc = vm["w_bg_hpc"].as<double>();
    }

    if (vm.count("w_bg_ei_hpc")) {
      std::cout << "w_bg_ei_hpc set to " 
		<< vm["w_bg_ei_hpc"].as<double>() << ".\n";
      w_bg_ei_hpc = vm["w_bg_ei_hpc"].as<double>();
    }

    if (vm.count("sparseness_bg_hpc")) {
      std::cout << "sparseness_bg_hpc set to " 
		<< vm["sparseness_bg_hpc"].as<double>() << ".\n";
      sparseness_bg_hpc = vm["sparseness_bg_hpc"].as<double>();
    }

    if (vm.count("recfile_bg_hpc")) {
      std::cout << "recfile_bg_hpc set to " 
		<< vm["recfile_bg_hpc"].as<string>() << ", ";
      recfile_bg_hpc = vm["recfile_bg_hpc"].as<string>();
    }

    if (vm.count("xi_bg_hpc")) {
      std::cout << "xi_bg_hpc set to " 
		<< vm["xi_bg_hpc"].as<double>() << ".\n";
      xi_bg_hpc = vm["xi_bg_hpc"].as<double>();
    }

    if (vm.count("recfile_ei_bg_hpc")) {
      std::cout << "recfile_ei_bg_hpc set to " 
		<< vm["recfile_ei_bg_hpc"].as<string>() << ", ";
      recfile_ei_bg_hpc = vm["recfile_ei_bg_hpc"].as<string>();
    }

    if (vm.count("xi_ei_bg_hpc")) {
      std::cout << "xi_ei_bg_hpc set to " 
		<< vm["xi_ei_bg_hpc"].as<double>() << ".\n";
      xi_ei_bg_hpc = vm["xi_ei_bg_hpc"].as<double>();
    }

    if (vm.count("rep_spike_mon")) {
      rep_spike_mon = true;
    }

    if (vm.count("size_rep")) {
      std::cout << "size_rep set to " 
		<< vm["size_rep"].as<int>() << ".\n";
      size_rep = vm["size_rep"].as<int>();
    }

    if (vm.count("repfile")) {
      std::cout << "repfile set to " 
		<< vm["repfile"].as<string>() << ", ";
      repfile = vm["repfile"].as<string>();
    }

    if (vm.count("w_rep_thl")) {
      std::cout << "w_rep_thl set to " 
		<< vm["w_rep_thl"].as<double>() << ".\n";
      w_rep_thl = vm["w_rep_thl"].as<double>();
    }

    if (vm.count("sparseness_rep_thl")) {
      std::cout << "sparseness_rep_thl set to " 
		<< vm["sparseness_rep_thl"].as<double>() << ".\n";
      sparseness_rep_thl = vm["sparseness_rep_thl"].as<double>();
    }

    if (vm.count("recfile_rep_thl")) {
      std::cout << "recfile_rep_thl set to " 
		<< vm["recfile_rep_thl"].as<string>() << ", ";
      recfile_rep_thl = vm["recfile_rep_thl"].as<string>();
    }

    if (vm.count("xi_rep_thl")) {
      std::cout << "xi_rep_thl set to " 
		<< vm["xi_rep_thl"].as<double>() << ".\n";
      xi_rep_thl = vm["xi_rep_thl"].as<double>();
    }

    if (vm.count("recfile_ei_rep_thl")) {
      std::cout << "recfile_ei_rep_thl set to " 
		<< vm["recfile_ei_rep_thl"].as<string>() << ", ";
      recfile_ei_rep_thl = vm["recfile_ei_rep_thl"].as<string>();
    }

    if (vm.count("xi_ei_rep_thl")) {
      std::cout << "xi_ei_rep_thl set to " 
		<< vm["xi_ei_rep_thl"].as<double>() << ".\n";
      xi_ei_rep_thl = vm["xi_ei_rep_thl"].as<double>();
    }

    if (vm.count("w_rep_ctx")) {
      std::cout << "w_rep_ctx set to " 
		<< vm["w_rep_ctx"].as<double>() << ".\n";
      w_rep_ctx = vm["w_rep_ctx"].as<double>();
    }

    if (vm.count("sparseness_rep_ctx")) {
      std::cout << "sparseness_rep_ctx set to " 
		<< vm["sparseness_rep_ctx"].as<double>() << ".\n";
      sparseness_rep_ctx = vm["sparseness_rep_ctx"].as<double>();
    }

    if (vm.count("recfile_rep_ctx")) {
      std::cout << "recfile_rep_ctx set to " 
		<< vm["recfile_rep_ctx"].as<string>() << ", ";
      recfile_rep_ctx = vm["recfile_rep_ctx"].as<string>();
    }

    if (vm.count("xi_rep_ctx")) {
      std::cout << "xi_rep_ctx set to " 
		<< vm["xi_rep_ctx"].as<double>() << ".\n";
      xi_rep_ctx = vm["xi_rep_ctx"].as<double>();
    }

    if (vm.count("recfile_ei_rep_ctx")) {
      std::cout << "recfile_ei_rep_ctx set to " 
		<< vm["recfile_ei_rep_ctx"].as<string>() << ", ";
      recfile_ei_rep_ctx = vm["recfile_ei_rep_ctx"].as<string>();
    }

    if (vm.count("xi_ei_rep_ctx")) {
      std::cout << "xi_ei_rep_ctx set to " 
		<< vm["xi_ei_rep_ctx"].as<double>() << ".\n";
      xi_ei_rep_ctx = vm["xi_ei_rep_ctx"].as<double>();
    }

    if (vm.count("w_rep_hpc")) {
      std::cout << "w_rep_hpc set to " 
		<< vm["w_rep_hpc"].as<double>() << ".\n";
      w_rep_hpc = vm["w_rep_hpc"].as<double>();
    }

    if (vm.count("sparseness_rep_hpc")) {
      std::cout << "sparseness_rep_hpc set to " 
		<< vm["sparseness_rep_hpc"].as<double>() << ".\n";
      sparseness_rep_hpc = vm["sparseness_rep_hpc"].as<double>();
    }

    if (vm.count("recfile_rep_hpc")) {
      std::cout << "recfile_rep_hpc set to " 
		<< vm["recfile_rep_hpc"].as<string>() << ", ";
      recfile_rep_hpc = vm["recfile_rep_hpc"].as<string>();
    }

    if (vm.count("xi_rep_hpc")) {
      std::cout << "xi_rep_hpc set to " 
		<< vm["xi_rep_hpc"].as<double>() << ".\n";
      xi_rep_hpc = vm["xi_rep_hpc"].as<double>();
    }

    if (vm.count("recfile_ei_rep_hpc")) {
      std::cout << "recfile_ei_rep_hpc set to " 
		<< vm["recfile_ei_rep_hpc"].as<string>() << ", ";
      recfile_ei_rep_hpc = vm["recfile_ei_rep_hpc"].as<string>();
    }

    if (vm.count("xi_ei_rep_hpc")) {
      std::cout << "xi_ei_rep_hpc set to " 
		<< vm["xi_ei_rep_hpc"].as<double>() << ".\n";
      xi_ei_rep_hpc = vm["xi_ei_rep_hpc"].as<double>();
    }

    if (vm.count("bgrate_rep")) {
      std::cout << "bgrate_rep set to " 
		<< vm["bgrate_rep"].as<double>() << ".\n";
      bgrate_rep = vm["bgrate_rep"].as<double>();
    }

    if (vm.count("bgrate_thl")) {
      std::cout << "bgrate_thl set to " 
		<< vm["bgrate_thl"].as<double>() << ".\n";
      bgrate_thl = vm["bgrate_thl"].as<double>();
    }

    if (vm.count("bgrate_ctx")) {
      std::cout << "bgrate_ctx set to " 
		<< vm["bgrate_ctx"].as<double>() << ".\n";
      bgrate_ctx = vm["bgrate_ctx"].as<double>();
    }

    if (vm.count("bgrate_hpc")) {
      std::cout << "bgrate_hpc set to " 
		<< vm["bgrate_hpc"].as<double>() << ".\n";
      bgrate_hpc = vm["bgrate_hpc"].as<double>();
    }

    if (vm.count("exc_size_thl")) {
      std::cout << "exc_size_thl set to " 
		<< vm["exc_size_thl"].as<int>() << ".\n";
      exc_size_thl = vm["exc_size_thl"].as<int>();
    }

    if (vm.count("eta_thl")) {
      std::cout << "eta_thl set to " 
		<< vm["eta_thl"].as<double>() << ".\n";
      eta_thl = vm["eta_thl"].as<double>();
    }

    if (vm.count("eta_stim_thl")) {
      std::cout << "eta_stim_thl set to " 
		<< vm["eta_stim_thl"].as<double>() << ".\n";
      eta_stim_thl = vm["eta_stim_thl"].as<double>();
    }

    if (vm.count("eta_exc_inh_thl")) {
      std::cout << "eta_exc_inh_thl set to " 
		<< vm["eta_exc_inh_thl"].as<double>() << ".\n";
      eta_exc_inh_thl = vm["eta_exc_inh_thl"].as<double>();
    }

    if (vm.count("exc_inh_thl")) {
      std::cout << "exc_inh_thl set to " 
		<< vm["exc_inh_thl"].as<int>() << ".\n";
      exc_inh_thl = vm["exc_inh_thl"].as<int>();
    }
    
    if (vm.count("alpha_thl")) {
      std::cout << "alpha_thl set to " 
		<< vm["alpha_thl"].as<double>() << ".\n";
      alpha_thl = vm["alpha_thl"].as<double>();
    } 

    if (vm.count("kappa_thl")) {
      std::cout << "kappa_thl set to " 
		<< vm["kappa_thl"].as<double>() << ".\n";
      kappa_thl = vm["kappa_thl"].as<double>();
    }

    if (vm.count("tauf_ee_thl")) {
      std::cout << "tauf_ee_thl set to " 
		<< vm["tauf_ee_thl"].as<double>() << ".\n";
      tauf_ee_thl = vm["tauf_ee_thl"].as<double>();
    }

    if (vm.count("taud_ee_thl")) {
      std::cout << "taud_ee_thl set to " 
		<< vm["taud_ee_thl"].as<double>() << ".\n";
      taud_ee_thl = vm["taud_ee_thl"].as<double>();
    }

    if (vm.count("tauh_ee_thl")) {
      std::cout << "tauh_ee_thl set to " 
		<< vm["tauh_ee_thl"].as<double>() << ".\n";
      tauh_ee_thl = vm["tauh_ee_thl"].as<double>();
    }

    if (vm.count("tauc_ee_thl")) {
      std::cout << "tauc_ee_thl set to " 
		<< vm["tauc_ee_thl"].as<double>() << ".\n";
      tauc_ee_thl = vm["tauc_ee_thl"].as<double>();
    }

    if (vm.count("ujump_ee_thl")) {
      std::cout << "ujump_ee_thl set to " 
		<< vm["ujump_ee_thl"].as<double>() << ".\n";
      ujump_ee_thl = vm["ujump_ee_thl"].as<double>();
    }

    if (vm.count("tauf_ee_stim_thl")) {
      std::cout << "tauf_ee_stim_thl set to " 
		<< vm["tauf_ee_stim_thl"].as<double>() << ".\n";
      tauf_ee_stim_thl = vm["tauf_ee_stim_thl"].as<double>();
    }

    if (vm.count("taud_ee_stim_thl")) {
      std::cout << "taud_ee_stim_thl set to " 
		<< vm["taud_ee_stim_thl"].as<double>() << ".\n";
      taud_ee_stim_thl = vm["taud_ee_stim_thl"].as<double>();
    }

    if (vm.count("tauh_ee_stim_thl")) {
      std::cout << "tauh_ee_stim_thl set to " 
		<< vm["tauh_ee_stim_thl"].as<double>() << ".\n";
      tauh_ee_stim_thl = vm["tauh_ee_stim_thl"].as<double>();
    }

    if (vm.count("tauc_ee_stim_thl")) {
      std::cout << "tauc_ee_stim_thl set to " 
		<< vm["tauc_ee_stim_thl"].as<double>() << ".\n";
      tauc_ee_stim_thl = vm["tauc_ee_stim_thl"].as<double>();
    }

    if (vm.count("ujump_ee_stim_thl")) {
      std::cout << "ujump_ee_stim_thl set to " 
		<< vm["ujump_ee_stim_thl"].as<double>() << ".\n";
      ujump_ee_stim_thl = vm["ujump_ee_stim_thl"].as<double>();
    }

    if (vm.count("beta_thl")) {
      std::cout << "beta_thl set to " 
		<< vm["beta_thl"].as<double>() << ".\n";
      beta_thl = vm["beta_thl"].as<double>();
    }

    if (vm.count("beta_stim_thl")) {
      std::cout << "beta_stim_thl set to " 
		<< vm["beta_stim_thl"].as<double>() << ".\n";
      beta_stim_thl = vm["beta_stim_thl"].as<double>();
    }

    if (vm.count("delta_thl")) {
      std::cout << "delta_thl set to " 
		<< vm["delta_thl"].as<double>() << ".\n";
      delta_thl = vm["delta_thl"].as<double>();
    }

    if (vm.count("weight_a_thl")) {
      std::cout << "weight_a_thl set to " 
		<< vm["weight_a_thl"].as<double>() << ".\n";
      weight_a_thl = vm["weight_a_thl"].as<double>();
    } 

    if (vm.count("weight_c_thl")) {
      std::cout << "weight_c_thl set to " 
		<< vm["weight_c_thl"].as<double>() << ".\n";
      weight_c_thl = vm["weight_c_thl"].as<double>();
    }

    if (vm.count("adapt1_thl")) {
      std::cout << "adapt1_thl set to " 
		<< vm["adapt1_thl"].as<double>() << ".\n";
      adapt1_thl = vm["adapt1_thl"].as<double>();
    }

    if (vm.count("adapt2_thl")) {
      std::cout << "adapt2_thl set to " 
		<< vm["adapt2_thl"].as<double>() << ".\n";
      adapt2_thl = vm["adapt2_thl"].as<double>();
    }

    if (vm.count("pot_strength_thl")) {
      std::cout << "pot_strength_thl set to " 
		<< vm["pot_strength_thl"].as<double>() << ".\n";
      pot_strength_thl = vm["pot_strength_thl"].as<double>();
    }

    if (vm.count("wmax_exc_thl")) {
      std::cout << "wmax_exc_thl set to " 
		<< vm["wmax_exc_thl"].as<double>() << ".\n";
      wmax_exc_thl = vm["wmax_exc_thl"].as<double>();
    }

    if (vm.count("wmin_exc_thl")) {
      std::cout << "wmin_exc_thl set to " 
		<< vm["wmin_exc_thl"].as<double>() << ".\n";
      wmin_exc_thl = vm["wmin_exc_thl"].as<double>();
    }

    if (vm.count("wmax_inh_thl")) {
      std::cout << "wmax_inh_thl set to " 
		<< vm["wmax_inh_thl"].as<double>() << ".\n";
      wmax_inh_thl = vm["wmax_inh_thl"].as<double>();
    }

    if (vm.count("wmin_inh_thl")) {
      std::cout << "wmin_inh_thl set to " 
		<< vm["wmin_inh_thl"].as<double>() << ".\n";
      wmin_inh_thl = vm["wmin_inh_thl"].as<double>();
    }

    if (vm.count("wee_thl")) {
      std::cout << "wee_thl set to " 
		<< vm["wee_thl"].as<double>() << ".\n";
      wee_thl = vm["wee_thl"].as<double>();
    } 

    if (vm.count("wei_thl")) {
      std::cout << "wei_thl set to " 
		<< vm["wei_thl"].as<double>() << ".\n";
      wei_thl = vm["wei_thl"].as<double>();
    }

    if (vm.count("wie_thl")) {
      std::cout << "wie_thl set to " 
		<< vm["wie_thl"].as<double>() << ".\n";
      wie_thl = vm["wie_thl"].as<double>();
    }

    if (vm.count("wii_thl")) {
      std::cout << "wii_thl set to " 
		<< vm["wii_thl"].as<double>() << ".\n";
      wii_thl = vm["wii_thl"].as<double>();
    }

    if (vm.count("wext_thl")) {
      std::cout << "wext_thl set to " 
		<< vm["wext_thl"].as<double>() << ".\n";
      wext_thl = vm["wext_thl"].as<double>();
    }

    if (vm.count("wext_ei_thl")) {
      std::cout << "wext_ei_thl set to " 
		<< vm["wext_ei_thl"].as<double>() << ".\n";
      wext_ei_thl = vm["wext_ei_thl"].as<double>();
    }

    if (vm.count("sparseness_int_ee_thl")) {
      std::cout << "sparseness_int_ee_thl set to " 
		<< vm["sparseness_int_ee_thl"].as<double>() << ".\n";
      sparseness_int_ee_thl = vm["sparseness_int_ee_thl"].as<double>();
    }

    if (vm.count("sparseness_int_ei_thl")) {
      std::cout << "sparseness_int_ei_thl set to " 
		<< vm["sparseness_int_ei_thl"].as<double>() << ".\n";
      sparseness_int_ei_thl = vm["sparseness_int_ei_thl"].as<double>();
    }

    if (vm.count("sparseness_int_ii_thl")) {
      std::cout << "sparseness_int_ii_thl set to " 
		<< vm["sparseness_int_ii_thl"].as<double>() << ".\n";
      sparseness_int_ii_thl = vm["sparseness_int_ii_thl"].as<double>();
    }

    if (vm.count("sparseness_int_ie_thl")) {
      std::cout << "sparseness_int_ie_thl set to " 
		<< vm["sparseness_int_ie_thl"].as<double>() << ".\n";
      sparseness_int_ie_thl = vm["sparseness_int_ie_thl"].as<double>();
    }

    if (vm.count("sparseness_ext_thl")) {
      std::cout << "sparseness_ext_thl set to " 
		<< vm["sparseness_ext_thl"].as<double>() << ".\n";
      sparseness_ext_thl = vm["sparseness_ext_thl"].as<double>();
    }

    if (vm.count("tauf_ei_thl")) {
      std::cout << "tauf_ei_thl set to " 
		<< vm["tauf_ei_thl"].as<double>() << ".\n";
      tauf_ei_thl = vm["tauf_ei_thl"].as<double>();
    }

    if (vm.count("taud_ei_thl")) {
      std::cout << "taud_ei_thl set to " 
		<< vm["taud_ei_thl"].as<double>() << ".\n";
      taud_ei_thl = vm["taud_ei_thl"].as<double>();
    }

    if (vm.count("ujump_ei_thl")) {
      std::cout << "ujump_ei_thl set to " 
		<< vm["ujump_ei_thl"].as<double>() << ".\n";
      ujump_ei_thl = vm["ujump_ei_thl"].as<double>();
    }

    if (vm.count("tauh_ie_thl")) {
      std::cout << "tauh_ie_thl set to " 
		<< vm["tauh_ie_thl"].as<double>() << ".\n";
      tauh_ie_thl = vm["tauh_ie_thl"].as<double>();
    }

    if (vm.count("taud_ei_stim_thl")) {
      std::cout << "taud_ei_stim_thl set to " 
		<< vm["taud_ei_stim_thl"].as<double>() << ".\n";
      taud_ei_stim_thl = vm["taud_ei_stim_thl"].as<double>();
    }

    if (vm.count("tauf_ei_stim_thl")) {
      std::cout << "tauf_ei_stim_thl set to " 
		<< vm["tauf_ei_stim_thl"].as<double>() << ".\n";
      tauf_ei_stim_thl = vm["tauf_ei_stim_thl"].as<double>();
    }

    if (vm.count("ujump_ei_stim_thl")) {
      std::cout << "ujump_ei_stim_thl set to " 
		<< vm["ujump_ei_stim_thl"].as<double>() << ".\n";
      ujump_ei_stim_thl = vm["ujump_ei_stim_thl"].as<double>();
    }

    if (vm.count("tau_ampa_e_thl")) {
      std::cout << "tau_ampa_e_thl set to " 
		<< vm["tau_ampa_e_thl"].as<double>() << ".\n";
      tau_ampa_e_thl = vm["tau_ampa_e_thl"].as<double>();
    }

    if (vm.count("tau_gaba_e_thl")) {
      std::cout << "tau_gaba_e_thl set to " 
		<< vm["tau_gaba_e_thl"].as<double>() << ".\n";
      tau_gaba_e_thl = vm["tau_gaba_e_thl"].as<double>();
    }

    if (vm.count("tau_nmda_e_thl")) {
      std::cout << "tau_nmda_e_thl set to " 
		<< vm["tau_nmda_e_thl"].as<double>() << ".\n";
      tau_nmda_e_thl = vm["tau_nmda_e_thl"].as<double>();
    }

    if (vm.count("ampa_nmda_e_thl")) {
      std::cout << "ampa_nmda_e_thl set to " 
		<< vm["ampa_nmda_e_thl"].as<double>() << ".\n";
      ampa_nmda_e_thl = vm["ampa_nmda_e_thl"].as<double>();
    }

    if (vm.count("tau_ampa_i_thl")) {
      std::cout << "tau_ampa_i_thl set to " 
		<< vm["tau_ampa_i_thl"].as<double>() << ".\n";
      tau_ampa_i_thl = vm["tau_ampa_i_thl"].as<double>();
    }

    if (vm.count("tau_gaba_i_thl")) {
      std::cout << "tau_gaba_i_thl set to " 
		<< vm["tau_gaba_i_thl"].as<double>() << ".\n";
      tau_gaba_i_thl = vm["tau_gaba_i_thl"].as<double>();
    }

    if (vm.count("tau_nmda_i_thl")) {
      std::cout << "tau_nmda_i_thl set to " 
		<< vm["tau_nmda_i_thl"].as<double>() << ".\n";
      tau_nmda_i_thl = vm["tau_nmda_i_thl"].as<double>();
    }

    if (vm.count("ampa_nmda_i_thl")) {
      std::cout << "ampa_nmda_i_thl set to " 
		<< vm["ampa_nmda_i_thl"].as<double>() << ".\n";
      ampa_nmda_i_thl = vm["ampa_nmda_i_thl"].as<double>();
    }
    
    if (vm.count("prefile_thl")) {
      std::cout << "prefile_thl set to " 
		<< vm["prefile_thl"].as<string>() << ", ";
      prefile_thl = vm["prefile_thl"].as<string>();
    }

    if (vm.count("chi_thl")) {
      std::cout << "chi_thl set to " 
		<< vm["chi_thl"].as<double>() << ".\n";
      chi_thl = vm["chi_thl"].as<double>();
    }

    if (vm.count("exc_size_ctx")) {
      std::cout << "exc_size_ctx set to " 
		<< vm["exc_size_ctx"].as<int>() << ".\n";
      exc_size_ctx = vm["exc_size_ctx"].as<int>();
    }

    if (vm.count("eta_ctx")) {
      std::cout << "eta_ctx set to " 
		<< vm["eta_ctx"].as<double>() << ".\n";
      eta_ctx = vm["eta_ctx"].as<double>();
    }

    if (vm.count("eta_stim_ctx")) {
      std::cout << "eta_stim_ctx set to " 
		<< vm["eta_stim_ctx"].as<double>() << ".\n";
      eta_stim_ctx = vm["eta_stim_ctx"].as<double>();
    }

    if (vm.count("eta_exc_inh_ctx")) {
      std::cout << "eta_exc_inh_ctx set to " 
		<< vm["eta_exc_inh_ctx"].as<double>() << ".\n";
      eta_exc_inh_ctx = vm["eta_exc_inh_ctx"].as<double>();
    }

    if (vm.count("exc_inh_ctx")) {
      std::cout << "exc_inh_ctx set to " 
		<< vm["exc_inh_ctx"].as<int>() << ".\n";
      exc_inh_ctx = vm["exc_inh_ctx"].as<int>();
    }
    
    if (vm.count("alpha_ctx")) {
      std::cout << "alpha_ctx set to " 
		<< vm["alpha_ctx"].as<double>() << ".\n";
      alpha_ctx = vm["alpha_ctx"].as<double>();
    } 

    if (vm.count("kappa_ctx")) {
      std::cout << "kappa_ctx set to " 
		<< vm["kappa_ctx"].as<double>() << ".\n";
      kappa_ctx = vm["kappa_ctx"].as<double>();
    }

    if (vm.count("tauf_ee_ctx")) {
      std::cout << "tauf_ee_ctx set to " 
		<< vm["tauf_ee_ctx"].as<double>() << ".\n";
      tauf_ee_ctx = vm["tauf_ee_ctx"].as<double>();
    }

    if (vm.count("taud_ee_ctx")) {
      std::cout << "taud_ee_ctx set to " 
		<< vm["taud_ee_ctx"].as<double>() << ".\n";
      taud_ee_ctx = vm["taud_ee_ctx"].as<double>();
    }

    if (vm.count("tauh_ee_ctx")) {
      std::cout << "tauh_ee_ctx set to " 
		<< vm["tauh_ee_ctx"].as<double>() << ".\n";
      tauh_ee_ctx = vm["tauh_ee_ctx"].as<double>();
    }

    if (vm.count("tauc_ee_ctx")) {
      std::cout << "tauc_ee_ctx set to " 
		<< vm["tauc_ee_ctx"].as<double>() << ".\n";
      tauc_ee_ctx = vm["tauc_ee_ctx"].as<double>();
    }

    if (vm.count("ujump_ee_ctx")) {
      std::cout << "ujump_ee_ctx set to " 
		<< vm["ujump_ee_ctx"].as<double>() << ".\n";
      ujump_ee_ctx = vm["ujump_ee_ctx"].as<double>();
    }

    if (vm.count("tauf_ee_stim_ctx")) {
      std::cout << "tauf_ee_stim_ctx set to " 
		<< vm["tauf_ee_stim_ctx"].as<double>() << ".\n";
      tauf_ee_stim_ctx = vm["tauf_ee_stim_ctx"].as<double>();
    }

    if (vm.count("taud_ee_stim_ctx")) {
      std::cout << "taud_ee_stim_ctx set to " 
		<< vm["taud_ee_stim_ctx"].as<double>() << ".\n";
      taud_ee_stim_ctx = vm["taud_ee_stim_ctx"].as<double>();
    }

    if (vm.count("tauh_ee_stim_ctx")) {
      std::cout << "tauh_ee_stim_ctx set to " 
		<< vm["tauh_ee_stim_ctx"].as<double>() << ".\n";
      tauh_ee_stim_ctx = vm["tauh_ee_stim_ctx"].as<double>();
    }

    if (vm.count("tauc_ee_stim_ctx")) {
      std::cout << "tauc_ee_stim_ctx set to " 
		<< vm["tauc_ee_stim_ctx"].as<double>() << ".\n";
      tauc_ee_stim_ctx = vm["tauc_ee_stim_ctx"].as<double>();
    }

    if (vm.count("ujump_ee_stim_ctx")) {
      std::cout << "ujump_ee_stim_ctx set to " 
		<< vm["ujump_ee_stim_ctx"].as<double>() << ".\n";
      ujump_ee_stim_ctx = vm["ujump_ee_stim_ctx"].as<double>();
    }

    if (vm.count("beta_ctx")) {
      std::cout << "beta_ctx set to " 
		<< vm["beta_ctx"].as<double>() << ".\n";
      beta_ctx = vm["beta_ctx"].as<double>();
    }

    if (vm.count("beta_stim_ctx")) {
      std::cout << "beta_stim_ctx set to " 
		<< vm["beta_stim_ctx"].as<double>() << ".\n";
      beta_stim_ctx = vm["beta_stim_ctx"].as<double>();
    }

    if (vm.count("delta_ctx")) {
      std::cout << "delta_ctx set to " 
		<< vm["delta_ctx"].as<double>() << ".\n";
      delta_ctx = vm["delta_ctx"].as<double>();
    }

    if (vm.count("weight_a_ctx")) {
      std::cout << "weight_a_ctx set to " 
		<< vm["weight_a_ctx"].as<double>() << ".\n";
      weight_a_ctx = vm["weight_a_ctx"].as<double>();
    } 

    if (vm.count("weight_c_ctx")) {
      std::cout << "weight_c_ctx set to " 
		<< vm["weight_c_ctx"].as<double>() << ".\n";
      weight_c_ctx = vm["weight_c_ctx"].as<double>();
    }

    if (vm.count("adapt1_ctx")) {
      std::cout << "adapt1_ctx set to " 
		<< vm["adapt1_ctx"].as<double>() << ".\n";
      adapt1_ctx = vm["adapt1_ctx"].as<double>();
    }

    if (vm.count("adapt2_ctx")) {
      std::cout << "adapt2_ctx set to " 
		<< vm["adapt2_ctx"].as<double>() << ".\n";
      adapt2_ctx = vm["adapt2_ctx"].as<double>();
    }

    if (vm.count("pot_strength_ctx")) {
      std::cout << "pot_strength_ctx set to " 
		<< vm["pot_strength_ctx"].as<double>() << ".\n";
      pot_strength_ctx = vm["pot_strength_ctx"].as<double>();
    }

    if (vm.count("wmax_exc_ctx")) {
      std::cout << "wmax_exc_ctx set to " 
		<< vm["wmax_exc_ctx"].as<double>() << ".\n";
      wmax_exc_ctx = vm["wmax_exc_ctx"].as<double>();
    }

    if (vm.count("wmin_exc_ctx")) {
      std::cout << "wmin_exc_ctx set to " 
		<< vm["wmin_exc_ctx"].as<double>() << ".\n";
      wmin_exc_ctx = vm["wmin_exc_ctx"].as<double>();
    }

    if (vm.count("wmax_inh_ctx")) {
      std::cout << "wmax_inh_ctx set to " 
		<< vm["wmax_inh_ctx"].as<double>() << ".\n";
      wmax_inh_ctx = vm["wmax_inh_ctx"].as<double>();
    }

    if (vm.count("wmin_inh_ctx")) {
      std::cout << "wmin_inh_ctx set to " 
		<< vm["wmin_inh_ctx"].as<double>() << ".\n";
      wmin_inh_ctx = vm["wmin_inh_ctx"].as<double>();
    }

    if (vm.count("wee_ctx")) {
      std::cout << "wee_ctx set to " 
		<< vm["wee_ctx"].as<double>() << ".\n";
      wee_ctx = vm["wee_ctx"].as<double>();
    } 

    if (vm.count("wei_ctx")) {
      std::cout << "wei_ctx set to " 
		<< vm["wei_ctx"].as<double>() << ".\n";
      wei_ctx = vm["wei_ctx"].as<double>();
    }

    if (vm.count("wie_ctx")) {
      std::cout << "wie_ctx set to " 
		<< vm["wie_ctx"].as<double>() << ".\n";
      wie_ctx = vm["wie_ctx"].as<double>();
    }

    if (vm.count("wii_ctx")) {
      std::cout << "wii_ctx set to " 
		<< vm["wii_ctx"].as<double>() << ".\n";
      wii_ctx = vm["wii_ctx"].as<double>();
    }

    if (vm.count("wext_ctx")) {
      std::cout << "wext_ctx set to " 
		<< vm["wext_ctx"].as<double>() << ".\n";
      wext_ctx = vm["wext_ctx"].as<double>();
    }

    if (vm.count("wext_ei_ctx")) {
      std::cout << "wext_ei_ctx set to " 
		<< vm["wext_ei_ctx"].as<double>() << ".\n";
      wext_ei_ctx = vm["wext_ei_ctx"].as<double>();
    }

    if (vm.count("sparseness_int_ee_ctx")) {
      std::cout << "sparseness_int_ee_ctx set to " 
		<< vm["sparseness_int_ee_ctx"].as<double>() << ".\n";
      sparseness_int_ee_ctx = vm["sparseness_int_ee_ctx"].as<double>();
    }

    if (vm.count("sparseness_int_ei_ctx")) {
      std::cout << "sparseness_int_ei_ctx set to " 
		<< vm["sparseness_int_ei_ctx"].as<double>() << ".\n";
      sparseness_int_ei_ctx = vm["sparseness_int_ei_ctx"].as<double>();
    }

    if (vm.count("sparseness_int_ii_ctx")) {
      std::cout << "sparseness_int_ii_ctx set to " 
		<< vm["sparseness_int_ii_ctx"].as<double>() << ".\n";
      sparseness_int_ii_ctx = vm["sparseness_int_ii_ctx"].as<double>();
    }

    if (vm.count("sparseness_int_ie_ctx")) {
      std::cout << "sparseness_int_ie_ctx set to " 
		<< vm["sparseness_int_ie_ctx"].as<double>() << ".\n";
      sparseness_int_ie_ctx = vm["sparseness_int_ie_ctx"].as<double>();
    }

    if (vm.count("sparseness_ext_ctx")) {
      std::cout << "sparseness_ext_ctx set to " 
		<< vm["sparseness_ext_ctx"].as<double>() << ".\n";
      sparseness_ext_ctx = vm["sparseness_ext_ctx"].as<double>();
    }

    if (vm.count("tauf_ei_ctx")) {
      std::cout << "tauf_ei_ctx set to " 
		<< vm["tauf_ei_ctx"].as<double>() << ".\n";
      tauf_ei_ctx = vm["tauf_ei_ctx"].as<double>();
    }

    if (vm.count("taud_ei_ctx")) {
      std::cout << "taud_ei_ctx set to " 
		<< vm["taud_ei_ctx"].as<double>() << ".\n";
      taud_ei_ctx = vm["taud_ei_ctx"].as<double>();
    }

    if (vm.count("ujump_ei_ctx")) {
      std::cout << "ujump_ei_ctx set to " 
		<< vm["ujump_ei_ctx"].as<double>() << ".\n";
      ujump_ei_ctx = vm["ujump_ei_ctx"].as<double>();
    }

    if (vm.count("tauh_ie_ctx")) {
      std::cout << "tauh_ie_ctx set to " 
		<< vm["tauh_ie_ctx"].as<double>() << ".\n";
      tauh_ie_ctx = vm["tauh_ie_ctx"].as<double>();
    }

    if (vm.count("taud_ei_stim_ctx")) {
      std::cout << "taud_ei_stim_ctx set to " 
		<< vm["taud_ei_stim_ctx"].as<double>() << ".\n";
      taud_ei_stim_ctx = vm["taud_ei_stim_ctx"].as<double>();
    }

    if (vm.count("tauf_ei_stim_ctx")) {
      std::cout << "tauf_ei_stim_ctx set to " 
		<< vm["tauf_ei_stim_ctx"].as<double>() << ".\n";
      tauf_ei_stim_ctx = vm["tauf_ei_stim_ctx"].as<double>();
    }

    if (vm.count("ujump_ei_stim_ctx")) {
      std::cout << "ujump_ei_stim_ctx set to " 
		<< vm["ujump_ei_stim_ctx"].as<double>() << ".\n";
      ujump_ei_stim_ctx = vm["ujump_ei_stim_ctx"].as<double>();
    }

    if (vm.count("tau_ampa_e_ctx")) {
      std::cout << "tau_ampa_e_ctx set to " 
		<< vm["tau_ampa_e_ctx"].as<double>() << ".\n";
      tau_ampa_e_ctx = vm["tau_ampa_e_ctx"].as<double>();
    }

    if (vm.count("tau_gaba_e_ctx")) {
      std::cout << "tau_gaba_e_ctx set to " 
		<< vm["tau_gaba_e_ctx"].as<double>() << ".\n";
      tau_gaba_e_ctx = vm["tau_gaba_e_ctx"].as<double>();
    }

    if (vm.count("tau_nmda_e_ctx")) {
      std::cout << "tau_nmda_e_ctx set to " 
		<< vm["tau_nmda_e_ctx"].as<double>() << ".\n";
      tau_nmda_e_ctx = vm["tau_nmda_e_ctx"].as<double>();
    }

    if (vm.count("ampa_nmda_e_ctx")) {
      std::cout << "ampa_nmda_e_ctx set to " 
		<< vm["ampa_nmda_e_ctx"].as<double>() << ".\n";
      ampa_nmda_e_ctx = vm["ampa_nmda_e_ctx"].as<double>();
    }

    if (vm.count("tau_ampa_i_ctx")) {
      std::cout << "tau_ampa_i_ctx set to " 
		<< vm["tau_ampa_i_ctx"].as<double>() << ".\n";
      tau_ampa_i_ctx = vm["tau_ampa_i_ctx"].as<double>();
    }

    if (vm.count("tau_gaba_i_ctx")) {
      std::cout << "tau_gaba_i_ctx set to " 
		<< vm["tau_gaba_i_ctx"].as<double>() << ".\n";
      tau_gaba_i_ctx = vm["tau_gaba_i_ctx"].as<double>();
    }

    if (vm.count("tau_nmda_i_ctx")) {
      std::cout << "tau_nmda_i_ctx set to " 
		<< vm["tau_nmda_i_ctx"].as<double>() << ".\n";
      tau_nmda_i_ctx = vm["tau_nmda_i_ctx"].as<double>();
    }

    if (vm.count("ampa_nmda_i_ctx")) {
      std::cout << "ampa_nmda_i_ctx set to " 
		<< vm["ampa_nmda_i_ctx"].as<double>() << ".\n";
      ampa_nmda_i_ctx = vm["ampa_nmda_i_ctx"].as<double>();
    }
    
    if (vm.count("prefile_ctx")) {
      std::cout << "prefile_ctx set to " 
		<< vm["prefile_ctx"].as<string>() << ", ";
      prefile_ctx = vm["prefile_ctx"].as<string>();
    }

    if (vm.count("chi_ctx")) {
      std::cout << "chi_ctx set to " 
		<< vm["chi_ctx"].as<double>() << ".\n";
      chi_ctx = vm["chi_ctx"].as<double>();
    }

    if (vm.count("exc_size_hpc")) {
      std::cout << "exc_size_hpc set to " 
		<< vm["exc_size_hpc"].as<int>() << ".\n";
      exc_size_hpc = vm["exc_size_hpc"].as<int>();
    }

    if (vm.count("eta_hpc")) {
      std::cout << "eta_hpc set to " 
		<< vm["eta_hpc"].as<double>() << ".\n";
      eta_hpc = vm["eta_hpc"].as<double>();
    }

    if (vm.count("eta_stim_hpc")) {
      std::cout << "eta_stim_hpc set to " 
		<< vm["eta_stim_hpc"].as<double>() << ".\n";
      eta_stim_hpc = vm["eta_stim_hpc"].as<double>();
    }

    if (vm.count("eta_exc_inh_hpc")) {
      std::cout << "eta_exc_inh_hpc set to " 
		<< vm["eta_exc_inh_hpc"].as<double>() << ".\n";
      eta_exc_inh_hpc = vm["eta_exc_inh_hpc"].as<double>();
    }

    if (vm.count("exc_inh_hpc")) {
      std::cout << "exc_inh_hpc set to " 
		<< vm["exc_inh_hpc"].as<int>() << ".\n";
      exc_inh_hpc = vm["exc_inh_hpc"].as<int>();
    }
    
    if (vm.count("alpha_hpc")) {
      std::cout << "alpha_hpc set to " 
		<< vm["alpha_hpc"].as<double>() << ".\n";
      alpha_hpc = vm["alpha_hpc"].as<double>();
    } 

    if (vm.count("kappa_hpc")) {
      std::cout << "kappa_hpc set to " 
		<< vm["kappa_hpc"].as<double>() << ".\n";
      kappa_hpc = vm["kappa_hpc"].as<double>();
    }

    if (vm.count("tauf_ee_hpc")) {
      std::cout << "tauf_ee_hpc set to " 
		<< vm["tauf_ee_hpc"].as<double>() << ".\n";
      tauf_ee_hpc = vm["tauf_ee_hpc"].as<double>();
    }

    if (vm.count("taud_ee_hpc")) {
      std::cout << "taud_ee_hpc set to " 
		<< vm["taud_ee_hpc"].as<double>() << ".\n";
      taud_ee_hpc = vm["taud_ee_hpc"].as<double>();
    }

    if (vm.count("tauh_ee_hpc")) {
      std::cout << "tauh_ee_hpc set to " 
		<< vm["tauh_ee_hpc"].as<double>() << ".\n";
      tauh_ee_hpc = vm["tauh_ee_hpc"].as<double>();
    }

    if (vm.count("tauc_ee_hpc")) {
      std::cout << "tauc_ee_hpc set to " 
		<< vm["tauc_ee_hpc"].as<double>() << ".\n";
      tauc_ee_hpc = vm["tauc_ee_hpc"].as<double>();
    }

    if (vm.count("ujump_ee_hpc")) {
      std::cout << "ujump_ee_hpc set to " 
		<< vm["ujump_ee_hpc"].as<double>() << ".\n";
      ujump_ee_hpc = vm["ujump_ee_hpc"].as<double>();
    }

    if (vm.count("tauf_ee_stim_hpc")) {
      std::cout << "tauf_ee_stim_hpc set to " 
		<< vm["tauf_ee_stim_hpc"].as<double>() << ".\n";
      tauf_ee_stim_hpc = vm["tauf_ee_stim_hpc"].as<double>();
    }

    if (vm.count("taud_ee_stim_hpc")) {
      std::cout << "taud_ee_stim_hpc set to " 
		<< vm["taud_ee_stim_hpc"].as<double>() << ".\n";
      taud_ee_stim_hpc = vm["taud_ee_stim_hpc"].as<double>();
    }

    if (vm.count("tauh_ee_stim_hpc")) {
      std::cout << "tauh_ee_stim_hpc set to " 
		<< vm["tauh_ee_stim_hpc"].as<double>() << ".\n";
      tauh_ee_stim_hpc = vm["tauh_ee_stim_hpc"].as<double>();
    }

    if (vm.count("tauc_ee_stim_hpc")) {
      std::cout << "tauc_ee_stim_hpc set to " 
		<< vm["tauc_ee_stim_hpc"].as<double>() << ".\n";
      tauc_ee_stim_hpc = vm["tauc_ee_stim_hpc"].as<double>();
    }

    if (vm.count("ujump_ee_stim_hpc")) {
      std::cout << "ujump_ee_stim_hpc set to " 
		<< vm["ujump_ee_stim_hpc"].as<double>() << ".\n";
      ujump_ee_stim_hpc = vm["ujump_ee_stim_hpc"].as<double>();
    }

    if (vm.count("beta_hpc")) {
      std::cout << "beta_hpc set to " 
		<< vm["beta_hpc"].as<double>() << ".\n";
      beta_hpc = vm["beta_hpc"].as<double>();
    }

    if (vm.count("beta_stim_hpc")) {
      std::cout << "beta_stim_hpc set to " 
		<< vm["beta_stim_hpc"].as<double>() << ".\n";
      beta_stim_hpc = vm["beta_stim_hpc"].as<double>();
    }

    if (vm.count("delta_hpc")) {
      std::cout << "delta_hpc set to " 
		<< vm["delta_hpc"].as<double>() << ".\n";
      delta_hpc = vm["delta_hpc"].as<double>();
    }

    if (vm.count("weight_a_hpc")) {
      std::cout << "weight_a_hpc set to " 
		<< vm["weight_a_hpc"].as<double>() << ".\n";
      weight_a_hpc = vm["weight_a_hpc"].as<double>();
    } 

    if (vm.count("weight_c_hpc")) {
      std::cout << "weight_c_hpc set to " 
		<< vm["weight_c_hpc"].as<double>() << ".\n";
      weight_c_hpc = vm["weight_c_hpc"].as<double>();
    }

    if (vm.count("adapt1_hpc")) {
      std::cout << "adapt1_hpc set to " 
		<< vm["adapt1_hpc"].as<double>() << ".\n";
      adapt1_hpc = vm["adapt1_hpc"].as<double>();
    }

    if (vm.count("adapt2_hpc")) {
      std::cout << "adapt2_hpc set to " 
		<< vm["adapt2_hpc"].as<double>() << ".\n";
      adapt2_hpc = vm["adapt2_hpc"].as<double>();
    }

    if (vm.count("pot_strength_hpc")) {
      std::cout << "pot_strength_hpc set to " 
		<< vm["pot_strength_hpc"].as<double>() << ".\n";
      pot_strength_hpc = vm["pot_strength_hpc"].as<double>();
    }

    if (vm.count("wmax_exc_hpc")) {
      std::cout << "wmax_exc_hpc set to " 
		<< vm["wmax_exc_hpc"].as<double>() << ".\n";
      wmax_exc_hpc = vm["wmax_exc_hpc"].as<double>();
    }

    if (vm.count("wmin_exc_hpc")) {
      std::cout << "wmin_exc_hpc set to " 
		<< vm["wmin_exc_hpc"].as<double>() << ".\n";
      wmin_exc_hpc = vm["wmin_exc_hpc"].as<double>();
    }

    if (vm.count("wmax_inh_hpc")) {
      std::cout << "wmax_inh_hpc set to " 
		<< vm["wmax_inh_hpc"].as<double>() << ".\n";
      wmax_inh_hpc = vm["wmax_inh_hpc"].as<double>();
    }

    if (vm.count("wmin_inh_hpc")) {
      std::cout << "wmin_inh_hpc set to " 
		<< vm["wmin_inh_hpc"].as<double>() << ".\n";
      wmin_inh_hpc = vm["wmin_inh_hpc"].as<double>();
    }

    if (vm.count("wee_hpc")) {
      std::cout << "wee_hpc set to " 
		<< vm["wee_hpc"].as<double>() << ".\n";
      wee_hpc = vm["wee_hpc"].as<double>();
    } 

    if (vm.count("wei_hpc")) {
      std::cout << "wei_hpc set to " 
		<< vm["wei_hpc"].as<double>() << ".\n";
      wei_hpc = vm["wei_hpc"].as<double>();
    }

    if (vm.count("wie_hpc")) {
      std::cout << "wie_hpc set to " 
		<< vm["wie_hpc"].as<double>() << ".\n";
      wie_hpc = vm["wie_hpc"].as<double>();
    }

    if (vm.count("wii_hpc")) {
      std::cout << "wii_hpc set to " 
		<< vm["wii_hpc"].as<double>() << ".\n";
      wii_hpc = vm["wii_hpc"].as<double>();
    }

    if (vm.count("wext_hpc")) {
      std::cout << "wext_hpc set to " 
		<< vm["wext_hpc"].as<double>() << ".\n";
      wext_hpc = vm["wext_hpc"].as<double>();
    }

    if (vm.count("wext_ei_hpc")) {
      std::cout << "wext_ei_hpc set to " 
		<< vm["wext_ei_hpc"].as<double>() << ".\n";
      wext_ei_hpc = vm["wext_ei_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ee_hpc")) {
      std::cout << "sparseness_int_ee_hpc set to " 
		<< vm["sparseness_int_ee_hpc"].as<double>() << ".\n";
      sparseness_int_ee_hpc = vm["sparseness_int_ee_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ei_hpc")) {
      std::cout << "sparseness_int_ei_hpc set to " 
		<< vm["sparseness_int_ei_hpc"].as<double>() << ".\n";
      sparseness_int_ei_hpc = vm["sparseness_int_ei_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ii_hpc")) {
      std::cout << "sparseness_int_ii_hpc set to " 
		<< vm["sparseness_int_ii_hpc"].as<double>() << ".\n";
      sparseness_int_ii_hpc = vm["sparseness_int_ii_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ie_hpc")) {
      std::cout << "sparseness_int_ie_hpc set to " 
		<< vm["sparseness_int_ie_hpc"].as<double>() << ".\n";
      sparseness_int_ie_hpc = vm["sparseness_int_ie_hpc"].as<double>();
    }

    if (vm.count("sparseness_ext_hpc")) {
      std::cout << "sparseness_ext_hpc set to " 
		<< vm["sparseness_ext_hpc"].as<double>() << ".\n";
      sparseness_ext_hpc = vm["sparseness_ext_hpc"].as<double>();
    }

    if (vm.count("tauf_ei_hpc")) {
      std::cout << "tauf_ei_hpc set to " 
		<< vm["tauf_ei_hpc"].as<double>() << ".\n";
      tauf_ei_hpc = vm["tauf_ei_hpc"].as<double>();
    }

    if (vm.count("taud_ei_hpc")) {
      std::cout << "taud_ei_hpc set to " 
		<< vm["taud_ei_hpc"].as<double>() << ".\n";
      taud_ei_hpc = vm["taud_ei_hpc"].as<double>();
    }

    if (vm.count("ujump_ei_hpc")) {
      std::cout << "ujump_ei_hpc set to " 
		<< vm["ujump_ei_hpc"].as<double>() << ".\n";
      ujump_ei_hpc = vm["ujump_ei_hpc"].as<double>();
    }

    if (vm.count("tauh_ie_hpc")) {
      std::cout << "tauh_ie_hpc set to " 
		<< vm["tauh_ie_hpc"].as<double>() << ".\n";
      tauh_ie_hpc = vm["tauh_ie_hpc"].as<double>();
    }

    if (vm.count("taud_ei_stim_hpc")) {
      std::cout << "taud_ei_stim_hpc set to " 
		<< vm["taud_ei_stim_hpc"].as<double>() << ".\n";
      taud_ei_stim_hpc = vm["taud_ei_stim_hpc"].as<double>();
    }

    if (vm.count("tauf_ei_stim_hpc")) {
      std::cout << "tauf_ei_stim_hpc set to " 
		<< vm["tauf_ei_stim_hpc"].as<double>() << ".\n";
      tauf_ei_stim_hpc = vm["tauf_ei_stim_hpc"].as<double>();
    }

    if (vm.count("ujump_ei_stim_hpc")) {
      std::cout << "ujump_ei_stim_hpc set to " 
		<< vm["ujump_ei_stim_hpc"].as<double>() << ".\n";
      ujump_ei_stim_hpc = vm["ujump_ei_stim_hpc"].as<double>();
    }

    if (vm.count("tau_ampa_e_hpc")) {
      std::cout << "tau_ampa_e_hpc set to " 
		<< vm["tau_ampa_e_hpc"].as<double>() << ".\n";
      tau_ampa_e_hpc = vm["tau_ampa_e_hpc"].as<double>();
    }

    if (vm.count("tau_gaba_e_hpc")) {
      std::cout << "tau_gaba_e_hpc set to " 
		<< vm["tau_gaba_e_hpc"].as<double>() << ".\n";
      tau_gaba_e_hpc = vm["tau_gaba_e_hpc"].as<double>();
    }

    if (vm.count("tau_nmda_e_hpc")) {
      std::cout << "tau_nmda_e_hpc set to " 
		<< vm["tau_nmda_e_hpc"].as<double>() << ".\n";
      tau_nmda_e_hpc = vm["tau_nmda_e_hpc"].as<double>();
    }

    if (vm.count("ampa_nmda_e_hpc")) {
      std::cout << "ampa_nmda_e_hpc set to " 
		<< vm["ampa_nmda_e_hpc"].as<double>() << ".\n";
      ampa_nmda_e_hpc = vm["ampa_nmda_e_hpc"].as<double>();
    }

    if (vm.count("tau_ampa_i_hpc")) {
      std::cout << "tau_ampa_i_hpc set to " 
		<< vm["tau_ampa_i_hpc"].as<double>() << ".\n";
      tau_ampa_i_hpc = vm["tau_ampa_i_hpc"].as<double>();
    }

    if (vm.count("tau_gaba_i_hpc")) {
      std::cout << "tau_gaba_i_hpc set to " 
		<< vm["tau_gaba_i_hpc"].as<double>() << ".\n";
      tau_gaba_i_hpc = vm["tau_gaba_i_hpc"].as<double>();
    }

    if (vm.count("tau_nmda_i_hpc")) {
      std::cout << "tau_nmda_i_hpc set to " 
		<< vm["tau_nmda_i_hpc"].as<double>() << ".\n";
      tau_nmda_i_hpc = vm["tau_nmda_i_hpc"].as<double>();
    }

    if (vm.count("ampa_nmda_i_hpc")) {
      std::cout << "ampa_nmda_i_hpc set to " 
		<< vm["ampa_nmda_i_hpc"].as<double>() << ".\n";
      ampa_nmda_i_hpc = vm["ampa_nmda_i_hpc"].as<double>();
    }
    
    if (vm.count("prefile_hpc")) {
      std::cout << "prefile_hpc set to " 
		<< vm["prefile_hpc"].as<string>() << ", ";
      prefile_hpc = vm["prefile_hpc"].as<string>();
    }

    if (vm.count("chi_hpc")) {
      std::cout << "chi_hpc set to " 
		<< vm["chi_hpc"].as<double>() << ".\n";
      chi_hpc = vm["chi_hpc"].as<double>();
    }

    if (vm.count("eta_exc_inh2_hpc")) {
      std::cout << "eta_exc_inh2_hpc set to " 
		<< vm["eta_exc_inh2_hpc"].as<double>() << ".\n";
      eta_exc_inh2_hpc = vm["eta_exc_inh2_hpc"].as<double>();
    }

    if (vm.count("exc_inh2_hpc")) {
      std::cout << "exc_inh2_hpc set to " 
		<< vm["exc_inh2_hpc"].as<int>() << ".\n";
      exc_inh2_hpc = vm["exc_inh2_hpc"].as<int>();
    }
    
    if (vm.count("alpha2_hpc")) {
      std::cout << "alpha2_hpc set to " 
		<< vm["alpha2_hpc"].as<double>() << ".\n";
      alpha2_hpc = vm["alpha2_hpc"].as<double>();
    }

    if (vm.count("tau_ampa_i2_hpc")) {
      std::cout << "tau_ampa_i2_hpc set to " 
		<< vm["tau_ampa_i2_hpc"].as<double>() << ".\n";
      tau_ampa_i2_hpc = vm["tau_ampa_i2_hpc"].as<double>();
    }

    if (vm.count("tau_gaba_i2_hpc")) {
      std::cout << "tau_gaba_i2_hpc set to " 
		<< vm["tau_gaba_i2_hpc"].as<double>() << ".\n";
      tau_gaba_i2_hpc = vm["tau_gaba_i2_hpc"].as<double>();
    }

    if (vm.count("tau_nmda_i2_hpc")) {
      std::cout << "tau_nmda_i2_hpc set to " 
		<< vm["tau_nmda_i2_hpc"].as<double>() << ".\n";
      tau_nmda_i2_hpc = vm["tau_nmda_i2_hpc"].as<double>();
    }

    if (vm.count("ampa_nmda_i2_hpc")) {
      std::cout << "ampa_nmda_i2_hpc set to " 
		<< vm["ampa_nmda_i2_hpc"].as<double>() << ".\n";
      ampa_nmda_i2_hpc = vm["ampa_nmda_i2_hpc"].as<double>();
    }

    if (vm.count("wei2_hpc")) {
      std::cout << "wei2_hpc set to " 
		<< vm["wei2_hpc"].as<double>() << ".\n";
      wei2_hpc = vm["wei2_hpc"].as<double>();
    }

    if (vm.count("wi2e_hpc")) {
      std::cout << "wi2e_hpc set to " 
		<< vm["wi2e_hpc"].as<double>() << ".\n";
      wi2e_hpc = vm["wi2e_hpc"].as<double>();
    }

    if (vm.count("wi2i2_hpc")) {
      std::cout << "wi2i2_hpc set to " 
		<< vm["wi2i2_hpc"].as<double>() << ".\n";
      wi2i2_hpc = vm["wi2i2_hpc"].as<double>();
    }

    if (vm.count("wii2_hpc")) {
      std::cout << "wii2_hpc set to " 
		<< vm["wii2_hpc"].as<double>() << ".\n";
      wii2_hpc = vm["wii2_hpc"].as<double>();
    }

    if (vm.count("wi2i_hpc")) {
      std::cout << "wi2i_hpc set to " 
		<< vm["wi2i_hpc"].as<double>() << ".\n";
      wi2i_hpc = vm["wi2i_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ei2_hpc")) {
      std::cout << "sparseness_int_ei2_hpc set to " 
		<< vm["sparseness_int_ei2_hpc"].as<double>() << ".\n";
      sparseness_int_ei2_hpc = vm["sparseness_int_ei2_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_i2e_hpc")) {
      std::cout << "sparseness_int_i2e_hpc set to " 
		<< vm["sparseness_int_i2e_hpc"].as<double>() << ".\n";
      sparseness_int_i2e_hpc = vm["sparseness_int_i2e_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_i2i2_hpc")) {
      std::cout << "sparseness_int_i2i2_hpc set to " 
		<< vm["sparseness_int_i2i2_hpc"].as<double>() << ".\n";
      sparseness_int_i2i2_hpc = vm["sparseness_int_i2i2_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_ii2_hpc")) {
      std::cout << "sparseness_int_ii2_hpc set to " 
		<< vm["sparseness_int_ii2_hpc"].as<double>() << ".\n";
      sparseness_int_ii2_hpc = vm["sparseness_int_ii2_hpc"].as<double>();
    }

    if (vm.count("sparseness_int_i2i_hpc")) {
      std::cout << "sparseness_int_i2i_hpc set to " 
		<< vm["sparseness_int_i2i_hpc"].as<double>() << ".\n";
      sparseness_int_i2i_hpc = vm["sparseness_int_i2i_hpc"].as<double>();
    }

    if (vm.count("tauf_ei2_hpc")) {
      std::cout << "tauf_ei2_hpc set to " 
		<< vm["tauf_ei2_hpc"].as<double>() << ".\n";
      tauf_ei2_hpc = vm["tauf_ei2_hpc"].as<double>();
    }

    if (vm.count("taud_ei2_hpc")) {
      std::cout << "taud_ei2_hpc set to " 
		<< vm["taud_ei2_hpc"].as<double>() << ".\n";
      taud_ei2_hpc = vm["taud_ei2_hpc"].as<double>();
    }

    if (vm.count("ujump_ei2_hpc")) {
      std::cout << "ujump_ei2_hpc set to " 
		<< vm["ujump_ei2_hpc"].as<double>() << ".\n";
      ujump_ei2_hpc = vm["ujump_ei2_hpc"].as<double>();
    }

    if (vm.count("tauh_i2e_hpc")) {
      std::cout << "tauh_i2e_hpc set to " 
		<< vm["tauh_i2e_hpc"].as<double>() << ".\n";
      tauh_i2e_hpc = vm["tauh_i2e_hpc"].as<double>();
    }

    if (vm.count("wmax_inh2_hpc")) {
      std::cout << "wmax_inh2_hpc set to " 
		<< vm["wmax_inh2_hpc"].as<double>() << ".\n";
      wmax_inh2_hpc = vm["wmax_inh2_hpc"].as<double>();
    }

    if (vm.count("wmin_inh2_hpc")) {
      std::cout << "wmin_inh2_hpc set to " 
		<< vm["wmin_inh2_hpc"].as<double>() << ".\n";
      wmin_inh2_hpc = vm["wmin_inh2_hpc"].as<double>();
    }

    if (vm.count("record_neuron_inh2_hpc")) {
      std::cout << "record_neuron_inh2_hpc set to " 
		<< vm["record_neuron_inh2_hpc"].as<int>() << ".\n";
      record_neuron_inh2_hpc = vm["record_neuron_inh2_hpc"].as<int>();
    }

    if (vm.count("block_inh2_hpc")) {
      std::cout << "block_inh2_hpc " 
		<< vm["block_inh2_hpc"].as<string>() << ".\n";
      block_inh2_hpc = vm["block_inh2_hpc"].as<string>();
    }

    if (vm.count("wee_thl_ctx")) {
      std::cout << "wee_thl_ctx set to " 
		<< vm["wee_thl_ctx"].as<double>() << ".\n";
      wee_thl_ctx = vm["wee_thl_ctx"].as<double>();
    }

    if (vm.count("wei_thl_ctx")) {
      std::cout << "wei_thl_ctx set to " 
		<< vm["wei_thl_ctx"].as<double>() << ".\n";
      wei_thl_ctx = vm["wei_thl_ctx"].as<double>();
    }

    if (vm.count("sparseness_thl_ctx")) {
      std::cout << "sparseness_thl_ctx set to " 
		<< vm["sparseness_thl_ctx"].as<double>() << ".\n";
      sparseness_thl_ctx = vm["sparseness_thl_ctx"].as<double>();
    }

    if (vm.count("recfile_thl_ctx")) {
      std::cout << "recfile_thl_ctx set to " 
		<< vm["recfile_thl_ctx"].as<string>() << ", ";
      recfile_thl_ctx = vm["recfile_thl_ctx"].as<string>();
    }

    if (vm.count("xi_thl_ctx")) {
      std::cout << "xi_thl_ctx set to " 
		<< vm["xi_thl_ctx"].as<double>() << ".\n";
      xi_thl_ctx = vm["xi_thl_ctx"].as<double>();
    }

    if (vm.count("eta_thl_ctx")) {
      std::cout << "eta_thl_ctx set to " 
		<< vm["eta_thl_ctx"].as<double>() << ".\n";
      eta_thl_ctx = vm["eta_thl_ctx"].as<double>();
    }

    if (vm.count("kappa_thl_ctx")) {
      std::cout << "kappa_thl_ctx set to " 
		<< vm["kappa_thl_ctx"].as<double>() << ".\n";
      kappa_thl_ctx = vm["kappa_thl_ctx"].as<double>();
    }

    if (vm.count("tauf_thl_ctx")) {
      std::cout << "tauf_thl_ctx set to " 
		<< vm["tauf_thl_ctx"].as<double>() << ".\n";
      tauf_thl_ctx = vm["tauf_thl_ctx"].as<double>();
    }

    if (vm.count("taud_thl_ctx")) {
      std::cout << "taud_thl_ctx set to " 
		<< vm["taud_thl_ctx"].as<double>() << ".\n";
      taud_thl_ctx = vm["taud_thl_ctx"].as<double>();
    }

    if (vm.count("tauh_thl_ctx")) {
      std::cout << "tauh_thl_ctx set to " 
		<< vm["tauh_thl_ctx"].as<double>() << ".\n";
      tauh_thl_ctx = vm["tauh_thl_ctx"].as<double>();
    }

    if (vm.count("tauc_thl_ctx")) {
      std::cout << "tauc_thl_ctx set to " 
		<< vm["tauc_thl_ctx"].as<double>() << ".\n";
      tauc_thl_ctx = vm["tauc_thl_ctx"].as<double>();
    }

    if (vm.count("ujump_thl_ctx")) {
      std::cout << "ujump_thl_ctx set to " 
		<< vm["ujump_thl_ctx"].as<double>() << ".\n";
      ujump_thl_ctx = vm["ujump_thl_ctx"].as<double>();
    }

    if (vm.count("beta_thl_ctx")) {
      std::cout << "beta_thl_ctx set to " 
		<< vm["beta_thl_ctx"].as<double>() << ".\n";
      beta_thl_ctx = vm["beta_thl_ctx"].as<double>();
    }

    if (vm.count("delta_thl_ctx")) {
      std::cout << "delta_thl_ctx set to " 
		<< vm["delta_thl_ctx"].as<double>() << ".\n";
      delta_thl_ctx = vm["delta_thl_ctx"].as<double>();
    }

    if (vm.count("weight_a_thl_ctx")) {
      std::cout << "weight_a_thl_ctx set to " 
		<< vm["weight_a_thl_ctx"].as<double>() << ".\n";
      weight_a_thl_ctx = vm["weight_a_thl_ctx"].as<double>();
    } 

    if (vm.count("weight_c_thl_ctx")) {
      std::cout << "weight_c_thl_ctx set to " 
		<< vm["weight_c_thl_ctx"].as<double>() << ".\n";
      weight_c_thl_ctx = vm["weight_c_thl_ctx"].as<double>();
    }

    if (vm.count("pot_strength_thl_ctx")) {
      std::cout << "pot_strength_thl_ctx set to " 
		<< vm["pot_strength_thl_ctx"].as<double>() << ".\n";
      pot_strength_thl_ctx = vm["pot_strength_thl_ctx"].as<double>();
    }

    if (vm.count("wmax_thl_ctx")) {
      std::cout << "wmax_thl_ctx set to " 
		<< vm["wmax_thl_ctx"].as<double>() << ".\n";
      wmax_thl_ctx = vm["wmax_thl_ctx"].as<double>();
    }

    if (vm.count("wmin_thl_ctx")) {
      std::cout << "wmin_thl_ctx set to " 
		<< vm["wmin_thl_ctx"].as<double>() << ".\n";
      wmin_thl_ctx = vm["wmin_thl_ctx"].as<double>();
    }

    if (vm.count("wee_ctx_thl")) {
      std::cout << "wee_ctx_thl set to " 
		<< vm["wee_ctx_thl"].as<double>() << ".\n";
      wee_ctx_thl = vm["wee_ctx_thl"].as<double>();
    }

    if (vm.count("wei_ctx_thl")) {
      std::cout << "wei_ctx_thl set to " 
		<< vm["wei_ctx_thl"].as<double>() << ".\n";
      wei_ctx_thl = vm["wei_ctx_thl"].as<double>();
    }

    if (vm.count("sparseness_ctx_thl")) {
      std::cout << "sparseness_ctx_thl set to " 
		<< vm["sparseness_ctx_thl"].as<double>() << ".\n";
      sparseness_ctx_thl = vm["sparseness_ctx_thl"].as<double>();
    }

    if (vm.count("recfile_ctx_thl")) {
      std::cout << "recfile_ctx_thl set to " 
		<< vm["recfile_ctx_thl"].as<string>() << ", ";
      recfile_ctx_thl = vm["recfile_ctx_thl"].as<string>();
    }

    if (vm.count("xi_ctx_thl")) {
      std::cout << "xi_ctx_thl set to " 
		<< vm["xi_ctx_thl"].as<double>() << ".\n";
      xi_ctx_thl = vm["xi_ctx_thl"].as<double>();
    }

    if (vm.count("eta_ctx_thl")) {
      std::cout << "eta_ctx_thl set to " 
		<< vm["eta_ctx_thl"].as<double>() << ".\n";
      eta_ctx_thl = vm["eta_ctx_thl"].as<double>();
    }

    if (vm.count("kappa_ctx_thl")) {
      std::cout << "kappa_ctx_thl set to " 
		<< vm["kappa_ctx_thl"].as<double>() << ".\n";
      kappa_ctx_thl = vm["kappa_ctx_thl"].as<double>();
    }

    if (vm.count("tauf_ctx_thl")) {
      std::cout << "tauf_ctx_thl set to " 
		<< vm["tauf_ctx_thl"].as<double>() << ".\n";
      tauf_ctx_thl = vm["tauf_ctx_thl"].as<double>();
    }

    if (vm.count("taud_ctx_thl")) {
      std::cout << "taud_ctx_thl set to " 
		<< vm["taud_ctx_thl"].as<double>() << ".\n";
      taud_ctx_thl = vm["taud_ctx_thl"].as<double>();
    }

    if (vm.count("tauh_ctx_thl")) {
      std::cout << "tauh_ctx_thl set to " 
		<< vm["tauh_ctx_thl"].as<double>() << ".\n";
      tauh_ctx_thl = vm["tauh_ctx_thl"].as<double>();
    }

    if (vm.count("tauc_ctx_thl")) {
      std::cout << "tauc_ctx_thl set to " 
		<< vm["tauc_ctx_thl"].as<double>() << ".\n";
      tauc_ctx_thl = vm["tauc_ctx_thl"].as<double>();
    }

    if (vm.count("ujump_ctx_thl")) {
      std::cout << "ujump_ctx_thl set to " 
		<< vm["ujump_ctx_thl"].as<double>() << ".\n";
      ujump_ctx_thl = vm["ujump_ctx_thl"].as<double>();
    }

    if (vm.count("beta_ctx_thl")) {
      std::cout << "beta_ctx_thl set to " 
		<< vm["beta_ctx_thl"].as<double>() << ".\n";
      beta_ctx_thl = vm["beta_ctx_thl"].as<double>();
    }

    if (vm.count("delta_ctx_thl")) {
      std::cout << "delta_ctx_thl set to " 
		<< vm["delta_ctx_thl"].as<double>() << ".\n";
      delta_ctx_thl = vm["delta_ctx_thl"].as<double>();
    }

    if (vm.count("weight_a_ctx_thl")) {
      std::cout << "weight_a_ctx_thl set to " 
		<< vm["weight_a_ctx_thl"].as<double>() << ".\n";
      weight_a_ctx_thl = vm["weight_a_ctx_thl"].as<double>();
    } 

    if (vm.count("weight_c_ctx_thl")) {
      std::cout << "weight_c_ctx_thl set to " 
		<< vm["weight_c_ctx_thl"].as<double>() << ".\n";
      weight_c_ctx_thl = vm["weight_c_ctx_thl"].as<double>();
    }

    if (vm.count("pot_strength_ctx_thl")) {
      std::cout << "pot_strength_ctx_thl set to " 
		<< vm["pot_strength_ctx_thl"].as<double>() << ".\n";
      pot_strength_ctx_thl = vm["pot_strength_ctx_thl"].as<double>();
    }

    if (vm.count("wmax_ctx_thl")) {
      std::cout << "wmax_ctx_thl set to " 
		<< vm["wmax_ctx_thl"].as<double>() << ".\n";
      wmax_ctx_thl = vm["wmax_ctx_thl"].as<double>();
    }

    if (vm.count("wmin_ctx_thl")) {
      std::cout << "wmin_ctx_thl set to " 
		<< vm["wmin_ctx_thl"].as<double>() << ".\n";
      wmin_ctx_thl = vm["wmin_ctx_thl"].as<double>();
    }

    if (vm.count("wee_thl_hpc")) {
      std::cout << "wee_thl_hpc set to " 
		<< vm["wee_thl_hpc"].as<double>() << ".\n";
      wee_thl_hpc = vm["wee_thl_hpc"].as<double>();
    }

    if (vm.count("wei_thl_hpc")) {
      std::cout << "wei_thl_hpc set to " 
		<< vm["wei_thl_hpc"].as<double>() << ".\n";
      wei_thl_hpc = vm["wei_thl_hpc"].as<double>();
    }

    if (vm.count("sparseness_thl_hpc")) {
      std::cout << "sparseness_thl_hpc set to " 
		<< vm["sparseness_thl_hpc"].as<double>() << ".\n";
      sparseness_thl_hpc = vm["sparseness_thl_hpc"].as<double>();
    }

    if (vm.count("recfile_thl_hpc")) {
      std::cout << "recfile_thl_hpc set to " 
		<< vm["recfile_thl_hpc"].as<string>() << ", ";
      recfile_thl_hpc = vm["recfile_thl_hpc"].as<string>();
    }

    if (vm.count("xi_thl_hpc")) {
      std::cout << "xi_thl_hpc set to " 
		<< vm["xi_thl_hpc"].as<double>() << ".\n";
      xi_thl_hpc = vm["xi_thl_hpc"].as<double>();
    }

    if (vm.count("eta_thl_hpc")) {
      std::cout << "eta_thl_hpc set to " 
		<< vm["eta_thl_hpc"].as<double>() << ".\n";
      eta_thl_hpc = vm["eta_thl_hpc"].as<double>();
    }

    if (vm.count("kappa_thl_hpc")) {
      std::cout << "kappa_thl_hpc set to " 
		<< vm["kappa_thl_hpc"].as<double>() << ".\n";
      kappa_thl_hpc = vm["kappa_thl_hpc"].as<double>();
    }

    if (vm.count("tauf_thl_hpc")) {
      std::cout << "tauf_thl_hpc set to " 
		<< vm["tauf_thl_hpc"].as<double>() << ".\n";
      tauf_thl_hpc = vm["tauf_thl_hpc"].as<double>();
    }

    if (vm.count("taud_thl_hpc")) {
      std::cout << "taud_thl_hpc set to " 
		<< vm["taud_thl_hpc"].as<double>() << ".\n";
      taud_thl_hpc = vm["taud_thl_hpc"].as<double>();
    }

    if (vm.count("tauh_thl_hpc")) {
      std::cout << "tauh_thl_hpc set to " 
		<< vm["tauh_thl_hpc"].as<double>() << ".\n";
      tauh_thl_hpc = vm["tauh_thl_hpc"].as<double>();
    }

    if (vm.count("tauc_thl_hpc")) {
      std::cout << "tauc_thl_hpc set to " 
		<< vm["tauc_thl_hpc"].as<double>() << ".\n";
      tauc_thl_hpc = vm["tauc_thl_hpc"].as<double>();
    }

    if (vm.count("ujump_thl_hpc")) {
      std::cout << "ujump_thl_hpc set to " 
		<< vm["ujump_thl_hpc"].as<double>() << ".\n";
      ujump_thl_hpc = vm["ujump_thl_hpc"].as<double>();
    }

    if (vm.count("beta_thl_hpc")) {
      std::cout << "beta_thl_hpc set to " 
		<< vm["beta_thl_hpc"].as<double>() << ".\n";
      beta_thl_hpc = vm["beta_thl_hpc"].as<double>();
    }

    if (vm.count("delta_thl_hpc")) {
      std::cout << "delta_thl_hpc set to " 
		<< vm["delta_thl_hpc"].as<double>() << ".\n";
      delta_thl_hpc = vm["delta_thl_hpc"].as<double>();
    }

    if (vm.count("weight_a_thl_hpc")) {
      std::cout << "weight_a_thl_hpc set to " 
		<< vm["weight_a_thl_hpc"].as<double>() << ".\n";
      weight_a_thl_hpc = vm["weight_a_thl_hpc"].as<double>();
    } 

    if (vm.count("weight_c_thl_hpc")) {
      std::cout << "weight_c_thl_hpc set to " 
		<< vm["weight_c_thl_hpc"].as<double>() << ".\n";
      weight_c_thl_hpc = vm["weight_c_thl_hpc"].as<double>();
    }

    if (vm.count("pot_strength_thl_hpc")) {
      std::cout << "pot_strength_thl_hpc set to " 
		<< vm["pot_strength_thl_hpc"].as<double>() << ".\n";
      pot_strength_thl_hpc = vm["pot_strength_thl_hpc"].as<double>();
    }

    if (vm.count("wmax_thl_hpc")) {
      std::cout << "wmax_thl_hpc set to " 
		<< vm["wmax_thl_hpc"].as<double>() << ".\n";
      wmax_thl_hpc = vm["wmax_thl_hpc"].as<double>();
    }

    if (vm.count("wmin_thl_hpc")) {
      std::cout << "wmin_thl_hpc set to " 
		<< vm["wmin_thl_hpc"].as<double>() << ".\n";
      wmin_thl_hpc = vm["wmin_thl_hpc"].as<double>();
    }

    if (vm.count("wee_hpc_thl")) {
      std::cout << "wee_hpc_thl set to " 
		<< vm["wee_hpc_thl"].as<double>() << ".\n";
      wee_hpc_thl = vm["wee_hpc_thl"].as<double>();
    }

    if (vm.count("wei_hpc_thl")) {
      std::cout << "wei_hpc_thl set to " 
		<< vm["wei_hpc_thl"].as<double>() << ".\n";
      wei_hpc_thl = vm["wei_hpc_thl"].as<double>();
    }

    if (vm.count("sparseness_hpc_thl")) {
      std::cout << "sparseness_hpc_thl set to " 
		<< vm["sparseness_hpc_thl"].as<double>() << ".\n";
      sparseness_hpc_thl = vm["sparseness_hpc_thl"].as<double>();
    }

    if (vm.count("recfile_hpc_thl")) {
      std::cout << "recfile_hpc_thl set to " 
		<< vm["recfile_hpc_thl"].as<string>() << ", ";
      recfile_hpc_thl = vm["recfile_hpc_thl"].as<string>();
    }

    if (vm.count("xi_hpc_thl")) {
      std::cout << "xi_hpc_thl set to " 
		<< vm["xi_hpc_thl"].as<double>() << ".\n";
      xi_hpc_thl = vm["xi_hpc_thl"].as<double>();
    }

    if (vm.count("eta_hpc_thl")) {
      std::cout << "eta_hpc_thl set to " 
		<< vm["eta_hpc_thl"].as<double>() << ".\n";
      eta_hpc_thl = vm["eta_hpc_thl"].as<double>();
    }

    if (vm.count("kappa_hpc_thl")) {
      std::cout << "kappa_hpc_thl set to " 
		<< vm["kappa_hpc_thl"].as<double>() << ".\n";
      kappa_hpc_thl = vm["kappa_hpc_thl"].as<double>();
    }

    if (vm.count("tauf_hpc_thl")) {
      std::cout << "tauf_hpc_thl set to " 
		<< vm["tauf_hpc_thl"].as<double>() << ".\n";
      tauf_hpc_thl = vm["tauf_hpc_thl"].as<double>();
    }

    if (vm.count("taud_hpc_thl")) {
      std::cout << "taud_hpc_thl set to " 
		<< vm["taud_hpc_thl"].as<double>() << ".\n";
      taud_hpc_thl = vm["taud_hpc_thl"].as<double>();
    }

    if (vm.count("tauh_hpc_thl")) {
      std::cout << "tauh_hpc_thl set to " 
		<< vm["tauh_hpc_thl"].as<double>() << ".\n";
      tauh_hpc_thl = vm["tauh_hpc_thl"].as<double>();
    }

    if (vm.count("tauc_hpc_thl")) {
      std::cout << "tauc_hpc_thl set to " 
		<< vm["tauc_hpc_thl"].as<double>() << ".\n";
      tauc_hpc_thl = vm["tauc_hpc_thl"].as<double>();
    }

    if (vm.count("ujump_hpc_thl")) {
      std::cout << "ujump_hpc_thl set to " 
		<< vm["ujump_hpc_thl"].as<double>() << ".\n";
      ujump_hpc_thl = vm["ujump_hpc_thl"].as<double>();
    }

    if (vm.count("beta_hpc_thl")) {
      std::cout << "beta_hpc_thl set to " 
		<< vm["beta_hpc_thl"].as<double>() << ".\n";
      beta_hpc_thl = vm["beta_hpc_thl"].as<double>();
    }

    if (vm.count("delta_hpc_thl")) {
      std::cout << "delta_hpc_thl set to " 
		<< vm["delta_hpc_thl"].as<double>() << ".\n";
      delta_hpc_thl = vm["delta_hpc_thl"].as<double>();
    }

    if (vm.count("weight_a_hpc_thl")) {
      std::cout << "weight_a_hpc_thl set to " 
		<< vm["weight_a_hpc_thl"].as<double>() << ".\n";
      weight_a_hpc_thl = vm["weight_a_hpc_thl"].as<double>();
    } 

    if (vm.count("weight_c_hpc_thl")) {
      std::cout << "weight_c_hpc_thl set to " 
		<< vm["weight_c_hpc_thl"].as<double>() << ".\n";
      weight_c_hpc_thl = vm["weight_c_hpc_thl"].as<double>();
    }

    if (vm.count("pot_strength_hpc_thl")) {
      std::cout << "pot_strength_hpc_thl set to " 
		<< vm["pot_strength_hpc_thl"].as<double>() << ".\n";
      pot_strength_hpc_thl = vm["pot_strength_hpc_thl"].as<double>();
    }

    if (vm.count("wmax_hpc_thl")) {
      std::cout << "wmax_hpc_thl set to " 
		<< vm["wmax_hpc_thl"].as<double>() << ".\n";
      wmax_hpc_thl = vm["wmax_hpc_thl"].as<double>();
    }

    if (vm.count("wmin_hpc_thl")) {
      std::cout << "wmin_hpc_thl set to " 
		<< vm["wmin_hpc_thl"].as<double>() << ".\n";
      wmin_hpc_thl = vm["wmin_hpc_thl"].as<double>();
    }
    
    if (vm.count("wee_ctx_hpc")) {
      std::cout << "wee_ctx_hpc set to " 
		<< vm["wee_ctx_hpc"].as<double>() << ".\n";
      wee_ctx_hpc = vm["wee_ctx_hpc"].as<double>();
    }

    if (vm.count("wei_ctx_hpc")) {
      std::cout << "wei_ctx_hpc set to " 
		<< vm["wei_ctx_hpc"].as<double>() << ".\n";
      wei_ctx_hpc = vm["wei_ctx_hpc"].as<double>();
    }

    if (vm.count("sparseness_ctx_hpc")) {
      std::cout << "sparseness_ctx_hpc set to " 
		<< vm["sparseness_ctx_hpc"].as<double>() << ".\n";
      sparseness_ctx_hpc = vm["sparseness_ctx_hpc"].as<double>();
    }

    if (vm.count("recfile_ctx_hpc")) {
      std::cout << "recfile_ctx_hpc set to " 
		<< vm["recfile_ctx_hpc"].as<string>() << ", ";
      recfile_ctx_hpc = vm["recfile_ctx_hpc"].as<string>();
    }

    if (vm.count("xi_ctx_hpc")) {
      std::cout << "xi_ctx_hpc set to " 
		<< vm["xi_ctx_hpc"].as<double>() << ".\n";
      xi_ctx_hpc = vm["xi_ctx_hpc"].as<double>();
    }

    if (vm.count("eta_ctx_hpc")) {
      std::cout << "eta_ctx_hpc set to " 
		<< vm["eta_ctx_hpc"].as<double>() << ".\n";
      eta_ctx_hpc = vm["eta_ctx_hpc"].as<double>();
    }

    if (vm.count("kappa_ctx_hpc")) {
      std::cout << "kappa_ctx_hpc set to " 
		<< vm["kappa_ctx_hpc"].as<double>() << ".\n";
      kappa_ctx_hpc = vm["kappa_ctx_hpc"].as<double>();
    }

    if (vm.count("tauf_ctx_hpc")) {
      std::cout << "tauf_ctx_hpc set to " 
		<< vm["tauf_ctx_hpc"].as<double>() << ".\n";
      tauf_ctx_hpc = vm["tauf_ctx_hpc"].as<double>();
    }

    if (vm.count("taud_ctx_hpc")) {
      std::cout << "taud_ctx_hpc set to " 
		<< vm["taud_ctx_hpc"].as<double>() << ".\n";
      taud_ctx_hpc = vm["taud_ctx_hpc"].as<double>();
    }

    if (vm.count("tauh_ctx_hpc")) {
      std::cout << "tauh_ctx_hpc set to " 
		<< vm["tauh_ctx_hpc"].as<double>() << ".\n";
      tauh_ctx_hpc = vm["tauh_ctx_hpc"].as<double>();
    }

    if (vm.count("tauc_ctx_hpc")) {
      std::cout << "tauc_ctx_hpc set to " 
		<< vm["tauc_ctx_hpc"].as<double>() << ".\n";
      tauc_ctx_hpc = vm["tauc_ctx_hpc"].as<double>();
    }

    if (vm.count("ujump_ctx_hpc")) {
      std::cout << "ujump_ctx_hpc set to " 
		<< vm["ujump_ctx_hpc"].as<double>() << ".\n";
      ujump_ctx_hpc = vm["ujump_ctx_hpc"].as<double>();
    }

    if (vm.count("beta_ctx_hpc")) {
      std::cout << "beta_ctx_hpc set to " 
		<< vm["beta_ctx_hpc"].as<double>() << ".\n";
      beta_ctx_hpc = vm["beta_ctx_hpc"].as<double>();
    }

    if (vm.count("delta_ctx_hpc")) {
      std::cout << "delta_ctx_hpc set to " 
		<< vm["delta_ctx_hpc"].as<double>() << ".\n";
      delta_ctx_hpc = vm["delta_ctx_hpc"].as<double>();
    }

    if (vm.count("weight_a_ctx_hpc")) {
      std::cout << "weight_a_ctx_hpc set to " 
		<< vm["weight_a_ctx_hpc"].as<double>() << ".\n";
      weight_a_ctx_hpc = vm["weight_a_ctx_hpc"].as<double>();
    } 

    if (vm.count("weight_c_ctx_hpc")) {
      std::cout << "weight_c_ctx_hpc set to " 
		<< vm["weight_c_ctx_hpc"].as<double>() << ".\n";
      weight_c_ctx_hpc = vm["weight_c_ctx_hpc"].as<double>();
    }

    if (vm.count("pot_strength_ctx_hpc")) {
      std::cout << "pot_strength_ctx_hpc set to " 
		<< vm["pot_strength_ctx_hpc"].as<double>() << ".\n";
      pot_strength_ctx_hpc = vm["pot_strength_ctx_hpc"].as<double>();
    }

    if (vm.count("wmax_ctx_hpc")) {
      std::cout << "wmax_ctx_hpc set to " 
		<< vm["wmax_ctx_hpc"].as<double>() << ".\n";
      wmax_ctx_hpc = vm["wmax_ctx_hpc"].as<double>();
    }

    if (vm.count("wmin_ctx_hpc")) {
      std::cout << "wmin_ctx_hpc set to " 
		<< vm["wmin_ctx_hpc"].as<double>() << ".\n";
      wmin_ctx_hpc = vm["wmin_ctx_hpc"].as<double>();
    }
    
    if (vm.count("wee_hpc_ctx")) {
      std::cout << "wee_hpc_ctx set to " 
		<< vm["wee_hpc_ctx"].as<double>() << ".\n";
      wee_hpc_ctx = vm["wee_hpc_ctx"].as<double>();
    }

    if (vm.count("wei_hpc_ctx")) {
      std::cout << "wei_hpc_ctx set to " 
		<< vm["wei_hpc_ctx"].as<double>() << ".\n";
      wei_hpc_ctx = vm["wei_hpc_ctx"].as<double>();
    }

    if (vm.count("sparseness_hpc_ctx")) {
      std::cout << "sparseness_hpc_ctx set to " 
		<< vm["sparseness_hpc_ctx"].as<double>() << ".\n";
      sparseness_hpc_ctx = vm["sparseness_hpc_ctx"].as<double>();
    }

    if (vm.count("recfile_hpc_ctx")) {
      std::cout << "recfile_hpc_ctx set to " 
		<< vm["recfile_hpc_ctx"].as<string>() << ", ";
      recfile_hpc_ctx = vm["recfile_hpc_ctx"].as<string>();
    }

    if (vm.count("xi_hpc_ctx")) {
      std::cout << "xi_hpc_ctx set to " 
		<< vm["xi_hpc_ctx"].as<double>() << ".\n";
      xi_hpc_ctx = vm["xi_hpc_ctx"].as<double>();
    }

    if (vm.count("eta_hpc_ctx")) {
      std::cout << "eta_hpc_ctx set to " 
		<< vm["eta_hpc_ctx"].as<double>() << ".\n";
      eta_hpc_ctx = vm["eta_hpc_ctx"].as<double>();
    }

    if (vm.count("kappa_hpc_ctx")) {
      std::cout << "kappa_hpc_ctx set to " 
		<< vm["kappa_hpc_ctx"].as<double>() << ".\n";
      kappa_hpc_ctx = vm["kappa_hpc_ctx"].as<double>();
    }

    if (vm.count("tauf_hpc_ctx")) {
      std::cout << "tauf_hpc_ctx set to " 
		<< vm["tauf_hpc_ctx"].as<double>() << ".\n";
      tauf_hpc_ctx = vm["tauf_hpc_ctx"].as<double>();
    }

    if (vm.count("taud_hpc_ctx")) {
      std::cout << "taud_hpc_ctx set to " 
		<< vm["taud_hpc_ctx"].as<double>() << ".\n";
      taud_hpc_ctx = vm["taud_hpc_ctx"].as<double>();
    }

    if (vm.count("tauh_hpc_ctx")) {
      std::cout << "tauh_hpc_ctx set to " 
		<< vm["tauh_hpc_ctx"].as<double>() << ".\n";
      tauh_hpc_ctx = vm["tauh_hpc_ctx"].as<double>();
    }

    if (vm.count("tauc_hpc_ctx")) {
      std::cout << "tauc_hpc_ctx set to " 
		<< vm["tauc_hpc_ctx"].as<double>() << ".\n";
      tauc_hpc_ctx = vm["tauc_hpc_ctx"].as<double>();
    }

    if (vm.count("ujump_hpc_ctx")) {
      std::cout << "ujump_hpc_ctx set to " 
		<< vm["ujump_hpc_ctx"].as<double>() << ".\n";
      ujump_hpc_ctx = vm["ujump_hpc_ctx"].as<double>();
    }

    if (vm.count("beta_hpc_ctx")) {
      std::cout << "beta_hpc_ctx set to " 
		<< vm["beta_hpc_ctx"].as<double>() << ".\n";
      beta_hpc_ctx = vm["beta_hpc_ctx"].as<double>();
    }

    if (vm.count("delta_hpc_ctx")) {
      std::cout << "delta_hpc_ctx set to " 
		<< vm["delta_hpc_ctx"].as<double>() << ".\n";
      delta_hpc_ctx = vm["delta_hpc_ctx"].as<double>();
    }

    if (vm.count("weight_a_hpc_ctx")) {
      std::cout << "weight_a_hpc_ctx set to " 
		<< vm["weight_a_hpc_ctx"].as<double>() << ".\n";
      weight_a_hpc_ctx = vm["weight_a_hpc_ctx"].as<double>();
    } 

    if (vm.count("weight_c_hpc_ctx")) {
      std::cout << "weight_c_hpc_ctx set to " 
		<< vm["weight_c_hpc_ctx"].as<double>() << ".\n";
      weight_c_hpc_ctx = vm["weight_c_hpc_ctx"].as<double>();
    }

    if (vm.count("pot_strength_hpc_ctx")) {
      std::cout << "pot_strength_hpc_ctx set to " 
		<< vm["pot_strength_hpc_ctx"].as<double>() << ".\n";
      pot_strength_hpc_ctx = vm["pot_strength_hpc_ctx"].as<double>();
    }

    if (vm.count("wmax_hpc_ctx")) {
      std::cout << "wmax_hpc_ctx set to " 
		<< vm["wmax_hpc_ctx"].as<double>() << ".\n";
      wmax_hpc_ctx = vm["wmax_hpc_ctx"].as<double>();
    }

    if (vm.count("wmin_hpc_ctx")) {
      std::cout << "wmin_hpc_ctx set to " 
		<< vm["wmin_hpc_ctx"].as<double>() << ".\n";
      wmin_hpc_ctx = vm["wmin_hpc_ctx"].as<double>();
    }

    if (vm.count("ontime")) {
      std::cout << "ontime set to " 
		<< vm["ontime"].as<double>() << ".\n";
      ontime = vm["ontime"].as<double>();
    }

    if (vm.count("offtime")) {
      std::cout << "offtime set to " 
		<< vm["offtime"].as<double>() << ".\n";
      offtime = vm["offtime"].as<double>();
    }

    if (vm.count("scale")) {
      std::cout << "scale set to " 
		<< vm["scale"].as<double>() << ".\n";
      scale = vm["scale"].as<double>();
    }

    if (vm.count("bgrate")) {
      std::cout << "bgrate set to " 
		<< vm["bgrate"].as<double>() << ".\n";
      bgrate = vm["bgrate"].as<double>();
    }

    if (vm.count("preferred")) {
      std::cout << "preferred set to " 
		<< vm["preferred"].as<int>() << ".\n";
      preferred = vm["preferred"].as<int>();
    }

    if (vm.count("stimfile")) {
      std::cout << "stimfile set to " 
		<< vm["stimfile"].as<string>() << ", ";
      stimfile = vm["stimfile"].as<string>();
      monfile_ctx = stimfile;
      premonfile_ctx = stimfile;
    }

    if (vm.count("recfile_stim_ctx")) {
      std::cout << "recfile_stim_ctx set to " 
		<< vm["recfile_stim_ctx"].as<string>() << ", ";
      recfile_stim_ctx = vm["recfile_stim_ctx"].as<string>();
    }

    if (vm.count("xi_stim_ctx")) {
      std::cout << "xi_stim_ctx set to " 
		<< vm["xi_stim_ctx"].as<double>() << ".\n";
      xi_stim_ctx = vm["xi_stim_ctx"].as<double>();
    }

    if (vm.count("recfile_stim_hpc")) {
      std::cout << "recfile_stim_hpc set to " 
		<< vm["recfile_stim_hpc"].as<string>() << ", ";
      recfile_stim_hpc = vm["recfile_stim_hpc"].as<string>();
    }

    if (vm.count("xi_stim_hpc")) {
      std::cout << "xi_stim_hpc set to " 
		<< vm["xi_stim_hpc"].as<double>() << ".\n";
      xi_stim_hpc = vm["xi_stim_hpc"].as<double>();
    }

    if (vm.count("recfile_stim_thl")) {
      std::cout << "recfile_stim_thl set to " 
		<< vm["recfile_stim_thl"].as<string>() << ", ";
      recfile_stim_thl = vm["recfile_stim_thl"].as<string>();
    }

    if (vm.count("xi_stim_thl")) {
      std::cout << "xi_stim_thl set to " 
		<< vm["xi_stim_thl"].as<double>() << ".\n";
      xi_stim_thl = vm["xi_stim_thl"].as<double>();
    }

    if (vm.count("prime_ontime")) {
      std::cout << "prime_ontime set to " 
		<< vm["prime_ontime"].as<double>() << ".\n";
      prime_ontime = vm["prime_ontime"].as<double>();
    }

    if (vm.count("prime_offtime")) {
      std::cout << "prime_offtime set to " 
		<< vm["prime_offtime"].as<double>() << ".\n";
      prime_offtime = vm["prime_offtime"].as<double>();
    }

    if (vm.count("prime_duration")) {
      std::cout << "prime_duration set to " 
		<< vm["prime_duration"].as<double>() << ".\n";
      prime_duration = vm["prime_duration"].as<double>();
    }

    if (vm.count("record_neuron_exc_ctx")) {
      std::cout << "record_neuron_exc_ctx set to " 
		<< vm["record_neuron_exc_ctx"].as<int>() << ".\n";
      record_neuron_exc_ctx = vm["record_neuron_exc_ctx"].as<int>();
    }

    if (vm.count("record_neuron_inh_ctx")) {
      std::cout << "record_neuron_inh_ctx set to " 
		<< vm["record_neuron_inh_ctx"].as<int>() << ".\n";
      record_neuron_inh_ctx = vm["record_neuron_inh_ctx"].as<int>();
    }

    if (vm.count("record_neuron_exc_hpc")) {
      std::cout << "record_neuron_exc_hpc set to " 
		<< vm["record_neuron_exc_hpc"].as<int>() << ".\n";
      record_neuron_exc_hpc = vm["record_neuron_exc_hpc"].as<int>();
    }

    if (vm.count("record_neuron_inh_hpc")) {
      std::cout << "record_neuron_inh_hpc set to " 
		<< vm["record_neuron_inh_hpc"].as<int>() << ".\n";
      record_neuron_inh_hpc = vm["record_neuron_inh_hpc"].as<int>();
    }

    if (vm.count("record_neuron_exc_thl")) {
      std::cout << "record_neuron_exc_thl set to " 
		<< vm["record_neuron_exc_thl"].as<int>() << ".\n";
      record_neuron_exc_thl = vm["record_neuron_exc_thl"].as<int>();
    }

    if (vm.count("record_neuron_inh_thl")) {
      std::cout << "record_neuron_inh_thl set to " 
		<< vm["record_neuron_inh_thl"].as<int>() << ".\n";
      record_neuron_inh_thl = vm["record_neuron_inh_thl"].as<int>();
    }

    if (vm.count("monfile_ctx")) {
      std::cout << "monfile_ctx set to " 
		<< vm["monfile_ctx"].as<string>() << ".\n";
      monfile_ctx = vm["monfile_ctx"].as<string>();
    } 

    if (vm.count("premonfile_ctx")) {
      std::cout << "premonfile_ctx set to " 
		<< vm["premonfile_ctx"].as<string>() << ".\n";
      premonfile_ctx = vm["premonfile_ctx"].as<string>();
    }

    if (vm.count("monfile_hpc")) {
      std::cout << "monfile_hpc set to " 
		<< vm["monfile_hpc"].as<string>() << ".\n";
      monfile_hpc = vm["monfile_hpc"].as<string>();
    } 

    if (vm.count("premonfile_hpc")) {
      std::cout << "premonfile_hpc set to " 
		<< vm["premonfile_hpc"].as<string>() << ".\n";
      premonfile_hpc = vm["premonfile_hpc"].as<string>();
    }

    if (vm.count("monfile_thl")) {
      std::cout << "monfile_thl set to " 
		<< vm["monfile_thl"].as<string>() << ".\n";
      monfile_thl = vm["monfile_thl"].as<string>();
    } 

    if (vm.count("premonfile_thl")) {
      std::cout << "premonfile_thl set to " 
		<< vm["premonfile_thl"].as<string>() << ".\n";
      premonfile_thl = vm["premonfile_thl"].as<string>();
    }

    if (vm.count("block_stim")) {
      std::cout << "block_stim " 
		<< vm["block_stim"].as<string>() << ".\n";
      block_stim = vm["block_stim"].as<string>();
    }

    if (vm.count("block_bg_thl")) {
      std::cout << "block_bg_thl " 
		<< vm["block_bg_thl"].as<string>() << ".\n";
      block_bg_thl = vm["block_bg_thl"].as<string>();
    }

    if (vm.count("block_bg_ctx")) {
      std::cout << "block_bg_ctx " 
		<< vm["block_bg_ctx"].as<string>() << ".\n";
      block_bg_ctx = vm["block_bg_ctx"].as<string>();
    }

    if (vm.count("block_bg_hpc")) {
      std::cout << "block_bg_hpc " 
		<< vm["block_bg_hpc"].as<string>() << ".\n";
      block_bg_hpc = vm["block_bg_hpc"].as<string>();
    }
    
    if (vm.count("block_rep")) {
      std::cout << "block_rep " 
		<< vm["block_rep"].as<string>() << ".\n";
      block_rep = vm["block_rep"].as<string>();
    }

    if (vm.count("block_exc_thl")) {
      std::cout << "block_exc_thl " 
		<< vm["block_exc_thl"].as<string>() << ".\n";
      block_exc_thl = vm["block_exc_thl"].as<string>();
    }

    if (vm.count("block_inh_thl")) {
      std::cout << "block_inh_thl " 
		<< vm["block_inh_thl"].as<string>() << ".\n";
      block_inh_thl = vm["block_inh_thl"].as<string>();
    }

    if (vm.count("block_exc_ctx")) {
      std::cout << "block_exc_ctx " 
		<< vm["block_exc_ctx"].as<string>() << ".\n";
      block_exc_ctx = vm["block_exc_ctx"].as<string>();
    }

    if (vm.count("block_inh_ctx")) {
      std::cout << "block_inh_ctx " 
		<< vm["block_inh_ctx"].as<string>() << ".\n";
      block_inh_ctx = vm["block_inh_ctx"].as<string>();
    }

    if (vm.count("block_exc_hpc")) {
      std::cout << "block_exc_hpc " 
		<< vm["block_exc_hpc"].as<string>() << ".\n";
      block_exc_hpc = vm["block_exc_hpc"].as<string>();
    }

    if (vm.count("block_inh_hpc")) {
      std::cout << "block_inh_hpc " 
		<< vm["block_inh_hpc"].as<string>() << ".\n";
      block_inh_hpc = vm["block_inh_hpc"].as<string>();
    }

    if (vm.count("inhibit_exc_hpc")) {
      std::cout << "inhibit_exc_hpc " 
		<< vm["inhibit_exc_hpc"].as<string>() << ".\n";
      inhibit_exc_hpc = vm["inhibit_exc_hpc"].as<string>();
    }
    
  }
  catch(std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    std::cerr << "Exception of unknown type!\n";
  }


  
  // start of auryn simulation
  auryn_init( ac, av, out_dir, binary, file_prefix );
  sys->set_master_seed(master_seed);
  logger->set_logfile_loglevel(VERBOSE);

  
  //log params

  // architecture
  #ifdef HAS_THL
   logger->parameter("has_thl", "true");
  #endif

  #ifndef HAS_THL
   logger->parameter("has_thl", "false");
  #endif
   
  #ifdef CON_STIM_THL_NONE
   logger->parameter("con_stim_thl", "none");
  #endif
   
  #ifdef CON_STIM_THL_SPARSEB
   logger->parameter("con_stim_thl", "sparseb");
  #endif

  #ifdef CON_STIM_THL_STPB
   logger->parameter("con_stim_thl", "stpb");
  #endif

  #ifdef CON_STIM_THL_P11B
   logger->parameter("con_stim_thl", "p11b");
  #endif
   
  #ifdef HAS_CTX
   logger->parameter("has_ctx", "true");
  #endif

  #ifndef HAS_CTX
   logger->parameter("has_ctx", "false");
  #endif
   
  #ifdef CON_STIM_CTX_NONE
   logger->parameter("con_stim_ctx", "none");
  #endif
   
  #ifdef CON_STIM_CTX_SPARSEB
   logger->parameter("con_stim_ctx", "sparseb");
  #endif

  #ifdef CON_STIM_CTX_STPB
   logger->parameter("con_stim_ctx", "stpb");
  #endif

  #ifdef CON_STIM_CTX_P11B
   logger->parameter("con_stim_ctx", "p11b");
  #endif

  #ifdef HAS_HPC
   logger->parameter("has_hpc", "true");
  #endif

  #ifndef HAS_HPC
   logger->parameter("has_hpc", "false");
  #endif

  #ifdef CON_STIM_HPC_NONE
   logger->parameter("con_stim_hpc", "none");
  #endif
   
  #ifdef CON_STIM_HPC_SPARSEB
   logger->parameter("con_stim_hpc", "sparseb");
  #endif

  #ifdef CON_STIM_HPC_STPB
   logger->parameter("con_stim_hpc", "stpb");
  #endif

  #ifdef CON_STIM_HPC_P11B
   logger->parameter("con_stim_hpc", "p11b");
  #endif

  #ifdef CON_THL_CTX_NONE
   logger->parameter("con_thl_ctx", "none");
  #endif
   
  #ifdef CON_THL_CTX_SPARSEB
   logger->parameter("con_thl_ctx", "sparseb");
  #endif

  #ifdef CON_THL_CTX_STPB
   logger->parameter("con_thl_ctx", "stpb");
  #endif

  #ifdef CON_THL_CTX_P11B
   logger->parameter("con_thl_ctx", "p11b");
  #endif

  #ifdef CON_CTX_THL_NONE
   logger->parameter("con_ctx_thl", "none");
  #endif
   
  #ifdef CON_CTX_THL_SPARSEB
   logger->parameter("con_ctx_thl", "sparseb");
  #endif

  #ifdef CON_CTX_THL_STPB
   logger->parameter("con_ctx_thl", "stpb");
  #endif

  #ifdef CON_CTX_THL_P11B
   logger->parameter("con_ctx_thl", "p11b");
  #endif

  #ifdef CON_THL_HPC_NONE
   logger->parameter("con_thl_hpc", "none");
  #endif
   
  #ifdef CON_THL_HPC_SPARSEB
   logger->parameter("con_thl_hpc", "sparseb");
  #endif

  #ifdef CON_THL_HPC_STPB
   logger->parameter("con_thl_hpc", "stpb");
  #endif

  #ifdef CON_THL_HPC_P11B
   logger->parameter("con_thl_hpc", "p11b");
  #endif

  #ifdef CON_HPC_THL_NONE
   logger->parameter("con_hpc_thl", "none");
  #endif
   
  #ifdef CON_HPC_THL_SPARSEB
   logger->parameter("con_hpc_thl", "sparseb");
  #endif

  #ifdef CON_HPC_THL_STPB
   logger->parameter("con_hpc_thl", "stpb");
  #endif

  #ifdef CON_HPC_THL_P11B
   logger->parameter("con_hpc_thl", "p11b");
  #endif

  #ifdef CON_CTX_HPC_NONE
   logger->parameter("con_ctx_hpc", "none");
  #endif
   
  #ifdef CON_CTX_HPC_SPARSEB
   logger->parameter("con_ctx_hpc", "sparseb");
  #endif

  #ifdef CON_CTX_HPC_STPB
   logger->parameter("con_ctx_hpc", "stpb");
  #endif

  #ifdef CON_CTX_HPC_P11B
   logger->parameter("con_ctx_hpc", "p11b");
  #endif

  #ifdef CON_HPC_CTX_NONE
   logger->parameter("con_hpc_ctx", "none");
  #endif
   
  #ifdef CON_HPC_CTX_SPARSEB
   logger->parameter("con_hpc_ctx", "sparseb");
  #endif

  #ifdef CON_HPC_CTX_STPB
   logger->parameter("con_hpc_ctx", "stpb");
  #endif

  #ifdef CON_HPC_CTX_P11B
   logger->parameter("con_hpc_ctx", "p11b");
  #endif

  #ifdef HAS_REPLAY
   logger->parameter("has_replay", "true");
  #endif

  #ifndef HAS_REPLAY
   logger->parameter("has_replay", "false");
  #endif

  #ifdef CON_REPLAY_THL_NONE
   logger->parameter("con_rep_thl", "none");
  #endif
   
  #ifdef CON_REPLAY_THL_SPARSEB
   logger->parameter("con_rep_thl", "sparseb");
  #endif

  #ifdef CON_REPLAY_CTX_NONE
   logger->parameter("con_rep_ctx", "none");
  #endif
   
  #ifdef CON_REPLAY_CTX_SPARSEB
   logger->parameter("con_rep_ctx", "sparseb");
  #endif

  #ifdef CON_REPLAY_HPC_NONE
   logger->parameter("con_rep_hpc", "none");
  #endif
   
  #ifdef CON_REPLAY_HPC_SPARSEB
   logger->parameter("con_rep_hpc", "sparseb");
  #endif

  #ifdef HAS_REC_EE
   logger->parameter("has_rec_ee", "true");
  #endif

  #ifndef HAS_REC_EE
   logger->parameter("has_rec_ee", "false");
  #endif

  #ifdef HAS_INH2
   logger->parameter("has_inh2", "true");
  #endif

  #ifndef HAS_INH2
   logger->parameter("has_inh2", "false");
  #endif

  // simulation
  logger->parameter("binary",binary);
  logger->parameter("out_dir",out_dir);
  logger->parameter("file_prefix",file_prefix);
  logger->parameter("file_prefix_hm",file_prefix_hm);
  logger->parameter("load_file",load_file);
  for (int index=0; index < simtimes.size(); ++index) {
    sprintf(strbuf, "simtime[%d]", index );
    string simtimestr = strbuf;
    logger->parameter(simtimestr,simtimes[index]);
  }

  // simulation flags
  logger->parameter("save",save);
  logger->parameter("save_without_hpc",save_without_hpc);
  logger->parameter("load_without_hpc",load_without_hpc);
  logger->parameter("chain",chain);
  logger->parameter("prime",prime);

  // initialization flags
  logger->parameter("noisy_initial_weights",noisy_initial_weights);
  logger->parameter("consolidate_initial_weights",consolidate_initial_weights);

  // plasticity flags
  logger->parameter("consolidation",consolidation);
  logger->parameter("isp_active",isp_active);

  logger->parameter("inh_input",inh_input);
  logger->parameter("quiet",quiet);

  // stim flags
  logger->parameter("stim_spike_mon",stim_spike_mon);

  // blocking flags
  logger->parameter("block_local",block_local);
  logger->parameter("block_cross_region",block_cross_region);

  // thl flags
  logger->parameter("weight_mon_ee_stim_thl",weight_mon_ee_stim_thl);
  logger->parameter("weightstat_mon_ee_stim_thl",weightstat_mon_ee_stim_thl);
  logger->parameter("weightpat_mon_ee_stim_thl",weightpat_mon_ee_stim_thl);
  logger->parameter("exc_spike_mon_thl",exc_spike_mon_thl);
  logger->parameter("exc_prate_mon_thl",exc_prate_mon_thl);
  logger->parameter("exc_pattern_mon_thl",exc_pattern_mon_thl);
  logger->parameter("exc_voltage_mon_thl",exc_voltage_mon_thl);
  logger->parameter("exc_g_ampa_mon_thl",exc_g_ampa_mon_thl);
  logger->parameter("exc_g_nmda_mon_thl",exc_g_nmda_mon_thl);
  logger->parameter("exc_g_gaba_mon_thl",exc_g_gaba_mon_thl);
  logger->parameter("exc_g_adapt1_mon_thl",exc_g_adapt1_mon_thl);
  logger->parameter("exc_g_adapt2_mon_thl",exc_g_adapt2_mon_thl);
  logger->parameter("exc_thr_mon_thl",exc_thr_mon_thl);
  logger->parameter("exc_ratechk_thl",exc_ratechk_thl);
  logger->parameter("inh_spike_mon_thl",inh_spike_mon_thl);
  logger->parameter("inh_prate_mon_thl",inh_prate_mon_thl);
  logger->parameter("inh_voltage_mon_thl",inh_voltage_mon_thl);
  logger->parameter("weight_mon_ee_thl",weight_mon_ee_thl);
  logger->parameter("weightstat_mon_ee_thl",weightstat_mon_ee_thl);
  logger->parameter("weightpat_mon_ee_thl",weightpat_mon_ee_thl);
  logger->parameter("ee_hom_mon_thl",ee_hom_mon_thl);
  logger->parameter("ei_weight_mon_thl",ei_weight_mon_thl);
  logger->parameter("ie_weight_mon_thl",ie_weight_mon_thl);
  logger->parameter("ie_weightstat_mon_thl",ie_weightstat_mon_thl);
  logger->parameter("ii_weight_mon_thl",ii_weight_mon_thl);

  // ctx flags
  logger->parameter("weight_mon_ee_stim_ctx",weight_mon_ee_stim_ctx);
  logger->parameter("weightstat_mon_ee_stim_ctx",weightstat_mon_ee_stim_ctx);
  logger->parameter("weightpat_mon_ee_stim_ctx",weightpat_mon_ee_stim_ctx);
  logger->parameter("exc_spike_mon_ctx",exc_spike_mon_ctx);
  logger->parameter("exc_prate_mon_ctx",exc_prate_mon_ctx);
  logger->parameter("exc_pattern_mon_ctx",exc_pattern_mon_ctx);
  logger->parameter("exc_voltage_mon_ctx",exc_voltage_mon_ctx);
  logger->parameter("exc_g_ampa_mon_ctx",exc_g_ampa_mon_ctx);
  logger->parameter("exc_g_nmda_mon_ctx",exc_g_nmda_mon_ctx);
  logger->parameter("exc_g_gaba_mon_ctx",exc_g_gaba_mon_ctx);
  logger->parameter("exc_g_adapt1_mon_ctx",exc_g_adapt1_mon_ctx);
  logger->parameter("exc_g_adapt2_mon_ctx",exc_g_adapt2_mon_ctx);
  logger->parameter("exc_thr_mon_ctx",exc_thr_mon_ctx);
  logger->parameter("exc_ratechk_ctx",exc_ratechk_ctx);
  logger->parameter("inh_spike_mon_ctx",inh_spike_mon_ctx);
  logger->parameter("inh_prate_mon_ctx",inh_prate_mon_ctx);
  logger->parameter("inh_voltage_mon_ctx",inh_voltage_mon_ctx);
  logger->parameter("weight_mon_ee_ctx",weight_mon_ee_ctx);
  logger->parameter("weightstat_mon_ee_ctx",weightstat_mon_ee_ctx);
  logger->parameter("weightpat_mon_ee_ctx",weightpat_mon_ee_ctx);
  logger->parameter("ee_hom_mon_ctx",ee_hom_mon_ctx);
  logger->parameter("ei_weight_mon_ctx",ei_weight_mon_ctx);
  logger->parameter("ie_weight_mon_ctx",ie_weight_mon_ctx);
  logger->parameter("ie_weightstat_mon_ctx",ie_weightstat_mon_ctx);
  logger->parameter("ii_weight_mon_ctx",ii_weight_mon_ctx);

  // hpc flags
  logger->parameter("weight_mon_ee_stim_hpc",weight_mon_ee_stim_hpc);
  logger->parameter("weightstat_mon_ee_stim_hpc",weightstat_mon_ee_stim_hpc);
  logger->parameter("weightpat_mon_ee_stim_hpc",weightpat_mon_ee_stim_hpc);
  logger->parameter("exc_spike_mon_hpc",exc_spike_mon_hpc);
  logger->parameter("exc_prate_mon_hpc",exc_prate_mon_hpc);
  logger->parameter("exc_pattern_mon_hpc",exc_pattern_mon_hpc);
  logger->parameter("exc_voltage_mon_hpc",exc_voltage_mon_hpc);
  logger->parameter("exc_g_ampa_mon_hpc",exc_g_ampa_mon_hpc);
  logger->parameter("exc_g_nmda_mon_hpc",exc_g_nmda_mon_hpc);
  logger->parameter("exc_g_gaba_mon_hpc",exc_g_gaba_mon_hpc);
  logger->parameter("exc_g_adapt1_mon_hpc",exc_g_adapt1_mon_hpc);
  logger->parameter("exc_g_adapt2_mon_hpc",exc_g_adapt2_mon_hpc);
  logger->parameter("exc_thr_mon_hpc",exc_thr_mon_hpc);
  logger->parameter("exc_ratechk_hpc",exc_ratechk_hpc);
  logger->parameter("inh_spike_mon_hpc",inh_spike_mon_hpc);
  logger->parameter("inh_prate_mon_hpc",inh_prate_mon_hpc);
  logger->parameter("inh_voltage_mon_hpc",inh_voltage_mon_hpc);
  logger->parameter("weight_mon_ee_hpc",weight_mon_ee_hpc);
  logger->parameter("weightstat_mon_ee_hpc",weightstat_mon_ee_hpc);
  logger->parameter("weightpat_mon_ee_hpc",weightpat_mon_ee_hpc);
  logger->parameter("ee_hom_mon_hpc",ee_hom_mon_hpc);
  logger->parameter("ei_weight_mon_hpc",ei_weight_mon_hpc);
  logger->parameter("ie_weight_mon_hpc",ie_weight_mon_hpc);
  logger->parameter("ie_weightstat_mon_hpc",ie_weightstat_mon_hpc);
  logger->parameter("ii_weight_mon_hpc",ii_weight_mon_hpc);

  // background flags
  logger->parameter("bg_spike_mon",bg_spike_mon);

  // replay flags
  logger->parameter("rep_spike_mon",rep_spike_mon);

  // monitors
  logger->parameter("record_neuron_exc_ctx",record_neuron_exc_ctx);
  logger->parameter("record_neuron_inh_ctx",record_neuron_inh_ctx);
  logger->parameter("record_neuron_exc_hpc",record_neuron_exc_hpc);
  logger->parameter("record_neuron_inh_hpc",record_neuron_inh_hpc);
  logger->parameter("record_neuron_exc_thl",record_neuron_exc_thl);
  logger->parameter("record_neuron_inh_thl",record_neuron_inh_thl);
  logger->parameter("monfile_ctx",monfile_ctx);
  logger->parameter("premonfile_ctx",premonfile_ctx);
  logger->parameter("monfile_hpc",monfile_hpc);
  logger->parameter("premonfile_hpc",premonfile_hpc);
  logger->parameter("monfile_thl",monfile_thl);
  logger->parameter("premonfile_thl",premonfile_thl);

  // pseudo-random number generation
  logger->parameter("stim_seed",stim_seed);
  logger->parameter("master_seed",master_seed);
  logger->parameter("rep_seed",rep_seed);
  logger->parameter("bg_thl_seed",bg_thl_seed);
  logger->parameter("bg_ctx_seed",bg_ctx_seed);
  logger->parameter("bg_hpc_seed",bg_hpc_seed);

  // stimulus
  logger->parameter("ontime",ontime);
  logger->parameter("offtime",offtime);
  logger->parameter("scale",scale);
  logger->parameter("bgrate",bgrate);
  logger->parameter("preferred",preferred);
  logger->parameter("stimfile",stimfile);
  logger->parameter("recfile_stim_ctx",recfile_stim_ctx);
  logger->parameter("xi_stim_ctx",xi_stim_ctx);
  logger->parameter("recfile_stim_hpc",recfile_stim_hpc);
  logger->parameter("xi_stim_hpc",xi_stim_hpc);
  logger->parameter("recfile_stim_thl",recfile_stim_thl);
  logger->parameter("xi_stim_thl",xi_stim_thl);
  logger->parameter("prime_ontime",prime_ontime);
  logger->parameter("prime_offtime",prime_offtime);
  logger->parameter("prime_duration",prime_duration);

  // background input
  logger->parameter("bgrate_thl",bgrate_thl);
  logger->parameter("bgfile_thl",bgfile_thl);
  logger->parameter("w_bg_thl",w_bg_thl);
  logger->parameter("w_bg_ei_thl",w_bg_ei_thl);
  logger->parameter("sparseness_bg_thl",sparseness_bg_thl);
  logger->parameter("recfile_bg_thl",recfile_bg_thl);
  logger->parameter("xi_bg_thl",xi_bg_thl);
  logger->parameter("recfile_ei_bg_thl",recfile_ei_bg_thl);
  logger->parameter("xi_ei_bg_thl",xi_ei_bg_thl);
  
  logger->parameter("bgrate_ctx",bgrate_ctx);
  logger->parameter("bgfile_ctx",bgfile_ctx);
  logger->parameter("w_bg_ctx",w_bg_ctx);
  logger->parameter("w_bg_ei_ctx",w_bg_ei_ctx);
  logger->parameter("sparseness_bg_ctx",sparseness_bg_ctx);
  logger->parameter("recfile_bg_ctx",recfile_bg_ctx);
  logger->parameter("xi_bg_ctx",xi_bg_ctx);
  logger->parameter("recfile_ei_bg_ctx",recfile_ei_bg_ctx);
  logger->parameter("xi_ei_bg_ctx",xi_ei_bg_ctx);
  
  logger->parameter("bgrate_hpc",bgrate_hpc);
  logger->parameter("bgfile_hpc",bgfile_hpc);
  logger->parameter("w_bg_hpc",w_bg_hpc);
  logger->parameter("w_bg_ei_hpc",w_bg_ei_hpc);
  logger->parameter("sparseness_bg_hpc",sparseness_bg_hpc);
  logger->parameter("recfile_bg_hpc",recfile_bg_hpc);
  logger->parameter("xi_bg_hpc",xi_bg_hpc);
  logger->parameter("recfile_ei_bg_hpc",recfile_ei_bg_hpc);
  logger->parameter("xi_ei_bg_hpc",xi_ei_bg_hpc);

  // replay
  logger->parameter("size_rep",size_rep);
  logger->parameter("repfile",repfile);
  logger->parameter("w_rep_thl",w_rep_thl);
  logger->parameter("sparseness_rep_thl",sparseness_rep_thl);
  logger->parameter("recfile_rep_thl",recfile_rep_thl);
  logger->parameter("xi_rep_thl",xi_rep_thl);
  logger->parameter("recfile_ei_rep_thl",recfile_ei_rep_thl);
  logger->parameter("xi_ei_rep_thl",xi_ei_rep_thl);
  logger->parameter("w_rep_ctx",w_rep_ctx);
  logger->parameter("sparseness_rep_ctx",sparseness_rep_ctx);
  logger->parameter("recfile_rep_ctx",recfile_rep_ctx);
  logger->parameter("xi_rep_ctx",xi_rep_ctx);
  logger->parameter("recfile_ei_rep_ctx",recfile_ei_rep_ctx);
  logger->parameter("xi_ei_rep_ctx",xi_ei_rep_ctx);
  logger->parameter("w_rep_hpc",w_rep_hpc);
  logger->parameter("sparseness_rep_hpc",sparseness_rep_hpc);
  logger->parameter("recfile_rep_hpc",recfile_rep_hpc);
  logger->parameter("xi_rep_hpc",xi_rep_hpc);
  logger->parameter("recfile_ei_rep_hpc",recfile_ei_rep_hpc);
  logger->parameter("xi_ei_rep_hpc",xi_ei_rep_hpc);
  logger->parameter("bgrate_rep",bgrate_rep);

  // thl network
  logger->parameter("exc_size_thl",exc_size_thl);
  logger->parameter("eta_thl",eta_thl);
  logger->parameter("eta_stim_thl",eta_stim_thl);
  logger->parameter("eta_exc_inh_thl",eta_exc_inh_thl);
  logger->parameter("exc_inh_thl",exc_inh_thl);
  logger->parameter("alpha_thl",alpha_thl);
  logger->parameter("kappa_thl",kappa_thl);
  logger->parameter("tauf_ee_thl",tauf_ee_thl);
  logger->parameter("taud_ee_thl",taud_ee_thl);
  logger->parameter("tauh_ee_thl",tauh_ee_thl);
  logger->parameter("tauc_ee_thl",tauc_ee_thl);
  logger->parameter("ujump_ee_thl",ujump_ee_thl);
  logger->parameter("tauf_ee_stim_thl",tauf_ee_stim_thl);
  logger->parameter("taud_ee_stim_thl",taud_ee_stim_thl);
  logger->parameter("tauh_ee_stim_thl",tauh_ee_stim_thl);
  logger->parameter("tauc_ee_stim_thl",tauc_ee_stim_thl);
  logger->parameter("ujump_ee_stim_thl",ujump_ee_stim_thl);
  logger->parameter("beta_thl",beta_thl);
  logger->parameter("beta_stim_thl",beta_stim_thl);
  logger->parameter("delta_thl",delta_thl);
  logger->parameter("weight_a_thl",weight_a_thl);
  logger->parameter("weight_c_thl",weight_c_thl);
  logger->parameter("adapt1_thl",adapt1_thl);
  logger->parameter("adapt2_thl",adapt2_thl);
  logger->parameter("pot_strength_thl",pot_strength_thl);
  logger->parameter("wmax_exc_thl",wmax_exc_thl);
  logger->parameter("wmin_exc_thl",wmin_exc_thl);
  logger->parameter("wmax_inh_thl",wmax_inh_thl);
  logger->parameter("wmin_inh_thl",wmin_inh_thl);
  logger->parameter("wee_thl",wee_thl);
  logger->parameter("wei_thl",wei_thl);
  logger->parameter("wie_thl",wie_thl);
  logger->parameter("wii_thl",wii_thl);
  logger->parameter("wext_thl",wext_thl);
  logger->parameter("wext_ei_thl",wext_ei_thl);
  logger->parameter("sparseness_int_ee_thl",sparseness_int_ee_thl);
  logger->parameter("sparseness_int_ei_thl",sparseness_int_ei_thl);
  logger->parameter("sparseness_int_ii_thl",sparseness_int_ii_thl);
  logger->parameter("sparseness_int_ie_thl",sparseness_int_ie_thl);
  logger->parameter("sparseness_ext_thl",sparseness_ext_thl);
  logger->parameter("tauf_ei_thl",tauf_ei_thl);
  logger->parameter("taud_ei_thl",taud_ei_thl);
  logger->parameter("ujump_ei_thl",ujump_ei_thl);
  logger->parameter("tauh_ie_thl",tauh_ie_thl);
  logger->parameter("taud_ei_stim_thl",taud_ei_stim_thl);
  logger->parameter("tauf_ei_stim_thl",tauf_ei_stim_thl);
  logger->parameter("ujump_ei_stim_thl",ujump_ei_stim_thl);
  logger->parameter("tau_ampa_e_thl",tau_ampa_e_thl);
  logger->parameter("tau_gaba_e_thl",tau_gaba_e_thl);
  logger->parameter("tau_nmda_e_thl",tau_nmda_e_thl);
  logger->parameter("ampa_nmda_e_thl",ampa_nmda_e_thl);
  logger->parameter("tau_ampa_i_thl",tau_ampa_i_thl);
  logger->parameter("tau_gaba_i_thl",tau_gaba_i_thl);
  logger->parameter("tau_nmda_i_thl",tau_nmda_i_thl);
  logger->parameter("ampa_nmda_i_thl",ampa_nmda_i_thl);
  logger->parameter("prefile_thl",prefile_thl);
  logger->parameter("chi_thl",chi_thl);

  // ctx network
  logger->parameter("exc_size_ctx",exc_size_ctx);
  logger->parameter("eta_ctx",eta_ctx);
  logger->parameter("eta_stim_ctx",eta_stim_ctx);
  logger->parameter("eta_exc_inh_ctx",eta_exc_inh_ctx);
  logger->parameter("exc_inh_ctx",exc_inh_ctx);
  logger->parameter("alpha_ctx",alpha_ctx);
  logger->parameter("kappa_ctx",kappa_ctx);
  logger->parameter("tauf_ee_ctx",tauf_ee_ctx);
  logger->parameter("taud_ee_ctx",taud_ee_ctx);
  logger->parameter("tauh_ee_ctx",tauh_ee_ctx);
  logger->parameter("tauc_ee_ctx",tauc_ee_ctx);
  logger->parameter("ujump_ee_ctx",ujump_ee_ctx);
  logger->parameter("tauf_ee_stim_ctx",tauf_ee_stim_ctx);
  logger->parameter("taud_ee_stim_ctx",taud_ee_stim_ctx);
  logger->parameter("tauh_ee_stim_ctx",tauh_ee_stim_ctx);
  logger->parameter("tauc_ee_stim_ctx",tauc_ee_stim_ctx);
  logger->parameter("ujump_ee_stim_ctx",ujump_ee_stim_ctx);
  logger->parameter("beta_ctx",beta_ctx);
  logger->parameter("beta_stim_ctx",beta_stim_ctx);
  logger->parameter("delta_ctx",delta_ctx);
  logger->parameter("weight_a_ctx",weight_a_ctx);
  logger->parameter("weight_c_ctx",weight_c_ctx);
  logger->parameter("adapt1_ctx",adapt1_ctx);
  logger->parameter("adapt2_ctx",adapt2_ctx);
  logger->parameter("pot_strength_ctx",pot_strength_ctx);
  logger->parameter("wmax_exc_ctx",wmax_exc_ctx);
  logger->parameter("wmin_exc_ctx",wmin_exc_ctx);
  logger->parameter("wmax_inh_ctx",wmax_inh_ctx);
  logger->parameter("wmin_inh_ctx",wmin_inh_ctx);
  logger->parameter("wee_ctx",wee_ctx);
  logger->parameter("wei_ctx",wei_ctx);
  logger->parameter("wie_ctx",wie_ctx);
  logger->parameter("wii_ctx",wii_ctx);
  logger->parameter("wext_ctx",wext_ctx);
  logger->parameter("wext_ei_ctx",wext_ei_ctx);
  logger->parameter("sparseness_int_ee_ctx",sparseness_int_ee_ctx);
  logger->parameter("sparseness_int_ei_ctx",sparseness_int_ei_ctx);
  logger->parameter("sparseness_int_ii_ctx",sparseness_int_ii_ctx);
  logger->parameter("sparseness_int_ie_ctx",sparseness_int_ie_ctx);
  logger->parameter("sparseness_ext_ctx",sparseness_ext_ctx);
  logger->parameter("tauf_ei_ctx",tauf_ei_ctx);
  logger->parameter("taud_ei_ctx",taud_ei_ctx);
  logger->parameter("ujump_ei_ctx",ujump_ei_ctx);
  logger->parameter("tauh_ie_ctx",tauh_ie_ctx);
  logger->parameter("taud_ei_stim_ctx",taud_ei_stim_ctx);
  logger->parameter("tauf_ei_stim_ctx",tauf_ei_stim_ctx);
  logger->parameter("ujump_ei_stim_ctx",ujump_ei_stim_ctx);
  logger->parameter("tau_ampa_e_ctx",tau_ampa_e_ctx);
  logger->parameter("tau_gaba_e_ctx",tau_gaba_e_ctx);
  logger->parameter("tau_nmda_e_ctx",tau_nmda_e_ctx);
  logger->parameter("ampa_nmda_e_ctx",ampa_nmda_e_ctx);
  logger->parameter("tau_ampa_i_ctx",tau_ampa_i_ctx);
  logger->parameter("tau_gaba_i_ctx",tau_gaba_i_ctx);
  logger->parameter("tau_nmda_i_ctx",tau_nmda_i_ctx);
  logger->parameter("ampa_nmda_i_ctx",ampa_nmda_i_ctx);
  logger->parameter("prefile_ctx",prefile_ctx);
  logger->parameter("chi_ctx",chi_ctx);

  // hpc network
  logger->parameter("exc_size_hpc",exc_size_hpc);
  logger->parameter("eta_hpc",eta_hpc);
  logger->parameter("eta_stim_hpc",eta_stim_hpc);
  logger->parameter("eta_exc_inh_hpc",eta_exc_inh_hpc);
  logger->parameter("exc_inh_hpc",exc_inh_hpc);
  logger->parameter("alpha_hpc",alpha_hpc);
  logger->parameter("kappa_hpc",kappa_hpc);
  logger->parameter("tauf_ee_hpc",tauf_ee_hpc);
  logger->parameter("taud_ee_hpc",taud_ee_hpc);
  logger->parameter("tauh_ee_hpc",tauh_ee_hpc);
  logger->parameter("tauc_ee_hpc",tauc_ee_hpc);
  logger->parameter("ujump_ee_hpc",ujump_ee_hpc);
  logger->parameter("tauf_ee_stim_hpc",tauf_ee_stim_hpc);
  logger->parameter("taud_ee_stim_hpc",taud_ee_stim_hpc);
  logger->parameter("tauh_ee_stim_hpc",tauh_ee_stim_hpc);
  logger->parameter("tauc_ee_stim_hpc",tauc_ee_stim_hpc);
  logger->parameter("ujump_ee_stim_hpc",ujump_ee_stim_hpc);
  logger->parameter("beta_hpc",beta_hpc);
  logger->parameter("beta_stim_hpc",beta_stim_hpc);
  logger->parameter("delta_hpc",delta_hpc);
  logger->parameter("weight_a_hpc",weight_a_hpc);
  logger->parameter("weight_c_hpc",weight_c_hpc);
  logger->parameter("adapt1_hpc",adapt1_hpc);
  logger->parameter("adapt2_hpc",adapt2_hpc);
  logger->parameter("pot_strength_hpc",pot_strength_hpc);
  logger->parameter("wmax_exc_hpc",wmax_exc_hpc);
  logger->parameter("wmin_exc_hpc",wmin_exc_hpc);
  logger->parameter("wmax_inh_hpc",wmax_inh_hpc);
  logger->parameter("wmin_inh_hpc",wmin_inh_hpc);
  logger->parameter("wee_hpc",wee_hpc);
  logger->parameter("wei_hpc",wei_hpc);
  logger->parameter("wie_hpc",wie_hpc);
  logger->parameter("wii_hpc",wii_hpc);
  logger->parameter("wext_hpc",wext_hpc);
  logger->parameter("wext_ei_hpc",wext_ei_hpc);
  logger->parameter("sparseness_int_ee_hpc",sparseness_int_ee_hpc);
  logger->parameter("sparseness_int_ei_hpc",sparseness_int_ei_hpc);
  logger->parameter("sparseness_int_ii_hpc",sparseness_int_ii_hpc);
  logger->parameter("sparseness_int_ie_hpc",sparseness_int_ie_hpc);
  logger->parameter("sparseness_ext_hpc",sparseness_ext_hpc);
  logger->parameter("tauf_ei_hpc",tauf_ei_hpc);
  logger->parameter("taud_ei_hpc",taud_ei_hpc);
  logger->parameter("ujump_ei_hpc",ujump_ei_hpc);
  logger->parameter("tauh_ie_hpc",tauh_ie_hpc);
  logger->parameter("taud_ei_stim_hpc",taud_ei_stim_hpc);
  logger->parameter("tauf_ei_stim_hpc",tauf_ei_stim_hpc);
  logger->parameter("ujump_ei_stim_hpc",ujump_ei_stim_hpc);
  logger->parameter("tau_ampa_e_hpc",tau_ampa_e_hpc);
  logger->parameter("tau_gaba_e_hpc",tau_gaba_e_hpc);
  logger->parameter("tau_nmda_e_hpc",tau_nmda_e_hpc);
  logger->parameter("ampa_nmda_e_hpc",ampa_nmda_e_hpc);
  logger->parameter("tau_ampa_i_hpc",tau_ampa_i_hpc);
  logger->parameter("tau_gaba_i_hpc",tau_gaba_i_hpc);
  logger->parameter("tau_nmda_i_hpc",tau_nmda_i_hpc);
  logger->parameter("ampa_nmda_i_hpc",ampa_nmda_i_hpc);
  logger->parameter("prefile_hpc",prefile_hpc);
  logger->parameter("chi_hpc",chi_hpc);
  // hpc inh2
  logger->parameter("eta_exc_inh2_hpc",eta_exc_inh2_hpc);
  logger->parameter("exc_inh2_hpc",exc_inh2_hpc);
  logger->parameter("alpha2_hpc",alpha2_hpc);
  logger->parameter("tau_ampa_i2_hpc",tau_ampa_i2_hpc);
  logger->parameter("tau_gaba_i2_hpc",tau_gaba_i2_hpc);
  logger->parameter("tau_nmda_i2_hpc",tau_nmda_i2_hpc);
  logger->parameter("ampa_nmda_i2_hpc",ampa_nmda_i2_hpc);
  logger->parameter("wei2_hpc",wei2_hpc);
  logger->parameter("wi2e_hpc",wi2e_hpc);
  logger->parameter("wi2i2_hpc",wi2i2_hpc);
  logger->parameter("wii2_hpc",wii2_hpc);
  logger->parameter("wi2i_hpc",wi2i_hpc);
  logger->parameter("sparseness_int_ei2_hpc",sparseness_int_ei2_hpc);
  logger->parameter("sparseness_int_i2e_hpc",sparseness_int_i2e_hpc);
  logger->parameter("sparseness_int_i2i2_hpc",sparseness_int_i2i2_hpc);
  logger->parameter("sparseness_int_ii2_hpc",sparseness_int_ii2_hpc);
  logger->parameter("sparseness_int_i2i_hpc",sparseness_int_i2i_hpc);
  logger->parameter("tauf_ei2_hpc",tauf_ei2_hpc);
  logger->parameter("taud_ei2_hpc",taud_ei2_hpc);
  logger->parameter("ujump_ei2_hpc",ujump_ei2_hpc);
  logger->parameter("tauh_i2e_hpc",tauh_i2e_hpc);
  logger->parameter("wmax_inh2_hpc",wmax_inh2_hpc);
  logger->parameter("wmin_inh2_hpc",wmin_inh2_hpc);
  logger->parameter("record_neuron_inh2_hpc",record_neuron_inh2_hpc);
  logger->parameter("block_inh2_hpc",block_inh2_hpc);
  

  // thl->ctx connection
  // used in case thl->ctx is: SPARSEB, STPB, P11B
  logger->parameter("wee_thl_ctx",wee_thl_ctx);
  logger->parameter("wei_thl_ctx",wei_thl_ctx);
  logger->parameter("sparseness_thl_ctx",sparseness_thl_ctx);
  logger->parameter("recfile_thl_ctx",recfile_thl_ctx);
  logger->parameter("xi_thl_ctx",xi_thl_ctx);
  // used in case ctx->thl is: P11B
  logger->parameter("eta_thl_ctx",eta_thl_ctx);
  logger->parameter("kappa_thl_ctx",kappa_thl_ctx);
  logger->parameter("delta_thl_ctx",delta_thl_ctx);
  logger->parameter("tauf_thl_ctx",tauf_thl_ctx);
  logger->parameter("taud_thl_ctx",taud_thl_ctx);
  logger->parameter("tauh_thl_ctx",tauh_thl_ctx);
  logger->parameter("tauc_thl_ctx",tauc_thl_ctx);
  logger->parameter("ujump_thl_ctx",ujump_thl_ctx);
  logger->parameter("beta_thl_ctx",beta_thl_ctx);
  logger->parameter("weight_a_thl_ctx",weight_a_thl_ctx);
  logger->parameter("weight_c_thl_ctx",weight_c_thl_ctx);
  logger->parameter("pot_strength_thl_ctx",pot_strength_thl_ctx);
  logger->parameter("wmax_thl_ctx",wmax_thl_ctx);
  logger->parameter("wmin_thl_ctx",wmin_thl_ctx);

  // ctx->thl connection
  // used in case ctx->thl is: SPARSEB, STPB, P11B
  logger->parameter("wee_ctx_thl",wee_ctx_thl);
  logger->parameter("wei_ctx_thl",wei_ctx_thl);
  logger->parameter("sparseness_ctx_thl",sparseness_ctx_thl);
  logger->parameter("recfile_ctx_thl",recfile_ctx_thl);
  logger->parameter("xi_ctx_thl",xi_ctx_thl);
  // used in case ctx->thl is: P11B
  logger->parameter("eta_ctx_thl",eta_ctx_thl);
  logger->parameter("kappa_ctx_thl",kappa_ctx_thl);
  logger->parameter("tauf_ctx_thl",tauf_ctx_thl);
  logger->parameter("taud_ctx_thl",taud_ctx_thl);
  logger->parameter("tauh_ctx_thl",tauh_ctx_thl);
  logger->parameter("tauc_ctx_thl",tauc_ctx_thl);
  logger->parameter("ujump_ctx_thl",ujump_ctx_thl);
  logger->parameter("beta_ctx_thl",beta_ctx_thl);
  logger->parameter("delta_ctx_thl",delta_ctx_thl);
  logger->parameter("weight_a_ctx_thl",weight_a_ctx_thl);
  logger->parameter("weight_c_ctx_thl",weight_c_ctx_thl);
  logger->parameter("pot_strength_ctx_thl",pot_strength_ctx_thl);
  logger->parameter("wmax_ctx_thl",wmax_ctx_thl);
  logger->parameter("wmin_ctx_thl",wmin_ctx_thl);

  // thl->hpc connection
  // used in case thl->hpc is: SPARSEB, STPB, P11B
  logger->parameter("wee_thl_hpc",wee_thl_hpc);
  logger->parameter("wei_thl_hpc",wei_thl_hpc);
  logger->parameter("sparseness_thl_hpc",sparseness_thl_hpc);
  logger->parameter("recfile_thl_hpc",recfile_thl_hpc);
  logger->parameter("xi_thl_hpc",xi_thl_hpc);
  // used in case hpc->thl is: P11B
  logger->parameter("eta_thl_hpc",eta_thl_hpc);
  logger->parameter("kappa_thl_hpc",kappa_thl_hpc);
  logger->parameter("delta_thl_hpc",delta_thl_hpc);
  logger->parameter("tauf_thl_hpc",tauf_thl_hpc);
  logger->parameter("taud_thl_hpc",taud_thl_hpc);
  logger->parameter("tauh_thl_hpc",tauh_thl_hpc);
  logger->parameter("tauc_thl_hpc",tauc_thl_hpc);
  logger->parameter("ujump_thl_hpc",ujump_thl_hpc);
  logger->parameter("beta_thl_hpc",beta_thl_hpc);
  logger->parameter("weight_a_thl_hpc",weight_a_thl_hpc);
  logger->parameter("weight_c_thl_hpc",weight_c_thl_hpc);
  logger->parameter("pot_strength_thl_hpc",pot_strength_thl_hpc);
  logger->parameter("wmax_thl_hpc",wmax_thl_hpc);
  logger->parameter("wmin_thl_hpc",wmin_thl_hpc);

  // hpc->thl connection
  // used in case hpc->thl is: SPARSEB, STPB, P11B
  logger->parameter("wee_hpc_thl",wee_hpc_thl);
  logger->parameter("wei_hpc_thl",wei_hpc_thl);
  logger->parameter("sparseness_hpc_thl",sparseness_hpc_thl);
  logger->parameter("recfile_hpc_thl",recfile_hpc_thl);
  logger->parameter("xi_hpc_thl",xi_hpc_thl);
  // used in case hpc->thl is: P11B
  logger->parameter("eta_hpc_thl",eta_hpc_thl);
  logger->parameter("kappa_hpc_thl",kappa_hpc_thl);
  logger->parameter("tauf_hpc_thl",tauf_hpc_thl);
  logger->parameter("taud_hpc_thl",taud_hpc_thl);
  logger->parameter("tauh_hpc_thl",tauh_hpc_thl);
  logger->parameter("tauc_hpc_thl",tauc_hpc_thl);
  logger->parameter("ujump_hpc_thl",ujump_hpc_thl);
  logger->parameter("beta_hpc_thl",beta_hpc_thl);
  logger->parameter("delta_hpc_thl",delta_hpc_thl);
  logger->parameter("weight_a_hpc_thl",weight_a_hpc_thl);
  logger->parameter("weight_c_hpc_thl",weight_c_hpc_thl);
  logger->parameter("pot_strength_hpc_thl",pot_strength_hpc_thl);
  logger->parameter("wmax_hpc_thl",wmax_hpc_thl);
  logger->parameter("wmin_hpc_thl",wmin_hpc_thl);

  // ctx->hpc connection
  // used in case ctx->hpc is: SPARSEB, STPB, P11B
  logger->parameter("wee_ctx_hpc",wee_ctx_hpc);
  logger->parameter("wei_ctx_hpc",wei_ctx_hpc);
  logger->parameter("sparseness_ctx_hpc",sparseness_ctx_hpc);
  logger->parameter("recfile_ctx_hpc",recfile_ctx_hpc);
  logger->parameter("xi_ctx_hpc",xi_ctx_hpc);
  // used in case ctx->hpc is: P11B
  logger->parameter("eta_ctx_hpc",eta_ctx_hpc);
  logger->parameter("kappa_ctx_hpc",kappa_ctx_hpc);
  logger->parameter("tauf_ctx_hpc",tauf_ctx_hpc);
  logger->parameter("taud_ctx_hpc",taud_ctx_hpc);
  logger->parameter("tauh_ctx_hpc",tauh_ctx_hpc);
  logger->parameter("tauc_ctx_hpc",tauc_ctx_hpc);
  logger->parameter("ujump_ctx_hpc",ujump_ctx_hpc);
  logger->parameter("beta_ctx_hpc",beta_ctx_hpc);
  logger->parameter("delta_ctx_hpc",delta_ctx_hpc);
  logger->parameter("weight_a_ctx_hpc",weight_a_ctx_hpc);
  logger->parameter("weight_c_ctx_hpc",weight_c_ctx_hpc);
  logger->parameter("pot_strength_ctx_hpc",pot_strength_ctx_hpc);
  logger->parameter("wmax_ctx_hpc",wmax_ctx_hpc);
  logger->parameter("wmin_ctx_hpc",wmin_ctx_hpc);

  // hpc->ctx connection
  // used in case hpc->ctx is: SPARSEB, STPB, P11B
  logger->parameter("wee_hpc_ctx",wee_hpc_ctx);
  logger->parameter("wei_hpc_ctx",wei_hpc_ctx);
  logger->parameter("sparseness_hpc_ctx",sparseness_hpc_ctx);
  logger->parameter("recfile_hpc_ctx",recfile_hpc_ctx);
  logger->parameter("xi_hpc_ctx",xi_hpc_ctx);
  // used in case ctx->hpc is: P11B
  logger->parameter("eta_hpc_ctx",eta_hpc_ctx);
  logger->parameter("kappa_hpc_ctx",kappa_hpc_ctx);
  logger->parameter("delta_hpc_ctx",delta_hpc_ctx);
  logger->parameter("tauf_hpc_ctx",tauf_hpc_ctx);
  logger->parameter("taud_hpc_ctx",taud_hpc_ctx);
  logger->parameter("tauh_hpc_ctx",tauh_hpc_ctx);
  logger->parameter("tauc_hpc_ctx",tauc_hpc_ctx);
  logger->parameter("ujump_hpc_ctx",ujump_hpc_ctx);
  logger->parameter("beta_hpc_ctx",beta_hpc_ctx);
  logger->parameter("weight_a_hpc_ctx",weight_a_hpc_ctx);
  logger->parameter("weight_c_hpc_ctx",weight_c_hpc_ctx);
  logger->parameter("pot_strength_hpc_ctx",pot_strength_hpc_ctx);
  logger->parameter("wmax_hpc_ctx",wmax_hpc_ctx);
  logger->parameter("wmin_hpc_ctx",wmin_hpc_ctx);

  // block neurons
  logger->parameter("block_stim",block_stim);
  logger->parameter("block_rep",block_rep);
  logger->parameter("block_bg_thl",block_rep);
  logger->parameter("block_bg_ctx",block_rep);
  logger->parameter("block_bg_hpc",block_rep);
  logger->parameter("block_exc_thl",block_exc_thl);
  logger->parameter("block_inh_thl",block_inh_thl);
  logger->parameter("block_exc_ctx",block_exc_ctx);
  logger->parameter("block_inh_ctx",block_inh_ctx);
  logger->parameter("block_exc_hpc",block_exc_hpc);
  logger->parameter("block_inh_hpc",block_inh_hpc);

  // inhibit neurons
  logger->parameter("inhibit_exc_hpc",inhibit_exc_hpc);
  
  // set up container for connections without hpc
  std::vector<Connection *> connections_hm;
  
  /*
  // set up container for spiking groups without hpc
  std::vector<SpikingGroup *> spiking_groups_hm;

  // set up container for devices without hpc
  std::vector<Device *> devices_hm;

  // set up container for checkers without hpc
  std::vector<Checker *> checkers_hm;
  */

  // build neuron groups
  #ifdef HAS_THL
    AIF2IGroup * neurons_e_thl = new AIF2IGroup(exc_size_thl);
    neurons_e_thl->dg_adapt1 = adapt1_thl;
    neurons_e_thl->dg_adapt2 = adapt2_thl;
    neurons_e_thl->set_tau_ampa(tau_ampa_e_thl);
    neurons_e_thl->set_tau_gaba(tau_gaba_e_thl);
    neurons_e_thl->set_tau_nmda(tau_nmda_e_thl);
    neurons_e_thl->set_ampa_nmda_ratio(ampa_nmda_e_thl);
    neurons_e_thl->set_name("E_THL");
    //spiking_groups_hm.push_back(neurons_e_thl);

    IFGroup * neurons_i_thl = new IFGroup(exc_size_thl/exc_inh_thl);
    neurons_i_thl->set_tau_ampa(tau_ampa_i_thl);
    neurons_i_thl->set_tau_gaba(tau_gaba_i_thl);
    neurons_i_thl->set_tau_nmda(tau_nmda_i_thl);
    neurons_i_thl->set_ampa_nmda_ratio(ampa_nmda_i_thl);
    neurons_i_thl->set_name("I_THL");
    //spiking_groups_hm.push_back(neurons_i_thl);

    #ifdef HAS_BACKGROUND
      StimulusGroup * bggroup_thl;
      sprintf(strbuf, "%s/%s.%d.bgtimes_thl", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      string bgtimefile_thl = strbuf;
      // bggroup_thl is initialized here
      // if bgfile_thl is empty no patterns are loaded
      // and it acts simply as PoissonGroup
      bggroup_thl = new StimulusGroup(exc_size_thl, bgtimefile_thl);
      bggroup_thl->set_mean_on_period(ontime);
      bggroup_thl->set_mean_off_period(offtime);
      bggroup_thl->binary_patterns = true;
      bggroup_thl->scale = scale;
      bggroup_thl->background_rate = bgrate_thl;
      bggroup_thl->background_during_stimulus = true;
      if (bg_thl_seed != 1) bggroup_thl->seed(bg_thl_seed);
      bggroup_thl->set_name("BG_THL");
      //spiking_groups_hm.push_back(bggroup_thl);
    #endif
      
  #endif
    
  #ifdef HAS_CTX
    AIF2IGroup * neurons_e_ctx = new AIF2IGroup(exc_size_ctx);
    neurons_e_ctx->dg_adapt1 = adapt1_ctx;
    neurons_e_ctx->dg_adapt2 = adapt2_ctx;
    neurons_e_ctx->set_tau_ampa(tau_ampa_e_ctx);
    neurons_e_ctx->set_tau_gaba(tau_gaba_e_ctx);
    neurons_e_ctx->set_tau_nmda(tau_nmda_e_ctx);
    neurons_e_ctx->set_ampa_nmda_ratio(ampa_nmda_e_ctx);
    neurons_e_ctx->set_name("E_CTX");
    //spiking_groups_hm.push_back(neurons_e_ctx);

    IFGroup * neurons_i_ctx = new IFGroup(exc_size_ctx/exc_inh_ctx);
    neurons_i_ctx->set_tau_ampa(tau_ampa_i_ctx);
    neurons_i_ctx->set_tau_gaba(tau_gaba_i_ctx);
    neurons_i_ctx->set_tau_nmda(tau_nmda_i_ctx);
    neurons_i_ctx->set_ampa_nmda_ratio(ampa_nmda_i_ctx);
    neurons_i_ctx->set_name("I_CTX");
    //spiking_groups_hm.push_back(neurons_i_ctx);

    #ifdef HAS_BACKGROUND
      StimulusGroup * bggroup_ctx;
      sprintf(strbuf, "%s/%s.%d.bgtimes_ctx", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      string bgtimefile_ctx = strbuf;
      // bggroup_ctx is initialized here
      // if bgfile_ctx is empty no patterns are loaded
      // and it acts simply as PoissonGroup
      bggroup_ctx = new StimulusGroup(exc_size_ctx, bgtimefile_ctx);
      bggroup_ctx->set_mean_on_period(ontime);
      bggroup_ctx->set_mean_off_period(offtime);
      bggroup_ctx->binary_patterns = true;
      bggroup_ctx->scale = scale;
      bggroup_ctx->background_rate = bgrate_ctx;
      bggroup_ctx->background_during_stimulus = true;
      if (bg_ctx_seed != 1) bggroup_ctx->seed(bg_ctx_seed);
      bggroup_ctx->set_name("BG_CTX");
      //spiking_groups_hm.push_back(bggroup_ctx);
    #endif
      
  #endif

  #ifdef HAS_HPC
    AIF2IGroup * neurons_e_hpc = new AIF2IGroup(exc_size_hpc);
    neurons_e_hpc->dg_adapt1 = adapt1_hpc;
    neurons_e_hpc->dg_adapt2 = adapt2_hpc;
    neurons_e_hpc->set_tau_ampa(tau_ampa_e_hpc);
    neurons_e_hpc->set_tau_gaba(tau_gaba_e_hpc);
    neurons_e_hpc->set_tau_nmda(tau_nmda_e_hpc);
    neurons_e_hpc->set_ampa_nmda_ratio(ampa_nmda_e_hpc);
    neurons_e_hpc->set_name("E_HPC");

    IFGroup * neurons_i_hpc = new IFGroup(exc_size_hpc/exc_inh_hpc);
    neurons_i_hpc->set_tau_ampa(tau_ampa_i_hpc);
    neurons_i_hpc->set_tau_gaba(tau_gaba_i_hpc);
    neurons_i_hpc->set_tau_nmda(tau_nmda_i_hpc);
    neurons_i_hpc->set_ampa_nmda_ratio(ampa_nmda_i_hpc);
    neurons_i_hpc->set_name("I_HPC");

    #ifdef HAS_INH2
      IFGroup * neurons_i2_hpc = new IFGroup(exc_size_hpc/exc_inh2_hpc);
      neurons_i2_hpc->set_tau_ampa(tau_ampa_i2_hpc);
      neurons_i2_hpc->set_tau_gaba(tau_gaba_i2_hpc);
      neurons_i2_hpc->set_tau_nmda(tau_nmda_i2_hpc);
      neurons_i2_hpc->set_ampa_nmda_ratio(ampa_nmda_i2_hpc);
      neurons_i2_hpc->set_name("I2_HPC");
    #endif


    #ifdef HAS_BACKGROUND
      StimulusGroup * bggroup_hpc;
      sprintf(strbuf, "%s/%s.%d.bgtimes_hpc", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      string bgtimefile_hpc = strbuf;
      // bggroup_hpc is initialized here
      // if bgfile_hpc is empty no patterns are loaded
      // and it acts simply as PoissonGroup
      bggroup_hpc = new StimulusGroup(exc_size_hpc, bgtimefile_hpc);
      bggroup_hpc->set_mean_on_period(ontime);
      bggroup_hpc->set_mean_off_period(offtime);
      bggroup_hpc->binary_patterns = true;
      bggroup_hpc->scale = scale;
      bggroup_hpc->background_rate = bgrate_hpc;
      bggroup_hpc->background_during_stimulus = true;
      if (bg_hpc_seed != 1) bggroup_hpc->seed(bg_hpc_seed);
      bggroup_hpc->set_name("BG_HPC");
      //spiking_groups_hm.push_back(bggroup_hpc);
    #endif
    
  #endif

  #ifdef HAS_REPLAY
    StimulusGroup * repgroup;
    sprintf(strbuf, "%s/%s.%d.reptimes", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank());
    string reptimefile = strbuf;
    // repgroup is initialized here
    // if repfile is empty no patterns are loaded
    // and it acts simply as PoissonGroup
    repgroup = new StimulusGroup(size_rep, reptimefile);
    repgroup->set_mean_on_period(ontime);
    repgroup->set_mean_off_period(offtime);
    repgroup->binary_patterns = true;
    repgroup->scale = scale;
    repgroup->background_rate = bgrate_rep;
    repgroup->background_during_stimulus = true;
    if (rep_seed != 1) repgroup->seed(rep_seed);
    repgroup->set_name("REP");
  #endif
  
  StimulusGroup * stimgroup;
  sprintf(strbuf, "%s/%s.%d.stimtimes", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
  string stimtimefile = strbuf;
  // stimgroup is initialized here
  // if stimfile is empty no patterns are loaded
  // and it acts simply as PoissonGroup
  stimgroup = new StimulusGroup(exc_size_ctx, stimtimefile);
  stimgroup->set_mean_on_period(ontime);
  stimgroup->set_mean_off_period(offtime);
  stimgroup->binary_patterns = true;
  stimgroup->scale = scale;
  stimgroup->background_rate = bgrate;
  stimgroup->background_during_stimulus = true;
  if (stim_seed != 1) stimgroup->seed(stim_seed);
  stimgroup->set_name("STIM");
  //spiking_groups_hm.push_back(stimgroup);

  
  // build connections
  #ifdef HAS_THL
    P11BConnection * con_ee_thl;
    con_ee_thl = new P11BConnection(neurons_e_thl,neurons_e_thl,
			       wee_thl,sparseness_int_ee_thl,
			       eta_thl,
			       kappa_thl,
			       wmax_exc_thl
			       );
    con_ee_thl->set_transmitter(AMPA);
    con_ee_thl->set_weight_a(weight_a_thl); 
    con_ee_thl->set_weight_c(weight_c_thl); 
    con_ee_thl->consolidation_active = consolidation;
    double wtmax_thl = 1.0/4*(weight_c_thl-weight_a_thl);
    double normalization_factor_thl = (wtmax_thl-weight_a_thl)*(wtmax_thl-(weight_a_thl+weight_c_thl)/2)*(wtmax_thl-weight_c_thl); 
    con_ee_thl->pot_strength = pot_strength_thl/normalization_factor_thl;
    logger->parameter("normalized pot_strength_thl", con_ee_thl->pot_strength);
    if ( noisy_initial_weights ) 
      con_ee_thl->random_data(wee_thl, wee_thl);
    if ( consolidate_initial_weights )
      con_ee_thl->consolidate();	
    con_ee_thl->set_tau_d(taud_ee_thl);
    con_ee_thl->set_tau_f(tauf_ee_thl);
    con_ee_thl->set_ujump(ujump_ee_thl);
    con_ee_thl->set_beta(beta_thl*eta_thl);
    con_ee_thl->delta = delta_thl*eta_thl;
    con_ee_thl->set_min_weight(wmin_exc_thl);
    con_ee_thl->set_tau_hom(tauh_ee_thl);
    con_ee_thl->set_tau_con(tauc_ee_thl);
    con_ee_thl->set_name("EE_THL");
    connections_hm.push_back(con_ee_thl);

    STPBConnection * con_ei_thl = new STPBConnection(neurons_e_thl,neurons_i_thl,wei_thl,sparseness_int_ei_thl,GLUT);
    con_ei_thl->set_tau_d(taud_ei_thl);
    con_ei_thl->set_tau_f(tauf_ei_thl);
    con_ei_thl->set_ujump(ujump_ei_thl);
    con_ei_thl->set_name("EI_THL");
    connections_hm.push_back(con_ei_thl);

    SparseBConnection * con_ii_thl = new SparseBConnection(neurons_i_thl,neurons_i_thl,wii_thl,sparseness_int_ii_thl,GABA);
    con_ii_thl->set_name("II_THL");
    connections_hm.push_back(con_ii_thl);

    GlobalPFBConnection * con_ie_thl;
    con_ie_thl = new GlobalPFBConnection(neurons_i_thl,neurons_e_thl,
				     wie_thl,sparseness_int_ie_thl,
				     tauh_ie_thl,
				     eta_thl/eta_exc_inh_thl,
				     alpha_thl, 
				     wmax_inh_thl,
				     GABA
				     );
    con_ie_thl->set_name("IE_THL");
    connections_hm.push_back(con_ie_thl);

    #ifndef CON_STIM_THL_NONE
     #ifdef HAS_CROSS_REGION_EI_CON
      STPBConnection * con_ei_stim_thl = new STPBConnection( stimgroup, 
							     neurons_i_thl, 
							     wext_ei_thl, 
							     sparseness_ext_thl, 
							     GLUT);
      con_ei_stim_thl->set_tau_d(taud_ei_stim_thl);
      con_ei_stim_thl->set_tau_f(tauf_ei_stim_thl);
      con_ei_stim_thl->set_ujump(ujump_ei_stim_thl);
      con_ei_stim_thl->set_name("EI_STIM->THL");
      connections_hm.push_back(con_ei_stim_thl);
     #endif
    #endif

    #ifdef CON_STIM_THL_SPARSEB
     SparseBConnection * con_ee_stim_thl = new SparseBConnection(stimgroup, neurons_e_thl,
							 wext_thl, sparseness_ext_thl, GLUT);
     con_ee_stim_thl->set_name("EE_STIM->THL");
     connections_hm.push_back(con_ee_stim_thl);
    #endif

    #ifdef CON_STIM_THL_STPB
     STPBConnection * con_ee_stim_thl = new STPBConnection(stimgroup,neurons_e_thl,wext_thl,sparseness_ext_thl,GLUT);
     con_ee_stim_thl->set_tau_d(taud_ee_stim_thl);
     con_ee_stim_thl->set_tau_f(tauf_ee_stim_thl);
     con_ee_stim_thl->set_ujump(ujump_ee_stim_thl);
     con_ee_stim_thl->set_name("EE_STIM->THL");
     connections_hm.push_back(con_ee_stim_thl);
    #endif

    #ifdef CON_STIM_THL_P11B
     P11BConnection * con_ee_stim_thl = NULL;
     con_ee_stim_thl = new P11BConnection( stimgroup, neurons_e_thl,
				     wext_thl,sparseness_ext_thl,
				     eta_stim_thl,
				     kappa_thl, // supposedly deprecated
				     wmax_exc_thl,
				     GLUT
				     );
     con_ee_stim_thl->set_weight_a(weight_a_thl);
     con_ee_stim_thl->set_weight_c(weight_c_thl);
     con_ee_stim_thl->set_tau_d(taud_ee_stim_thl);
     con_ee_stim_thl->set_tau_f(tauf_ee_stim_thl);
     con_ee_stim_thl->set_ujump(ujump_ee_stim_thl);
     con_ee_stim_thl->set_beta(beta_stim_thl*eta_stim_thl);
     con_ee_stim_thl->delta = delta_thl*eta_stim_thl;
     con_ee_stim_thl->set_min_weight(wmin_exc_thl);
     if ( noisy_initial_weights ) 
       con_ee_stim_thl->random_data(wext_thl, wext_thl);
     con_ee_stim_thl->consolidation_active = consolidation;
     con_ee_stim_thl->pot_strength = pot_strength_thl/normalization_factor_thl;
     con_ee_stim_thl->set_tau_hom(tauh_ee_stim_thl);
     con_ee_stim_thl->set_tau_con(tauc_ee_stim_thl);
     if ( consolidate_initial_weights )
       con_ee_stim_thl->consolidate();
     con_ee_stim_thl->set_name("EE_STIM->THL");
     connections_hm.push_back(con_ee_stim_thl);
    #endif

    #ifdef HAS_CTX
      
      #ifndef CON_THL_CTX_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
        STPBConnection * con_ei_thl_ctx = new STPBConnection(neurons_e_thl,neurons_i_ctx,wei_thl_ctx,sparseness_thl_ctx,GLUT);
	con_ei_thl_ctx->set_tau_d(taud_thl_ctx);
	con_ei_thl_ctx->set_tau_f(tauf_thl_ctx);
	con_ei_thl_ctx->set_ujump(ujump_thl_ctx);
	con_ei_thl_ctx->set_name("EI_THL->CTX");
	connections_hm.push_back(con_ei_thl_ctx);
       #endif
      #endif
     
      #ifdef CON_THL_CTX_SPARSEB
       SparseBConnection * con_ee_thl_ctx = new SparseBConnection(neurons_e_thl,neurons_e_ctx,wee_thl_ctx,sparseness_thl_ctx,GLUT);
       con_ee_thl_ctx->set_name("EE_THL->CTX");
       connections_hm.push_back(con_ee_thl_ctx);
      #endif

      #ifdef CON_THL_CTX_STPB
       STPBConnection * con_ee_thl_ctx = new STPBConnection(neurons_e_thl,neurons_e_ctx,wee_thl_ctx,sparseness_thl_ctx,GLUT);
       con_ee_thl_ctx->set_tau_d(taud_thl_ctx);
       con_ee_thl_ctx->set_tau_f(tauf_thl_ctx);
       con_ee_thl_ctx->set_ujump(ujump_thl_ctx);
       con_ee_thl_ctx->set_name("EE_THL->CTX");
       connections_hm.push_back(con_ee_thl_ctx);
      #endif

      #ifdef CON_THL_CTX_P11B
       P11BConnection * con_ee_thl_ctx;
       con_ee_thl_ctx = new P11BConnection(neurons_e_thl,neurons_e_ctx,
					  wee_thl_ctx,sparseness_thl_ctx,
					  eta_thl_ctx,
					  kappa_thl_ctx,
					  wmax_thl_ctx
					  );
       con_ee_thl_ctx->set_transmitter(AMPA);
       con_ee_thl_ctx->set_name("EE_THL->CTX");
       con_ee_thl_ctx->set_weight_a(weight_a_thl_ctx); 
       con_ee_thl_ctx->set_weight_c(weight_c_thl_ctx); 
       con_ee_thl_ctx->consolidation_active = consolidation;
       double wtmax_thl_ctx = 1.0/4*(weight_c_thl_ctx-weight_a_thl_ctx);
       double normalization_factor_thl_ctx = (wtmax_thl_ctx-weight_a_thl_ctx)*(wtmax_thl_ctx-(weight_a_thl_ctx+weight_c_thl_ctx)/2)*(wtmax_thl_ctx-weight_c_thl_ctx); 
       con_ee_thl_ctx->pot_strength = pot_strength_thl_ctx/normalization_factor_thl_ctx;
       logger->parameter("normalized pot_strength_thl_ctx", con_ee_thl_ctx->pot_strength);
       if ( noisy_initial_weights ) 
	 con_ee_thl_ctx->random_data(wee_thl_ctx, wee_thl_ctx);
       if ( consolidate_initial_weights )
	 con_ee_thl_ctx->consolidate();	
       con_ee_thl_ctx->set_tau_d(taud_thl_ctx);
       con_ee_thl_ctx->set_tau_f(tauf_thl_ctx);
       con_ee_thl_ctx->set_ujump(ujump_thl_ctx);
       con_ee_thl_ctx->set_beta(beta_thl_ctx*eta_thl_ctx);
       con_ee_thl_ctx->delta = delta_thl_ctx*eta_thl_ctx;
       con_ee_thl_ctx->set_min_weight(wmin_thl_ctx);
       con_ee_thl_ctx->set_tau_hom(tauh_thl_ctx);
       con_ee_thl_ctx->set_tau_con(tauc_thl_ctx);
       connections_hm.push_back(con_ee_thl_ctx);
      #endif

      #ifndef CON_CTX_THL_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
       STPBConnection * con_ei_ctx_thl = new STPBConnection(neurons_e_ctx,neurons_i_thl,wei_ctx_thl,sparseness_ctx_thl,GLUT);
	con_ei_ctx_thl->set_tau_d(taud_ctx_thl);
	con_ei_ctx_thl->set_tau_f(tauf_ctx_thl);
	con_ei_ctx_thl->set_ujump(ujump_ctx_thl);
	con_ei_ctx_thl->set_name("EI_CTX->THL");
	connections_hm.push_back(con_ei_ctx_thl);
       #endif
      #endif
      
      #ifdef CON_CTX_THL_SPARSEB
       SparseBConnection * con_ee_ctx_thl = new SparseBConnection(neurons_e_ctx,neurons_e_thl,wee_ctx_thl,sparseness_ctx_thl,GLUT);
       con_ee_ctx_thl->set_name("EE_CTX->THL");
       connections_hm.push_back(con_ee_ctx_thl);
      #endif

      #ifdef CON_CTX_THL_STPB
       STPBConnection * con_ee_ctx_thl = new STPBConnection(neurons_e_ctx,neurons_e_thl,wee_ctx_thl,sparseness_ctx_thl,GLUT);
       con_ee_ctx_thl->set_tau_d(taud_ctx_thl);
       con_ee_ctx_thl->set_tau_f(tauf_ctx_thl);
       con_ee_ctx_thl->set_ujump(ujump_ctx_thl);
       con_ee_ctx_thl->set_name("EE_CTX->THL");
       connections_hm.push_back(con_ee_ctx_thl);
      #endif

      #ifdef CON_CTX_THL_P11B
       P11BConnection * con_ee_ctx_thl;
       con_ee_ctx_thl = new P11BConnection(neurons_e_ctx,neurons_e_thl,
					  wee_ctx_thl,sparseness_ctx_thl,
					  eta_ctx_thl,
					  kappa_ctx_thl,
					  wmax_ctx_thl
					  );
       con_ee_ctx_thl->set_transmitter(AMPA);
       con_ee_ctx_thl->set_name("EE_CTX->THL");
       con_ee_ctx_thl->set_weight_a(weight_a_ctx_thl); 
       con_ee_ctx_thl->set_weight_c(weight_c_ctx_thl); 
       con_ee_ctx_thl->consolidation_active = consolidation;
       double wtmax_ctx_thl = 1.0/4*(weight_c_ctx_thl-weight_a_ctx_thl);
       double normalization_factor_ctx_thl = (wtmax_ctx_thl-weight_a_ctx_thl)*(wtmax_ctx_thl-(weight_a_ctx_thl+weight_c_ctx_thl)/2)*(wtmax_ctx_thl-weight_c_ctx_thl); 
       con_ee_ctx_thl->pot_strength = pot_strength_ctx_thl/normalization_factor_ctx_thl;
       logger->parameter("normalized pot_strength_ctx_thl", con_ee_ctx_thl->pot_strength);
       if ( noisy_initial_weights ) 
	 con_ee_ctx_thl->random_data(wee_ctx_thl, wee_ctx_thl);
       if ( consolidate_initial_weights )
	 con_ee_ctx_thl->consolidate();	
       con_ee_ctx_thl->set_tau_d(taud_ctx_thl);
       con_ee_ctx_thl->set_tau_f(tauf_ctx_thl);
       con_ee_ctx_thl->set_ujump(ujump_ctx_thl);
       con_ee_ctx_thl->set_beta(beta_ctx_thl*eta_ctx_thl);
       con_ee_ctx_thl->delta = delta_ctx_thl*eta_ctx_thl;
       con_ee_ctx_thl->set_min_weight(wmin_ctx_thl);
       con_ee_ctx_thl->set_tau_hom(tauh_ctx_thl);
       con_ee_ctx_thl->set_tau_con(tauc_ctx_thl);
       connections_hm.push_back(con_ee_ctx_thl);
      #endif
    #endif

    #ifdef HAS_HPC

      #ifndef CON_THL_HPC_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
        STPBConnection * con_ei_thl_hpc = new STPBConnection(neurons_e_thl,neurons_i_hpc,wei_thl_hpc,sparseness_thl_hpc,GLUT);
	con_ei_thl_hpc->set_tau_d(taud_thl_hpc);
	con_ei_thl_hpc->set_tau_f(tauf_thl_hpc);
	con_ei_thl_hpc->set_ujump(ujump_thl_hpc);
	con_ei_thl_hpc->set_name("EI_THL->HPC");
	connections_hm.push_back(con_ei_thl_hpc);
       #endif
      #endif
      
      #ifdef CON_THL_HPC_SPARSEB
       SparseBConnection * con_ee_thl_hpc = new SparseBConnection(neurons_e_thl,neurons_e_hpc,wee_thl_hpc,sparseness_thl_hpc,GLUT);
       con_ee_thl_hpc->set_name("EE_THL->HPC");
      #endif

      #ifdef CON_THL_HPC_STPB
       STPBConnection * con_ee_thl_hpc = new STPBConnection(neurons_e_thl,neurons_e_hpc,wee_thl_hpc,sparseness_thl_hpc,GLUT);
       con_ee_thl_hpc->set_tau_d(taud_thl_hpc);
       con_ee_thl_hpc->set_tau_f(tauf_thl_hpc);
       con_ee_thl_hpc->set_ujump(ujump_thl_hpc);
       con_ee_thl_hpc->set_name("EE_THL->HPC");
      #endif

      #ifdef CON_THL_HPC_P11B
       P11BConnection * con_ee_thl_hpc;
       con_ee_thl_hpc = new P11BConnection(neurons_e_thl,neurons_e_hpc,
					  wee_thl_hpc,sparseness_thl_hpc,
					  eta_thl_hpc,
					  kappa_thl_hpc,
					  wmax_thl_hpc
					  );
       con_ee_thl_hpc->set_transmitter(AMPA);
       con_ee_thl_hpc->set_name("EE_THL->HPC");
       con_ee_thl_hpc->set_weight_a(weight_a_thl_hpc); 
       con_ee_thl_hpc->set_weight_c(weight_c_thl_hpc); 
       con_ee_thl_hpc->consolidation_active = consolidation;
       double wtmax_thl_hpc = 1.0/4*(weight_c_thl_hpc-weight_a_thl_hpc);
       double normalization_factor_thl_hpc = (wtmax_thl_hpc-weight_a_thl_hpc)*(wtmax_thl_hpc-(weight_a_thl_hpc+weight_c_thl_hpc)/2)*(wtmax_thl_hpc-weight_c_thl_hpc); 
       con_ee_thl_hpc->pot_strength = pot_strength_thl_hpc/normalization_factor_thl_hpc;
       logger->parameter("normalized pot_strength_thl_hpc", con_ee_thl_hpc->pot_strength);
       if ( noisy_initial_weights ) 
	 con_ee_thl_hpc->random_data(wee_thl_hpc, wee_thl_hpc);
       if ( consolidate_initial_weights )
	 con_ee_thl_hpc->consolidate();	
       con_ee_thl_hpc->set_tau_d(taud_thl_hpc);
       con_ee_thl_hpc->set_tau_f(tauf_thl_hpc);
       con_ee_thl_hpc->set_ujump(ujump_thl_hpc);
       con_ee_thl_hpc->set_beta(beta_thl_hpc*eta_thl_hpc);
       con_ee_thl_hpc->delta = delta_thl_hpc*eta_thl_hpc;
       con_ee_thl_hpc->set_min_weight(wmin_thl_hpc);
       con_ee_thl_hpc->set_tau_hom(tauh_thl_hpc);
       con_ee_thl_hpc->set_tau_con(tauc_thl_hpc);
      #endif

      #ifndef CON_HPC_THL_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
        STPBConnection * con_ei_hpc_thl = new STPBConnection(neurons_e_hpc,neurons_i_thl,wei_hpc_thl,sparseness_hpc_thl,GLUT);
	con_ei_hpc_thl->set_tau_d(taud_hpc_thl);
	con_ei_hpc_thl->set_tau_f(tauf_hpc_thl);
	con_ei_hpc_thl->set_ujump(ujump_hpc_thl);
	con_ei_hpc_thl->set_name("EI_HPC->THL");
	connections_hm.push_back(con_ei_hpc_thl);
       #endif
      #endif
      
      #ifdef CON_HPC_THL_SPARSEB
       SparseBConnection * con_ee_hpc_thl = new SparseBConnection(neurons_e_hpc,neurons_e_thl,wee_hpc_thl,sparseness_hpc_thl,GLUT);
       con_ee_hpc_thl->set_name("EE_HPC->THL");
      #endif

      #ifdef CON_HPC_THL_STPB
       STPBConnection * con_ee_hpc_thl = new STPBConnection(neurons_e_hpc,neurons_e_thl,wee_hpc_thl,sparseness_hpc_thl,GLUT);
       con_ee_hpc_thl->set_tau_d(taud_hpc_thl);
       con_ee_hpc_thl->set_tau_f(tauf_hpc_thl);
       con_ee_hpc_thl->set_ujump(ujump_hpc_thl);
       con_ee_hpc_thl->set_name("EE_HPC->THL");
      #endif

      #ifdef CON_HPC_THL_P11B
       P11BConnection * con_ee_hpc_thl;
       con_ee_hpc_thl = new P11BConnection(neurons_e_hpc,neurons_e_thl,
					  wee_hpc_thl,sparseness_hpc_thl,
					  eta_hpc_thl,
					  kappa_hpc_thl,
					  wmax_hpc_thl
					  );
       con_ee_hpc_thl->set_transmitter(AMPA);
       con_ee_hpc_thl->set_name("EE_HPC->THL");
       con_ee_hpc_thl->set_weight_a(weight_a_hpc_thl); 
       con_ee_hpc_thl->set_weight_c(weight_c_hpc_thl); 
       con_ee_hpc_thl->consolidation_active = consolidation;
       double wtmax_hpc_thl = 1.0/4*(weight_c_hpc_thl-weight_a_hpc_thl);
       double normalization_factor_hpc_thl = (wtmax_hpc_thl-weight_a_hpc_thl)*(wtmax_hpc_thl-(weight_a_hpc_thl+weight_c_hpc_thl)/2)*(wtmax_hpc_thl-weight_c_hpc_thl); 
       con_ee_hpc_thl->pot_strength = pot_strength_hpc_thl/normalization_factor_hpc_thl;
       logger->parameter("normalized pot_strength_hpc_thl", con_ee_hpc_thl->pot_strength);
       if ( noisy_initial_weights ) 
	 con_ee_hpc_thl->random_data(wee_hpc_thl, wee_hpc_thl);
       if ( consolidate_initial_weights )
	 con_ee_hpc_thl->consolidate();	
       con_ee_hpc_thl->set_tau_d(taud_hpc_thl);
       con_ee_hpc_thl->set_tau_f(tauf_hpc_thl);
       con_ee_hpc_thl->set_ujump(ujump_hpc_thl);
       con_ee_hpc_thl->set_beta(beta_hpc_thl*eta_hpc_thl);
       con_ee_hpc_thl->delta = delta_hpc_thl*eta_hpc_thl;
       con_ee_hpc_thl->set_min_weight(wmin_hpc_thl);
       con_ee_hpc_thl->set_tau_hom(tauh_hpc_thl);
       con_ee_hpc_thl->set_tau_con(tauc_hpc_thl);
      #endif
    #endif

    #ifdef HAS_REPLAY
     #ifdef CON_REPLAY_THL_SPARSEB
       
      SparseBConnection * con_ee_rep_thl = new SparseBConnection(repgroup, neurons_e_thl,
								 w_rep_thl, sparseness_rep_thl,
								 GLUT);
      con_ee_rep_thl->set_name("EE_REPLAY->THL");
      connections_hm.push_back(con_ee_rep_thl);
      
      SparseBConnection * con_ei_rep_thl = new SparseBConnection(repgroup, neurons_i_thl,
								 w_rep_thl, sparseness_rep_thl,
								 GLUT);
      con_ei_rep_thl->set_name("EI_REPLAY->THL");
      connections_hm.push_back(con_ei_rep_thl);

     #endif
    #endif

    #ifdef HAS_BACKGROUND
      
     SparseBConnection * con_ee_bg_thl = new SparseBConnection(bggroup_thl, neurons_e_thl,
							       w_bg_thl, sparseness_bg_thl,
							       GLUT);
     con_ee_bg_thl->set_name("EE_BG->THL");
     connections_hm.push_back(con_ee_bg_thl);

     #ifdef HAS_CROSS_REGION_EI_CON
     
      STPBConnection * con_ei_bg_thl = new STPBConnection(bggroup_thl, 
							  neurons_i_thl, 
							  w_bg_ei_thl, 
							  sparseness_bg_thl, 
							  GLUT);
      con_ei_bg_thl->set_tau_d(taud_ei_stim_thl);
      con_ei_bg_thl->set_tau_f(tauf_ei_stim_thl);
      con_ei_bg_thl->set_ujump(ujump_ei_stim_thl);
      con_ei_bg_thl->set_name("EI_BG->THL");
      connections_hm.push_back(con_ei_bg_thl);
     #endif
      
    #endif
  #endif

  
  #ifdef HAS_CTX
    P11BConnection * con_ee_ctx;
    con_ee_ctx = new P11BConnection(neurons_e_ctx,neurons_e_ctx,
			       wee_ctx,sparseness_int_ee_ctx,
			       eta_ctx,
			       kappa_ctx,
			       wmax_exc_ctx
			       );
    con_ee_ctx->set_transmitter(AMPA);
    con_ee_ctx->set_weight_a(weight_a_ctx); 
    con_ee_ctx->set_weight_c(weight_c_ctx); 
    con_ee_ctx->consolidation_active = consolidation;
    double wtmax_ctx = 1.0/4*(weight_c_ctx-weight_a_ctx);
    double normalization_factor_ctx = (wtmax_ctx-weight_a_ctx)*(wtmax_ctx-(weight_a_ctx+weight_c_ctx)/2)*(wtmax_ctx-weight_c_ctx); 
    con_ee_ctx->pot_strength = pot_strength_ctx/normalization_factor_ctx;
    logger->parameter("normalized pot_strength_ctx", con_ee_ctx->pot_strength);
    if ( noisy_initial_weights ) 
      con_ee_ctx->random_data(wee_ctx, wee_ctx);
    if ( consolidate_initial_weights )
      con_ee_ctx->consolidate();	
    con_ee_ctx->set_tau_d(taud_ee_ctx);
    con_ee_ctx->set_tau_f(tauf_ee_ctx);
    con_ee_ctx->set_ujump(ujump_ee_ctx);
    con_ee_ctx->set_beta(beta_ctx*eta_ctx);
    con_ee_ctx->delta = delta_ctx*eta_ctx;
    con_ee_ctx->set_min_weight(wmin_exc_ctx);
    con_ee_ctx->set_tau_hom(tauh_ee_ctx);
    con_ee_ctx->set_tau_con(tauc_ee_ctx);
    con_ee_ctx->set_name("EE_CTX");
    connections_hm.push_back(con_ee_ctx);

    STPBConnection * con_ei_ctx = new STPBConnection(neurons_e_ctx,neurons_i_ctx,wei_ctx,sparseness_int_ei_ctx,GLUT);
    con_ei_ctx->set_tau_d(taud_ei_ctx);
    con_ei_ctx->set_tau_f(tauf_ei_ctx);
    con_ei_ctx->set_ujump(ujump_ei_ctx);
    con_ei_ctx->set_name("EI_CTX");
    connections_hm.push_back(con_ei_ctx);

    SparseBConnection * con_ii_ctx = new SparseBConnection(neurons_i_ctx,neurons_i_ctx,wii_ctx,sparseness_int_ii_ctx,GABA);
    con_ii_ctx->set_name("II_CTX");
    connections_hm.push_back(con_ii_ctx);

    GlobalPFBConnection * con_ie_ctx;
    con_ie_ctx = new GlobalPFBConnection(neurons_i_ctx,neurons_e_ctx,
				     wie_ctx,sparseness_int_ie_ctx,
				     tauh_ie_ctx,
				     eta_ctx/eta_exc_inh_ctx,
				     alpha_ctx, 
				     wmax_inh_ctx,
				     GABA
				     );
    con_ie_ctx->set_name("IE_CTX");
    connections_hm.push_back(con_ie_ctx);

    #ifndef CON_STIM_CTX_NONE
     #ifdef HAS_CROSS_REGION_EI_CON
      STPBConnection * con_ei_stim_ctx = new STPBConnection( stimgroup, 
							     neurons_i_ctx, 
							     wext_ei_ctx, 
							     sparseness_ext_ctx, 
							     GLUT);
      con_ei_stim_ctx->set_tau_d(taud_ei_stim_ctx);
      con_ei_stim_ctx->set_tau_f(tauf_ei_stim_ctx);
      con_ei_stim_ctx->set_ujump(ujump_ei_stim_ctx);
      con_ei_stim_ctx->set_name("EI_STIM->CTX");
      connections_hm.push_back(con_ei_stim_ctx);
     #endif
    #endif

    #ifdef CON_STIM_CTX_SPARSEB
     SparseBConnection * con_ee_stim_ctx = new SparseBConnection(stimgroup, neurons_e_ctx,
							 wext_ctx, sparseness_ext_ctx, GLUT);
     con_ee_stim_ctx->set_name("EE_STIM->CTX");
     connections_hm.push_back(con_ee_stim_ctx);
    #endif

    #ifdef CON_STIM_CTX_STPB
     STPBConnection * con_ee_stim_ctx = new STPBConnection(stimgroup,neurons_e_ctx,wext_ctx,sparseness_ext_ctx,GLUT);
     con_ee_stim_ctx->set_tau_d(taud_ee_stim_ctx);
     con_ee_stim_ctx->set_tau_f(tauf_ee_stim_ctx);
     con_ee_stim_ctx->set_ujump(ujump_ee_stim_ctx);
     con_ee_stim_ctx->set_name("EE_STIM->CTX");
     connections_hm.push_back(con_ee_stim_ctx);
    #endif

    #ifdef CON_STIM_CTX_P11B
     P11BConnection * con_ee_stim_ctx = NULL;
     con_ee_stim_ctx = new P11BConnection( stimgroup, neurons_e_ctx,
				     wext_ctx,sparseness_ext_ctx,
				     eta_stim_ctx,
				     kappa_ctx, // supposedly deprecated
				     wmax_exc_ctx,
				     GLUT
				     );
     con_ee_stim_ctx->set_weight_a(weight_a_ctx);
     con_ee_stim_ctx->set_weight_c(weight_c_ctx);
     con_ee_stim_ctx->set_tau_d(taud_ee_stim_ctx);
     con_ee_stim_ctx->set_tau_f(tauf_ee_stim_ctx);
     con_ee_stim_ctx->set_ujump(ujump_ee_stim_ctx);
     con_ee_stim_ctx->set_beta(beta_stim_ctx*eta_stim_ctx);
     con_ee_stim_ctx->delta = delta_ctx*eta_stim_ctx;
     con_ee_stim_ctx->set_min_weight(wmin_exc_ctx);
     if ( noisy_initial_weights ) 
       con_ee_stim_ctx->random_data(wext_ctx, wext_ctx);
     con_ee_stim_ctx->consolidation_active = consolidation;
     con_ee_stim_ctx->pot_strength = pot_strength_ctx/normalization_factor_ctx;
     con_ee_stim_ctx->set_tau_hom(tauh_ee_stim_ctx);
     con_ee_stim_ctx->set_tau_con(tauc_ee_stim_ctx);
     if ( consolidate_initial_weights )
       con_ee_stim_ctx->consolidate();
     con_ee_stim_ctx->set_name("EE_STIM->CTX");
     connections_hm.push_back(con_ee_stim_ctx);
    #endif

    #ifdef HAS_REPLAY
     #ifdef CON_REPLAY_CTX_SPARSEB
     
      SparseBConnection * con_ee_rep_ctx = new SparseBConnection(repgroup, neurons_e_ctx,
								 w_rep_ctx, sparseness_rep_ctx,
								 GLUT);
      con_ee_rep_ctx->set_name("EE_REPLAY->CTX");
      connections_hm.push_back(con_ee_rep_ctx);

      SparseBConnection * con_ei_rep_ctx = new SparseBConnection(repgroup, neurons_i_ctx,
								 w_rep_ctx, sparseness_rep_ctx,
								 GLUT);
      con_ei_rep_ctx->set_name("EI_REPLAY->CTX");
      connections_hm.push_back(con_ei_rep_ctx);
       
     #endif
    #endif

    #ifdef HAS_BACKGROUND
     SparseBConnection * con_ee_bg_ctx = new SparseBConnection(bggroup_ctx, neurons_e_ctx,
							       w_bg_ctx, sparseness_bg_ctx,
							       GLUT);
     con_ee_bg_ctx->set_name("EE_BG->CTX");
     connections_hm.push_back(con_ee_bg_ctx);

     #ifdef HAS_CROSS_REGION_EI_CON
     
      STPBConnection * con_ei_bg_ctx = new STPBConnection(bggroup_ctx, 
							  neurons_i_ctx, 
							  w_bg_ei_ctx, 
							  sparseness_bg_ctx, 
							  GLUT);
      con_ei_bg_ctx->set_tau_d(taud_ei_stim_ctx);
      con_ei_bg_ctx->set_tau_f(tauf_ei_stim_ctx);
      con_ei_bg_ctx->set_ujump(ujump_ei_stim_ctx);
      con_ei_bg_ctx->set_name("EI_BG->CTX");
      connections_hm.push_back(con_ei_bg_ctx);
      
     #endif
    #endif
  #endif
  
  #ifdef HAS_HPC
   double wtmax_hpc = 1.0/4*(weight_c_hpc-weight_a_hpc);
   double normalization_factor_hpc = (wtmax_hpc-weight_a_hpc)*(wtmax_hpc-(weight_a_hpc+weight_c_hpc)/2)*(wtmax_hpc-weight_c_hpc); 
   #ifdef HAS_REC_EE
      P11BConnection * con_ee_hpc;
      con_ee_hpc = new P11BConnection(neurons_e_hpc,neurons_e_hpc,
				      wee_hpc,sparseness_int_ee_hpc,
				      eta_hpc,
				      kappa_hpc,
				      wmax_exc_hpc
				      );
      con_ee_hpc->set_transmitter(AMPA);
      con_ee_hpc->set_name("EE_HPC");
      con_ee_hpc->set_weight_a(weight_a_hpc); 
      con_ee_hpc->set_weight_c(weight_c_hpc); 
      con_ee_hpc->consolidation_active = consolidation;
      con_ee_hpc->pot_strength = pot_strength_hpc/normalization_factor_hpc;
      logger->parameter("normalized pot_strength_hpc", con_ee_hpc->pot_strength);
      if ( noisy_initial_weights ) 
	con_ee_hpc->random_data(wee_hpc, wee_hpc);
      if ( consolidate_initial_weights )
	con_ee_hpc->consolidate();	
      con_ee_hpc->set_tau_d(taud_ee_hpc);
      con_ee_hpc->set_tau_f(tauf_ee_hpc);
      con_ee_hpc->set_ujump(ujump_ee_hpc);
      con_ee_hpc->set_beta(beta_hpc*eta_hpc);
      con_ee_hpc->delta = delta_hpc*eta_hpc;
      con_ee_hpc->set_min_weight(wmin_exc_hpc);
      con_ee_hpc->set_tau_hom(tauh_ee_hpc);
      con_ee_hpc->set_tau_con(tauc_ee_hpc);
   #endif
      

   STPBConnection * con_ei_hpc = new STPBConnection(neurons_e_hpc,neurons_i_hpc,wei_hpc,sparseness_int_ei_hpc,GLUT);
   con_ei_hpc->set_tau_d(taud_ei_hpc);
   con_ei_hpc->set_tau_f(tauf_ei_hpc);
   con_ei_hpc->set_ujump(ujump_ei_hpc);
   con_ei_hpc->set_name("EI_HPC");
     
   SparseBConnection * con_ii_hpc = new SparseBConnection(neurons_i_hpc,neurons_i_hpc,wii_hpc,sparseness_int_ii_hpc,GABA);
   con_ii_hpc->set_name("II_HPC");

   GlobalPFBConnection * con_ie_hpc;
   con_ie_hpc = new GlobalPFBConnection(neurons_i_hpc,neurons_e_hpc,
				       wie_hpc,sparseness_int_ie_hpc,
				       tauh_ie_hpc,
				       eta_hpc/eta_exc_inh_hpc,
				       alpha_hpc, 
				       wmax_inh_hpc,
				       GABA
				       );
   con_ie_hpc->set_name("IE_HPC");

   #ifdef HAS_INH2
     STPBConnection * con_ei2_hpc = new STPBConnection(neurons_e_hpc,neurons_i2_hpc,wei2_hpc,sparseness_int_ei2_hpc,GLUT);
     con_ei2_hpc->set_tau_d(taud_ei2_hpc);
     con_ei2_hpc->set_tau_f(tauf_ei2_hpc);
     con_ei2_hpc->set_ujump(ujump_ei2_hpc);
     con_ei2_hpc->set_name("EI2_HPC");

     SparseBConnection * con_i2i2_hpc = new SparseBConnection(neurons_i2_hpc,neurons_i2_hpc,wi2i2_hpc,sparseness_int_i2i2_hpc,GABA);
     con_i2i2_hpc->set_name("I2I2_HPC");

     GlobalPF2BConnection * con_i2e_hpc;
     con_i2e_hpc = new GlobalPF2BConnection(neurons_i2_hpc,neurons_e_hpc,
					  wi2e_hpc,sparseness_int_i2e_hpc,
					  tauh_i2e_hpc,
					  eta_hpc/eta_exc_inh2_hpc,
					  alpha2_hpc, 
					  wmax_inh2_hpc,
					  GABA
					  );
     con_i2e_hpc->set_name("I2E_HPC");

     SparseBConnection * con_ii2_hpc = new SparseBConnection(neurons_i_hpc,neurons_i2_hpc,wii2_hpc,sparseness_int_ii2_hpc,GABA);
     con_ii2_hpc->set_name("II2_HPC");

     SparseBConnection * con_i2i_hpc = new SparseBConnection(neurons_i2_hpc,neurons_i_hpc,wi2i_hpc,sparseness_int_i2i_hpc,GABA);
     con_i2i_hpc->set_name("I2I_HPC");
   #endif

   #ifndef CON_STIM_HPC_NONE
    #ifdef HAS_CROSS_REGION_EI_CON
     STPBConnection * con_ei_stim_hpc = new STPBConnection( stimgroup,
							    neurons_i_hpc, 
							    wext_ei_hpc, 
							    sparseness_ext_hpc, 
							    GLUT);
     con_ei_stim_hpc->set_tau_d(taud_ei_stim_hpc);
     con_ei_stim_hpc->set_tau_f(tauf_ei_stim_hpc);
     con_ei_stim_hpc->set_ujump(ujump_ei_stim_hpc);
     con_ei_stim_hpc->set_name("EI_STIM->HPC");
    #endif
   #endif

   #ifdef CON_STIM_HPC_SPARSEB
    SparseBConnection * con_ee_stim_hpc = new SparseBConnection(stimgroup, neurons_e_hpc,
							wext_hpc, sparseness_ext_hpc, GLUT);
    con_ee_stim_hpc->set_name("EE_STIM->HPC");
   #endif

   #ifdef CON_STIM_HPC_STPB
    STPBConnection * con_ee_stim_hpc = new STPBConnection(stimgroup,neurons_e_hpc,wext_hpc,sparseness_ext_hpc,GLUT);
    con_ee_stim_hpc->set_tau_d(taud_ee_stim_hpc);
    con_ee_stim_hpc->set_tau_f(tauf_ee_stim_hpc);
    con_ee_stim_hpc->set_ujump(ujump_ee_stim_hpc);
    con_ee_stim_hpc->set_name("EE_STIM->HPC");
   #endif

   #ifdef CON_STIM_HPC_P11B
    P11BConnection * con_ee_stim_hpc = NULL;
    con_ee_stim_hpc = new P11BConnection( stimgroup, neurons_e_hpc,
				    wext_hpc,sparseness_ext_hpc,
				    eta_stim_hpc,
				    kappa_hpc, // supposedly deprecated
				    wmax_exc_hpc,
				    GLUT
				    );
    con_ee_stim_hpc->set_weight_a(weight_a_hpc);
    con_ee_stim_hpc->set_weight_c(weight_c_hpc);
    con_ee_stim_hpc->set_tau_d(taud_ee_stim_hpc);
    con_ee_stim_hpc->set_tau_f(tauf_ee_stim_hpc);
    con_ee_stim_hpc->set_ujump(ujump_ee_stim_hpc);
    con_ee_stim_hpc->set_beta(beta_stim_hpc*eta_stim_hpc);
    con_ee_stim_hpc->delta = delta_hpc*eta_stim_hpc;
    con_ee_stim_hpc->set_min_weight(wmin_exc_hpc);
    if ( noisy_initial_weights ) 
      con_ee_stim_hpc->random_data(wext_hpc, wext_hpc);
    con_ee_stim_hpc->set_name("EE_STIM->HPC");
    con_ee_stim_hpc->consolidation_active = consolidation;
    con_ee_stim_hpc->pot_strength = pot_strength_hpc/normalization_factor_hpc;
    con_ee_stim_hpc->set_tau_hom(tauh_ee_stim_hpc);
    con_ee_stim_hpc->set_tau_con(tauc_ee_stim_hpc);
    if ( consolidate_initial_weights )
      con_ee_stim_hpc->consolidate();
   #endif

   #ifdef HAS_CTX
    
     #ifndef CON_CTX_HPC_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
        STPBConnection * con_ei_ctx_hpc = new STPBConnection(neurons_e_ctx,neurons_i_hpc,wei_ctx_hpc,sparseness_ctx_hpc,GLUT);
	con_ei_ctx_hpc->set_tau_d(taud_ctx_hpc);
	con_ei_ctx_hpc->set_tau_f(tauf_ctx_hpc);
	con_ei_ctx_hpc->set_ujump(ujump_ctx_hpc);
	con_ei_ctx_hpc->set_name("EI_CTX->HPC");
       #endif
     #endif
     
     #ifdef CON_CTX_HPC_SPARSEB
      SparseBConnection * con_ee_ctx_hpc = new SparseBConnection(neurons_e_ctx,neurons_e_hpc,wee_ctx_hpc,sparseness_ctx_hpc,GLUT);
      con_ee_ctx_hpc->set_name("EE_CTX->HPC");
     #endif

     #ifdef CON_CTX_HPC_STPB
      STPBConnection * con_ee_ctx_hpc = new STPBConnection(neurons_e_ctx,neurons_e_hpc,wee_ctx_hpc,sparseness_ctx_hpc,GLUT);
      con_ee_ctx_hpc->set_tau_d(taud_ctx_hpc);
      con_ee_ctx_hpc->set_tau_f(tauf_ctx_hpc);
      con_ee_ctx_hpc->set_ujump(ujump_ctx_hpc);
      con_ee_ctx_hpc->set_name("EE_CTX->HPC");
     #endif

     #ifdef CON_CTX_HPC_P11B
      P11BConnection * con_ee_ctx_hpc;
      con_ee_ctx_hpc = new P11BConnection(neurons_e_ctx,neurons_e_hpc,
					 wee_ctx_hpc,sparseness_ctx_hpc,
					 eta_ctx_hpc,
					 kappa_ctx_hpc,
					 wmax_ctx_hpc
					 );
      con_ee_ctx_hpc->set_transmitter(AMPA);
      con_ee_ctx_hpc->set_name("EE_CTX->HPC");
      con_ee_ctx_hpc->set_weight_a(weight_a_ctx_hpc); 
      con_ee_ctx_hpc->set_weight_c(weight_c_ctx_hpc); 
      con_ee_ctx_hpc->consolidation_active = consolidation;
      double wtmax_ctx_hpc = 1.0/4*(weight_c_ctx_hpc-weight_a_ctx_hpc);
      double normalization_factor_ctx_hpc = (wtmax_ctx_hpc-weight_a_ctx_hpc)*(wtmax_ctx_hpc-(weight_a_ctx_hpc+weight_c_ctx_hpc)/2)*(wtmax_ctx_hpc-weight_c_ctx_hpc); 
      con_ee_ctx_hpc->pot_strength = pot_strength_ctx_hpc/normalization_factor_ctx_hpc;
      logger->parameter("normalized pot_strength_ctx_hpc", con_ee_ctx_hpc->pot_strength);
      if ( noisy_initial_weights ) 
	con_ee_ctx_hpc->random_data(wee_ctx_hpc, wee_ctx_hpc);
      if ( consolidate_initial_weights )
	con_ee_ctx_hpc->consolidate();	
      con_ee_ctx_hpc->set_tau_d(taud_ctx_hpc);
      con_ee_ctx_hpc->set_tau_f(tauf_ctx_hpc);
      con_ee_ctx_hpc->set_ujump(ujump_ctx_hpc);
      con_ee_ctx_hpc->set_beta(beta_ctx_hpc*eta_ctx_hpc);
      con_ee_ctx_hpc->delta = delta_ctx_hpc*eta_ctx_hpc;
      con_ee_ctx_hpc->set_min_weight(wmin_ctx_hpc);
      con_ee_ctx_hpc->set_tau_hom(tauh_ctx_hpc);
      con_ee_ctx_hpc->set_tau_con(tauc_ctx_hpc);
     #endif

     #ifndef CON_HPC_CTX_NONE
       #ifdef HAS_CROSS_REGION_EI_CON
        STPBConnection * con_ei_hpc_ctx = new STPBConnection(neurons_e_hpc,neurons_i_ctx,wei_hpc_ctx,sparseness_hpc_ctx,GLUT);
	con_ei_hpc_ctx->set_tau_d(taud_hpc_ctx);
	con_ei_hpc_ctx->set_tau_f(tauf_hpc_ctx);
	con_ei_hpc_ctx->set_ujump(ujump_hpc_ctx);
	con_ei_hpc_ctx->set_name("EI_HPC->CTX");
       #endif
     #endif
     
     #ifdef CON_HPC_CTX_SPARSEB
      SparseBConnection * con_ee_hpc_ctx = new SparseBConnection(neurons_e_hpc,neurons_e_ctx,wee_hpc_ctx,sparseness_hpc_ctx,GLUT);
      con_ee_hpc_ctx->set_name("EE_HPC->CTX");
     #endif

     #ifdef CON_HPC_CTX_STPB
      STPBConnection * con_ee_hpc_ctx = new STPBConnection(neurons_e_hpc,neurons_e_ctx,wee_hpc_ctx,sparseness_hpc_ctx,GLUT);
      con_ee_hpc_ctx->set_tau_d(taud_hpc_ctx);
      con_ee_hpc_ctx->set_tau_f(tauf_hpc_ctx);
      con_ee_hpc_ctx->set_ujump(ujump_hpc_ctx);
      con_ee_hpc_ctx->set_name("EE_HPC->CTX");
     #endif

     #ifdef CON_HPC_CTX_P11B
      P11BConnection * con_ee_hpc_ctx;
      con_ee_hpc_ctx = new P11BConnection(neurons_e_hpc,neurons_e_ctx,
					 wee_hpc_ctx,sparseness_hpc_ctx,
					 eta_hpc_ctx,
					 kappa_hpc_ctx,
					 wmax_hpc_ctx
					 );
      con_ee_hpc_ctx->set_transmitter(AMPA);
      con_ee_hpc_ctx->set_name("EE_HPC->CTX");
      con_ee_hpc_ctx->set_weight_a(weight_a_hpc_ctx); 
      con_ee_hpc_ctx->set_weight_c(weight_c_hpc_ctx); 
      con_ee_hpc_ctx->consolidation_active = consolidation;
      double wtmax_hpc_ctx = 1.0/4*(weight_c_hpc_ctx-weight_a_hpc_ctx);
      double normalization_factor_hpc_ctx = (wtmax_hpc_ctx-weight_a_hpc_ctx)*(wtmax_hpc_ctx-(weight_a_hpc_ctx+weight_c_hpc_ctx)/2)*(wtmax_hpc_ctx-weight_c_hpc_ctx); 
      con_ee_hpc_ctx->pot_strength = pot_strength_hpc_ctx/normalization_factor_hpc_ctx;
      logger->parameter("normalized pot_strength_hpc_ctx", con_ee_hpc_ctx->pot_strength);
      if ( noisy_initial_weights ) 
	con_ee_hpc_ctx->random_data(wee_hpc_ctx, wee_hpc_ctx);
      if ( consolidate_initial_weights )
	con_ee_hpc_ctx->consolidate();	
      con_ee_hpc_ctx->set_tau_d(taud_hpc_ctx);
      con_ee_hpc_ctx->set_tau_f(tauf_hpc_ctx);
      con_ee_hpc_ctx->set_ujump(ujump_hpc_ctx);
      con_ee_hpc_ctx->set_beta(beta_hpc_ctx*eta_hpc_ctx);
      con_ee_hpc_ctx->delta = delta_hpc_ctx*eta_hpc_ctx;
      con_ee_hpc_ctx->set_min_weight(wmin_hpc_ctx);
      con_ee_hpc_ctx->set_tau_hom(tauh_hpc_ctx);
      con_ee_hpc_ctx->set_tau_con(tauc_hpc_ctx);
     #endif
   #endif

   #ifdef HAS_REPLAY
    #ifdef CON_REPLAY_HPC_SPARSEB
     SparseBConnection * con_ee_rep_hpc = new SparseBConnection(repgroup, neurons_e_hpc,
								w_rep_hpc, sparseness_rep_hpc,
								GLUT);
     con_ee_rep_hpc->set_name("EE_REPLAY->HPC");

     SparseBConnection * con_ei_rep_hpc = new SparseBConnection(repgroup, neurons_i_hpc,
								w_rep_hpc, sparseness_rep_hpc,
								GLUT);
     con_ei_rep_hpc->set_name("EI_REPLAY->HPC");
     
    #endif 
   #endif

   #ifdef HAS_BACKGROUND
    SparseBConnection * con_ee_bg_hpc = new SparseBConnection(bggroup_hpc, neurons_e_hpc,
							      w_bg_hpc, sparseness_bg_hpc,
							      GLUT);
    con_ee_bg_hpc->set_name("EE_BG->HPC");

    #ifdef HAS_CROSS_REGION_EI_CON

     STPBConnection * con_ei_bg_hpc = new STPBConnection(bggroup_hpc, 
							 neurons_i_hpc, 
							 w_bg_ei_hpc, 
							 sparseness_bg_hpc, 
							 GLUT);
     con_ei_bg_hpc->set_tau_d(taud_ei_stim_hpc);
     con_ei_bg_hpc->set_tau_f(tauf_ei_stim_hpc);
     con_ei_bg_hpc->set_ujump(ujump_ei_stim_hpc);
     con_ei_bg_hpc->set_name("EI_BG->HPC");
     
    #endif
   #endif
  #endif
    

  // set up stimulus protocol
  if (!stimfile.empty()) {
    logger->msg("Setting up stimulus ...",PROGRESS,true);
    stimgroup->load_patterns(stimfile.c_str());
    stimgroup->set_next_action_time(10); // let network settle for some time    
    // gives the first 3 patterns half of the probability
    if ( preferred > 0 ) { 
      std::vector<double> dist = stimgroup->get_distribution();
      int r = preferred;
      for ( int i = 0 ; i < dist.size() ; ++i ) {
	if ( i == r ) 
	  dist[i] = 1.0;
	else
	  dist[i] = 1.0/dist.size();
      }
      stimgroup->set_distribution(dist);
    }
  }

  #ifdef HAS_REPLAY
   // set up replay protocol
   if (!repfile.empty()) {
     logger->msg("Setting up replay ...",PROGRESS,true);
     repgroup->load_patterns(repfile.c_str());
     repgroup->set_next_action_time(10); // let network settle for some time    
     // gives the first 3 patterns half of the probability
     if ( preferred > 0 ) { 
       std::vector<double> dist = repgroup->get_distribution();
       int r = preferred;
       for ( int i = 0 ; i < dist.size() ; ++i ) {
	 if ( i == r ) 
	   dist[i] = 1.0;
	 else
	   dist[i] = 1.0/dist.size();
       }
       repgroup->set_distribution(dist);
     }
   }
  #endif

  #ifdef HAS_BACKGROUND
   
   #ifdef HAS_THL
    // set up thl background protocol
    if (!bgfile_thl.empty()) {
      logger->msg("Setting up thl background ...",PROGRESS,true);
      bggroup_thl->load_patterns(bgfile_thl.c_str());
      bggroup_thl->set_next_action_time(10); // let network settle for some time    
      // gives the first 3 patterns half of the probability
      if ( preferred > 0 ) { 
        std::vector<double> dist = bggroup_thl->get_distribution();
        int r = preferred;
        for ( int i = 0 ; i < dist.size() ; ++i ) {
	  if ( i == r ) 
	    dist[i] = 1.0;
	  else
	    dist[i] = 1.0/dist.size();
        }
        bggroup_thl->set_distribution(dist);
      }
    }
   #endif

   #ifdef HAS_CTX
    // set up ctx background protocol
    if (!bgfile_ctx.empty()) {
      logger->msg("Setting up ctx background ...",PROGRESS,true);
      bggroup_ctx->load_patterns(bgfile_ctx.c_str());
      bggroup_ctx->set_next_action_time(10); // let network settle for some time    
      // gives the first 3 patterns half of the probability
      if ( preferred > 0 ) { 
        std::vector<double> dist = bggroup_ctx->get_distribution();
        int r = preferred;
        for ( int i = 0 ; i < dist.size() ; ++i ) {
	  if ( i == r ) 
	    dist[i] = 1.0;
	  else
	    dist[i] = 1.0/dist.size();
        }
        bggroup_ctx->set_distribution(dist);
      }
    }
   #endif

   #ifdef HAS_HPC
    // set up hpc background protocol
    if (!bgfile_hpc.empty()) {
      logger->msg("Setting up hpc background ...",PROGRESS,true);
      bggroup_hpc->load_patterns(bgfile_hpc.c_str());
      bggroup_hpc->set_next_action_time(10); // let network settle for some time    
      // gives the first 3 patterns half of the probability
      if ( preferred > 0 ) { 
        std::vector<double> dist = bggroup_hpc->get_distribution();
        int r = preferred;
        for ( int i = 0 ; i < dist.size() ; ++i ) {
	  if ( i == r ) 
	    dist[i] = 1.0;
	  else
	    dist[i] = 1.0/dist.size();
        }
        bggroup_hpc->set_distribution(dist);
      }
    }
   #endif
   
  #endif

   
  /*
  if (stim_spike_mon)
    devices_hm.push_back(smon_s);

  #ifndef CON_STIM_CTX_NONE
   if (weightstat_mon_ee_stim_ctx)
     devices_hm.push_back(ee_stim_ctx_mon_ws);

   if (weight_mon_ee_stim_ctx)
     devices_hm.push_back(ee_stim_ctx_mon_w);
  #endif

  if ( !monfile_ctx.empty() && exc_pattern_mon_ctx)
    devices_hm.push_back(exc_ctx_mon_pat);

  if (ei_weight_mon_ctx)
    devices_hm.push_back(ei_ctx_mon_w);

  if (weight_mon_ee_ctx)
    devices_hm.push_back(ee_ctx_mon_w);

  if (ee_hom_mon_ctx)
    devices_hm.push_back(ee_ctx_mon_hom);

  if (exc_voltage_mon_ctx)
    devices_hm.push_back(exc_ctx_mon_mem);

  if (inh_voltage_mon_ctx)
    devices_hm.push_back(inh_ctx_mon_mem);

  if (ie_weight_mon_ctx)
    devices_hm.push_back(ie_ctx_mon_w);

  if (ii_weight_mon_ctx)
    devices_hm.push_back(ii_ctx_mon_w);

  if (weightstat_mon_ee_ctx)
    devices_hm.push_back(ee_ctx_mon_ws);

  if (ie_weightstat_mon_ctx)
    devices_hm.push_back(ie_ctx_mon_ws);

  if ( !monfile_ctx.empty() && weightpat_mon_ee_ctx)
    devices_hm.push_back(ee_ctx_mon_wp);

  #ifndef CON_STIM_CTX_NONE
   if ( !premonfile_ctx.empty() && !monfile_ctx.empty() && weightpat_mon_ee_stim_ctx)
     devices_hm.push_back(stim_ctx_mon_wp);
  #endif

  if (exc_spike_mon_ctx)
    devices_hm.push_back(exc_ctx_mon_sp);

  if (inh_spike_mon_ctx)
    devices_hm.push_back(inh_ctx_mon_sp);

  if (exc_prate_mon_ctx)
    devices_hm.push_back(exc_ctx_mon_pr);

  if (inh_prate_mon_ctx)
    devices_hm.push_back(inh_ctx_mon_pr);


  if (exc_ratechk_ctx)
    checkers_hm.push_back(exc_ctx_rchk);
  */

  // load preset configurations
  // load network state if appropriate
  if (!load_file.empty()) {
    if (!load_without_hpc) {
      logger->msg("Loading from file ...",PROGRESS,true);
      sys->load_network_state(load_file.c_str());
    }
    else {
      bool has_hm_net = false;
      
      #ifdef HAS_CTX
       has_hm_net = true;
      #endif

      #ifdef HAS_THL
       has_hm_net = true;
      #endif

      if (has_hm_net) {
       auryn::logger->msg("Loading network state without hpc", NOTIFICATION);

       std::string netstate_filename;
       {
	 sprintf(strbuf, "%s", load_file.c_str());
	 string basename = strbuf;
	 std::stringstream oss;
	 oss << basename
	     << "." << sys->mpi_rank()
	     << ".netstate";
	 netstate_filename = oss.str();
       } // oss goes out of scope

       std::ifstream ifs(netstate_filename.c_str());

       if ( !ifs.is_open() ) {
	 std::stringstream oss;
	 oss << "Error opening netstate file: "
	     << netstate_filename;
	 auryn::logger->msg(oss.str(),ERROR);
	 throw AurynOpenFileException();
       }

       boost::archive::binary_iarchive ia(ifs);

       // verify simulator version information 
       bool pass_version = true;
       int tmp_version;
       ia >> tmp_version;
       pass_version = pass_version && sys->build.version==tmp_version;
       ia >> tmp_version;
       pass_version = pass_version && sys->build.subversion==tmp_version;
       ia >> tmp_version;
       pass_version = pass_version && sys->build.revision_number==tmp_version;

       if ( !pass_version ) {
	 auryn::logger->msg("WARNING: Version check failed! Current Auryn version " 
			    "does not match the version which created the file. "
			    "This could pose a problem. " 
			    "Proceed with caution!" ,WARNING);
       }

       // verify communicator information 
       bool pass_comm = true;
       unsigned int tmp_int;
       ia >> tmp_int;
       pass_comm = pass_comm && (tmp_int == sys->mpi_size());
       ia >> tmp_int;
       pass_comm = pass_comm && (tmp_int == sys->mpi_rank());

       if ( !pass_comm ) {
	 auryn::logger->msg("ERROR: Communicator size or rank do not match! "
			    "Presumably you are trying to load the network "
			    "state netstate from a simulation which was run "
			    "on a different number of cores." ,ERROR);
       }

       /*
       auryn::logger->msg("Loading SpikingGroups ...",VERBOSE);

       {
       std::stringstream oss;
       oss << "Loading SpikingGroup:  neurons_e_ctx "
	   << neurons_e_ctx->get_name();
       auryn::logger->msg(oss.str(),VERBOSE);
       ia >> *neurons_e_ctx;
       }

       {
       std::stringstream oss;
       oss << "Loading SpikingGroup: neurons_i_ctx "
	   << neurons_i_ctx->get_name();
       auryn::logger->msg(oss.str(),VERBOSE);      
       ia >> *neurons_i_ctx;
       }

       {
       std::stringstream oss;
       oss << "Loading SpikingGroup: stimgroup " 
	   << stimgroup->get_name();
       auryn::logger->msg(oss.str(),VERBOSE);      
       ia >> *stimgroup;
       }
       */

       auryn::logger->msg("Loading Connections ...",VERBOSE);
       for ( unsigned int i = 0 ; i < connections_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Loading connection "
	     <<  i 
	     << ": " 
	     << connections_hm[i]->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);

	 ia >> *(connections_hm[i]);
	 connections_hm[i]->finalize();
       }

       auryn::logger->msg("Loading SpikingGroups ...",VERBOSE);

       #ifdef HAS_CTX
	{
	  std::string netstate_filename;
	  {
	    sprintf(strbuf, "%s_e_ctx", load_file.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ifstream ifs(netstate_filename.c_str());

	  if ( !ifs.is_open() ) {
	    std::stringstream oss;
	    oss << "Error opening netstate file: "
		<< netstate_filename;
	    auryn::logger->msg(oss.str(),ERROR);
	    throw AurynOpenFileException();
	  }

	  boost::archive::binary_iarchive ia(ifs);

	  std::stringstream oss;
	  oss << "Loading SpikingGroup:  neurons_e_ctx "
	      << neurons_e_ctx->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);
	  ia >> *neurons_e_ctx;

	  ifs.close();
	}

	{
	  std::string netstate_filename;
	  {
	    sprintf(strbuf, "%s_i_ctx", load_file.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ifstream ifs(netstate_filename.c_str());

	  if ( !ifs.is_open() ) {
	    std::stringstream oss;
	    oss << "Error opening netstate file: "
		<< netstate_filename;
	    auryn::logger->msg(oss.str(),ERROR);
	    throw AurynOpenFileException();
	  }

	  boost::archive::binary_iarchive ia(ifs);

	  std::stringstream oss;	
	  oss << "Loading SpikingGroup: neurons_i_ctx "
	      << neurons_i_ctx->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);      
	  ia >> *neurons_i_ctx;

	  ifs.close();
	}
       #endif

       #ifdef HAS_THL
	{
	  std::string netstate_filename;
	  {
	    sprintf(strbuf, "%s_e_thl", load_file.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ifstream ifs(netstate_filename.c_str());

	  if ( !ifs.is_open() ) {
	    std::stringstream oss;
	    oss << "Error opening netstate file: "
		<< netstate_filename;
	    auryn::logger->msg(oss.str(),ERROR);
	    throw AurynOpenFileException();
	  }

	  boost::archive::binary_iarchive ia(ifs);

	  std::stringstream oss;
	  oss << "Loading SpikingGroup:  neurons_e_thl "
	      << neurons_e_thl->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);
	  ia >> *neurons_e_thl;

	  ifs.close();
	}

	{
	  std::string netstate_filename;
	  {
	    sprintf(strbuf, "%s_i_thl", load_file.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ifstream ifs(netstate_filename.c_str());

	  if ( !ifs.is_open() ) {
	    std::stringstream oss;
	    oss << "Error opening netstate file: "
		<< netstate_filename;
	    auryn::logger->msg(oss.str(),ERROR);
	    throw AurynOpenFileException();
	  }

	  boost::archive::binary_iarchive ia(ifs);

	  std::stringstream oss;	
	  oss << "Loading SpikingGroup: neurons_i_thl "
	      << neurons_i_thl->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);      
	  ia >> *neurons_i_thl;

	  ifs.close();
	}
       #endif

       {
	 std::string netstate_filename;
	 {
	   sprintf(strbuf, "%s_stim", load_file.c_str());
	   string basename = strbuf;
	   std::stringstream oss;
	   oss << basename
	       << "." << sys->mpi_rank()
	       << ".netstate";
	   netstate_filename = oss.str();
	 } // oss goes out of scope

	 std::ifstream ifs(netstate_filename.c_str());

	 if ( !ifs.is_open() ) {
	   std::stringstream oss;
	   oss << "Error opening netstate file: "
	       << netstate_filename;
	   auryn::logger->msg(oss.str(),ERROR);
	   throw AurynOpenFileException();
	 }

	 boost::archive::binary_iarchive ia(ifs);

	 std::stringstream oss;
	 oss << "Loading SpikingGroup: stimgroup " 
	     << stimgroup->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);      
	 ia >> *stimgroup;

	 ifs.close();
       }

       /*
       for ( unsigned int i = 0 ; i < spiking_groups_hm.size() ; ++i ) {
	 std::stringstream oss;
	 oss << "Loading group "
	     <<  i 
	     << ": "
	     << spiking_groups_hm[i]->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);

	 ia >> *(spiking_groups_hm[i]);
	 std::cout << "\nloaded hm spiking group " << spiking_groups_hm[i]->get_name()
		   << " from " << netstate_filename << "\n";
       }
       */

       /*
       // Loading Devices states
       auryn::logger->msg("Loading Devices ...",VERBOSE);
       for ( unsigned int i = 0 ; i < devices_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Loading Device "
	     <<  i;
	 auryn::logger->msg(oss.str(),VERBOSE);

	 ia >> *(devices_hm[i]);
       }

       auryn::logger->msg("Loading Checkers ...",VERBOSE);
       for ( unsigned int i = 0 ; i < checkers_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Loading Checker "
	     <<  i;
	 auryn::logger->msg(oss.str(),VERBOSE);

	 ia >> *(checkers_hm[i]);
       }
       */

       ifs.close();
      }
      
    }
  }

  // load receptive fields and scale weights
  #ifdef HAS_THL
  
   #ifndef CON_STIM_THL_NONE
    if ( !recfile_stim_thl.empty() ) {

      #ifdef CON_STIM_THL_SPARSEB
       con_ee_stim_thl->load_from_complete_file(recfile_stim_thl);
      #endif

      #ifdef CON_STIM_THL_STPB
       con_ee_stim_thl->load_from_complete_file(recfile_stim_thl);
      #endif

      #ifdef CON_STIM_THL_P11B
       con_ee_stim_thl->load_fragile_matrix(recfile_stim_thl);
      #endif

    }

    if ( xi_stim_thl > -1.0 ) {

      con_ee_stim_thl->scale_all(xi_stim_thl);
      
    }
   #endif

   #ifdef HAS_CTX

    #ifndef CON_THL_CTX_NONE
     if ( !recfile_thl_ctx.empty() ) {

       #ifdef CON_THL_CTX_SPARSEB
	con_ee_thl_ctx->load_from_complete_file(recfile_thl_ctx);
       #endif

       #ifdef CON_THL_CTX_STPB
	con_ee_thl_ctx->load_from_complete_file(recfile_thl_ctx);
       #endif

       #ifdef CON_THL_CTX_P11B
	con_ee_thl_ctx->load_fragile_matrix(recfile_thl_ctx);
       #endif

     }

     if ( xi_thl_ctx > -1.0 ) {

       con_ee_thl_ctx->scale_all(xi_thl_ctx);
       
     }
    #endif

    #ifndef CON_CTX_THL_NONE
     if ( !recfile_ctx_thl.empty() ) {

       #ifdef CON_CTX_THL_SPARSEB
	con_ee_ctx_thl->load_from_complete_file(recfile_ctx_thl);
       #endif

       #ifdef CON_CTX_THL_STPB
	con_ee_ctx_thl->load_from_complete_file(recfile_ctx_thl);
       #endif

       #ifdef CON_CTX_THL_P11B
	con_ee_ctx_thl->load_fragile_matrix(recfile_ctx_thl);
       #endif

     }

     if ( xi_ctx_thl > -1.0 ) {

       con_ee_ctx_thl->scale_all(xi_ctx_thl);
       
     }
    #endif

   #endif

   #ifdef HAS_HPC

    #ifndef CON_THL_HPC_NONE
     if ( !recfile_thl_hpc.empty() ) {

       #ifdef CON_THL_HPC_SPARSEB
	con_ee_thl_hpc->load_from_complete_file(recfile_thl_hpc);
       #endif

       #ifdef CON_THL_HPC_STPB
	con_ee_thl_hpc->load_from_complete_file(recfile_thl_hpc);
       #endif

       #ifdef CON_THL_HPC_P11B
	con_ee_thl_hpc->load_fragile_matrix(recfile_thl_hpc);
       #endif

     }

     if ( xi_thl_hpc > -1.0 ) {

       con_ee_thl_hpc->scale_all(xi_thl_hpc);
       
     }
    #endif

    #ifndef CON_HPC_THL_NONE
     if ( !recfile_hpc_thl.empty() ) {

       #ifdef CON_HPC_THL_SPARSEB
	con_ee_hpc_thl->load_from_complete_file(recfile_hpc_thl);
       #endif

       #ifdef CON_HPC_THL_STPB
	con_ee_hpc_thl->load_from_complete_file(recfile_hpc_thl);
       #endif

       #ifdef CON_HPC_THL_P11B
	con_ee_hpc_thl->load_fragile_matrix(recfile_hpc_thl);
       #endif

     }

     if ( xi_hpc_thl > -1.0 ) {

       con_ee_hpc_thl->scale_all(xi_hpc_thl);
       
     }
    #endif

   #endif

   #ifdef HAS_REPLAY
    if ( !recfile_rep_thl.empty() ) {
       #ifdef CON_REPLAY_THL_SPARSEB
	con_ee_rep_thl->load_from_complete_file(recfile_rep_thl);
       #endif
    }
    if ( xi_rep_thl > -1.0 ) {
      #ifdef CON_REPLAY_THL_SPARSEB
	con_ee_rep_thl->scale_all(xi_rep_thl);
      #endif
    }
    
    if ( !recfile_ei_rep_thl.empty() ) {
      #ifdef CON_REPLAY_THL_SPARSEB
	con_ei_rep_thl->load_from_complete_file(recfile_ei_rep_thl);
      #endif
    }
    if ( xi_ei_rep_thl > -1.0 ) {
      #ifdef CON_REPLAY_THL_SPARSEB
	con_ei_rep_thl->scale_all(xi_ei_rep_thl);
      #endif
    }
   #endif

   #ifdef HAS_BACKGROUND
    if ( !recfile_bg_thl.empty() ) {
       con_ee_bg_thl->load_from_complete_file(recfile_bg_thl);
    }
    if ( xi_bg_thl > -1.0 ) {
       con_ee_bg_thl->scale_all(xi_bg_thl);
    }
    
    #ifdef HAS_CROSS_REGION_EI_CON
     if ( !recfile_ei_bg_thl.empty() ) {
        con_ei_bg_thl->load_from_complete_file(recfile_ei_bg_thl);
     }
     if ( xi_ei_bg_thl > -1.0 ) {
        con_ei_bg_thl->scale_all(xi_ei_bg_thl);
     }
    #endif
   #endif

  #endif

  #ifdef HAS_HPC

   #ifndef CON_STIM_HPC_NONE
    if ( !recfile_stim_hpc.empty() ) {

      #ifdef CON_STIM_HPC_SPARSEB
       con_ee_stim_hpc->load_from_complete_file(recfile_stim_hpc);
      #endif

      #ifdef CON_STIM_HPC_STPB
       con_ee_stim_hpc->load_from_complete_file(recfile_stim_hpc);
      #endif

      #ifdef CON_STIM_HPC_P11B
       con_ee_stim_hpc->load_fragile_matrix(recfile_stim_hpc);
      #endif

    }

    if ( xi_stim_hpc > -1.0 ) {

      con_ee_stim_hpc->scale_all(xi_stim_hpc);
      
    }
   #endif

   #ifdef HAS_REPLAY
    if ( !recfile_rep_hpc.empty() ) {
       #ifdef CON_REPLAY_HPC_SPARSEB
	con_ee_rep_hpc->load_from_complete_file(recfile_rep_hpc);
       #endif
    }
    if ( xi_rep_hpc > -1.0 ) {
       #ifdef CON_REPLAY_HPC_SPARSEB
	con_ee_rep_hpc->scale_all(xi_rep_hpc);
       #endif
    }

    if ( !recfile_ei_rep_hpc.empty() ) {
       #ifdef CON_REPLAY_HPC_SPARSEB
	con_ei_rep_hpc->load_from_complete_file(recfile_ei_rep_hpc);
       #endif
    }
    if ( xi_ei_rep_hpc > -1.0 ) {
      #ifdef CON_REPLAY_HPC_SPARSEB
	con_ei_rep_hpc->scale_all(xi_ei_rep_hpc);
      #endif
    }
   #endif

   #ifdef HAS_BACKGROUND
    if ( !recfile_bg_hpc.empty() ) {
       con_ee_bg_hpc->load_from_complete_file(recfile_bg_hpc);
    }
    if ( xi_bg_hpc > -1.0 ) {
       con_ee_bg_hpc->scale_all(xi_bg_hpc);
    }

    #ifdef HAS_CROSS_REGION_EI_CON
     if ( !recfile_ei_bg_hpc.empty() ) {
        con_ei_bg_hpc->load_from_complete_file(recfile_ei_bg_hpc);
     }
     if ( xi_ei_bg_hpc > -1.0 ) {
        con_ei_bg_hpc->scale_all(xi_ei_bg_hpc);
     }
    #endif
   #endif

   #ifdef HAS_CTX
    #ifndef CON_CTX_HPC_NONE
     if ( !recfile_ctx_hpc.empty() ) {

       #ifdef CON_CTX_HPC_SPARSEB
	con_ee_ctx_hpc->load_from_complete_file(recfile_ctx_hpc);
       #endif

       #ifdef CON_CTX_HPC_STPB
	con_ee_ctx_hpc->load_from_complete_file(recfile_ctx_hpc);
       #endif

       #ifdef CON_CTX_HPC_P11B
	con_ee_ctx_hpc->load_fragile_matrix(recfile_ctx_hpc);
       #endif

     }

     if ( xi_ctx_hpc > -1.0 ) {

       con_ee_ctx_hpc->scale_all(xi_ctx_hpc);
       
     }
    #endif

    #ifndef CON_HPC_CTX_NONE
     if ( !recfile_hpc_ctx.empty() ) {

       #ifdef CON_HPC_CTX_SPARSEB
	con_ee_hpc_ctx->load_from_complete_file(recfile_hpc_ctx);
       #endif

       #ifdef CON_HPC_CTX_STPB
	con_ee_hpc_ctx->load_from_complete_file(recfile_hpc_ctx);
       #endif

       #ifdef CON_HPC_CTX_P11B
	con_ee_hpc_ctx->load_fragile_matrix(recfile_hpc_ctx);
       #endif

     }

     if ( xi_hpc_ctx > -1.0 ) {

       con_ee_hpc_ctx->scale_all(xi_hpc_ctx);
       
     }
    #endif
   #endif
  #endif

  #ifdef HAS_CTX
     
   #ifndef CON_STIM_CTX_NONE
    if ( !recfile_stim_ctx.empty() ) {

      #ifdef CON_STIM_CTX_SPARSEB
       con_ee_stim_ctx->load_from_complete_file(recfile_stim_ctx);
      #endif

      #ifdef CON_STIM_CTX_STPB
       con_ee_stim_ctx->load_from_complete_file(recfile_stim_ctx);
      #endif

      #ifdef CON_STIM_CTX_P11B
       con_ee_stim_ctx->load_fragile_matrix(recfile_stim_ctx);
      #endif

    }

    if ( xi_stim_ctx > -1.0 ) {

      con_ee_stim_ctx->scale_all(xi_stim_ctx);
      
    }
   #endif

   #ifdef HAS_REPLAY
    if ( !recfile_rep_ctx.empty() ) {
       #ifdef CON_REPLAY_CTX_SPARSEB
	con_ee_rep_ctx->load_from_complete_file(recfile_rep_ctx);
       #endif
    }
    if ( xi_rep_ctx > -1.0 ) {
       #ifdef CON_REPLAY_CTX_SPARSEB
	con_ee_rep_ctx->scale_all(xi_rep_ctx);
       #endif
    }

    if ( !recfile_ei_rep_ctx.empty() ) {
      #ifdef CON_REPLAY_CTX_SPARSEB
	con_ei_rep_ctx->load_from_complete_file(recfile_ei_rep_ctx);
      #endif
    }
    if ( xi_ei_rep_ctx > -1.0 ) {
      #ifdef CON_REPLAY_CTX_SPARSEB
	con_ei_rep_ctx->scale_all(xi_ei_rep_ctx);
      #endif
    }
   #endif

   #ifdef HAS_BACKGROUND
    if ( !recfile_bg_ctx.empty() ) {
       con_ee_bg_ctx->load_from_complete_file(recfile_bg_ctx);
    }
    if ( xi_bg_ctx > -1.0 ) {
       con_ee_bg_ctx->scale_all(xi_bg_ctx);
    }

    #ifdef HAS_CROSS_REGION_EI_CON
     if ( !recfile_ei_bg_ctx.empty() ) {
        con_ei_bg_ctx->load_from_complete_file(recfile_ei_bg_ctx);
     }
     if ( xi_ei_bg_ctx > -1.0 ) {
        con_ei_bg_ctx->scale_all(xi_ei_bg_ctx);
     }
    #endif
   #endif
   
  #endif

  #ifdef HAS_CTX
   // load ctx_ee connection pattern if appropriate
   if ( !prefile_ctx.empty() && chi_ctx > 0.0 ) {
     con_ee_ctx->patterns_ignore_gamma = true;
     con_ee_ctx->load_patterns(prefile_ctx,chi_ctx);
   }

   #ifndef CON_STIM_CTX_NONE
    // load stim_ctx connection pattern if appropriate
    if ( !prefile_ctx.empty() && xi_stim_ctx > -1.0 ) {
      con_ee_stim_ctx->patterns_ignore_gamma = true;
      con_ee_stim_ctx->load_patterns(prefile_ctx,xi_stim_ctx,false);

      #ifdef CON_STIM_CTX_P11B
       if ( consolidate_initial_weights )
	 con_ee_stim_ctx->consolidate();
      #endif
    }
   #endif
  #endif
  
  #ifdef HAS_HPC
   #ifdef HAS_REC_EE
    // load hpc_ee connection pattern if appropriate
    if ( !prefile_hpc.empty() && chi_hpc > 0.0 ) {
      con_ee_hpc->patterns_ignore_gamma = true;
      con_ee_hpc->load_patterns(prefile_hpc,chi_hpc);
    }
   #endif

   #ifndef CON_STIM_HPC_NONE
    // load stim_hpc connection pattern if appropriate
    if ( !prefile_hpc.empty() && xi_stim_hpc > -1.0 ) {
      con_ee_stim_hpc->patterns_ignore_gamma = true;
      con_ee_stim_hpc->load_patterns(prefile_hpc,xi_stim_hpc,false);

      #ifdef CON_STIM_HPC_P11B
       if ( consolidate_initial_weights )
	 con_ee_stim_hpc->consolidate();
      #endif
    }
   #endif
  #endif

  #ifdef HAS_THL
   // load thl_ee connection pattern if appropriate
   if ( !prefile_thl.empty() && chi_thl > 0.0 ) {
     con_ee_thl->patterns_ignore_gamma = true;
     con_ee_thl->load_patterns(prefile_thl,chi_thl);
   }

   #ifndef CON_STIM_THL_NONE
    // load stim_thl connection pattern if appropriate
    if ( !prefile_thl.empty() && xi_stim_thl > -1.0 ) {
      con_ee_stim_thl->patterns_ignore_gamma = true;
      con_ee_stim_thl->load_patterns(prefile_thl,xi_stim_thl,false);

      #ifdef CON_STIM_THL_P11B
       if ( consolidate_initial_weights )
	 con_ee_stim_thl->consolidate();
      #endif
    }
   #endif
  #endif

  // set up monitors
  if (stim_spike_mon) {
    sprintf(strbuf, "%s/%s.%d.s.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    BinarySpikeMonitor * smon_s = new BinarySpikeMonitor( stimgroup,
							  string(strbuf),
							  exc_size_ctx );
  }

  #ifdef HAS_REPLAY
   if (rep_spike_mon) {
     sprintf(strbuf, "%s/%s.%d.r.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     BinarySpikeMonitor * rmon_s = new BinarySpikeMonitor( repgroup,
							   string(strbuf),
							   size_rep );
   }
  #endif

  #ifdef HAS_BACKGROUND
   if (bg_spike_mon) {
     
     #ifdef HAS_THL
      sprintf(strbuf, "%s/%s.%d.bg.thl.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * bg_thl_mon_s = new BinarySpikeMonitor( bggroup_thl,
								  string(strbuf),
								  exc_size_thl );
     #endif

     #ifdef HAS_CTX
      sprintf(strbuf, "%s/%s.%d.bg.ctx.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * bg_ctx_mon_s = new BinarySpikeMonitor( bggroup_ctx,
								  string(strbuf),
								  exc_size_ctx );
     #endif

     #ifdef HAS_HPC
      sprintf(strbuf, "%s/%s.%d.bg.hpc.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * bg_hpc_mon_s = new BinarySpikeMonitor( bggroup_hpc,
								  string(strbuf),
								  exc_size_hpc );
     #endif
   }
  #endif

  #ifdef HAS_THL
    
    #ifndef CON_STIM_THL_NONE
     if (weightstat_mon_ee_stim_thl) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_thl.ws", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightStatsMonitor * ee_stim_thl_mon_ws = new WeightStatsMonitor( con_ee_stim_thl, string(strbuf) );
     }

     if (weight_mon_ee_stim_thl) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_thl.w", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightMonitor * ee_stim_thl_mon_w = new WeightMonitor( con_ee_stim_thl, string(strbuf), 1.0 ); 
       ee_stim_thl_mon_w->add_equally_spaced(50);
       if (!monfile_thl.empty())
	 if (!stimfile.empty()) 
	   ee_stim_thl_mon_w->load_pattern_connections(stimfile,monfile_thl,20,20,ASSEMBLIES_ONLY); // true for assemblies only
	 else 
	   ee_stim_thl_mon_w->load_pattern_connections(monfile_thl,20,20,ASSEMBLIES_ONLY); // true for assemblies only
     }
    #endif

    if ( !monfile_thl.empty() && exc_pattern_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.pact", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PatternMonitor * exc_thl_mon_pat = new PatternMonitor( neurons_e_thl, string(strbuf) , monfile_thl.c_str(), 100);
    }

    if (ei_weight_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.ei.thl.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ei_thl_mon_w = new WeightMonitor( con_ei_thl, 0, 100, strbuf, 1.0, DATARANGE);
    }

    if (weight_mon_ee_thl) {
      sprintf(strbuf, "%s/%s.%d.ee.thl.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ee_thl_mon_w = new WeightMonitor( con_ee_thl, string(strbuf), 1.0);
      ee_thl_mon_w->add_equally_spaced(50);
      if ( !monfile_thl.empty() ) 
	ee_thl_mon_w->load_pattern_connections(monfile_thl,10,10,ASSEMBLIES_ONLY); // true for assemblies only
    }

    if (ee_hom_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.ee.thl.hom", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_hom = new StateMonitor( con_ee_thl->hom, record_neuron_exc_thl, string(strbuf), 1 );
    }

    if (exc_g_ampa_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.gampa", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_g_ampa = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "g_ampa", string(strbuf) );
    }

    if (exc_g_nmda_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.gnmda", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_g_nmda = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "g_nmda", string(strbuf) );
    }

    if (exc_g_gaba_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.ggaba", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_g_gaba = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "g_gaba", string(strbuf) );
    }

    if (exc_g_adapt1_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.gadapt1", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_g_adapt1 = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "g_adapt1", string(strbuf) );
    }

    if (exc_g_adapt2_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.gadapt2", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_g_adapt2 = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "g_adapt2", string(strbuf) );
    }

    if (exc_thr_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.thr", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_thl_mon_thr = new StateMonitor( neurons_e_thl, record_neuron_exc_thl, "thr", string(strbuf) );
    }

    if (exc_voltage_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      VoltageMonitor * exc_thl_mon_mem = new VoltageMonitor( neurons_e_thl, record_neuron_exc_thl, string(strbuf) ); 
      //exc_thl_mon_mem->record_for(10); // stops recording after 10s
    }

    if (inh_voltage_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.i.thl.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      VoltageMonitor * inh_thl_mon_mem = new VoltageMonitor( neurons_i_thl, record_neuron_inh_thl, string(strbuf) ); 
      //inh_thl_mon_mem->record_for(10); // stops recording after 10s
    }

    if (ie_weight_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.ie.thl.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ie_thl_mon_w = new WeightMonitor( con_ie_thl, string(strbuf) );
      ie_thl_mon_w->add_equally_spaced(50);
    }

    if (ii_weight_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.ii.thl.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      //WeightMonitor * ii_thl_mon_w = new WeightMonitor( con_ii_thl, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor
    }

    if (weightstat_mon_ee_thl) {
      sprintf(strbuf, "%s/%s.%d.ee.thl.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightStatsMonitor * ee_thl_mon_ws = new WeightStatsMonitor( con_ee_thl, string(strbuf) );
    }

    if (ie_weightstat_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.ie.thl.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightStatsMonitor * ie_thl_mon_ws = new WeightStatsMonitor( con_ie_thl, string(strbuf) );
    }

    if ( !monfile_thl.empty() && weightpat_mon_ee_thl) {
      sprintf(strbuf, "%s/%s.%d.ee.thl.wp", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightPatternMonitor * ee_thl_mon_wp = new WeightPatternMonitor(con_ee_thl, string(strbuf), 60);
      ee_thl_mon_wp->load_patterns(monfile_thl);
    }

    #ifndef CON_STIM_THL_NONE
     if ( !premonfile_thl.empty() && !monfile_thl.empty() && weightpat_mon_ee_stim_thl) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_thl.wp", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightPatternMonitor * stim_thl_mon_wp = new WeightPatternMonitor( con_ee_stim_thl,
									  string(strbuf), 60 );
       stim_thl_mon_wp->load_pre_patterns(premonfile_thl);
       stim_thl_mon_wp->load_post_patterns(monfile_thl);
     }
    #endif

    if (exc_spike_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * exc_thl_mon_sp = new BinarySpikeMonitor( neurons_e_thl, string(strbuf), exc_size_thl );
    }


    if (inh_spike_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.i.thl.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * inh_thl_mon_sp = new BinarySpikeMonitor( neurons_i_thl, string(strbuf), exc_size_thl / exc_inh_thl);
    }

    if (exc_prate_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.e.thl.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PopulationRateMonitor * exc_thl_mon_pr = new PopulationRateMonitor( neurons_e_thl, string(strbuf), 0.1 );
    }

    if (inh_prate_mon_thl) {
      sprintf(strbuf, "%s/%s.%d.i.thl.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PopulationRateMonitor * inh_thl_mon_pr = new PopulationRateMonitor( neurons_i_thl, string(strbuf), 0.1 );
    }
  #endif

  #ifdef HAS_CTX
    
    #ifndef CON_STIM_CTX_NONE
     if (weightstat_mon_ee_stim_ctx) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_ctx.ws", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightStatsMonitor * ee_stim_ctx_mon_ws = new WeightStatsMonitor( con_ee_stim_ctx, string(strbuf) );
     }

     if (weight_mon_ee_stim_ctx) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_ctx.w", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightMonitor * ee_stim_ctx_mon_w = new WeightMonitor( con_ee_stim_ctx, string(strbuf), 1.0 ); 
       ee_stim_ctx_mon_w->add_equally_spaced(50);
       if (!monfile_ctx.empty())
	 if (!stimfile.empty()) 
	   ee_stim_ctx_mon_w->load_pattern_connections(stimfile,monfile_ctx,20,20,ASSEMBLIES_ONLY); // true for assemblies only
	 else 
	   ee_stim_ctx_mon_w->load_pattern_connections(monfile_ctx,20,20,ASSEMBLIES_ONLY); // true for assemblies only
     }
    #endif

    if ( !monfile_ctx.empty() && exc_pattern_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.pact", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PatternMonitor * exc_ctx_mon_pat = new PatternMonitor( neurons_e_ctx, string(strbuf) , monfile_ctx.c_str(), 100);
    }

    if (ei_weight_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.ei.ctx.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ei_ctx_mon_w = new WeightMonitor( con_ei_ctx, 0, 100, strbuf, 1.0, DATARANGE);
    }

    if (weight_mon_ee_ctx) {
      sprintf(strbuf, "%s/%s.%d.ee.ctx.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ee_ctx_mon_w = new WeightMonitor( con_ee_ctx, string(strbuf), 1.0);
      ee_ctx_mon_w->add_equally_spaced(50);
      if ( !monfile_ctx.empty() ) 
	ee_ctx_mon_w->load_pattern_connections(monfile_ctx,10,10,ASSEMBLIES_ONLY); // true for assemblies only
    }

    if (ee_hom_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.ee.ctx.hom", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_hom = new StateMonitor( con_ee_ctx->hom, record_neuron_exc_ctx, string(strbuf), 1 );
    }

    if (exc_g_ampa_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.gampa", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_g_ampa = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "g_ampa", string(strbuf) );
    }

    if (exc_g_nmda_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.gnmda", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_g_nmda = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "g_nmda", string(strbuf) );
    }

    if (exc_g_gaba_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.ggaba", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_g_gaba = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "g_gaba", string(strbuf) );
    }

    if (exc_g_adapt1_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.gadapt1", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_g_adapt1 = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "g_adapt1", string(strbuf) );
    }

    if (exc_g_adapt2_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.gadapt2", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_g_adapt2 = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "g_adapt2", string(strbuf) );
    }

    if (exc_thr_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.thr", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_ctx_mon_thr = new StateMonitor( neurons_e_ctx, record_neuron_exc_ctx, "thr", string(strbuf) );
    }

    if (exc_voltage_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      VoltageMonitor * exc_ctx_mon_mem = new VoltageMonitor( neurons_e_ctx, record_neuron_exc_ctx, string(strbuf) ); 
      //exc_ctx_mon_mem->record_for(10); // stops recording after 10s
    }

    if (inh_voltage_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.i.ctx.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      VoltageMonitor * inh_ctx_mon_mem = new VoltageMonitor( neurons_i_ctx, record_neuron_inh_ctx, string(strbuf) ); 
      //inh_ctx_mon_mem->record_for(10); // stops recording after 10s
    }

    if (ie_weight_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.ie.ctx.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ie_ctx_mon_w = new WeightMonitor( con_ie_ctx, string(strbuf) );
      ie_ctx_mon_w->add_equally_spaced(50);
    }

    if (ii_weight_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.ii.ctx.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      //WeightMonitor * ii_ctx_mon_w = new WeightMonitor( con_ii_ctx, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor
    }

    if (weightstat_mon_ee_ctx) {
      sprintf(strbuf, "%s/%s.%d.ee.ctx.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightStatsMonitor * ee_ctx_mon_ws = new WeightStatsMonitor( con_ee_ctx, string(strbuf) );
    }

    if (ie_weightstat_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.ie.ctx.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightStatsMonitor * ie_ctx_mon_ws = new WeightStatsMonitor( con_ie_ctx, string(strbuf) );
    }

    if ( !monfile_ctx.empty() && weightpat_mon_ee_ctx) {
      sprintf(strbuf, "%s/%s.%d.ee.ctx.wp", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightPatternMonitor * ee_ctx_mon_wp = new WeightPatternMonitor(con_ee_ctx, string(strbuf), 60);
      ee_ctx_mon_wp->load_patterns(monfile_ctx);
    }

    #ifndef CON_STIM_CTX_NONE
     if ( !premonfile_ctx.empty() && !monfile_ctx.empty() && weightpat_mon_ee_stim_ctx) {
       sprintf(strbuf, "%s/%s.%d.ee.stim_ctx.wp", out_dir.c_str(), file_prefix.c_str(),
	       sys->mpi_rank() );
       WeightPatternMonitor * stim_ctx_mon_wp = new WeightPatternMonitor( con_ee_stim_ctx,
									  string(strbuf), 60 );
       stim_ctx_mon_wp->load_pre_patterns(premonfile_ctx);
       stim_ctx_mon_wp->load_post_patterns(monfile_ctx);
     }
    #endif

    if (exc_spike_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * exc_ctx_mon_sp = new BinarySpikeMonitor( neurons_e_ctx, string(strbuf), exc_size_ctx );
    }


    if (inh_spike_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.i.ctx.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      BinarySpikeMonitor * inh_ctx_mon_sp = new BinarySpikeMonitor( neurons_i_ctx, string(strbuf), exc_size_ctx / exc_inh_ctx);
    }

    if (exc_prate_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.e.ctx.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PopulationRateMonitor * exc_ctx_mon_pr = new PopulationRateMonitor( neurons_e_ctx, string(strbuf), 0.1 );
    }

    if (inh_prate_mon_ctx) {
      sprintf(strbuf, "%s/%s.%d.i.ctx.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      PopulationRateMonitor * inh_ctx_mon_pr = new PopulationRateMonitor( neurons_i_ctx, string(strbuf), 0.1 );
    }
  #endif
  
  #ifdef HAS_HPC

   #ifndef CON_STIM_HPC_NONE
    if (weightstat_mon_ee_stim_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.stim_hpc.ws", out_dir.c_str(), file_prefix.c_str(),
	      sys->mpi_rank() );
      WeightStatsMonitor * ee_stim_hpc_mon_ws = new WeightStatsMonitor( con_ee_stim_hpc,
									string(strbuf) );
    }

    if (weight_mon_ee_stim_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.stim_hpc.w", out_dir.c_str(), file_prefix.c_str(),
	      sys->mpi_rank() );
      WeightMonitor * ee_stim_hpc_mon_w = new WeightMonitor( con_ee_stim_hpc,
							     string(strbuf), 1.0 );
      ee_stim_hpc_mon_w->add_equally_spaced(50);
      if (!monfile_hpc.empty())
	if (!stimfile.empty()) 
	  ee_stim_hpc_mon_w->load_pattern_connections(stimfile,monfile_hpc,20,20,ASSEMBLIES_ONLY); // true for assemblies only
	else 
	  ee_stim_hpc_mon_w->load_pattern_connections(monfile_hpc,20,20,ASSEMBLIES_ONLY); // true for assemblies only
    }
   #endif
  
   if ( !monfile_hpc.empty() && exc_pattern_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.pact", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     PatternMonitor * exc_hpc_mon_pat = new PatternMonitor( neurons_e_hpc, string(strbuf) , monfile_hpc.c_str(), 100);
   }

   if (ei_weight_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.ei.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     WeightMonitor * ei_hpc_mon_w = new WeightMonitor( con_ei_hpc, 0, 100, strbuf, 1.0, DATARANGE);
   }

   #ifdef HAS_REC_EE
    if (weight_mon_ee_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightMonitor * ee_hpc_mon_w = new WeightMonitor( con_ee_hpc, string(strbuf), 1.0);
      ee_hpc_mon_w->add_equally_spaced(50);
      if ( !monfile_hpc.empty() ) 
	ee_hpc_mon_w->load_pattern_connections(monfile_hpc,10,10,ASSEMBLIES_ONLY); // true for assemblies only
    }

    if (ee_hom_mon_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.hpc.hom", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      StateMonitor * ee_hpc_mon_hom = new StateMonitor( con_ee_hpc->hom, record_neuron_exc_hpc, string(strbuf), 1 );
    }
   #endif

   if (exc_g_ampa_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.gampa", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_g_ampa = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "g_ampa", string(strbuf) );
   }

   if (exc_g_nmda_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.gnmda", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_g_nmda = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "g_nmda", string(strbuf) );
   }

   if (exc_g_gaba_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.ggaba", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_g_gaba = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "g_gaba", string(strbuf) );
   }

   if (exc_g_adapt1_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.gadapt1", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_g_adapt1 = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "g_adapt1", string(strbuf) );
   }

   if (exc_g_adapt2_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.gadapt2", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_g_adapt2 = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "g_adapt2", string(strbuf) );
   }

   if (exc_thr_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.thr", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     StateMonitor * ee_hpc_mon_thr = new StateMonitor( neurons_e_hpc, record_neuron_exc_hpc, "thr", string(strbuf) );
   }

   if (exc_voltage_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     VoltageMonitor * exc_hpc_mon_mem = new VoltageMonitor( neurons_e_hpc, record_neuron_exc_hpc, string(strbuf) ); 
     //exc_hpc_mon_mem->record_for(10); // stops recording after 10s
   }

   if (inh_voltage_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.i.hpc.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     VoltageMonitor * inh_hpc_mon_mem = new VoltageMonitor( neurons_i_hpc, record_neuron_inh_hpc, string(strbuf) );
     //inh_hpc_mon_mem->record_for(10); // stops recording after 10s
   }

   if (ie_weight_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.ie.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     WeightMonitor * ie_hpc_mon_w = new WeightMonitor( con_ie_hpc, string(strbuf) );
     ie_hpc_mon_w->add_equally_spaced(50);
   }

   if (ii_weight_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.ii.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     //WeightMonitor * ii_hpc_mon_w = new WeightMonitor( con_ii_hpc, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor
   }
     
   #ifdef HAS_INH2
     if (ei_weight_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.ei2.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       WeightMonitor * ei2_hpc_mon_w = new WeightMonitor( con_ei2_hpc, 0, 100, strbuf, 1.0, DATARANGE);
     }

     if (ie_weight_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2e.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       WeightMonitor * i2e_hpc_mon_w = new WeightMonitor( con_i2e_hpc, string(strbuf) );
       i2e_hpc_mon_w->add_equally_spaced(50);
     }

     if (ii_weight_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2i2.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       //WeightMonitor * i2i2_hpc_mon_w = new WeightMonitor( con_i2i2_hpc, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor

       sprintf(strbuf, "%s/%s.%d.ii2.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       //WeightMonitor * ii2_hpc_mon_w = new WeightMonitor( con_ii2_hpc, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor

       sprintf(strbuf, "%s/%s.%d.i2i.hpc.w", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       //WeightMonitor * i2i_hpc_mon_w = new WeightMonitor( con_i2i_hpc, 0, 100, strbuf, 1.0, DATARANGE); TO-DO: include support for SparseBConnection in WeightMonitor
     }

     if (ie_weightstat_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2e.hpc.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       WeightStatsMonitor * i2e_hpc_mon_ws = new WeightStatsMonitor( con_i2e_hpc, string(strbuf) );
     }

     if (inh_spike_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2.hpc.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       BinarySpikeMonitor * inh2_hpc_mon_sp = new BinarySpikeMonitor( neurons_i2_hpc, string(strbuf), exc_size_hpc / exc_inh2_hpc);
     }

     if (inh_prate_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2.hpc.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       PopulationRateMonitor * inh2_hpc_mon_pr = new PopulationRateMonitor( neurons_i2_hpc, string(strbuf), 0.1 );
     }

     if (inh_voltage_mon_hpc) {
       sprintf(strbuf, "%s/%s.%d.i2.hpc.mem", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       VoltageMonitor * inh2_hpc_mon_mem = new VoltageMonitor( neurons_i2_hpc, record_neuron_inh2_hpc, string(strbuf) );
       //inh2_hpc_mon_mem->record_for(10); // stops recording after 10s
     }
   #endif

   #ifdef HAS_REC_EE
    if (weightstat_mon_ee_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.hpc.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightStatsMonitor * ee_hpc_mon_ws = new WeightStatsMonitor( con_ee_hpc, string(strbuf) );
    }

    if ( !monfile_hpc.empty() && weightpat_mon_ee_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.hpc.wp", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      WeightPatternMonitor * ee_hpc_mon_wp = new WeightPatternMonitor(con_ee_hpc, string(strbuf), 60);
      ee_hpc_mon_wp->load_patterns(monfile_hpc);
    }
   #endif

   if (ie_weightstat_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.ie.hpc.ws", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     WeightStatsMonitor * ie_hpc_mon_ws = new WeightStatsMonitor( con_ie_hpc, string(strbuf) );
   }

   #ifndef CON_STIM_HPC_NONE
    if ( !premonfile_hpc.empty() && !monfile_hpc.empty() && weightpat_mon_ee_stim_hpc) {
      sprintf(strbuf, "%s/%s.%d.ee.stim_hpc.wp", out_dir.c_str(), file_prefix.c_str(),
	      sys->mpi_rank() );
      WeightPatternMonitor * stim_hpc_mon_wp = new WeightPatternMonitor( con_ee_stim_hpc,
									 string(strbuf), 60 );
      stim_hpc_mon_wp->load_pre_patterns(premonfile_hpc);
      stim_hpc_mon_wp->load_post_patterns(monfile_hpc);
    }
   #endif

   if (exc_spike_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     BinarySpikeMonitor * exc_hpc_mon_sp = new BinarySpikeMonitor( neurons_e_hpc, string(strbuf), exc_size_hpc );
   }

   if (inh_spike_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.i.hpc.spk", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     BinarySpikeMonitor * inh_hpc_mon_sp = new BinarySpikeMonitor( neurons_i_hpc, string(strbuf), exc_size_hpc / exc_inh_hpc);
   }

   if (exc_prate_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.e.hpc.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     PopulationRateMonitor * exc_hpc_mon_pr = new PopulationRateMonitor( neurons_e_hpc, string(strbuf), 0.1 );
   }

   if (inh_prate_mon_hpc) {
     sprintf(strbuf, "%s/%s.%d.i.hpc.prate", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     PopulationRateMonitor * inh_hpc_mon_pr = new PopulationRateMonitor( neurons_i_hpc, string(strbuf), 0.1 );
   }
  #endif

  // set up checkers
  #ifdef HAS_CTX
   if (exc_ratechk_ctx)
     RateChecker * exc_ctx_rchk = new RateChecker( neurons_e_ctx , -1 , 20. , 0.1);
  #endif

  #ifdef HAS_HPC
   if (exc_ratechk_hpc)
     RateChecker * exc_hpc_rchk = new RateChecker( neurons_e_hpc , -1 , 20. , 0.1);
  #endif

  #ifdef HAS_THL
   if (exc_ratechk_thl)
     RateChecker * exc_thl_rchk = new RateChecker( neurons_e_thl , -1 , 20. , 0.1);
  #endif
  
  // prime if appropriate
  if ( prime ) {
    // tries to decrease training time by initially rapidly decorrelating patterns
    // Note that this was not done in the original publication
    logger->msg("High intensity priming ...",PROGRESS,true);
    stimgroup->set_mean_on_period(prime_ontime);
    stimgroup->set_mean_off_period(prime_offtime);

    if (!sys->run(prime_duration,false))
      errcode = 1;
  }


  // reiterate stimulation protocol (is this redundant?)
  stimgroup->set_mean_on_period(ontime);
  stimgroup->set_mean_off_period(offtime);

  #ifdef HAS_REPLAY
   // reiterate replay protocol (is this redundant?)
   repgroup->set_mean_on_period(ontime);
   repgroup->set_mean_off_period(offtime);
  #endif

  #ifdef HAS_BACKGROUND
   
   #ifdef HAS_THL
    // reiterate thl background protocol (is this redundant?)
    bggroup_thl->set_mean_on_period(ontime);
    bggroup_thl->set_mean_off_period(offtime);
   #endif

   #ifdef HAS_CTX
    // reiterate ctx background protocol (is this redundant?)
    bggroup_ctx->set_mean_on_period(ontime);
    bggroup_ctx->set_mean_off_period(offtime);
   #endif

   #ifdef HAS_HPC
    // reiterate hpc background protocol (is this redundant?)
    bggroup_hpc->set_mean_on_period(ontime);
    bggroup_hpc->set_mean_off_period(offtime);
   #endif
   
  #endif


  // toggle plasticity
  #ifdef HAS_THL

   if ( eta_thl > 0 ) {
     con_ee_thl->stdp_active = true;
   } else {
     con_ee_thl->stdp_active = false;
   }

   if ( eta_thl/eta_exc_inh_thl > 0 ) {
     con_ie_thl->stdp_active = isp_active;
   } else {
     con_ie_thl->stdp_active = false;
   }

   #ifdef CON_STIM_THL_P11B
    if ( eta_stim_thl > 0 ) {
      con_ee_stim_thl->stdp_active = true;
    } else {
      con_ee_stim_thl->stdp_active = false;
    }
   #endif

   #ifdef HAS_CTX

    #ifdef CON_THL_CTX_P11B
     if ( eta_thl_ctx > 0 )
       con_ee_thl_ctx->stdp_active = true;
     else
       con_ee_thl_ctx->stdp_active = false;
    #endif

    #ifdef CON_CTX_THL_P11B
     if ( eta_ctx_thl > 0 )
       con_ee_ctx_thl->stdp_active = true;
     else
       con_ee_ctx_thl->stdp_active = false;
    #endif

   #endif

   #ifdef HAS_HPC

    #ifdef CON_THL_HPC_P11B
     if ( eta_thl_hpc > 0 )
       con_ee_thl_hpc->stdp_active = true;
     else
       con_ee_thl_hpc->stdp_active = false;
    #endif

    #ifdef CON_HPC_THL_P11B
     if ( eta_hpc_thl > 0 )
       con_ee_hpc_thl->stdp_active = true;
     else
       con_ee_hpc_thl->stdp_active = false;
    #endif

   #endif

  #endif
  
  #ifdef HAS_CTX

   if ( eta_ctx > 0 ) {
     con_ee_ctx->stdp_active = true;
   } else {
     con_ee_ctx->stdp_active = false;
   }

   if ( eta_ctx/eta_exc_inh_ctx > 0 ) {
     con_ie_ctx->stdp_active = isp_active;
   } else {
     con_ie_ctx->stdp_active = false;
   }

   #ifdef CON_STIM_CTX_P11B
    if ( eta_stim_ctx > 0 ) {
      con_ee_stim_ctx->stdp_active = true;
    } else {
      con_ee_stim_ctx->stdp_active = false;
    }
   #endif

  #endif
  
  #ifdef HAS_HPC

   #ifdef HAS_REC_EE
    if ( eta_hpc > 0 ) {
      con_ee_hpc->stdp_active = true;      
    } else {
      con_ee_hpc->stdp_active = false;
    }
   #endif

   if ( eta_hpc/eta_exc_inh_hpc > 0 ) {
     con_ie_hpc->stdp_active = isp_active;
   } else {
     con_ie_hpc->stdp_active = false;
   }

   #ifdef HAS_INH2
     if ( eta_hpc/eta_exc_inh2_hpc > 0 ) {
       con_i2e_hpc->stdp_active = isp_active;
     } else {
       con_i2e_hpc->stdp_active = false;
     }
   #endif

   #ifdef CON_STIM_HPC_P11B
    if ( eta_stim_hpc > 0 ) {
      con_ee_stim_hpc->stdp_active = true;
    } else {
      con_ee_stim_hpc->stdp_active = false;
    }
   #endif

   #ifdef HAS_CTX

    #ifdef CON_CTX_HPC_P11B
     if ( eta_ctx_hpc > 0 )
       con_ee_ctx_hpc->stdp_active = true;
     else
       con_ee_ctx_hpc->stdp_active = false;
    #endif

    #ifdef CON_HPC_CTX_P11B
     if ( eta_hpc_ctx > 0 )
       con_ee_hpc_ctx->stdp_active = true;
     else
       con_ee_hpc_ctx->stdp_active = false;
    #endif

   #endif

  #endif

  // Block and/or inhibit neurons, log has_forward_propagation and can_spike , and save connectivity matrices
  logger->msg("Blocking neurons, inhibiting neurons, logging spike propagation, and writing connectivity matrices upon initialization...",PROGRESS,true);

  #ifdef HAS_THL

    if ( !block_exc_thl.empty() ) {
      if ( block_local ) {
	con_ee_thl->block_pre_neurons( block_exc_thl );
	con_ei_thl->block_pre_neurons( block_exc_thl );
      }
    }
    con_ee_thl->log_has_fwd_prop("final");
    con_ei_thl->log_has_fwd_prop("final");

    if ( !block_inh_thl.empty() ) {
      if ( block_local ) {
	con_ii_thl->block_pre_neurons( block_inh_thl );
	con_ie_thl->block_pre_neurons( block_inh_thl );
      }
    }
    con_ii_thl->log_has_fwd_prop("final");
    con_ie_thl->log_has_fwd_prop("final");

    sprintf(strbuf, "%s/%s.%d.ee.thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ee_thl->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ei.thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ei_thl->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ii.thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ii_thl->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ie.thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ie_thl->write_to_file(strbuf);

    #ifndef CON_STIM_THL_NONE
    
      if ( !block_stim.empty() ) {
	if ( block_cross_region ) {
	  con_ee_stim_thl->block_pre_neurons( block_stim );
	}
      }
      con_ee_stim_thl->log_has_fwd_prop("final");
      
      sprintf(strbuf, "%s/%s.%d.ee.stim_thl.wmat", out_dir.c_str(), file_prefix.c_str(),
	      sys->mpi_rank() );
      con_ee_stim_thl->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
    
        if ( !block_stim.empty() ) {
	  if ( block_cross_region ) {
	    con_ei_stim_thl->block_pre_neurons( block_stim );
	  }
	}
	con_ei_stim_thl->log_has_fwd_prop("final");

	sprintf(strbuf, "%s/%s.%d.ei.stim_thl.wmat", out_dir.c_str(), file_prefix.c_str(),
		sys->mpi_rank() );
	con_ei_stim_thl->write_to_file(strbuf);
      
      #endif
      
    #endif

    #ifdef HAS_REPLAY
      #ifndef CON_REPLAY_THL_NONE

        if ( !block_rep.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_rep_thl->block_pre_neurons( block_rep );
	    con_ei_rep_thl->block_pre_neurons( block_rep );
	  }
	}
	con_ee_rep_thl->log_has_fwd_prop("final");
	con_ei_rep_thl->log_has_fwd_prop("final");
	
        sprintf(strbuf, "%s/%s.%d.ee.rep_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
        con_ee_rep_thl->write_to_file(strbuf);
	
        sprintf(strbuf, "%s/%s.%d.ei.rep_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
        con_ei_rep_thl->write_to_file(strbuf);
	
      #endif
    #endif

    #ifdef HAS_BACKGROUND

      if ( !block_bg_thl.empty() ) {
	if ( block_cross_region ) {
	  con_ee_bg_thl->block_pre_neurons( block_bg_thl );
	}
      }
      con_ee_bg_thl->log_has_fwd_prop("final");

      sprintf(strbuf, "%s/%s.%d.ee.bg_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_ee_bg_thl->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
       if ( !block_bg_thl.empty() ) {
	 if ( block_cross_region ) {
	   con_ei_bg_thl->block_pre_neurons( block_bg_thl );
	 }
       }
       con_ei_bg_thl->log_has_fwd_prop("final");
       
       sprintf(strbuf, "%s/%s.%d.ei.bg_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
       con_ei_bg_thl->write_to_file(strbuf);
      #endif
	
    #endif

    #ifdef HAS_CTX

      #ifndef CON_THL_CTX_NONE

	if ( !block_exc_thl.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_thl_ctx->block_pre_neurons( block_exc_thl );
	  }
	}
	con_ee_thl_ctx->log_has_fwd_prop("final");
	
	sprintf(strbuf, "%s/%s.%d.ee.thl_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_thl_ctx->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	 if ( !block_exc_thl.empty() ) {
	   if ( block_cross_region ) {
	     con_ei_thl_ctx->block_pre_neurons( block_exc_thl );
	   }
	 }
	 con_ei_thl_ctx->log_has_fwd_prop("final");
	
	 sprintf(strbuf, "%s/%s.%d.ei.thl_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	 con_ei_thl_ctx->write_to_file(strbuf);
	#endif
	
      #endif

      #ifndef CON_CTX_THL_NONE
	if ( !block_exc_ctx.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_ctx_thl->block_pre_neurons( block_exc_ctx );
	  }
	}
	con_ee_ctx_thl->log_has_fwd_prop("final");
	
	sprintf(strbuf, "%s/%s.%d.ee.ctx_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_ctx_thl->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	  if ( !block_exc_ctx.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_ctx_thl->block_pre_neurons( block_exc_ctx );
	    }
	  }
	  con_ei_ctx_thl->log_has_fwd_prop("final");
	  
	  sprintf(strbuf, "%s/%s.%d.ei.ctx_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_ctx_thl->write_to_file(strbuf);
	#endif
	
      #endif

    #endif

    #ifdef HAS_HPC

      #ifndef CON_THL_HPC_NONE

	if ( !block_exc_thl.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_thl_hpc->block_pre_neurons( block_exc_thl );
	  }
	}
	con_ee_thl_hpc->log_has_fwd_prop("final");
	
	sprintf(strbuf, "%s/%s.%d.ee.thl_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_thl_hpc->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	  if ( !block_exc_thl.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_thl_hpc->block_pre_neurons( block_exc_thl );
	    }
	  }
	  con_ei_thl_hpc->log_has_fwd_prop("final");
	
	  sprintf(strbuf, "%s/%s.%d.ei.thl_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_thl_hpc->write_to_file(strbuf);
	#endif
	
      #endif

      #ifndef CON_HPC_THL_NONE

	if ( !block_exc_hpc.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_hpc_thl->block_pre_neurons( block_exc_hpc );
	  }
	}
	con_ee_hpc_thl->log_has_fwd_prop("final");
	
	sprintf(strbuf, "%s/%s.%d.ee.hpc_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_hpc_thl->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	  if ( !block_exc_hpc.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_hpc_thl->block_pre_neurons( block_exc_hpc );
	    }
	  }
	  con_ei_hpc_thl->log_has_fwd_prop("final");
	
	  sprintf(strbuf, "%s/%s.%d.ei.hpc_thl.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_hpc_thl->write_to_file(strbuf);
	#endif
	
      #endif

    #endif

  #endif

  #ifdef HAS_CTX

    if ( !block_exc_ctx.empty() ) {
      if ( block_local ) {
	con_ee_ctx->block_pre_neurons( block_exc_ctx );
	con_ei_ctx->block_pre_neurons( block_exc_ctx );
      }
    }
    con_ee_ctx->log_has_fwd_prop("final");
    con_ei_ctx->log_has_fwd_prop("final");

    if ( !block_inh_ctx.empty() ) {
      if ( block_local ) {
	con_ii_ctx->block_pre_neurons( block_inh_ctx );
	con_ie_ctx->block_pre_neurons( block_inh_ctx );
      }
    }
    con_ii_ctx->log_has_fwd_prop("final");
    con_ie_ctx->log_has_fwd_prop("final");

    sprintf(strbuf, "%s/%s.%d.ee.ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ee_ctx->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ei.ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ei_ctx->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ii.ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ii_ctx->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ie.ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ie_ctx->write_to_file(strbuf);

    #ifndef CON_STIM_CTX_NONE
    
      if ( !block_stim.empty() ) {
	if ( block_cross_region ) {
	  con_ee_stim_ctx->block_pre_neurons( block_stim );
	}
      }
      con_ee_stim_ctx->log_has_fwd_prop("final");
    
      sprintf(strbuf, "%s/%s.%d.ee.stim_ctx.wmat", out_dir.c_str(), file_prefix.c_str(),
	      sys->mpi_rank() );
      con_ee_stim_ctx->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
    
        if ( !block_stim.empty() ) {
	  if ( block_cross_region ) {
	    con_ei_stim_ctx->block_pre_neurons( block_stim );
	  }
	}
	con_ei_stim_ctx->log_has_fwd_prop("final");

	sprintf(strbuf, "%s/%s.%d.ei.stim_ctx.wmat", out_dir.c_str(), file_prefix.c_str(),
		sys->mpi_rank() );
	con_ei_stim_ctx->write_to_file(strbuf);
      
    #endif
      
    #endif

    #ifdef HAS_REPLAY
      #ifndef CON_REPLAY_CTX_NONE

        if ( !block_rep.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_rep_ctx->block_pre_neurons( block_rep );
	    con_ei_rep_ctx->block_pre_neurons( block_rep );
	  }
        }
        con_ee_rep_ctx->log_has_fwd_prop("final");
	con_ei_rep_ctx->log_has_fwd_prop("final");
      
        sprintf(strbuf, "%s/%s.%d.ee.rep_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
        con_ee_rep_ctx->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.ei.rep_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ei_rep_ctx->write_to_file(strbuf);
	
      #endif
    #endif
	
    #ifdef HAS_BACKGROUND

      if ( !block_bg_ctx.empty() ) {
	if ( block_cross_region ) {
	  con_ee_bg_ctx->block_pre_neurons( block_bg_ctx );
	}
      }
      con_ee_bg_ctx->log_has_fwd_prop("final");

      sprintf(strbuf, "%s/%s.%d.ee.bg_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_ee_bg_ctx->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
        if ( !block_bg_ctx.empty() ) {
	  if ( block_cross_region ) {
	    con_ei_bg_ctx->block_pre_neurons( block_bg_ctx );
	  }
	}
	con_ei_bg_ctx->log_has_fwd_prop("final");

	sprintf(strbuf, "%s/%s.%d.ei.bg_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ei_bg_ctx->write_to_file(strbuf);
      #endif
	
    #endif

  #endif

  #ifdef HAS_HPC

    if ( !block_exc_hpc.empty() ) {
      if ( block_local ) {
	#ifdef HAS_REC_EE
	 con_ee_hpc->block_pre_neurons( block_exc_hpc );
	#endif
	con_ei_hpc->block_pre_neurons( block_exc_hpc );
      }
    }
    #ifdef HAS_REC_EE
     con_ee_hpc->log_has_fwd_prop("final");
    #endif
    con_ei_hpc->log_has_fwd_prop("final");

    if ( !inhibit_exc_hpc.empty() ) {
      neurons_e_hpc->inhibit_neurons( inhibit_exc_hpc );
    }
    neurons_e_hpc->log_can_spike("final");

    if ( !block_inh_hpc.empty() ) {
      if ( block_local ) {
	con_ii_hpc->block_pre_neurons( block_inh_hpc );
	con_ie_hpc->block_pre_neurons( block_inh_hpc );
      }
    }
    con_ii_hpc->log_has_fwd_prop("final");
    con_ie_hpc->log_has_fwd_prop("final");

    #ifdef HAS_INH2
      if ( !block_exc_hpc.empty() ) {
	if ( block_local ) {
	  con_ei2_hpc->block_pre_neurons( block_exc_hpc );
	}
      }
      con_ei2_hpc->log_has_fwd_prop("final");
    
      if ( !block_inh_hpc.empty() ) {
	if ( block_local ) {
	  con_ii2_hpc->block_pre_neurons( block_inh_hpc );
	}
      }
      con_ii2_hpc->log_has_fwd_prop("final");
      
      if ( !block_inh2_hpc.empty() ) {
	if ( block_local ) {
	  con_i2i2_hpc->block_pre_neurons( block_inh2_hpc );
	  con_i2e_hpc->block_pre_neurons( block_inh2_hpc );
	  con_i2i_hpc->block_pre_neurons( block_inh2_hpc );
	}
      }
      con_i2i2_hpc->log_has_fwd_prop("final");
      con_i2e_hpc->log_has_fwd_prop("final");
      con_i2i_hpc->log_has_fwd_prop("final");
    #endif

    #ifdef HAS_REC_EE
     sprintf(strbuf, "%s/%s.%d.ee.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
     con_ee_hpc->write_to_file(strbuf);
    #endif

    sprintf(strbuf, "%s/%s.%d.ei.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ei_hpc->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ii.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ii_hpc->write_to_file(strbuf);

    sprintf(strbuf, "%s/%s.%d.ie.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
    con_ie_hpc->write_to_file(strbuf);

    #ifdef HAS_INH2
      sprintf(strbuf, "%s/%s.%d.ei2.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_ei2_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.i2i2.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_i2i2_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.i2e.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_i2e_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ii2.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_ii2_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.i2i.hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_i2i_hpc->write_to_file(strbuf);
    #endif

    #ifndef CON_STIM_HPC_NONE
    
      if ( !block_stim.empty() ) {
	if ( block_cross_region ) {
	  con_ee_stim_hpc->block_pre_neurons( block_stim );
	}
      }
      con_ee_stim_hpc->log_has_fwd_prop("final");
      
      sprintf(strbuf, "%s/%s.%d.ee.stim_hpc.wmat", out_dir.c_str(), file_prefix.c_str(),
	    sys->mpi_rank() );
      con_ee_stim_hpc->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
        if ( !block_stim.empty() ) {
	  if ( block_cross_region ) {
	    con_ei_stim_hpc->block_pre_neurons( block_stim );
	  }
	}
	con_ei_stim_hpc->log_has_fwd_prop("final");

	sprintf(strbuf, "%s/%s.%d.ei.stim_hpc.wmat", out_dir.c_str(), file_prefix.c_str(),
	    sys->mpi_rank() );
	con_ei_stim_hpc->write_to_file(strbuf);
      #endif
      
    #endif

    #ifdef HAS_REPLAY
      #ifndef CON_REPLAY_HPC_NONE

        if ( !block_rep.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_rep_hpc->block_pre_neurons( block_rep );
	    con_ei_rep_hpc->block_pre_neurons( block_rep );
	  }
        }
        con_ee_rep_hpc->log_has_fwd_prop("final");
	con_ei_rep_hpc->log_has_fwd_prop("final");
	
        sprintf(strbuf, "%s/%s.%d.ee.rep_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
        con_ee_rep_hpc->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.ei.rep_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
        con_ei_rep_hpc->write_to_file(strbuf);
	
      #endif
    #endif
	
    #ifdef HAS_BACKGROUND

      if ( !block_bg_hpc.empty() ) {
	if ( block_cross_region ) {
	  con_ee_bg_hpc->block_pre_neurons( block_bg_hpc );
	}
      }
      con_ee_bg_hpc->log_has_fwd_prop("final");

      sprintf(strbuf, "%s/%s.%d.ee.bg_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
      con_ee_bg_hpc->write_to_file(strbuf);

      #ifdef HAS_CROSS_REGION_EI_CON
          if ( !block_bg_hpc.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_bg_hpc->block_pre_neurons( block_bg_hpc );
	    }
	  }
	  con_ei_bg_hpc->log_has_fwd_prop("final");

	  sprintf(strbuf, "%s/%s.%d.ei.bg_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_bg_hpc->write_to_file(strbuf);
      #endif
	
    #endif

    #ifdef HAS_CTX

      #ifndef CON_CTX_HPC_NONE

	if ( !block_exc_ctx.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_ctx_hpc->block_pre_neurons( block_exc_ctx );
	  }
        }
        con_ee_ctx_hpc->log_has_fwd_prop("final");

	sprintf(strbuf, "%s/%s.%d.ee.ctx_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_ctx_hpc->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	  if ( !block_exc_ctx.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_ctx_hpc->block_pre_neurons( block_exc_ctx );
	    }
	  }
	  con_ei_ctx_hpc->log_has_fwd_prop("final");

	  sprintf(strbuf, "%s/%s.%d.ei.ctx_hpc.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_ctx_hpc->write_to_file(strbuf);
	#endif
	
      #endif

      #ifndef CON_HPC_CTX_NONE

	if ( !block_exc_hpc.empty() ) {
	  if ( block_cross_region ) {
	    con_ee_hpc_ctx->block_pre_neurons( block_exc_hpc );
	  }
        }
        con_ee_hpc_ctx->log_has_fwd_prop("final");
	
	sprintf(strbuf, "%s/%s.%d.ee.hpc_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	con_ee_hpc_ctx->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	  if ( !block_exc_hpc.empty() ) {
	    if ( block_cross_region ) {
	      con_ei_hpc_ctx->block_pre_neurons( block_exc_hpc );
	    }
	  }
	  con_ei_hpc_ctx->log_has_fwd_prop("final");
	
	  sprintf(strbuf, "%s/%s.%d.ei.hpc_ctx.wmat", out_dir.c_str(), file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_hpc_ctx->write_to_file(strbuf);
	#endif
	
      #endif

    #endif

  #endif
  
	
  // run simulation and save
  double previous_simtime = 0.;
  double simtime = 0;
  for (int index=0; index < simtimes.size(); ++index) {

    simtime = simtimes[index];
    
    logger->msg("Main simtime ...",PROGRESS,true);
    if (!sys->run(simtime-previous_simtime,false)) 
      errcode = 1;
    
    sprintf(strbuf, "%s-%d", file_prefix.c_str(), static_cast<int>(simtime) );
    string full_file_prefix = strbuf;
    
    logger->msg("Saving ...",PROGRESS,true);
    if ( save ) {
      // save state of complete network
      sys->set_output_dir(out_dir);
      sys->save_network_state(full_file_prefix);
    }
    
    if (save_without_hpc) {

      bool has_hm_net = false;
      
      #ifdef HAS_CTX
       has_hm_net = true;
      #endif

      #ifdef HAS_THL
       has_hm_net = true;
      #endif

      if (has_hm_net) {
      
       sprintf(strbuf, "%s-%d", file_prefix_hm.c_str(), static_cast<int>(simtime) );
       string full_file_prefix_hm = strbuf;

       auryn::logger->msg("Saving network state without hpc", NOTIFICATION);

       std::string netstate_filename;
       {	
	 sprintf(strbuf, "%s", full_file_prefix_hm.c_str());
	 string basename = strbuf;
	 std::stringstream oss;
	 oss << out_dir
	     << "/" << basename
	     << "." << sys->mpi_rank()
	     << ".netstate";
	 netstate_filename = oss.str();
       } // oss goes out of scope

       auryn::logger->msg("Opening output stream ...",VERBOSE);
       std::ofstream ofs(netstate_filename.c_str());
       boost::archive::binary_oarchive oa(ofs);

       auryn::logger->msg("Saving version information ...",VERBOSE);
       oa << sys->build.version;
       oa << sys->build.subversion;
       oa << sys->build.revision_number;

       auryn::logger->msg("Saving communicator information ...",VERBOSE);
       unsigned int tmp_int = sys->mpi_size();
       oa << tmp_int;
       tmp_int = sys->mpi_rank();
       oa << tmp_int;

       /*
	 auryn::logger->msg("Saving SpikingGroups ...",VERBOSE);

	 {
	 std::stringstream oss;
	 oss << "Saving SpikingGroup: neurons_e_ctx " 
	 << neurons_e_ctx->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);
	 oa << *neurons_e_ctx;
	 }

	 {
	 std::stringstream oss;
	 oss << "Saving SpikingGroup: neurons_i_ctx " 
	 << neurons_i_ctx->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);      
	 oa << *neurons_i_ctx;
	 }

	 {
	 std::stringstream oss;
	 oss << "Saving SpikingGroup: stimgroup "
	 << stimgroup->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);      
	 oa << *stimgroup;
	 }
       */

       auryn::logger->msg("Saving Connections ...",VERBOSE);
       for ( unsigned int i = 0 ; i < connections_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Saving connection "
	     <<  i 
	     << " '"
	     << connections_hm[i]->get_name()
	     << "' "
	     << " to stream";
	 auryn::logger->msg(oss.str(),VERBOSE);

	 oa << *(connections_hm[i]);
       }

       auryn::logger->msg("Saving SpikingGroups ...",VERBOSE);

       #ifdef HAS_THL
	{
	  std::string netstate_filename;
	  {	
	    sprintf(strbuf, "%s_e_thl", full_file_prefix_hm.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << out_dir
		<< "/" << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ofstream ofs(netstate_filename.c_str());
	  boost::archive::binary_oarchive oa(ofs);

	  std::stringstream oss;
	  oss << "Saving SpikingGroup: neurons_e_thl " 
	      << neurons_e_thl->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);
	  oa << *neurons_e_thl;

	  ofs.close();
	}

	{
	  std::string netstate_filename;
	  {	
	    sprintf(strbuf, "%s_i_thl", full_file_prefix_hm.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << out_dir
		<< "/" << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ofstream ofs(netstate_filename.c_str());
	  boost::archive::binary_oarchive oa(ofs);

	  std::stringstream oss;
	  oss << "Saving SpikingGroup: neurons_i_thl " 
	      << neurons_i_thl->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);      
	  oa << *neurons_i_thl;

	  ofs.close();
	}
       #endif

       #ifdef HAS_CTX
	{
	  std::string netstate_filename;
	  {	
	    sprintf(strbuf, "%s_e_ctx", full_file_prefix_hm.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << out_dir
		<< "/" << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ofstream ofs(netstate_filename.c_str());
	  boost::archive::binary_oarchive oa(ofs);

	  std::stringstream oss;
	  oss << "Saving SpikingGroup: neurons_e_ctx " 
	      << neurons_e_ctx->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);
	  oa << *neurons_e_ctx;

	  ofs.close();
	}

	{
	  std::string netstate_filename;
	  {	
	    sprintf(strbuf, "%s_i_ctx", full_file_prefix_hm.c_str());
	    string basename = strbuf;
	    std::stringstream oss;
	    oss << out_dir
		<< "/" << basename
		<< "." << sys->mpi_rank()
		<< ".netstate";
	    netstate_filename = oss.str();
	  } // oss goes out of scope

	  std::ofstream ofs(netstate_filename.c_str());
	  boost::archive::binary_oarchive oa(ofs);

	  std::stringstream oss;
	  oss << "Saving SpikingGroup: neurons_i_ctx " 
	      << neurons_i_ctx->get_name();
	  auryn::logger->msg(oss.str(),VERBOSE);      
	  oa << *neurons_i_ctx;

	  ofs.close();
	}
       #endif

       {
	 std::string netstate_filename;
	 {	
	   sprintf(strbuf, "%s_stim", full_file_prefix_hm.c_str());
	   string basename = strbuf;
	   std::stringstream oss;
	   oss << out_dir
	       << "/" << basename
	       << "." << sys->mpi_rank()
	       << ".netstate";
	   netstate_filename = oss.str();
	 } // oss goes out of scope

	 std::ofstream ofs(netstate_filename.c_str());
	 boost::archive::binary_oarchive oa(ofs);

	 std::stringstream oss;
	 oss << "Saving SpikingGroup: stimgroup "
	     << stimgroup->get_name();
	 auryn::logger->msg(oss.str(),VERBOSE);      
	 oa << *stimgroup;

	 ofs.close();
       }

       /*
	 for ( unsigned int i = 0 ; i < spiking_groups_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Saving SpikingGroup "
	 <<  i 
	 << " ("
	 << spiking_groups_hm[i]->get_name()
	 << ")"
	 << " to stream";
	 auryn::logger->msg(oss.str(),VERBOSE);

	 oa << *(spiking_groups_hm[i]);
	 }
       */

       /*
	 auryn::logger->msg("Saving Devices ...",VERBOSE);
	 for ( unsigned int i = 0 ; i < devices_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Saving Device "
	 <<  i 
	 << " to stream";
	 auryn::logger->msg(oss.str(),VERBOSE);

	 oa << *(devices_hm[i]);
	 }

	 auryn::logger->msg("Saving Checkers ...",VERBOSE);
	 for ( unsigned int i = 0 ; i < checkers_hm.size() ; ++i ) {

	 std::stringstream oss;
	 oss << "Saving Checker "
	 <<  i 
	 << " to stream";
	 auryn::logger->msg(oss.str(),VERBOSE);

	 oa << *(checkers_hm[i]);
	 }
       */

       ofs.close();

      }
    } 

    logger->msg("Writing connectivity matrices at the end of simulation...",PROGRESS,true);

    #ifdef HAS_THL

      sprintf(strbuf, "%s/%s.%d.ee.thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ee_thl->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ei.thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ei_thl->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ii.thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ii_thl->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ie.thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ie_thl->write_to_file(strbuf);

      #ifndef CON_STIM_THL_NONE
       sprintf(strbuf, "%s/%s.%d.ee.stim_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(),
	       sys->mpi_rank() );
       con_ee_stim_thl->write_to_file(strbuf);

       #ifdef HAS_CROSS_REGION_EI_CON
        sprintf(strbuf, "%s/%s.%d.ei.stim_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(),
		sys->mpi_rank() );
	con_ei_stim_thl->write_to_file(strbuf);
       #endif
      #endif

      #ifdef HAS_REPLAY
	#ifndef CON_REPLAY_THL_NONE
	  sprintf(strbuf, "%s/%s.%d.ee.rep_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ee_rep_thl->write_to_file(strbuf);

	  sprintf(strbuf, "%s/%s.%d.ei.rep_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_rep_thl->write_to_file(strbuf);
	#endif
      #endif

      #ifdef HAS_BACKGROUND
        sprintf(strbuf, "%s/%s.%d.ee.bg_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
        con_ee_bg_thl->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	 sprintf(strbuf, "%s/%s.%d.ei.bg_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ei_bg_thl->write_to_file(strbuf);
	#endif
      #endif

      #ifdef HAS_CTX

       #ifndef CON_THL_CTX_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.thl_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_thl_ctx->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.thl_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_thl_ctx->write_to_file(strbuf);
	 #endif
       #endif

       #ifndef CON_CTX_THL_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.ctx_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_ctx_thl->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.ctx_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_ctx_thl->write_to_file(strbuf);
	 #endif
       #endif

      #endif

      #ifdef HAS_HPC

       #ifndef CON_THL_HPC_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.thl_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_thl_hpc->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.thl_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_thl_hpc->write_to_file(strbuf);
	 #endif
       #endif

       #ifndef CON_HPC_THL_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.hpc_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_hpc_thl->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.hpc_thl.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_hpc_thl->write_to_file(strbuf);
	 #endif
       #endif

      #endif

    #endif

    #ifdef HAS_CTX

      sprintf(strbuf, "%s/%s.%d.ee.ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ee_ctx->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ei.ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ei_ctx->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ii.ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ii_ctx->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ie.ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ie_ctx->write_to_file(strbuf);

      #ifndef CON_STIM_CTX_NONE
       sprintf(strbuf, "%s/%s.%d.ee.stim_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(),
	       sys->mpi_rank() );
       con_ee_stim_ctx->write_to_file(strbuf);

       #ifdef HAS_CROSS_REGION_EI_CON
        sprintf(strbuf, "%s/%s.%d.ei.stim_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(),
		sys->mpi_rank() );
	con_ei_stim_ctx->write_to_file(strbuf);
       #endif
      #endif

      #ifdef HAS_REPLAY
	#ifndef CON_REPLAY_CTX_NONE
	  sprintf(strbuf, "%s/%s.%d.ee.rep_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ee_rep_ctx->write_to_file(strbuf);

	  sprintf(strbuf, "%s/%s.%d.ei.rep_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_rep_ctx->write_to_file(strbuf);
	#endif
      #endif
	  
      #ifdef HAS_BACKGROUND
        sprintf(strbuf, "%s/%s.%d.ee.bg_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
        con_ee_bg_ctx->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	 sprintf(strbuf, "%s/%s.%d.ei.bg_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ei_bg_ctx->write_to_file(strbuf);
	#endif
      #endif

    #endif
  
    #ifdef HAS_HPC

      #ifdef HAS_REC_EE
       sprintf(strbuf, "%s/%s.%d.ee.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
       con_ee_hpc->write_to_file(strbuf);
      #endif

      sprintf(strbuf, "%s/%s.%d.ei.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ei_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ii.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ii_hpc->write_to_file(strbuf);

      sprintf(strbuf, "%s/%s.%d.ie.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
      con_ie_hpc->write_to_file(strbuf);
      
      #ifdef HAS_INH2
        sprintf(strbuf, "%s/%s.%d.ei2.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	con_ei2_hpc->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.i2i2.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	con_i2i2_hpc->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.i2e.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	con_i2e_hpc->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.ii2.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	con_ii2_hpc->write_to_file(strbuf);

	sprintf(strbuf, "%s/%s.%d.i2i.hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	con_i2i_hpc->write_to_file(strbuf);
      #endif

      #ifndef CON_STIM_HPC_NONE
	sprintf(strbuf, "%s/%s.%d.ee.stim_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(),
		sys->mpi_rank() );
	con_ee_stim_hpc->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
         sprintf(strbuf, "%s/%s.%d.ei.stim_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(),
		 sys->mpi_rank() );
	 con_ei_stim_hpc->write_to_file(strbuf);
        #endif
      #endif

      #ifdef HAS_REPLAY
	#ifndef CON_REPLAY_HPC_NONE
	  sprintf(strbuf, "%s/%s.%d.ee.rep_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ee_rep_hpc->write_to_file(strbuf);

	  sprintf(strbuf, "%s/%s.%d.ei.rep_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_rep_hpc->write_to_file(strbuf);
	#endif
      #endif

      #ifdef HAS_BACKGROUND
        sprintf(strbuf, "%s/%s.%d.ee.bg_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
        con_ee_bg_hpc->write_to_file(strbuf);

	#ifdef HAS_CROSS_REGION_EI_CON
	 sprintf(strbuf, "%s/%s.%d.ei.bg_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ei_bg_hpc->write_to_file(strbuf);
	#endif
      #endif

      #ifdef HAS_CTX

       #ifndef CON_CTX_HPC_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.ctx_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_ctx_hpc->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.ctx_hpc.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_ctx_hpc->write_to_file(strbuf);
	 #endif
       #endif

       #ifndef CON_HPC_CTX_NONE
	 sprintf(strbuf, "%s/%s.%d.ee.hpc_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	 con_ee_hpc_ctx->write_to_file(strbuf);

	 #ifdef HAS_CROSS_REGION_EI_CON
	  sprintf(strbuf, "%s/%s.%d.ei.hpc_ctx.wmat", out_dir.c_str(), full_file_prefix.c_str(), sys->mpi_rank() );
	  con_ei_hpc_ctx->write_to_file(strbuf);
	 #endif
       #endif

      #endif

    #endif

    previous_simtime = simtime;
  }

  // final checks and cleaning up
  if (errcode) auryn_abort(errcode);

  logger->msg("Freeing ...",PROGRESS,true);
  auryn_free();

  return errcode;

}

