/* 
* Source code modified by Douglas Feitosa Tomé in 2023
*
* The modified code is released under the the GNU General Public License
* as published by the Free Software Foundation, either version 3 
* of the License, or (at your option) any later version.
*
* Copyright 2014-2018 Friedemann Zenke
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
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "AIF2IGroup.h"

using namespace auryn;


AIF2IGroup::AIF2IGroup( NeuronID size, NodeDistributionMode distmode ) : AIFGroup(size, distmode)
{
	// auryn::sys->register_spiking_group(this); // already registered in AIFGroup
	if ( evolve_locally() ) {
	  init();
	  init_can_spike();
	}
}

void AIF2IGroup::init_can_spike()
{
	for (int i = 0 ; i < get_size() ; ++i) {
	  can_spike.push_back(true);
	}
	
	log_can_spike("initialization");
}

void AIF2IGroup::calculate_scale_constants()
{
	AIFGroup::calculate_scale_constants();
	scale_adapt2 = exp(-auryn_timestep/tau_adapt2);
}

void AIF2IGroup::init()
{
	tau_adapt2 = 20.0;
	dg_adapt2  = 0.002;

	calculate_scale_constants();
	g_adapt2 = get_state_vector ("g_adapt2");

	clear();
}

void AIF2IGroup::clear()
{
	AIFGroup::clear();
	for (NeuronID i = 0; i < get_rank_size(); i++) {
	   auryn_vector_float_set (g_adapt2, i, 0.);
	 }
}


void AIF2IGroup::random_adapt(AurynState mean, AurynState sigma)
{
	boost::mt19937 ng_gen(42); // produces same series every time 
	boost::normal_distribution<> dist((double)mean, (double)sigma);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die(ng_gen, dist);
	AurynState rv;

	for ( AurynLong i = 0 ; i<get_rank_size() ; ++i ) {
		rv = die();
		if ( rv>0 ) 
			g_adapt1->set( i, rv ); 
		rv = die();
		if ( rv>0 ) 
			g_adapt2->set( i, rv ); 
	}

	init_state();
}

void AIF2IGroup::free()
{
}

AIF2IGroup::~AIF2IGroup()
{
	if ( evolve_locally() ) free();
}

void AIF2IGroup::integrate_linear_nmda_synapses()
{
	// decay of ampa and gaba channel, i.e. multiply by exp(-auryn_timestep/tau)
    auryn_vector_float_scale(scale_ampa,g_ampa);
    auryn_vector_float_scale(scale_gaba,g_gaba);
    auryn_vector_float_scale(scale_adapt1,g_adapt1);
    auryn_vector_float_scale(scale_adapt2,g_adapt2);

    // compute dg_nmda = (g_ampa-g_nmda)*auryn_timestep/tau_nmda and add to g_nmda
	AurynFloat mul_nmda = auryn_timestep/tau_nmda;
    auryn_vector_float_saxpy(mul_nmda,g_ampa,g_nmda);
	auryn_vector_float_saxpy(-mul_nmda,g_nmda,g_nmda);

    // excitatory
    auryn_vector_float_copy(g_ampa,t_exc);
    auryn_vector_float_scale(-A_ampa,t_exc);
    auryn_vector_float_saxpy(-A_nmda,g_nmda,t_exc);
    auryn_vector_float_mul(t_exc,mem);
    
    // inhibitory
    auryn_vector_float_copy(g_gaba,t_leak); // using t_leak as temp here
    auryn_vector_float_saxpy(1,g_adapt1,t_leak);
    auryn_vector_float_saxpy(1,g_adapt2,t_leak);
    auryn_vector_float_copy(mem,t_inh);
    auryn_vector_float_add_constant(t_inh,-e_rev);
    auryn_vector_float_mul(t_inh,t_leak);
}


void AIF2IGroup::check_thresholds()
{
	auryn_vector_float_clip( mem, e_rev );

	AurynState * thr_ptr = thr->data;
	for ( AurynState * i = mem->data ; i != mem->data+get_rank_size() ; ++i ) { // it's important to use rank_size here otherwise there might be spikes from units that do not exist
    	if ( *i > ( thr_rest + *thr_ptr ) ) {
			NeuronID unit = i-mem->data;
			if ( can_spike.at(rank2global(unit)) ) {
			  push_spike(unit);
			}
		    mem->set( unit, e_rest); // reset
	        thr->set( unit, dthr); //refractory
			g_adapt1->add_specific( unit, dg_adapt1);
			g_adapt2->add_specific( unit, dg_adapt2);
		} 
		thr_ptr++;
	}

}

void AIF2IGroup::evolve()
{
	integrate_linear_nmda_synapses();
	integrate_membrane();
	check_thresholds();
}

void AIF2IGroup::log_can_spike(std::string stage)
{
	std::stringstream oss;
	if ( !stage.empty() ) {
		oss << get_log_name() << "SpikingGroup " << get_name() << ": can_spike at " << stage;
		logger->msg(oss.str(),SETTINGS);
		oss.str(std::string());
	}
	
	int sum = 0;
	
	for (int i = 0 ; i < get_size() ; ++i ) {
		oss << get_log_name() << "can_spike.at(" << i << ") = " << can_spike.at(i);
		logger->msg(oss.str(),SETTINGS);
		oss.str(std::string());
		sum += can_spike.at(i);
	}
	
	oss << get_log_name() << "sum(can_spike) = " << sum;
	logger->msg(oss.str(),SETTINGS);
}

void AIF2IGroup::inhibit_neurons(std::string filename, int nb_max_patterns)
{
	std::vector<type_pattern> patterns = load_pattern_file( filename, nb_max_patterns );
	
	for ( int i = 0 ; i < patterns.size() ; ++i ) {
		for ( std::vector<pattern_member>::iterator it = patterns[i].begin() ; it != patterns[i].end() ; ++it ) {
			can_spike.at((*it).i) = false;
		}
	}
	
	log_can_spike("inhibiting neurons");
}


std::vector<type_pattern> AIF2IGroup::load_pattern_file( string filename, int nb_max_patterns)
{
	std::vector<type_pattern> patterns;

	std::ifstream fin (filename.c_str());
	if (!fin) {
		std::stringstream oss2;
		oss2 << get_log_name() << "There was a problem opening file " << filename << " for reading.";
		auryn::logger->msg(oss2.str(),WARNING);
		return patterns;
	} else {
		std::stringstream oss;
		oss << get_log_name() << "Loading patterns from " << filename << " ...";
		auryn::logger->msg(oss.str(),NOTIFICATION);
	}

	unsigned int patcount = 0 ;

	//NeuronID mindimension = std::min( get_m_rows()*patterns_every_pre, get_n_cols()*patterns_every_post );
	//bool istoolarge = false;
	

	type_pattern pattern;
	char buffer[256];
	std::string line;

	while(!fin.eof()) {
		line.clear();
		fin.getline (buffer,255);
		line = buffer;

		if(line[0] == '#') continue;
		if ( patcount >= nb_max_patterns ) break;
		if (line == "") { 
			if ( pattern.size() > 0 ) {
				// put_pattern( &pattern, strength, overwrite );
				patterns.push_back(pattern);
				patcount++;
				pattern.clear();
			}
			continue;
		}

		pattern_member pm;
		std::stringstream iss (line);
		pm.gamma = 1 ; 
		iss >>  pm.i ;
		//if ( !wrap_patterns && !istoolarge && pm.i > mindimension ) { 
		if ( pm.i > get_size() - 1 ) { 
			std::stringstream oss;
			oss << get_log_name() << "Some elements of pattern " << patcount << " are larger than the underlying NeuronGroup!";
			auryn::logger->msg(oss.str(),WARNING);
			//istoolarge = true;
		}
		iss >>  pm.gamma ;
		//if ( patterns_ignore_gamma ) 
		//	pm.gamma = 1;
		pattern.push_back(pm) ;
	}

	fin.close();

	return patterns;
}
