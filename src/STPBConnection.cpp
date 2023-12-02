/* 
* Source code modified by Douglas Feitosa Tomé in 2022
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

#include "STPBConnection.h"

using namespace auryn;

void STPBConnection::init() 
{
	if ( src->get_rank_size() > 0 ) {
		// init of STPB stuff
		tau_d = 0.2;
		tau_f = 1.0;
		Ujump = 0.01;
		state_x = auryn_vector_float_alloc( src->get_rank_size() );
		state_u = auryn_vector_float_alloc( src->get_rank_size() );
		state_temp = auryn_vector_float_alloc( src->get_rank_size() );
		clear();
	}

	// Registering the right amount of spike attributes
	// this line is very important finding bugs due to 
	// this being wrong or missing is hard 
	add_number_of_spike_attributes(1);

}

void STPBConnection::init_has_fwd_prop()
{
	for (int i = 0 ; i < src->get_size() ; ++i) {
	  has_forward_prop.push_back(true);
	}
	
	log_has_fwd_prop("initialization");
}


STPBConnection::STPBConnection(const char * filename) 
: SparseConnection(filename)
{
	init();
	init_has_fwd_prop();
}

STPBConnection::STPBConnection(SpikingGroup * source, NeuronGroup * destination, 
		TransmitterType transmitter) 
: SparseConnection(source, destination, transmitter)
{
	init_has_fwd_prop();
}

STPBConnection::STPBConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename , 
		TransmitterType transmitter) 
: SparseConnection(source, destination, filename, transmitter)
{
	init();
	init_has_fwd_prop();
}


STPBConnection::STPBConnection(NeuronID rows, NeuronID cols) 
: SparseConnection(rows,cols)
{
	init();
	init_has_fwd_prop();
}

STPBConnection::STPBConnection( SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		TransmitterType transmitter, std::string name) 
: SparseConnection(source,destination,weight,sparseness,transmitter, name)
{
	init();
	init_has_fwd_prop();
}

void STPBConnection::free()
{
	if ( src->get_rank_size() > 0 ) {
		auryn_vector_float_free (state_x);
		auryn_vector_float_free (state_u);
		auryn_vector_float_free (state_temp);
	}
}



STPBConnection::~STPBConnection()
{
	free();
}

void STPBConnection::push_attributes()
{
	// need to push one attribute for each spike
	SpikeContainer * spikes = src->get_spikes_immediate();
	for (SpikeContainer::const_iterator spike = spikes->begin() ;
			spike != spikes->end() ; ++spike ) {
		// dynamics 
		NeuronID spk = src->global2rank(*spike);
		double x = auryn_vector_float_get( state_x, spk );
		double u = auryn_vector_float_get( state_u, spk );
		auryn_vector_float_set( state_x, spk, x-u*x );
		auryn_vector_float_set( state_u, spk, u+Ujump*(1-u) );

		// TODO spike translation or introduce local_spikes 
		// function in SpikingGroup and implement this there ... (better option)
		src->push_attribute( x*u ); 

	}

	// If we had two spike attributes in this connection we push 
	// the second attribute for each spike here:
	//
	// SpikeContainer * spikes = src->get_spikes_immediate();
	// for (SpikeContainer::const_iterator spike = spikes->begin() ;
	// 		spike != spikes->end() ; ++spike ) {
	// 	AurynFloat other_attribute = foo+bar;
	// 	src->push_attribute( other_attribute ); 
	// }
}

void STPBConnection::evolve()
{
	if ( src->evolve_locally() ) {
		// dynamics of x
		auryn_vector_float_set_all( state_temp, 1);
		auryn_vector_float_saxpy(-1,state_x,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_d,state_temp,state_x);

		// dynamics of u
		auryn_vector_float_set_all( state_temp, Ujump);
		auryn_vector_float_saxpy(-1,state_u,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_f,state_temp,state_u);
	}
}

void STPBConnection::propagate()
{
	if ( src->evolve_locally()) {
		push_attributes(); // stuffs all attributes into the SpikeDelays for sync
	}

	if ( dst->evolve_locally() ) { // necessary 
		NeuronID * ind = w->get_row_begin(0); // first element of index array
		AurynWeight * data = w->get_data_begin(); // first element of data array

		// loop over spikes
		for (NeuronID i = 0 ; i < src->get_spikes()->size() ; ++i ) {
			// get spike at pos i in SpikeContainer
			NeuronID spike = src->get_spikes()->at(i);

			//if ( true ) {
			if ( has_forward_prop.at(spike) ) {
				// extract spike attribute from attribute stack;
				AurynFloat attribute = get_spike_attribute(i);

				// loop over postsynaptic targets
				for (NeuronID * c = w->get_row_begin(spike) ; 
						c != w->get_row_end(spike) ; 
						++c ) {
					AurynWeight value = data[c-ind] * attribute;
					transmit( *c , value );
				}
			}
		}
	}
}

void STPBConnection::set_tau_f(AurynFloat tauf) {
	tau_f = tauf;
}

void STPBConnection::set_tau_d(AurynFloat taud) {
	tau_d = taud;
}

void STPBConnection::set_ujump(AurynFloat r) {
	Ujump = r;
}

void STPBConnection::clear() 
{
	for (NeuronID i = 0; i < src->get_rank_size() ; i++)
	{
		   auryn_vector_float_set (state_x, i, 1 ); // TODO
		   auryn_vector_float_set (state_u, i, Ujump );
	}
}

void STPBConnection::clone_parameters(STPBConnection * con) 
{
	tau_d = con->tau_d;
	tau_f = con->tau_f;
	Ujump = con->Ujump;
	clear();
}

void STPBConnection::log_has_fwd_prop(string stage)
{
	std::stringstream oss;
	if ( !stage.empty() ) {
		oss << get_log_name() << "Connection " << src->get_name() << "->" << dst->get_name() << ": has_forward_prop at " << stage;
		logger->msg(oss.str(),SETTINGS);
		oss.str(std::string());
	}
	
	int sum = 0;
	
	for (int i = 0 ; i < src->get_size() ; ++i ) {
		oss << get_log_name() << "has_forward_prop.at(" << i << ") = " << has_forward_prop.at(i);
		logger->msg(oss.str(),SETTINGS);
		oss.str(std::string());
		sum += has_forward_prop.at(i);
	}
	
	oss << get_log_name() << "sum(has_forward_prop) = " << sum;
	logger->msg(oss.str(),SETTINGS);
}

void STPBConnection::block_pre_neurons(std::string filename, int nb_max_patterns)
{
	std::vector<type_pattern> patterns = load_pattern_file( filename, nb_max_patterns );
	
	for ( int i = 0 ; i < patterns.size() ; ++i ) {
		for ( std::vector<pattern_member>::iterator it = patterns[i].begin() ; it != patterns[i].end() ; ++it ) {
			has_forward_prop.at((*it).i) = false;
		}
	}
	
	log_has_fwd_prop("blocking pre neurons");
}
