/* 
* Source code modified by Douglas Feitosa Tomé in 2022
*
* The modified code is released under the the GNU General Public License
* as published by the Free Software Foundation, either version 3 
* of the License, or (at your option) any later version.
*
* Copyright 2014 Friedemann Zenke
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

#include "P11BConnection.h"

using namespace auryn;

void P11BConnection::init(AurynFloat eta, AurynFloat kappa, AurynFloat maxweight)
{
	set_name("P11BConnection");

	tau_plus  = 20e-3;
	tau_minus = 20e-3;

	tau_long  = 100e-3;

	set_tau_con(1200);

	set_max_weight(maxweight);

	pot_strength = 20.0;



	double fudge = 1;
	eta *= fudge;
	A3_plus  = 1*eta;
	A2_minus = 1*eta; // comes with hom vector implementation -- 10Hz corresponds to A2_minus=1
	set_beta(1e-1);


	tau_hom = 1200.0 ; 
	target_rate = kappa;
	hom = dst->get_post_trace(tau_hom);
	hom->set_all(1.0*tau_hom);
	// auryn_vector_float_set_all(hom,1.0); // initialize threshold at some value
	// dst->randomize_state_vector_gauss("P11hom_vector",0.1,0.5);
	// delta_hom = auryn_timestep/tau_hom;

	if ( dst->get_post_size() ) {
		tr_pre = src->get_pre_trace(tau_plus);
		tr_post = dst->get_post_trace(tau_minus);
		tr_post2 = dst->get_post_trace(tau_long);
	}


	delta    = 0.02*eta;


	w_solid_matrix = new ForwardMatrix ( w ); 
	set_weight_a(0.0);
	set_weight_c(0.5);
	w_solid_matrix->set_all(weight_a);

	stdp_active = true;
	consolidation_active = true;

	
	// cases where dst->evolve_locally() == true will be registered in SparseConnection
	// TODO write a check in System that Connections cannot be registered twice
	if ( src->evolve_locally() == true && dst->evolve_locally() == false )
		sys->register_connection(this);

	if ( src->get_rank_size() ) {
		// init of STP stuff
		tau_d = 0.2;
		tau_f = 1.0;
		Urest = 0.2;
		Ujump = 0.2;
		state_x = auryn_vector_float_alloc( src->get_vector_size() );
		state_u = auryn_vector_float_alloc( src->get_vector_size() );
		state_temp = auryn_vector_float_alloc( src->get_vector_size() );
		for (NeuronID i = 0; i < src->get_rank_size() ; i++)
		{
			   auryn_vector_float_set (state_x, i, 1 ); // TODO
			   auryn_vector_float_set (state_u, i, Ujump );
		}

	}

	// Registering the right number of spike attributes
	add_number_of_spike_attributes(1);
}

void P11BConnection::finalize() 
{
	DuplexConnection::finalize();
	init_shortcuts();
}

void P11BConnection::init_shortcuts() 
{
	if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

	fwd_ind = w->get_row_begin(0); 
	fwd_data = w->get_data_begin();

	bkw_ind = bkw->get_row_begin(0); 
	bkw_data = bkw->get_data_begin();
}

void P11BConnection::init_has_fwd_prop()
{
	for (int i = 0 ; i < src->get_size() ; ++i) {
	  has_forward_prop.push_back(true);
	}
	
	log_has_fwd_prop("initialization");
}

void P11BConnection::free()
{
	if ( src->get_rank_size() > 0 ) {
		auryn_vector_float_free (state_x);
		auryn_vector_float_free (state_u);
		auryn_vector_float_free (state_temp);
	}

	delete w_solid_matrix;
}

P11BConnection::P11BConnection(SpikingGroup * source, NeuronGroup * destination, TransmitterType transmitter) : DuplexConnection(source, destination, transmitter)
{
	init_has_fwd_prop();
}

P11BConnection::P11BConnection(SpikingGroup * source, NeuronGroup * destination, 
		const char * filename, 
		AurynFloat eta, 
		AurynFloat kappa, 
		AurynFloat maxweight , 
		TransmitterType transmitter) 
: DuplexConnection(source, 
		destination, 
		filename, 
		transmitter)
{
	init(eta, kappa, maxweight);
	init_shortcuts();
	init_has_fwd_prop();
}

P11BConnection::P11BConnection(SpikingGroup * source, NeuronGroup * destination, 
		AurynWeight weight, AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat kappa, AurynFloat maxweight , 
		TransmitterType transmitter,
		string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init( eta, kappa, maxweight);
	init_shortcuts();
	init_has_fwd_prop();
}

P11BConnection::~P11BConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}



inline AurynWeight P11BConnection::dw_pre(const NeuronID post, const AurynWeight * w)
{
	const AurynDouble mul = std::pow(hom->normalized_get(post)/target_rate,2);

	AurynDouble dw = -A2_minus*( (tr_post->get(post))*mul ) + delta;

	return dw;
}

inline AurynWeight P11BConnection::dw_post(const NeuronID pre, const NeuronID post, const AurynWeight * w, const AurynWeight w0)
{
	AurynFloat p = tr_post->get(post);
	// AurynDouble dw = A3_plus*p*tr_pre->get(pre)-beta_fudge*pow(p,3);
	AurynFloat l = tr_post2->get(post);
	AurynFloat dw = A3_plus*(l)*tr_pre->get(pre) -beta_fudge*pow(p,3)*(*w-w0) ;
	return dw;
}

void P11BConnection::propagate_forward()
{
  // loop over spikes
  for (int i = 0 ; i < src->get_spikes()->size() ; ++i ) {
    // get spike at pos i in SpikeContainer
    NeuronID spike = src->get_spikes()->at(i);
    
    // extract spike attribute from attribute stack;
    AurynFloat attribute = get_spike_attribute(i);
    
    // loop over postsynaptic targets
    for (NeuronID * c = w->get_row_begin(spike) ; 
	 c != w->get_row_end(spike) ; 
	 ++c ) {
      
      //if ( true ) {
      if ( has_forward_prop.at(spike) ) {
	AurynWeight value = fwd_data[c-fwd_ind] * attribute; 
	transmit( *c , value );
      }
      
      if ( stdp_active ) {
	NeuronID translated_spike = dst->global2rank(*c); // only to be used for post traces
	fwd_data[c-fwd_ind] += dw_pre(translated_spike,&fwd_data[c-fwd_ind]);
	if ( fwd_data[c-fwd_ind] < 0 ) 
	  fwd_data[c-fwd_ind] = 0.;
      }
    }
  }
}

void P11BConnection::propagate_backward()
{
	SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
	// process spikes
	for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin() ; // spike = post_spike
			spike != spikes_end ; ++spike ) {
		NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
		if (stdp_active) {
			for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {

				#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
				_mm_prefetch(bkw_data[c-bkw_ind+2],  _MM_HINT_NTA);
				#endif

				AurynWeight * cur = bkw_data[c-bkw_ind];
				AurynWeight cur0 = w_solid_matrix->get_data_begin()[cur-fwd_data];
				*cur = *cur + dw_post(*c,translated_spike,cur,cur0);
				if (*bkw_data[c-bkw_ind]>get_max_weight()) *bkw_data[c-bkw_ind]=get_max_weight();
				if (*bkw_data[c-bkw_ind]<get_min_weight()) *bkw_data[c-bkw_ind]=get_min_weight();
			}
		}
	}
}

void P11BConnection::propagate()
{
	if ( src->evolve_locally()) {
		push_attributes(); // stuffs all attributes into the SpikeDelays for sync
	}
	if ( dst->evolve_locally() ) { // necessary 
		// remember this connection can be registered although post might be empty on this node
		propagate_forward();
		propagate_backward();
	}
}

void P11BConnection::evolve()
{
	if ( src->evolve_locally() ) {

		// dynamics of x
		auryn_vector_float_set_all( state_temp, 1);
		auryn_vector_float_saxpy(-1,state_x,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_d,state_temp,state_x);

		// dynamics of u
		auryn_vector_float_set_all( state_temp, Urest);
		auryn_vector_float_saxpy(-1,state_u,state_temp);
		auryn_vector_float_saxpy(auryn_timestep/tau_f,state_temp,state_u);

	}

	// consolidation dynamics
	if ( sys->get_clock()%timestep_consolidation == 0 && stdp_active && consolidation_active ) {
		for (AurynLong i = 0 ; i < w_solid_matrix->get_nonzero() ; ++i ) {
			AurynWeight * fragile = w->get_data_begin()+i;
			AurynWeight * solid   = w_solid_matrix->get_data_begin()+i;
			AurynFloat dw = ( ( *fragile - *solid ) + ( pot_strength*(weight_a-*solid)*(weight_b-*solid)*(weight_c-*solid) ) ) * delta_consolidation;
			*solid += dw;
		}
	}

	// compute homeostatic threshold shift
	// auryn_vector_float_copy(  tr_post2->get_state_ptr(), state_temp );
	// auryn_vector_float_mul(  state_temp, tr_post2->get_state_ptr() );
	// auryn_vector_float_saxpy( -1.0, hom, state_temp );
	// auryn_vector_float_saxpy( auryn_timestep/tau_hom, state_temp, hom );

	// if ( sys->get_clock()%10000==0 ) {
	// 	for ( int i = 0 ; i < 100 ; ++i )
	// 		cout << -delta_hom*auryn_vector_float_get(state_temp,i) << endl;
	// }
}

void P11BConnection::consolidate() {
	for (AurynLong i = 0 ; i < w_solid_matrix->get_nonzero() ; ++i ) {
		AurynWeight * fragile = w->get_data_begin()+i;
		AurynWeight * solid   = w_solid_matrix->get_data_begin()+i;
		*solid = *fragile;
	}
}

void P11BConnection::push_attributes()
{
	SpikeContainer * spikes = src->get_spikes_immediate();
	for (SpikeContainer::const_iterator spike = spikes->begin() ;
			spike != spikes->end() ; ++spike ) {
		// dynamics 
		NeuronID spk = src->global2rank(*spike);
		double x = auryn_vector_float_get( state_x, spk );
		double u = auryn_vector_float_get( state_u, spk );
		auryn_vector_float_set( state_u, spk, u+Ujump*(1-u) );
		auryn_vector_float_set( state_x, spk, x-u*x );

		// TODO spike translation or introduce local_spikes function in SpikingGroup and implement this there ... (better option)
		// one attribute per spike - make sure to set set_num_spike_attributes for src
		double valueToSend = x*u;
		src->push_attribute( valueToSend ); 

		// cout.precision(5);
		// cout << x << " " << u << " " << valueToSend << endl;

	}
}

void P11BConnection::set_tau_f(AurynFloat tauf) {
	tau_f = tauf;
}

void P11BConnection::set_tau_d(AurynFloat taud) {
	tau_d = taud;
}

void P11BConnection::set_ujump(AurynFloat r) {
	Ujump = r;
}

void P11BConnection::set_urest(AurynFloat r) {
	Urest = r;
}

void P11BConnection::set_weight_a(AurynFloat w0) {
	weight_a = w0;
	weight_b = (weight_a+weight_c)/2;
}

void P11BConnection::set_weight_c(AurynFloat w2) {
	weight_c = w2;
	weight_b = (weight_a+weight_c)/2;
}

void P11BConnection::set_beta(AurynFloat b)
{
	beta = b;
	beta_fudge = beta; 
	logger->parameter("beta_fudge",beta_fudge);
}

bool P11BConnection::write_to_file(string filename)
{

	std::stringstream oss;
	oss << filename << "2";

	SparseConnection::write_to_file(w_solid_matrix,oss.str());

	oss.str("");
	oss << filename << ".cstate";
	std::ofstream outfile;
	outfile.open(oss.str().c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output file " << filename << std::endl;
	  throw AurynOpenFileException();
	}

	boost::archive::text_oarchive oa(outfile);
	for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
		oa << state_x->data[i];
		oa << state_u->data[i];
	}

	outfile.close();

	return SparseConnection::write_to_file(filename);
}

bool P11BConnection::load_from_file(string filename)
{

	std::stringstream oss;
	oss << filename << "2";

	SparseConnection::load_from_file(w_solid_matrix,oss.str());

	oss.str("");
	oss << filename << ".cstate";
	std::ifstream infile (oss.str().c_str());
	if (!infile) {
		std::stringstream oes;
		oes << "Can't open input file " << filename;
		logger->msg(oes.str(),ERROR);
		throw AurynOpenFileException();
	}

	boost::archive::text_iarchive ia(infile);
	for (NeuronID i = 0 ; i < src->get_rank_size() ; ++i ) {
		ia >> state_x->data[i];
		ia >> state_u->data[i];
	}

	infile.close();

	bool returnvalue = SparseConnection::load_from_file(filename);


	return returnvalue;
}

void P11BConnection::load_fragile_matrix(string filename)
{
	// load fragile (w) from complete file 
	SparseConnection::load_from_complete_file(filename);

	// now adapt the size of the solid matrix to the new w matrix
	w_solid_matrix->resize_buffer_and_clear(w->get_nonzero());
	// copy element positions
	w_solid_matrix->copy(w);
	// this replaces finalize
	w_solid_matrix->fill_zeros();
	// set all elements to the lower fixed point
	w_solid_matrix->set_all(weight_a);
}

void P11BConnection::set_tau_hom(AurynFloat tau)
{
	tau_hom = tau;
	hom->set_timeconstant(tau_hom);
}

void P11BConnection::set_tau_con(AurynFloat tau)
{
	tau_consolidation = tau;
	timestep_consolidation = 1e-3*tau_consolidation/auryn_timestep;
	delta_consolidation = 1.0*timestep_consolidation/tau_consolidation*auryn_timestep;;
	//logger->parameter("timestep_consolidation",(int)timestep_consolidation);
}

/*
void P11BConnection::print_has_fwd_prop()
{
	std::cout << "src->get_size() = " << src->get_size() << "\n";
	std::cout << "src->get_rank_size() = " << src->get_rank_size() << "\n";
	
	std::cout << "has_forward_prop\n\n";
	
	std::cout << "9: " << has_forward_prop.at(9) << "\n";
	std::cout << "146: " << has_forward_prop.at(146) << "\n";
	std::cout << "2080: " << has_forward_prop.at(2080) << "\n\n";

	std::cout << "6: " << has_forward_prop.at(6) << "\n";
	std::cout << "583: " << has_forward_prop.at(583) << "\n";
	std::cout << "999: " << has_forward_prop.at(999) << "\n\n";

	std::cout << "0: " << has_forward_prop.at(0) << "\n";
	std::cout << "149: " << has_forward_prop.at(149) << "\n";
	std::cout << "3049: " << has_forward_prop.at(3049) << "\n\n";

	int sum = 0;
	for (int i = 0 ; i < src->get_size() ; ++i ) {
	  sum += has_forward_prop.at(i);
	}
	std::cout << "sum: " << sum << "\n\n";
}
*/

void P11BConnection::log_has_fwd_prop(std::string stage)
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

void P11BConnection::block_pre_neurons(std::string filename, int nb_max_patterns)
{
	std::vector<type_pattern> patterns = load_pattern_file( filename, nb_max_patterns );
	
	for ( int i = 0 ; i < patterns.size() ; ++i ) {
		for ( std::vector<pattern_member>::iterator it = patterns[i].begin() ; it != patterns[i].end() ; ++it ) {
			has_forward_prop.at((*it).i) = false;
		}
	}
	
	log_has_fwd_prop("blocking pre neurons");
}
