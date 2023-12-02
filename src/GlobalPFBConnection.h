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

#ifndef GLOBALPFBCONNECTION_H_
#define GLOBALPFBCONNECTION_H_

#include "auryn.h"


namespace auryn {

class GlobalPFBConnection : public DuplexConnection
{

private:


	void init(AurynFloat tau_hom, AurynFloat eta, AurynFloat kappa, AurynFloat maxweight);
	void init_shortcuts();



protected:

	AurynFloat tau_post;
	AurynDouble post_factor_mul;
	AurynFloat  expected_spikes;

	AurynFloat post_factor;

	NeuronID * fwd_ind; 
	AurynWeight * fwd_data;

	NeuronID * bkw_ind; 
	AurynWeight ** bkw_data;

	AurynDouble target_rate;

	Trace * tr_pre;
	Trace * tr_post;

	inline AurynWeight dw_pre(NeuronID post);
	inline AurynWeight dw_post(NeuronID pre);

	void propagate_forward();
	inline void propagate_backward();

    void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
    {
        DuplexConnection::virtual_serialize(ar,version);
		ar & post_factor ;
    }

    void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
    {
        DuplexConnection::virtual_serialize(ar,version);
		ar & post_factor ;
    }


public:
	AurynFloat zeta; //!< Presynaptically triggered decay variable similar to Vogels et al. 2011
	AurynFloat xi; //!< Floor parameter in G modulated part (default=1.0).

	AurynFloat learning_rate;


	/*! Toggle stdp active/inactive. When inactive traces are still updated, but weights are not. */
	bool stdp_active;

	GlobalPFBConnection(SpikingGroup * source, NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	GlobalPFBConnection(SpikingGroup * source, NeuronGroup * destination, 
			const char * filename, 
			AurynFloat tau_hom=10, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT);

	/*! Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * @param source the presynaptic neurons.
	 * @param destinatino the postsynaptic neurons.
	 * @param weight the initial synaptic weight.
	 * @param sparseness the sparseness of the connection (probability of connection).
	 * @param tau_hom the timescale of the homeostatic rate estimate (moving average).
	 * @param eta the relaive learning rate (default=1).
	 * @param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * @param name a sensible identifier for the connection used in debug output.
	 */
	GlobalPFBConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			AurynFloat tau_hom=1, 
			AurynFloat eta=1, 
			AurynFloat kappa=3., AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "GlobalPFBConnection" );

	virtual ~GlobalPFBConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

	virtual bool load_from_file(string filename);
	virtual bool write_to_file(string filename);

	std::vector<bool> has_forward_prop;

	void init_has_fwd_prop();

	void log_has_fwd_prop(string stage="");

	void block_pre_neurons(std::string filename, int nb_max_patterns=1000);

};

}

#endif /*GLOBALPFBCONNECTION_H_*/
