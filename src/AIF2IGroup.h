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

#ifndef AIF2IGROUP_H_
#define AIF2IGROUP_H_

#include "auryn.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

namespace auryn {

/*! \brief An adaptive integrate and fire group comparable to AIFGroup but with two independent adaptation timescales
 */
class AIF2IGroup : public AIFGroup
{
private:
	AurynFloat scale_adapt2;
	AurynFloat tau_adapt2;

	void init();
	void free();

protected:
	auryn_vector_float * g_adapt2;

	void calculate_scale_constants();
	void integrate_linear_nmda_synapses();
	void check_thresholds();

public:
	AIF2IGroup( NeuronID size, NodeDistributionMode distmode = AUTO);
	virtual ~AIF2IGroup();

	void random_adapt(AurynState mean, AurynState sigma);

	AurynFloat dg_adapt2;

	void clear();
	virtual void evolve();

	std::vector<bool> can_spike;

	void init_can_spike();

	void log_can_spike(string stage="");

	void inhibit_neurons(std::string filename, int nb_max_patterns=1000);

	std::vector<type_pattern> load_pattern_file( string filename, int nb_max_patterns);
};

}

#endif /*AIF2IGROUP_H_*/

