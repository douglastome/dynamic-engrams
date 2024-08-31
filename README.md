# Dynamic and selective engrams emerge with memory consolidation

This repository contains the code used to perform the simulations and data analyses reported in our manuscript:

```
Tomé, D.F., Zhang, Y., Aida, T., Mosto, O., Lu, Y., Chen, M., Sadeh, S., Roy, D. S., Clopath, C.
Dynamic and selective engrams emerge with memory consolidation. Nature Neuroscience. 2024.
```

This file provides specific instructions to reproduce our key simulation results (Fig. 1d-k, Extended Data Fig. 2a-g, and Extended Data Fig. 3j-m). The remaining simulations and data analyses in the manuscript (including analyses of experimental data) can also be reproduced with this source code by modifying simulation and data analysis parameters (see below).

This code has been tested on machines running Ubuntu 18.04.5 LTS with at least 16 cores. If running on a machine with less than 16 cores, reset the number of MPI ranks in `globalvars.sh` accordingly.

## Installing prerequisites

Download the spiking network simulator Auryn version `6928b97` available at https://github.com/fzenke/auryn/tree/6928b97de024b47f696091c1d3de18ff535ffc91 and place it in your home directory. 

Unzip the Auryn source code and install Auryn:

```
cd ~
unzip auryn-6928b97de024b47f696091c1d3de18ff535ffc91
mv auryn-6928b97de024b47f696091c1d3de18ff535ffc91 auryn
sudo apt-get install cmake git build-essential libboost-all-dev
cd auryn/build/release
./bootstrap.sh && make
```

If you experience issues when installing Auryn, you may refer to its official documentation at https://fzenke.net/auryn/doku.php?id=quick_start.

Now create a Python 3 (version 3.11) virtual environment `venv_sim` in your home directory and install the necessary packages:

```
cd ~
sudo apt-get install python3-venv
python3 -m venv venv_sim
source venv_sim/bin/activate
pip install cython
pip install numpy
pip install pandas
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install statsmodels
pip install bioinfokit
pip install pingouin
deactivate
```

Lastly, place the source code directory `src` in your home directory, create the simulation directory in the preset path, and move the source code directory `src` there:

```
cd ~
mkdir -p projects/dynamic-engrams/simulations/sim_rc_p11/run-001
mv src/ projects/dynamic-engrams/simulations/sim_rc_p11/run-001/
```


## Running a simulation

Run the simulation in Fig. 1b:

```
cd ~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/
make clean
make
./run.sh 11
```

This will produce ensemble overlap and memory recall curves in the individual directories of trials 0-9 in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/`.

## Combining simulation trials

Take the steps below:

1) Open the file `run.sh` using a text editor and set the variable `HAS_SIMULATION` to `false` and the variable `HAS_MERGE` to `true`. Save `run.sh`.
2) Open the file `analyze_run.sh` using a text editor and set the variable `HAS_MERGE` to `true`. Save `analyze_run.sh`.
3) Merge trials:

```
cd ~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/
./run.sh 11
```

This will produce average ensemble overlap and memory recall curves with 99% confidence intervals in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/`.

## Analysis of simulation

For the discrimination curve in Fig. 1j, the stimulus overlap plot in Fig. 1d, and the cumulative distributions in Extended Data Fig. 3j-m:

```
cd ~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/
~/venv_sim/bin/python data_analyzer.py
```

This will plot average discrimination with 99% confidence intervals in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/`, stimulus overlap in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/data`, and cumulative distributions in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/0`.

For the discrimination curve in Extended Data Fig. 2f, set `neuron_type` to `inh` in `data_analyzer.py` and repeat the procedure above.

## Analysis of experimental data

The experimental data in Fig. 3, Fig. 5e-j, Fig. 6e-j, Extended Data Fig. 6i-j, and Extended Data Fig. 8b-c is provided in `src/data/experimental-data.csv`. A description of the data in this file is provided in `src/data/experimental-data-description.csv`. To plot the experimental data in this file, set the variable `has_simulation_analysis` to `False` and the variable `has_experiment_analysis` to `True` in `data_analyzer.py` and then:

```
cd ~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/
~/venv_sim/bin/python data_analyzer.py
```

This will plot Fig. 3b-d/f, Fig. 5f-g/i-j, Fig. 6f-g, Extended Data Fig. 6j, and Extended Data Fig. 8b-c in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/data`.

To perform statistical tests of the experimental data in `src/data/experimental-data.csv`, choose the data to be compared and the test to be conducted using the variables `data1`, `data2`, and `data3` in `data_analyzer.py` and repeat the procedure above.

The longitudinal imaging data in Fig. 4 and Extended Data Fig. 9 is provided in `src/data/mouse*.csv`. A description of the data in these files is provided in `src/data/imaging-data-description.txt`. To analyze the imaging data in these files and to perform the associated statistical tests:

```
cd ~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/
~/venv_sim/bin/python imaging_analyzer.py
```

This will plot Fig. 4b, Fig. 4c (dynamic engram), Fig. 4d (dynamic engram), and Extended Data Fig. 9b-g in `~/projects/dynamic-engrams/simulations/sim_rc_p11/run-001/src/data`. Output of statistical tests will be printed in the terminal. To plot Fig. 4c (stable engram) and Fig. 4d (stable engram), set `engram_type` to `stable` in `imaging_analyzer.py` and repeat the procedure above. To plot Fig. 4c (random neurons) and Fig. 4d (random neurons), set `engram_type` to `random` in `imaging_analyzer.py` and repeat the procedure above. To plot Extended Data Fig. 9a, set `engram_id` to `nmf` in `imaging_analyzer.py` and repeat the procedure above.

## Further data analyses and simulations

To further analyze the simulation above (Fig. 1l, Extended Data Fig. 1e-h, Extended Data Fig. 2h), modify parameters in `run.sh` and `analyze_run.sh` accordingly and run `./run.sh 11` from the source code location as in the procedure to combine trials outlined above.

To change the time window used to identify engram cells in the training phase of the simulation (Extended Data Fig. 1a-d), modify the method `parse_run_10_11` in `parser.py` accordingly and re-run the simulation as outlined above.

To use an NMF-based approach to identify engram cells in the simulation (Extended Data Fig. 4j-o, Extended Data Fig. 2i-n), set `CELL_ASSEMB_METHOD` to `nmf` in `analyze_run.sh` and re-run the simulation as outlined above.

To block training-activated engram cells during recall or to reactivate training-activated engram cells during recall without cue presentations (Fig. 2), add the appropriate command-line arguments to `run-test_cue.sh` (see `sim_rc_p11.cpp`), modify parameters in `globalvars.sh` accordingly, and re-run the simulation as outlined above. 

To block inhibitory neurons during recall (Fig. 5a-d), add the appropriate command-line arguments to `run-test_cue.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To block inhibitory synaptic plasticity during consolidation (Fig. 6a-d), set `ETA_EXC_INH_HPC_CON` to `-50.0` in `globalvars.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To block sensory reactivation during consolidation (Extended Data Fig. 3a-i), add the appropriate command-line arguments to `run-consolidation.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To block long-term potentiation during consolidation (Extended Data Fig. 4a-i), modify the implementation of excitatory synaptic plasticity in `P11BConnection.cpp` and re-run the simulation for the consolidation, probing, and recall phases.

To block triplet STDP in the entire simulation (Extended Data Fig. 5a-b), modify the implementation of excitatory synaptic plasticity in `P11BConnection.cpp` and re-run the simulation as outlined above.

To block heterosynaptic plasticity in the entire simulation (Extended Data Fig. 5c-d), set `BETA_HPC` and `BETA_STIM_HPC` to `0` in `globalvars.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To block transmitter-induced plasticity in the entire simulation (Extended Data Fig. 5e-f), set `DELTA_HPC` to `0` in `globalvars.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To block inhibitory synaptic plasticity in the entire simulation (Extended Data Fig. 6a-h), set `ETA_EXC_INH_HPC` to `-50.0` and `ETA_EXC_INH_HPC_CON` to `-50.0` in `globalvars.sh` (see `sim_rc_p11.cpp`) and re-run the simulation as outlined above.

To simulate the network with the alternative form of inhibitory synaptic plasticity described in the manuscript (Extended Data Fig. 7), modify the implementation of inhibitory synaptic plasticity in `GlobalPFBConnection.cpp` and re-run the simulation as outlined above.

To simulate the expanded network with parvalbumin- and cholecystokinin-expressing interneurons (Extended Data Fig. 10), uncomment `HAS_INH2` in  `sim_rc_p11.cpp`, modify the appropriate parameters in `globalvars.sh`, and repeat the procedure above to compile the code and run the simulation.
