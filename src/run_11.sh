#!/bin/bash

# command-line arguments:
# 1: trial id
# 2: run type (1,2,...)
# 3: run id (e.g., run-001)

TRIAL=$1
RUN=$2
RUN_ID=$3

# output directory
OUT_DIR="$HOME/projects/dynamic-engrams/simulations/sim_rc_p11/$RUN_ID"

# source directory
SOURCE="src"
DIR=$OUT_DIR/$SOURCE

$DIR/run-burn_in.sh $TRIAL $RUN $RUN_ID
$DIR/run-learn.sh $TRIAL $RUN $RUN_ID
$DIR/run-consolidation.sh $TRIAL $RUN $RUN_ID
$DIR/run-probe.sh $TRIAL $RUN $RUN_ID
$DIR/run-test_cue.sh $TRIAL $RUN $RUN_ID
