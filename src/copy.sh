#!/bin/bash

. ./globalvars.sh

SRC_DIR=$OUT_DIR/src

mkdir -p $SRC_DIR

echo src_dir: $SRC_DIR

cp -r * $SRC_DIR
