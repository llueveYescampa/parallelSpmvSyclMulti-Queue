#!/bin/bash
if [ "$#" -lt 1 ]
then
  echo $#
  echo "Usage: $0 matrixName [nStreams]"
  exit 1
fi

matrix=$1
nStreams=$2

cd buildDouble


../runTest.sh parallelSpmvMulti-Streams  $matrix $nStreams

../runTest.sh myIpcsr  $matrix

../runTest.sh cudaSparseCSR  $matrix

../runTest.sh cudaSparseHyb  $matrix

