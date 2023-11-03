#!/bin/bash
if [ "$#" -lt 2 ]
then
  echo "Usage: $0 parallelSpmvMulti-Streams matrixName [#_of_Streams"]
  exit 1
fi

source setCuda
export CUDA_VISIBLE_DEVICES=1

tempFilename=$(hostname)'_anyTempFileNameWillWork.txt'
outputFilename=$1.txt

nloops=5


rm -f $tempFilename
for j in  `seq 1 $nloops`; do
    #echo run number: $j, using 4 processes 
    echo run number: $j
    echo  $1  ../matrices/$2".mm_bin" ../matrices/$2".in_bin"  $3
    $1  ../matrices/$2".mm_bin" ../matrices/$2".in_bin"   ../matrices/$2".out_bin" $3 | grep was >>  $tempFilename
done

mkdir -p ../plots/$(hostname)/$2

#cat $tempFilename | awk 'BEGIN{}   { printf("%d %f\n", $5,$7)}  END{}' |  sort  -k1,1n -k2,2n |  awk 'BEGIN{ prev=-1} { if ($1 != prev) { print $0; prev=$1}  } END{}' > ../plots/$(hostname)/$2/$outputFilename
cat $tempFilename | awk 'BEGIN{}   { printf("%f  %f\n", $4,$7)}  END{}' |  sort  -k1,1n -k2,2n |   head -1  > ../plots/$(hostname)/$2/$outputFilename

rm $tempFilename

