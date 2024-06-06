#!/bin/bash

generatePartition(){
	add=1
	val=`expr $1 + $add`
	# ViT Base [1,48]
	echo "1,$1,$val,48"
	# ViT Large [1,96]
	# echo "1,$1,$val,96"
}

# sbatch upload_eval.job 1,47,48,96 0

for bit in 2 4 6 8 16
do
	# ViT Base [1,47]
	for pt in `seq 1 47`
	# ViT Large [1,95]
	# for pt in `seq 1 95`
	do
		pt=$(generatePartition $pt)
		sbatch upload_eval.job $pt $bit
	done
done		
