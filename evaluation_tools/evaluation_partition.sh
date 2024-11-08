#!/bin/bash

# vit-base
# generatePartition(){
# 	add=1
# 	val=`expr $1 + $add`
# 	# ViT Base [1,48]
# 	echo "1,$1,$val,48"
# }

# for bit in 16 8 6 4 2
# do
# 	for e in `seq 0 $((bit - 1))`
# 	do
# 		# ViT Base [1,47]
# 		for pt in `seq 1 47`
# 		do
# 			pt=$(generatePartition $pt)
# 			sbatch upload_eval_clamp.job $pt $bit $e
# 			sbatch upload_eval_noClamp.job $pt $bit $e
# 		done
# 	done
# done

# vit-large
# generatePartition(){
# 	add=1
# 	val=`expr $1 + $add`
# 	# ViT Base [1,96]
# 	echo "1,$1,$val,96"
# }

# for bit in 16 8 6 4 2
# do
# 	for e in `seq 0 $((bit - 1))`
# 	do
# 		# ViT Large [1,95]
# 		for pt in `seq 1 95`
# 		do
# 			pt=$(generatePartition $pt)
# 			# sbatch upload_eval_clamp.job $pt $bit $e
# 			sbatch upload_eval_noClamp.job $pt $bit $e
# 		done
# 	done
# done

# resnet18
# generatePartition(){
# 	add=1
# 	val=`expr $1 + $add`
# 	# resnet 18 [1,21]
# 	echo "1,$1,$val,21"
# }

# for bit in 16 8 6 4 2
# do
# 	for e in `seq 0 $((bit - 1))`
# 	do
# 		# resnet 18 [1,20]
# 		for pt in `seq 1 20`
# 		do
# 			pt=$(generatePartition $pt)
# 			sbatch upload_eval_clamp.job $pt $bit $e
# 			sbatch upload_eval_noClamp.job $pt $bit $e
# 		done
# 	done
# done

# resnet50
generatePartition(){
	add=1
	val=`expr $1 + $add`
	# resnet 50 [1,54]
	echo "1,$1,$val,54"
}

for bit in 16 8 6 4 2
do
	for e in `seq 0 $((bit - 1))`
	do
		# resnet 50 [1,53]
		for pt in `seq 1 53`
		do
			pt=$(generatePartition $pt)
			sbatch upload_eval_clamp.job $pt $bit $e
			sbatch upload_eval_noClamp.job $pt $bit $e
		done
	done
done
