#!/bin/bash
nodes=$3 #"nodes"
n=$1
model=$2
port=29500
master=''
num_machines=0
while read node; do
	num_machines=$((num_machines+1))
	if [ $num_machines -eq 1 ]
	then
		master=$node
	fi
done < $nodes
num_gpus=$(((num_machines-1)*2+1))
echo "Total number of GPUs for workers in this deployment: $num_gpus"
node_per_gpu=$((n/num_gpus))
echo "Number of nodes per GPU $node_per_gpu"
if [ $((n%node_per_gpu)) != 0 ]
then
	echo "WARNING: Choose a value of n that is divisible by $node_per_gpu in this setup; the current value for n is $n"
fi
if [ $node_per_gpu -gt 16 ]
then
        echo "WARNING: The number of nodes per GPU to be placed is HUGE...the maximum number is 16...You can go to 25 nodes on 96 GB GPU.."
fi
pwd=`pwd`
common="python3 $pwd/md-gan.py --size $n --master $master --model $model --port $port"
#for i in `seq 0 $((n-1))`;
#   do
#     $common --rank $i &
#   done
r=0
while read node; do
	if [ $r -lt $node_per_gpu ]
	then
	   node_per_machine=$node_per_gpu
	else
	   node_per_machine=$((node_per_gpu*2))
	fi
	for i in `seq 0 $((node_per_machine-1))`;
	     do
		cmd="$common --rank $r"
                echo "running $cmd on $node"
		ssh $node $cmd < /dev/tty &
		r=$((r+1))
	     done
done < $nodes
