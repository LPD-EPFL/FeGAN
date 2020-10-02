#!/bin/bash
nodes=$9 #"nodes"
n=$1
E=$2
C=$3
B=$4
s=$5
w=$6
model=$7
magic=$8
num_servers=$n
iid=${10}
job_id=${11}
port=2379
master=''
fid_master=''
num_machines=0
while read node; do
	num_machines=$((num_machines+1))
	if [ $num_machines -eq 1 ]
	then
		master=$node
                fid_master=$node
	fi
#        if [ $num_machines -eq 2 ]
#        then
#                fid_master=$node
#        fi
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
common="$pwd/dist-lsgan.py --size $n --master $fid_master --local_steps $E --frac_workers $C --batch_size $B --sample $s --weight_avg $w --model $model --port $port --magic_num $magic --iid $iid"
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
                if false; #[ $r -lt $num_servers ]
                then
          		cmd="python -m torchelastic.distributed.launch --nnodes=$num_servers --nproc_per_node=1 --rdzv_id=$job_id --rdzv_backend=etcd --rdzv_endpoint=$master:$port $common --rank $r"
                else
                        cmd="python3 $common --rank $r"
                fi
                echo "running $cmd on $node"
		ssh $node $cmd < /dev/tty &
		r=$((r+1))
	     done
done < $nodes
