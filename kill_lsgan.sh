#!/bin/bash
nodes=$1
echo "Killing nodes in file $nodes"
pwd=`pwd`
while read p; do
	ssh $p "pkill -f $pwd/dist-lsgan.py" < /dev/tty
done < $nodes
