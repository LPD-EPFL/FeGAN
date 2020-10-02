nodes=$1
echo "Killing nodes in file $nodes"
pwd=`pwd`
while read p; do
	ssh $p "pkill -f $pwd/md-gan.py" < /dev/tty
done < $nodes
