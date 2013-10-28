#!/bin/sh

while echo $1 | grep ^- > /dev/null; do
    eval $( echo $1 | sed 's/-//g' | tr -d '\012')=$2
    shift
    shift
done

echo nClusters = $c
echo nPoints = $p

echo "Running..."
echo "kmeans"
time ./kmeans $c $p > x
echo ""
echo "kmeans_cuda"
time ./kmeans_cuda $c $p > y
diff -y x y | tail -$c | grep "|"
