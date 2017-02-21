#!/bin/bash
for f in *.py
do
	#echo "Executing test $f" &>> out_test
	n=1
	while (($n<=32))
	do
	        #python $f $n &>> out_test
		(/usr/bin/time --format '%E %U %S ' python $f $n) &>> out_test
		wait
		n=$((n*2))
	done
done
