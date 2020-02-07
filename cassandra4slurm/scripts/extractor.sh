#!/bin/bash
FILENAME=${1}
FIRST_NUM=${2}
LAST_NUM=${3}
COUNTER=$FIRST_NUM

function usage () {
    echo "This script prints the result of consecutive cassandra-stress logs."
    echo "Usage: bash extractor.sh basename startingnumber endingnumber"
    echo "i.e.: bash extractor.sh WR-10M-SimpleStrategy-3nodes-exec1node- 1 4"
    echo "It will print WR-10M-SimpleStrategy-3nodes-exec1node-{1-4}.log results"
}

if [ "$#" -ne 3 ]; then
    echo "ERROR: Incorrect number of parameters."
    usage
    exit
fi

while [ "$COUNTER" -le "$LAST_NUM" ]; do
    cat "$FILENAME""$COUNTER".log | grep "Op rate" | awk '{ print $4" "$5}' | sed s/,/./g
    ((COUNTER++))
done
if [ ! -f "$FILENAME"0.log ]; then
    echo "WARN: This solution set looks incomplete..."
else
    cat "$FILENAME"0.log
fi
