#!/bin/bash

count=0

while [ $count -lt 10 ]; do
	echo "Loop is at $count"
	let count=count+1
done

echo "Done..."

count=0

while [ $count -le 10 ]; do
	echo "Loop is at $count"
	let count=count+1
done

count2=25

until [ $count2 -lt 10 ]; do
	echo $count2
	let count2-=1
done

count=0

while [ $count -le 10 ]; do
	let count=count+1

	if [ $count -eq 5 ];
	then
		break
		#continue
	fi

	echo "Loop is at $count"
done