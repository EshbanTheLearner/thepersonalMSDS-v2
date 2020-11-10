#!/bin/bash

arr=( "Batman" "Ironman" "Spiderman" )

size=${#arr[@]}

echo $size

: 'index=1

val1=${arr[${index}]}
'
echo $val1

for (( i=0; i<$size; i++ )); do
	echo ${arr[${i}]}
done