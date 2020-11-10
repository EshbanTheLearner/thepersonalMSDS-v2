#!/bin/bash

var1="Hello"
var2="World"

result="$var1 $var2"

echo $result

str="Hello World"
sub=${str:6:3}
sub1=${str:6:5}

echo $sub
echo $sub1