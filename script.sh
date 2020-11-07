#!/bin/bash

SEMESTER=$1
DAY=$2

git add .
git commit -m "'Semester $SEMESTER - Day $DAY'"
git push origin main