#!/bin/bash
curdate=`date +%Y%m%d-%H%M`
source activate py36

runfile=$1.py

python ${runfile} > log_${curdate} 2>&1 &
