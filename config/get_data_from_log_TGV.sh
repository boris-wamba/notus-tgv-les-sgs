#!/bin/bash

# execute the awk script to get data from notus redirected simulation information in log
# you can customize the notus_log_pp.awk for your needs

LOG_FILE=$1
FNAME=$2

echo "# time, MeanKE, MeanKEDR, MeanP, MeanFKEDR" > $FNAME
echo "awk -f notus_log_pp_TGV.awk $LOG_FILE >> $FNAME"
awk -f notus_log_pp_TGV.awk $LOG_FILE >> $FNAME
