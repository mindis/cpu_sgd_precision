info_log=$1
startsec=$2

# OUPUT DURATION
end=$(date)
endsec=$(date +%s)
diff=$(( endsec - startsec ))

echo >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo "END: $end (DURATION: $diff seconds)" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
