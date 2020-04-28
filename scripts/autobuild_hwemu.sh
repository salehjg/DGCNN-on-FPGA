start=$SECONDS
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee autobuild_hwemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hwemu_log.txt
make compile_hwemu | tee -a autobuild_hwemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hwemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hwemu_log.txt
make link_hwemu | tee -a autobuild_hwemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hwemu_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hwemu_log.txt

RUN_PASTEBIN_OR_NOT=@PASTEBIN_0_1@
if [ "$RUN_PASTEBIN_OR_NOT" -eq 1 ] ; then
	echo "**Running PasteBin agent..."
	python PasteBinAgentForAutoBuilds.py
	echo "**Done."
else
	echo "PasteBin agent is disabled."
fi

echo "DONE. RUNNING POWEROFF COMMAND..."
stop=$SECONDS
elapsed=$[stop-start]
hours=$(bc <<< "scale=3;${elapsed}/3600.0")
echo "Total Elapsed Seconds: ${elapsed}"
echo "Total Elapsed Hours: ${hours}"
sudo poweroff
