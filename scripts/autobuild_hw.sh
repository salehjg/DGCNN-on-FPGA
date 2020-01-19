printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee autobuild_hw_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hw_log.txt
make compile_hw | tee -a autobuild_hw_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hw_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hw_log.txt
make link_hw | tee -a autobuild_hw_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hw_log.txt
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = | tee -a autobuild_hw_log.txt

RUN_PASTEBIN_OR_NOT=@PASTEBIN_0_1@
if [ "$RUN_PASTEBIN_OR_NOT" -eq 1 ] ; then
	echo "**Running PasteBin agent..."
	python PasteBinAgentForAutoBuilds.py
	echo "**Done."
else
	echo "PasteBin agent is disabled."
fi

echo "DONE. RUNNING POWEROFF COMMAND..."
sudo poweroff
