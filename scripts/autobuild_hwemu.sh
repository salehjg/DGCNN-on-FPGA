echo "====================================================================================" | tee autobuild_hwemu_log.txt
echo "====================================================================================" | tee -a autobuild_hwemu_log.txt
make compile_hwemu | tee -a autobuild_hwemu_log.txt
echo "====================================================================================" | tee -a autobuild_hwemu_log.txt
echo "====================================================================================" | tee -a autobuild_hwemu_log.txt
make link_hwemu | tee -a autobuild_hwemu_log.txt
echo "====================================================================================" | tee -a autobuild_hwemu_log.txt
echo "====================================================================================" | tee -a autobuild_hwemu_log.txt
sudo poweroff
