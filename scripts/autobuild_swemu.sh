echo "====================================================================================" | tee autobuild_swemu_log.txt
echo "====================================================================================" | tee -a autobuild_swemu_log.txt
make compile_swemu | tee -a autobuild_swemu_log.txt
echo "====================================================================================" | tee -a autobuild_swemu_log.txt
echo "====================================================================================" | tee -a autobuild_swemu_log.txt
make link_swemu | tee -a autobuild_swemu_log.txt
echo "====================================================================================" | tee -a autobuild_swemu_log.txt
echo "====================================================================================" | tee -a autobuild_swemu_log.txt
sudo poweroff
