echo "gather_results.sh: Gathering results(*.log and *.csv) in a zip file..."
timestamp=`date +%Y_%m_%d.%H_%M_%S`
zipname="fpga_run_${timestamp}.zip"
find ./ -maxdepth 1 -type f \( -iname \*.log -o -iname \*.csv \) -exec zip -r ${zipname} "{}"  \;
echo "gather_results.sh: ${zipname} has been created successfully."
echo "gather_results.sh: done."