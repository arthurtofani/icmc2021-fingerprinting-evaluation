# todo
#$dbase=../sample.pklz
#find /home/arthur/Projects/amatch/data/Sample -type f > sample.txt
python audfprint.py new --dbase ../8k.pklz --ncores=6 -k -l ../8k.txt
#python audfprint.py match --dbase $dbase -x 5 --ncores=6 -l ../sample.txt -o ../sample.out2.csv
