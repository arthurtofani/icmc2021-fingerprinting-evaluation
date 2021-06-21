#python queries.py --dataset datasets/fingerprints/ytc.txt --results exp/ytc
#python build.py --dataset datasets/fingerprints/ytc.txt --results exp/ytc --db_idx=4
#python match.py --dataset datasets/fingerprints/ytc.txt --results exp/ytc --db_idx=4 --query_size=3

python compare.py --dataset datasets/fingerprints/ytc.txt --results exp/ytc --db_idx 4 --tau 5.5
python compare.py --dataset datasets/fingerprints/ytc.txt --results exp/ytc --db_idx 4 --tau 0
