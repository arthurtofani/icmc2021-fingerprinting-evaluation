python build.py --dataset datasets/fingerprints/fma_small.txt --results exp/fma_small2 --db_idx 1
python compare.py --dataset datasets/fingerprints/fma_small.txt --results exp/fma_small2 --db_idx 1 --tau 7
python compare.py --dataset datasets/fingerprints/fma_small.txt --results exp/fma_small2 --db_idx 1 --tau 0

