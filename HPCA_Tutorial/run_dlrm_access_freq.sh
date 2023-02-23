python dlrm_access_freq.py --arch-sparse-feature-size=16 \
								--arch-embedding-size="1460-583-10131227-2202608-305-24-12517-633-3-93145-5683-8351593-3194-27-14992-5461306-10-5652-2173-4-7046547-18-15-286181-105-142572" \
								--arch-mlp-bot="13-512-256-64-16" \
								--arch-mlp-top="512-256-1" \
								--data-generation=dataset \
								--data-set=kaggle_tutorial \
								--processed-data-file-tutorial=./input/kaggle/train.npz \
								--emb-characterization=2
