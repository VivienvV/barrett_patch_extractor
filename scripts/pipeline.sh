python wsi2biopsy.py --root_dir '../../../data/archief/AMC-data/Barrett' --out_dir '../../../data/ml/AMC-data/Barrett/Barrett20x' --magnification 20 --extract_stroma --verbose
python biopsy2patches.py --root_dir '../../../data/ml/AMC-data/Barrett/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.33 --verbose --dont_save_patches --dataset_probing