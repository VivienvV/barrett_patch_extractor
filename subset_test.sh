python wsi2biopsy.py --root_dir 'TIFFs' --out_dir 'data/Barrett20x' --magnification 20 --extract_stroma --verbose --save_fig

python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.1 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.25 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.33 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.5 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.66 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.75 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing
python biopsy2patches.py --root_dir 'data/Barrett20x' --patch_size 224 224 --stride 112 112 --threshold 0.9 --verbose --datasets 'ASL' 'Bolero' 'LANS' 'RBE' 'RBE_Nieuw' --dont_save_patches --save_gif --dataset_probing