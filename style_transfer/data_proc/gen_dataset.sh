# For xia
python export_train.py --dataset xia --bvh_path ../data/mocap_xia --output_path ../data/xia --window 32 --window_step 8 --dataset_config ../global_info/xia_dataset.yml
# For bfa
python export_train.py --dataset bfa --bvh_path ../data/mocap_bfa --output_path ../data/bfa --window 32 --window_step 8 --dataset_config ../global_info/bfa_dataset.yml
