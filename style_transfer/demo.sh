python style_transfer/test.py --content_src style_transfer/data/xia_test/depressed_13_000.bvh --style_src style_transfer/data/xia_test/strutting_01_000.bvh --output_dir style_transfer/demo_results/demo_3d_1
python style_transfer/test.py --content_src style_transfer/data/xia_test/neutral_16_000.bvh --style_src style_transfer/data/xia_test/angry_13_000.bvh --output_dir style_transfer/demo_results/demo_3d_2

python style_transfer/test.py --content_src style_transfer/data/xia_test/old_13_000.bvh --style_src style_transfer/data/xia_test/sexy_01_001.bvh --output_dir style_transfer/demo_results/comp_3d_1
python style_transfer/test.py --content_src style_transfer/data/xia_test/sexy_01_000.bvh --style_src style_transfer/data/xia_test/depressed_18_000.bvh --output_dir style_transfer/demo_results/comp_3d_2

python style_transfer/test.py --content_src style_transfer/data/xia_test/neutral_01_000.bvh --style_src style_transfer/data/treadmill/json_inputs/27 --output_dir style_transfer/demo_results/demo_video_1
python style_transfer/test.py --content_src style_transfer/data/xia_test/neutral_01_000.bvh --style_src style_transfer/data/treadmill/json_inputs/95 --output_dir style_transfer/demo_results/demo_video_2

python style_transfer/test.py --content_src style_transfer/data/xia_test/neutral_01_000.bvh --style_src style_transfer/data/treadmill/json_inputs/27 --output_dir style_transfer/demo_results/comp_video_1
python style_transfer/test.py --content_src style_transfer/data/xia_test/neutral_01_000.bvh --style_src style_transfer/data/treadmill/json_inputs/32 --output_dir style_transfer/demo_results/comp_video_2
