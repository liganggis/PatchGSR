# python train.py -s ../../data/DTU/scan24/ -m exps/DTU5/scan24_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan24/ -m exps/DTU/scan24_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan24/ -m exps/DTU4/scan24_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan24/ -m exps/DTU4/scan24_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan37/ -m exps/DTU/scan37_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml -DDN --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan37/ -m exps/DTU/scan37_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan37/ -m exps/DTU4/scan37_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan37/ -m exps/DTU4/scan37_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan40/ -m exps/DTU/scan40_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan40/ -m exps/DTU/scan40_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan40/ -m exps/DTU4/scan40_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan40/ -m exps/DTU4/scan40_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan55/ -m exps/DTU/scan55_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan55/ -m exps/DTU/scan55_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.97
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan55/ -m exps/DTU4/scan55_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.97
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan55/ -m exps/DTU4/scan55_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan63/ -m exps/DTU/scan63_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml -DDN --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan63/ -m exps/DTU/scan63_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99 --num_cluster 4
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan63/ -m exps/DTU4/scan63_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99 --num_cluster 4
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan63/ -m exps/DTU4/scan63_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan65/ -m exps/DTU/scan65_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan65/ -m exps/DTU/scan65_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan65/ -m exps/DTU4/scan65_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan65/ -m exps/DTU4/scan65_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan69/ -m exps/DTU/scan69_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan69/ -m exps/DTU/scan69_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.975
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan69/ -m exps/DTU4/scan69_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.975
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan69/ -m exps/DTU4/scan69_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan83/ -m exps/DTU/scan83_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan83/ -m exps/DTU4/scan83_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99 --num_cluster 2
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan83/ -m exps/DTU4/scan83_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99 --num_cluster 2
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan83/ -m exps/DTU4/scan83_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan97/ -m exps/DTU/scan97_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan97/ -m exps/DTU/scan97_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan97/ -m exps/DTU4/scan97_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan97/ -m exps/DTU4/scan97_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan105/ -m exps/DTU/scan105_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan105/ -m exps/DTU/scan105_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.98
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan105/ -m exps/DTU4/scan105_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.98
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan105/ -m exps/DTU4/scan105_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan106/ -m exps/DTU/scan106_v1 -r 2 --cfg_path cfg/DTU/106.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan106/ -m exps/DTU/scan106_v1 -r 2 --cfg_path cfg/DTU/106.yaml --iterations 12000 -UDF -DT 0.85
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan106/ -m exps/DTU4/scan106_v1 -r 2 --cfg_path cfg/DTU/106.yaml --iterations 12000 -UDF -DT 0.85
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan106/ -m exps/DTU4/scan106_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan110/ -m exps/DTU/scan110_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan110/ -m exps/DTU/scan110_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan110/ -m exps/DTU4/scan110_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.99
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan110/ -m exps/DTU4/scan110_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan114/ -m exps/DTU/scan114_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan114/ -m exps/DTU/scan114_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan114/ -m exps/DTU4/scan114_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan114/ -m exps/DTU4/scan114_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan118/ -m exps/DTU/scan118_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan118/ -m exps/DTU/scan118_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan118/ -m exps/DTU4/scan118_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan118/ -m exps/DTU4/scan118_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan122/ -m exps/DTU/scan122_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan122/ -m exps/DTU/scan122_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python extract_dtu_tsdf_m.py -s ../../data/DTU/scan122/ -m exps/DTU4/scan122_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan122/ -m exps/DTU4/scan122_v1 --DTU ../../data/DTU_real -f tsdf_m --iteration 12000 -UDF
