import os




# python train.py -s ../../data/TNT/Barn -m ./exps/TNT/Barn_v1 -r2 --cfg_path cfg/TNT/Barn.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Barn_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Barn.yaml --iteration 15000 -UDF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Barn --traj-path ../../data/TNT/Barn/Barn_COLMAP_SfM.log --ply-path ./exps/TNT/Barn_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Barn_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Barn.yaml --iteration 15000 -UDF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Barn --traj-path ../../data/TNT/Barn/Barn_COLMAP_SfM.log --ply-path ./exps/TNT/Barn_v1/tsdf_m/tsdf.ply

# ulimit -n 2048
# python train.py -s ../../data/TNT/Courthouse -m ./exps/TNT/Courthouse_v1 -r2 --cfg_path cfg/TNT/Courthouse.yaml --iterations 15000 -w
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Courthouse_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Courthouse.yaml --iteration 15000 -UDF -UNF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Courthouse --traj-path ../../data/TNT/Courthouse/Courthouse_COLMAP_SfM.log --ply-path ./exps/TNT/Courthouse_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Courthouse_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Courthouse.yaml --iteration 15000 -UDF -UNF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Courthouse --traj-path ../../data/TNT/Courthouse/Courthouse_COLMAP_SfM.log --ply-path ./exps/TNT/Courthouse_v1/tsdf_m/tsdf.ply

# python train.py -s ../../data/TNT_by_gof/Courthouse -m ./exps/TNT/Courthouse_v1 -r2 --cfg_path cfg/TNT/Courthouse.yaml --iterations 15000 -w
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Courthouse_v1 --num_cluster 2 -r 2 --cfg_path cfg/TNT/Courthouse.yaml --iteration 15000 -UDF -UNF -DT 0.985
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Courthouse --traj-path ../../data/TNT_by_gof/Courthouse/Courthouse_COLMAP_SfM.log --ply-path ./exps/TNT/Courthouse_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Courthouse_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Courthouse.yaml --iteration 15000 -UDF -UNF -DT 0.985 --num_cluster 2
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Courthouse --traj-path ../../data/TNT_by_gof/Courthouse/Courthouse_COLMAP_SfM.log --ply-path ./exps/TNT/Courthouse_v1/tsdf_m/tsdf.ply


# python train.py -s ../../data/TNT/Caterpillar -m ./exps/TNT/Caterpillar_v1 -r2 --cfg_path cfg/TNT/Caterpillar.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Caterpillar_v1 --num_cluster 2 -r 2 --cfg_path cfg/TNT/Caterpillar.yaml --iteration 15000 -UDF -UNF -DT 0.975 --tsdf 2.0
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Caterpillar --traj-path ../../data/TNT/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path ./exps/TNT/Caterpillar_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Caterpillar_v1 --num_cluster 2 -r 2 --cfg_path cfg/TNT/Caterpillar.yaml --iteration 15000 -UDF -UNF -DT 0.975 --tsdf 2.0
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Caterpillar --traj-path ../../data/TNT/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path ./exps/TNT/Caterpillar_v1/tsdf_m/tsdf.ply

# python train.py -s ../../data/TNT/Ignatius -m ./exps/TNT/Ignatius_v1 -r2 --cfg_path cfg/TNT/Ignatius.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Ignatius_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Ignatius.yaml --iteration 15000 -UDF  -DT 0.98
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Ignatius --traj-path ../../data/TNT/Ignatius/Ignatius_COLMAP_SfM.log --ply-path ./exps/TNT/Ignatius_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Ignatius_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Ignatius.yaml --iteration 15000 -UDF  -DT 0.98
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Ignatius --traj-path ../../data/TNT/Ignatius/Ignatius_COLMAP_SfM.log --ply-path ./exps/TNT/Ignatius_v1/tsdf_m/tsdf.ply

# python train.py -s ../../data/TNT/Meetingroom -m ./exps/TNT/Meetingroom_v1 -r2 --cfg_path cfg/TNT/Meetingroom.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Meetingroom_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Meetingroom.yaml --iteration 15000 -UDF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Meetingroom --traj-path ../../data/TNT/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path ./exps/TNT/Meetingroom_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Meetingroom_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Meetingroom.yaml --iteration 15000 -UDF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Meetingroom --traj-path ../../data/TNT/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path ./exps/TNT/Meetingroom_v1/tsdf_m/tsdf.ply

# python train.py -s ../../data/TNT/Truck -m ./exps/TNT/Truck_v1 -r2 --cfg_path cfg/TNT/Truck.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT/Truck_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Truck.yaml --iteration 15000 -UDF -UNF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Truck --traj-path ../../data/TNT/Truck/Truck_COLMAP_SfM.log --ply-path ./exps/TNT/Truck_v1/tsdf_u/tsdf.ply
# python extract_tnt_tsdf_m.py -m ./exps/TNT/Truck_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Truck.yaml --iteration 15000 -UDF -UNF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Truck --traj-path ../../data/TNT/Truck/Truck_COLMAP_SfM.log --ply-path ./exps/TNT/Truck_v1/tsdf_m/tsdf.ply

# python render_mesh.py -s ../../data/TNT/Truck -n Truck --mesh_path ../../Meshes/TnT/PatchGS --output_dir ../../Meshes/TnT_images/PatchGS -r 1
# python render_mesh.py -s ../../data/TNT/Truck -n Truck --mesh_path ../../Meshes/TnT/GOF --output_dir ../../Meshes/TnT_images/GOF -r 1
# python render_mesh.py -s ../../data/TNT/Truck -n Truck --mesh_path ../../Meshes/TnT/PGSR --output_dir ../../Meshes/TnT_images/PGSR -r 1
# python render_mesh.py -s ../../data/TNT/Truck -n Truck --mesh_path ../../Meshes/TnT/GeoSVR --output_dir ../../Meshes/TnT_images/GeoSVR -r 1

# scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
scenes = ['Caterpillar']
# scenes = ["scan24", "scan37", "scan63", "scan65", "scan69", "scan97", "scan105", "scan110"]
# scenes = ["scan24"]

# methods = ["PatchGS", "GeoSVR", "PGSR", "GOF"]
methods = ["PatchGS"]
# python render_mesh.py -s ../../data/TNT/Caterpillar -n 0 --mesh_path ../../Meshes/TnT/PatchGS/Caterpillar --output_dir ../../Meshes/TnT/PatchGS/Caterpillar -r 1 --data_device cpu
# python render_mesh.py -s ../../data/DTU/scan37 -n TNT2 --mesh_path ../../Meshes/DTU/PatchGS/scan37 --output_dir ../../Meshes/DTU/PatchGS/Images -r 1
# for _, scene in enumerate(scenes):
#     for id, method in enumerate(methods):
#         # cmd = f'python render_mesh.py -s ../../data/TNT/{scene} -n {scene} --mesh_path ../../Meshes/TnT/{method} --output_dir ../../Meshes/TnT_images/{method} -r 1 --data_device cpu'
#         # cmd = f'python render_mesh.py -s ../../data/DTU/{scene} -n {scene} --mesh_path ../../Meshes/DTU/{method} --output_dir ../../Meshes/DTU/Images/{method} -r 1 --data_device cpu'
#         print(cmd)
#         os.system(cmd)

for id in range(410):
    scene = id + 1
    for id, method in enumerate(methods):
        cmd = f'python render_point.py -s ../../data/TNT/Barn -n {scene} --mesh_path ./exps/TNT/Barn_v1/tsdf_u --output_dir ./exps/TNT/Barn_v1/tsdf_u_img -r 1 --data_device cpu'
        print(cmd)
        os.system(cmd)

# python train.py -s ../../data/DTU/scan24/ -m exps/TNT2/scan24_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan24/ -m exps/TNT2/scan24_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF -DT 0.9
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan24/ -m exps/TNT2/scan24_v1 --DTU ../../data/DTU_real -f tsdf_u --iteration 12000 -UDF

# python train.py -s ../../data/DTU/scan37/ -m exps/TNT2/scan37_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml -DDN --iterations 12000
# python extract_dtu_tsdf_u.py -s ../../data/DTU/scan37/ -m exps/TNT2/scan37_v1 -r 2 --cfg_path cfg/DTU/dtu_mesh.yaml --iterations 12000 -UDF
# python evaluate_dtu_mesh.py -s ../../data/DTU/scan37/ -m exps/TNT2/scan37_v1 --DTU ../../data/DTU_real -f tsdf_u --iteration 12000 -UDF

# python train.py -s ../../data/TNT/Truck -m ./exps/TNT3/Truck_v1 -r2 --cfg_path cfg/TNT/Truck.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT3/Truck_v1 --num_cluster 1 -r 2 --cfg_path cfg/TNT/Truck.yaml --iteration 15000 -UDF -UNF -DT 0.975
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Truck --traj-path ../../data/TNT/Truck/Truck_COLMAP_SfM.log --ply-path ./exps/TNT3/Truck_v1/tsdf_u/tsdf.ply

# python train.py -s ../../data/TNT/Caterpillar -m ./exps/TNT2/Caterpillar_v5 -r2 --cfg_path cfg/TNT/Caterpillar.yaml --iterations 15000
# python extract_tnt_tsdf_u.py -m ./exps/TNT2/Caterpillar_v1 --num_cluster 2 -r 2 --cfg_path cfg/TNT/Caterpillar.yaml --iteration 15000 -UDF -UNF -DT 0.975 --tsdf 2.0
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Caterpillar --traj-path ../../data/TNT/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path ./exps/TNT2/Caterpillar_v5/tsdf_u/tsdf.ply

# python render_point.py -s ../../data/TNT/Barn -n 0 --mesh_path ./exps/TNT/Barn_v1/tsdf_u --output_dir ./exps/TNT/Barn_v1/tsdf_u_img -r 1 --data_device cpu
# python render_point.py -s ../../data/TNT/Barn -n 0 --mesh_path ./exps/TNT/Barn_v1/tsdf_m --output_dir ./exps/TNT/Barn_v1/tsdf_m_img -r 1 --data_device cpu
