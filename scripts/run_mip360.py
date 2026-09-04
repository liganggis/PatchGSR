import os
    
scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path='mip360'
out_base_path='output_mip360'
out_name='test'
gpu_id=0

# for id, scene in enumerate(scenes):

#     cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
#     print(cmd)
#     os.system(cmd)

#     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name}  --eval -r{factors[id]} --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0004'
#     print(cmd)
#     os.system(cmd)

#     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name}'
#     print(cmd)
#     os.system(cmd)
    
#     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
#     print(cmd)
#     os.system(cmd)


# python train.py -s ../../data/360_v2/bonsai -m ./exps/360/bonsai_v1 -r 2 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000
# python metrics2.py -s ../../data/360_v2/bonsai/ -m ./exps/360/bonsai_v1 -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000
# python extract_360_tsdf_u.py -m ./exps/360/bonsai_v1 --num_cluster 1 -r 2 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/counter -m ./exps/360/counter_v1 -r 2 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000
# python metrics2.py -s ../../data/360_v2/counter/ -m ./exps/360/counter_v1 -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000
# python extract_360_tsdf_u.py -m ./exps/360/counter_v1 --num_cluster 1 -r 2 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/kitchen -m ./exps/360/kitchen_v1 -r 2 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000
# python metrics2.py -s ../../data/360_v2/kitchen/ -m ./exps/360/kitchen_v1 -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000
# python extract_360_tsdf_u.py -m ./exps/360/kitchen_v1 --num_cluster 1 -r 2 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/room -m ./exps/360/room_v1 -r 2 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000
# python metrics2.py -s ../../data/360_v2/room/ -m ./exps/360/room_v1 -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000
# python extract_360_tsdf_u.py -m ./exps/360/room_v1 --num_cluster 1 -r 2 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/bicycle -m ./exps/360/bicycle_v1 -r 4 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000 --data_device cpu
# python metrics2.py -s ../../data/360_v2/bicycle/ -m ./exps/360/bicycle_v1 -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu
# python extract_360_tsdf_u.py -m ./exps/360/bicycle_v1 --num_cluster 1 -r 4 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu --eval

# python train.py -s ../../data/360_v2/garden -m ./exps/360/garden_v1 -r 4 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000 --data_device cpu
# python metrics2.py -s ../../data/360_v2/garden/ -m ./exps/360/garden_v1 -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu
# python extract_360_tsdf_u.py -m ./exps/360/garden_v1 --num_cluster 2 -r 4 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/stump -m ./exps/360/stump_v1 -r 4 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000 --data_device cpu
# python metrics2.py -s ../../data/360_v2/stump/ -m ./exps/360/stump_v1 -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu
# python extract_360_tsdf_u.py -m ./exps/360/stump_v1 --num_cluster 2 -r 4 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/flowers -m ./exps/360/flowers_v1 -r 4 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000 --data_device cpu
# python metrics2.py -s ../../data/360_v2/flowers/ -m ./exps/360/flowers_v1 -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu
# python extract_360_tsdf_u.py -m ./exps/360/flowers_v1 --num_cluster 1 -r 4 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval

# python train.py -s ../../data/360_v2/treehill -m ./exps/360/treehill_v1 -r 4 --cfg_path cfg/Mip360/360.yaml --eval --iterations 15000 --data_device cpu
# python metrics2.py -s ../../data/360_v2/treehill/ -m ./exps/360/treehill_v1 -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 15000 --data_device cpu
# python extract_360_tsdf_u.py -m ./exps/360/treehill_v1 --num_cluster 1 -r 4 --cfg_path cfg/Mip360/360.yaml --iteration 15000 --eval


# python render_mesh.py -s ../../data/360_v2/bicycle -n bicycle --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 4 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/treehill -n treehill --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 4 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/garden -n garden --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 4 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/stump -n stump --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 4 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/flowers -n flowers --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 4 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/bonsai -n bonsai --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 2 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/kitchen -n kitchen --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 2 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/counter -n counter --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 2 --data_device cpu
# python render_mesh.py -s ../../data/360_v2/room -n room --mesh_path ../../Meshes/360/PatchGSR --output_dir ../../Meshes/360_images/PatchGSR -r 2 --data_device cpu