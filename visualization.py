import os
import torch
import open3d as o3d

def visualization(address, scene_name, cur, gaps):
    data_path = os.path.join(address, scene_name + '_git')
    save_path = os.path.join(data_path, 'pair_file_sf')
    cur_file = f'gap_{1:02d}_cur_{cur:05d}.pt'
    cur_tensor = torch.load(os.path.join(save_path, cur_file))
    point_cur = cur_tensor['global_p1']
    img_cur = cur_tensor['img_1']    
    point_whole = torch.cat((point_cur, img_cur), axis=-1).view(-1, 6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_whole[:,:3].numpy())
    pcd.colors = o3d.utility.Vector3dVector(point_whole[:,3:].numpy())
    if not os.path.exists('test_img/points'):
        os.makedirs('test_img/points')
    o3d.io.write_point_cloud(f'test_img/points/gap_{00:02d}_cur_{cur:05d}.ply', pcd)
    
    for gap in gaps:
        ref = cur + gap
        ref_gap = -1 * gap
        ref_file = f'gap_{ref_gap:02d}_cur_{ref:05d}.pt'
        ref_tensor = torch.load(os.path.join(save_path, ref_file))
        sf_ref = ref_tensor['scene_flow_1_2']
        point_ref = ref_tensor['global_p1']
        img_ref = ref_tensor['img_1']
        mask = ref_tensor['mask']
        ref_cur_point = torch.cat((point_ref + sf_ref, img_ref), axis=-1)  * mask[..., None]
        ref_cur_point = ref_cur_point.view(-1, 6)
        point_whole = torch.cat((point_whole, ref_cur_point), axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_whole[:,:3].numpy())
        pcd.colors = o3d.utility.Vector3dVector(point_whole[:,3:].numpy())
        o3d.io.write_point_cloud(f'test_img/points/gap_{gap:02d}_cur_{cur:05d}.ply', pcd)

if __name__ == '__main__':
    address = '/root/autodl-tmp/kaichen/data/davis'
    scene_name = 'train' # Scene to be Used
    cur = 16 # Frame Number to be Used
    gaps = [-4, -3, -2, -1, 1, 2, 3, 4]
    visualization(address, scene_name, cur, gaps)