import os
import torch
import numpy as np


def save_xyz_file(path, positions, h, node_mask, rot=None, names=None, suffix=''):
    'Write rigid-fragment states to an XYZ-style file.'
    try:
        os.makedirs(path, exist_ok=True)

        
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            node_mask = node_mask.unsqueeze(0)
            h = h.unsqueeze(0)
            if rot is not None: rot = rot.unsqueeze(0)

        batch_size = positions.shape[0]
        if names is None: names = [f'mol_{i}' for i in range(batch_size)]

        for i in range(batch_size):
            file_path = os.path.join(path, f"{names[i]}{suffix}.xyz")

            pos = positions[i].detach().cpu().numpy()
            mask = node_mask[i].detach().cpu().numpy().flatten()
            feats = h[i].detach().cpu().numpy()

            
            if rot is not None:
                r_mats = rot[i].detach().cpu().numpy().reshape(-1, 9)
            else:
                
                r_mats = np.eye(3).flatten()[None, :].repeat(pos.shape[0], axis=0)

            valid = mask > 0.5
            v_pos, v_feats, v_rots = pos[valid], feats[valid], r_mats[valid]

            with open(file_path, 'w') as f:
                f.write(f"{len(v_pos)}\n\n")
                for j in range(len(v_pos)):
                    x, y, z = v_pos[j]
                    
                    type_id = np.argmax(v_feats[j])
                    
                    rot_str = " ".join([f"{val:.6f}" for val in v_rots[j]])
                    f.write(f"C {x:.6f} {y:.6f} {z:.6f} {rot_str} {type_id}\n")

    except Exception as e:
        print(f"Error saving FragDiff XYZ: {e}")


def visualize_chain(path, chain, node_mask, prefix=''):
    'Export selected frames from a sampling trajectory.'
    if chain is None: return
    os.makedirs(path, exist_ok=True)

    
    n_steps = chain.shape[0]
    n_mols = chain.shape[1]

    for i in range(n_mols):
        mol_dir = os.path.join(path, f"{prefix}mol_{i}")
        os.makedirs(mol_dir, exist_ok=True)

        for t in range(n_steps):
            
            state = chain[t, i]
            pos = state[:, 0:3]
            rot_flat = state[:, 3:12]
            h = state[:, 12:]

            save_xyz_file(
                path=mol_dir,
                positions=pos,
                h=h,
                rot=rot_flat.reshape(-1, 3, 3),
                node_mask=node_mask[i],
                names=[f'frame_{t:03d}']
            )
