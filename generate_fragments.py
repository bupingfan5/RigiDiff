# # pdb18
# import argparse
# import os
# import numpy as np
# import torch
# import copy
# from scipy.spatial.distance import pdist
# from rdkit import Chem
# from rdkit.Chem import rdForceFieldHelpers
# from rdkit.Chem import QED
# from rdkit.Chem import Descriptors
# from tqdm import tqdm
# import sys
# from rdkit import RDConfig
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdDetermineBonds
#
# # 引入合成可及性打分
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# try:
#     from src import sascorer
# except ImportError:
#     import sascorer
#
# from src import const
# from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule, PDBbindDataset
# from src.lightning import DDPM
# from src.utils import FoundNaNException
# from src.visualizer import save_xyz_file
#
# from openfold.np import residue_constants as rc
# from openfold.utils.rigid_utils import Rotation, Rigid
# from openfold.utils.tensor_utils import batched_gather
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--output', type=str, default='./results')
# parser.add_argument('--n_samples', type=int, default=5)
# parser.add_argument('--n_steps', type=int, default=None)
# parser.add_argument('--linker_size', type=str, default='5')
# parser.add_argument('--use_dataset', action='store_true')
# parser.add_argument('--data_dir', type=str, default='data')
# parser.add_argument('--dataset_prefix', type=str, default='PDBbind_pocket10')
# parser.add_argument('--split_file', type=str, default=None)
# parser.add_argument('--limit', type=int, default=-1)
# parser.add_argument('--fragments', type=str, default=None)
# parser.add_argument('--pocket', type=str, default=None)
# parser.add_argument('--anchors', type=str, default=None)
#
#
# def force_field_assisted_reconstruction(mol):
#     """
#     方案二：通过力场预优化消除扩散位阻应变，随后由官方引擎精确感知拓扑。
#     """
#     try:
#         # 1. 先用最宽松的条件清理一下游离原子
#         Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#
#         # 2. 加氢以准确计算空间位阻
#         mol_with_h = Chem.AddHs(mol, addCoords=True)
#
#         # 3. 极小步数的 UFF 能量最小化（熨平局部畸变，保留整体姿态）
#         AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=50)
#
#         # 4. 移除氢原子
#         mol_relaxed = Chem.RemoveHs(mol_with_h)
#     except Exception as e:
#         # print(f"力场预松弛干预失败，退回原始坐标: {e}")
#         mol_relaxed = mol
#
#     # 5. 坐标平整后，调用官方算法分配单双键和芳香键
#     try:
#         rdDetermineBonds.DetermineBondOrders(mol_relaxed, charge=0)
#         return mol_relaxed
#     except Exception:
#         return mol  # 若失败，原样返回，交给后续的 3D 雷达兜底
#
#
# def assemble_via_xyz2mol(atomic_numbers, coords, charge=0):
#     """
#     方案三：使用 xyz2mol 的试探性启发式搜索在高度带噪体系中寻取最优的含芳香键重构
#     (注意：使用此函数前需要在终端执行 pip install xyz2mol)
#     """
#     try:
#         import xyz2mol as x2m
#         mols = x2m.xyz2mol(atomic_numbers, coords, charge=charge,
#                            use_graph=True,
#                            allow_charged_fragments=True,
#                            use_huckel=True)
#         if mols:
#             best_mol = mols[0]
#             Chem.SanitizeMol(best_mol)
#             return best_mol
#     except Exception as e:
#         print(f"xyz2mol 超启发推演异常: {e}")
#         pass
#     return None
#
# def enhanced_aromaticity_perception_for_diffusion(mol, coords, ring_indices):
#     """
#     为扩散模型定制的强化版三维芳香性感知雷达
#     """
#     # 提取目标环的原子三维坐标
#     ring_coords = coords[list(ring_indices)]
#
#     # 利用 SVD 分解计算最小二乘平面并求取 RMSD
#     centroid = np.mean(ring_coords, axis=0)
#     centered = ring_coords - centroid
#     U, S, Vt = np.linalg.svd(centered)
#     normal = Vt[-1]  # 获取平面的法向量
#
#     distances = np.dot(centered, normal)
#     rmsd = np.sqrt(np.mean(distances ** 2))
#
#     # 【核心修改点1】：将平面容忍度阈值从 0.05 提升至 0.25，吸收高斯噪声
#     # 【核心修改点2】：限定探测范围为 5 元或 6 元环
#     if rmsd < 0.25 and len(ring_indices) in [5, 6]:
#
#         valid_sp2_candidates = 0
#         # 遍历环上的每一个原子，进行辅助拓扑验证
#         for idx in ring_indices:
#             atom = mol.GetAtomWithIdx(idx)
#             # 获取原子的度数（不包含由外部隐式/显式计算推断的氢）
#             degree = atom.GetDegree()
#             # 典型的芳香环原子 (如苯中的碳、吡啶中的氮)，在分子图骨架中度数常为 2 或 3
#             if degree in [2, 3]:
#                 valid_sp2_candidates += 1
#
#         # 若环内有高达 80% 的原子满足几何与杂化特征的潜力，认定其为扩散生成的芳香环
#         if valid_sp2_candidates / len(ring_indices) >= 0.8:
#             for i in range(len(ring_indices)):
#                 u_idx = ring_indices[i]
#                 v_idx = ring_indices[(i + 1) % len(ring_indices)]
#                 bond = mol.GetBondBetweenAtoms(u_idx, v_idx)
#                 if bond:
#                     # 显式重写键级为芳香键
#                     bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
#                 mol.GetAtomWithIdx(u_idx).SetIsAromatic(True)
#             return True
#
#     return False
#
# def svdd_qed_guidance(x_t, t, base_prediction, data, decoder, node_mask, num_samples=15, scale=3.5):
#     """
#     终极多目标 SVDD 引导机制：
#     1. 【加大探索】：num_samples 提升至 15，scale 提升至 3.5，增强在严苛条件下的寻优能力。
#     2. 【黄金漏斗】：LogP不仅在边缘有高压电网，在 [1.0, 3.0] 内部也有指向 2.0 的“磁吸奖励”。
#     3. 【权重重组】：大幅拉升 QED 的奖励（* 25.0）和 SA 的惩罚（* 2.0）。
#     """
#     device = x_t.device
#     B = x_t.shape[0]
#     guided_prediction = base_prediction.clone()
#
#     if isinstance(t, torch.Tensor):
#         t_val = t[0].item() if t.numel() > 1 else t.item()
#     else:
#         t_val = t
#
#     # 放宽作用时间范围，使得在去噪的中后半段能够持续进行微调
#     if t_val < 80:
#         with torch.no_grad():
#             for b in range(B):
#                 best_eps = base_prediction[b]
#                 best_score = -10000.0  # 扩大下限以适应严厉的惩罚
#                 mask = node_mask[b].bool().cpu().view(-1)
#
#                 if not mask.any(): continue
#
#                 # 增加单次SVDD的采样量，扩大在陡峭梯度面的搜索范围
#                 for _ in range(num_samples):
#                     cand_eps = base_prediction[b] + torch.randn_like(base_prediction[b]) * 0.08
#                     cand_pos = (x_t[b] - cand_eps)[mask].detach().cpu().numpy()
#
#                     if 'one_hot' in data:
#                         probs = torch.softmax(data['one_hot'][b][mask] / 0.5, dim=-1)
#                         b_type_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
#                     else:
#                         b_type_idx = np.zeros(len(cand_pos), dtype=np.int64)
#
#                     try:
#                         rw_mol = Chem.RWMol()
#                         valid_coords = []
#                         for i in range(len(cand_pos)):
#                             template = decoder.ideal_coords[b_type_idx[i]]
#                             atom_info = template[0][0]
#                             elem_symbol = "".join([c for c in atom_info.strip() if c.isalpha()]).capitalize()
#                             if elem_symbol == 'H': continue
#                             rw_mol.AddAtom(Chem.Atom(elem_symbol))
#                             valid_coords.append(cand_pos[i])
#
#                         num_atoms = rw_mol.GetNumAtoms()
#                         for idx_i in range(num_atoms):
#                             for idx_j in range(idx_i + 1, num_atoms):
#                                 dist = np.linalg.norm(valid_coords[idx_i] - valid_coords[idx_j])
#                                 if 0.4 < dist < 1.75:
#                                     rw_mol.AddBond(idx_i, idx_j, Chem.rdchem.BondType.SINGLE)
#
#                         mol = rw_mol.GetMol()
#                         Chem.SanitizeMol(mol,
#                                          sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#
#                         c_qed = QED.qed(mol)
#                         c_sa = sascorer.calculateScore(mol)
#                         c_logp = Descriptors.MolLogP(mol)
#
#                         # =========================================================
#                         # 🌟 核心优化：LogP 黄金漏斗势阱 + QED/SA 强力权重重组
#                         # =========================================================
#                         logp_score = 0.0
#                         if 1.0 <= c_logp <= 3.0:
#                             # 【磁吸效应】：在安全区内，LogP 越靠近完美的 2.0，给的附加奖励越高！
#                             # 避免模型在 1~3 之间“躺平”，主动去配合 QED 的提升。
#                             logp_score = 5.0 - ((c_logp - 2.0) ** 2) * 5.0
#                         elif c_logp < 1.0:
#                             # 【高压电网】：一旦越界，线性 + 二次方双重暴击扣分！
#                             diff = 1.0 - c_logp
#                             logp_score = -(diff * 30.0 + (diff ** 2) * 50.0)
#                         else:
#                             diff = c_logp - 3.0
#                             logp_score = -(diff * 30.0 + (diff ** 2) * 50.0)
#
#                         # 💥 综合打分大洗牌：
#                         # QED 权重拉爆到 25.0 (满分极高诱惑)
#                         # SA  惩罚拉到 2.0 (SA 越低，扣分越少，逼迫模型走向易合成结构)
#                         cand_score = (c_qed * 25.0) - (c_sa * 2.0) + logp_score
#
#                     except Exception:
#                         cand_score = -10000.0
#
#                     if cand_score > best_score:
#                         best_score = cand_score
#                         best_eps = cand_eps
#
#                 if best_score > -10000.0:
#                     guided_prediction[b] = base_prediction[b] + scale * (best_eps - base_prediction[b])
#
#     return guided_prediction
#
#
# class RigidAssemblyDecoder:
#     def __init__(self):
#         self.num2aa = const.NUM2AA
#         self.ideal_coords = const.IDEAL_COORDS
#         self.aa_bonds = const.AA_BONDS
#         self.frag_bond_cache = {}  # 缓存片段的真实化学键类型
#
#     def get_template_bond_type(self, f_type, start_name, end_name):
#         """
#         🌟 智能芳香环检测：结合 3D 平面度分析，合适的位置给芳香键，不合适的给单键！
#         """
#         if f_type not in self.frag_bond_cache:
#             try:
#                 from rdkit.Chem import rdDetermineBonds
#                 import numpy as np  # 引入 numpy 计算平面度
#
#                 temp_mol = Chem.RWMol()
#                 idx_map = {}
#                 coords = []
#
#                 for i, atom_info in enumerate(self.ideal_coords[f_type]):
#                     raw_name = atom_info[0].strip()
#                     elem_symbol = "".join([c for c in raw_name if c.isalpha()])
#                     if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
#                         elem_symbol = elem_symbol[0]
#                     idx = temp_mol.AddAtom(Chem.Atom(elem_symbol.capitalize()))
#                     idx_map[raw_name] = idx
#                     coords.append(atom_info[2])
#
#                 conf = Chem.Conformer(temp_mol.GetNumAtoms())
#                 for i, c in enumerate(coords):
#                     conf.SetAtomPosition(i, np.array(c).astype(float))
#                 temp_mol.AddConformer(conf)
#                 mol = temp_mol.GetMol()
#
#                 try:
#                     rdDetermineBonds.DetermineConnectivity(mol)
#                 except Exception:
#                     pass
#
#                 determined_officially = False
#                 try:
#                     # 尝试官方连双键
#                     rdDetermineBonds.DetermineBondOrders(mol, charge=0)
#                     determined_officially = True
#                 except Exception:
#                     # 官方失败时，先用单键兜底
#                     Chem.SanitizeMol(mol,
#                                      sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#
#                # 核心：调用强化版 3D 几何与杂化感知雷达
#
#                 conf = mol.GetConformer()
#                 all_coords = np.array([list(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())])
#
#                 for ring in mol.GetRingInfo().AtomRings():
#                     enhanced_aromaticity_perception_for_diffusion(mol, all_coords, ring)
#
#                 b_dict = {}
#                 for bond in mol.GetBonds():
#                     idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#                     n1_list = [k for k, v in idx_map.items() if v == idx1]
#                     n2_list = [k for k, v in idx_map.items() if v == idx2]
#                     if n1_list and n2_list:
#                         b_dict[tuple(sorted((n1_list[0], n2_list[0])))] = bond.GetBondType()
#                 self.frag_bond_cache[f_type] = b_dict
#             except Exception:
#                 self.frag_bond_cache[f_type] = {}
#
#         pair = tuple(sorted((start_name.strip(), end_name.strip())))
#         return self.frag_bond_cache[f_type].get(pair, Chem.rdchem.BondType.SINGLE)
#
#     def assemble(self, frag_pos, frag_rot, frag_types, node_mask):
#         batch_mols =[]
#         B = frag_pos.shape[0]
#
#         for b in range(B):
#             mask = node_mask[b].bool().cpu().view(-1)
#             b_pos = frag_pos[b][mask].detach().cpu().numpy()
#             b_rot = frag_rot[b][mask].detach().cpu().numpy()
#             temp = 0.25
#             probs = torch.softmax(frag_types[b][mask] / temp, dim=-1)
#
#             best_score = -10000.0
#             best_mol_candidate = None
#             best_coords_candidate = None
#
#             # 🌟 核心调优：增加抽卡次数至 20 次，确保能在严格的 LogP 限制下筛出极品
#             for attempt in range(20):
#                 b_type_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
#
#                 rw_mol = Chem.RWMol()
#                 all_final_coords =[]
#                 centers =[]
#                 atom_to_frag = {}
#
#                 for i in range(len(b_pos)):
#                     f_type = b_type_idx[i]
#                     t = b_pos[i].copy()
#                     R_raw = b_rot[i]
#
#                     # 宏观碰撞排斥
#                     max_relax_iters = 10
#                     for _ in range(max_relax_iters):
#                         collision_found = False
#                         for prev_t in centers:
#                             dist = np.linalg.norm(t - prev_t)
#                             if dist < 3.8:
#                                 collision_found = True
#                                 direction = t - prev_t
#                                 norm_dir = np.linalg.norm(direction)
#                                 if norm_dir < 1e-4: direction = np.random.randn(3)
#                                 norm_dir = np.linalg.norm(direction)
#                                 direction = direction / norm_dir
#                                 # 加入 0.5 的阻尼系数，慢慢推，防止剧烈震荡
#                                 t += direction * (4.0 - dist) * 0.5
#                         if not collision_found:
#                             break  # 如果大家隔得都挺远，安全退出
#                     centers.append(t.copy())
#
#                     try:
#                         U, _, Vt = np.linalg.svd(R_raw)
#                         R = np.dot(U, Vt)
#                         if np.linalg.det(R) < 0:
#                             Vt[-1, :] *= -1
#                             R = np.dot(U, Vt)
#                     except:
#                         R = np.eye(3)
#
#                     template = self.ideal_coords[f_type]
#                     f_bonds = self.aa_bonds[f_type]
#                     local_atom_map = {}
#
#                     for atom_info in template:
#                         raw_name = atom_info[0].strip()
#                         elem_symbol = "".join([c for c in raw_name if c.isalpha()])
#                         if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
#                             elem_symbol = elem_symbol[0]
#                         if elem_symbol.upper() == 'H': continue
#
#                         local_pos = np.array(atom_info[2])
#                         global_pos = np.dot(R, local_pos) + t
#
#                         try:
#                             rd_idx = rw_mol.AddAtom(Chem.Atom(elem_symbol.capitalize()))
#                             all_final_coords.append(global_pos)
#                             local_atom_map[raw_name] = rd_idx
#                             atom_to_frag[rd_idx] = i
#                         except:
#                             continue
#
#                     for start_name, end_name in f_bonds:
#                         u = local_atom_map.get(start_name.strip())
#                         v = local_atom_map.get(end_name.strip())
#                         if u is not None and v is not None:
#                             if not rw_mol.GetBondBetweenAtoms(u, v):
#                                 correct_bond_type = self.get_template_bond_type(f_type, start_name, end_name)
#                                 rw_mol.AddBond(u, v, correct_bond_type)
#
#                                 # 🌟 核心绝杀1：强制印刻原子的芳香性标签，对抗RDKit的三维歧视
#                                 if correct_bond_type == Chem.rdchem.BondType.AROMATIC:
#                                     rw_mol.GetAtomWithIdx(u).SetIsAromatic(True)
#                                     rw_mol.GetAtomWithIdx(v).SetIsAromatic(True)
#
#                 if rw_mol.GetNumAtoms() == 0: continue
#
#                 final_mol = rw_mol.GetMol()
#                 conf = Chem.Conformer(final_mol.GetNumAtoms())
#                 for idx, coord in enumerate(all_final_coords):
#                     conf.SetAtomPosition(idx, coord.astype(float))
#                 final_mol.AddConformer(conf)
#                 fallback_mol = Chem.Mol(final_mol)
#
#                 rw_mol_constrained = Chem.RWMol(fallback_mol)
#                 num_atoms = rw_mol_constrained.GetNumAtoms()
#                 max_valence = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}
#
#                 connected_frags = set()
#                 frag_connection_count = {}
#
#                 # 片段之间的拼接
#                 for idx_i in range(num_atoms):
#                     for idx_j in range(idx_i + 1, num_atoms):
#                         if not rw_mol_constrained.GetBondBetweenAtoms(idx_i, idx_j):
#                             p1 = np.array(conf.GetAtomPosition(idx_i))
#                             p2 = np.array(conf.GetAtomPosition(idx_j))
#                             dist = np.linalg.norm(p1 - p2)
#
#                             if 0.4 < dist < 1.65:
#                                 frag_i = atom_to_frag.get(idx_i, -1)
#                                 frag_j = atom_to_frag.get(idx_j, -2)
#
#                                 if frag_i != frag_j:
#                                     pair = tuple(sorted((frag_i, frag_j)))
#                                     if pair in connected_frags: continue
#
#                                     MAX_FRAG_DEGREE = 2
#                                     if frag_connection_count.get(frag_i, 0) >= MAX_FRAG_DEGREE or \
#                                             frag_connection_count.get(frag_j, 0) >= MAX_FRAG_DEGREE:
#                                         continue
#
#                                 atom_i = rw_mol_constrained.GetAtomWithIdx(idx_i)
#                                 atom_j = rw_mol_constrained.GetAtomWithIdx(idx_j)
#                                 limit_i = max_valence.get(atom_i.GetSymbol(), 4)
#                                 limit_j = max_valence.get(atom_j.GetSymbol(), 4)
#
#                                 if atom_i.GetDegree() < limit_i and atom_j.GetDegree() < limit_j:
#                                     rw_mol_constrained.AddBond(idx_i, idx_j, Chem.rdchem.BondType.SINGLE)
#
#                                     if frag_i != frag_j:
#                                         connected_frags.add(pair)
#                                         frag_connection_count[frag_i] = frag_connection_count.get(frag_i, 0) + 1
#                                         frag_connection_count[frag_j] = frag_connection_count.get(frag_j, 0) + 1
#                 if len(all_final_coords) > 1:
#                     from scipy.spatial.distance import pdist
#                     distances = pdist(np.array(all_final_coords))
#                     # 如果任意两个原子距离小于 0.45 埃，说明发生了极其严重的物理重叠
#                     if np.min(distances) < 0.45:
#                         continue  # 直接抛弃这个废品，进入下一个 attempt，绝不送给 RDKit 折磨它！
#                 try:
#                     raw_mol = rw_mol_constrained.GetMol()
#                     # 让 UFF 力场先轻微活动一下扭曲的原子，然后自动推断芳香键
#                     final_mol = force_field_assisted_reconstruction(raw_mol)
#
#                     # 彻底进行标准化学检查，因为键级已经被正确分配了
#                     Chem.SanitizeMol(final_mol)
#                 except:
#                     # 如果方案二崩溃，尝试调用方案三 (xyz2mol) 作为终极兜底
#                     try:
#                         atoms_list = [atom.GetAtomicNum() for atom in raw_mol.GetAtoms()]
#                         coords_list = [list(conf.GetAtomPosition(i)) for i in range(raw_mol.GetNumAtoms())]
#                         fallback_mol = assemble_via_xyz2mol(atoms_list, coords_list)
#                         if fallback_mol is not None:
#                             final_mol = fallback_mol
#                         else:
#                             continue
#                     except:
#                         continue
#
#                 try:
#                     frags = Chem.GetMolFrags(final_mol, asMols=True)
#                     if not frags: continue
#                     largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
#                     if largest_frag.GetNumAtoms() < 5: continue
#                 except:
#                     continue
#
#                 # =========================================================
#                 # 绝杀3：基于重构特征的多目标打分（LogP平底锅 + 芳香环奖励）
#                 # =========================================================
#                 try:
#                     c_qed = QED.qed(largest_frag)
#                     c_sa = sascorer.calculateScore(largest_frag)
#                     c_logp = Descriptors.MolLogP(largest_frag)
#
#                     # 执行极高梯度的 LogP 二次惩罚
#                     if 1.0 <= c_logp <= 3.0:
#                         logp_penalty = 0.0
#                     elif c_logp < 1.0:
#                         logp_penalty = ((1.0 - c_logp) ** 2) * 15.0
#                     else:
#                         logp_penalty = ((c_logp - 3.0) ** 2) * 15.0
#
#                     ring_info = largest_frag.GetRingInfo()
#                     atom_rings = ring_info.AtomRings()
#
#                     good_rings = 0
#                     bad_rings = 0
#                     aromatic_rings = 0
#
#                     for ring in atom_rings:
#                         if len(ring) in [5, 6]:
#                             good_rings += 1
#                         else:
#                             bad_rings += 1
#                         # 现在原子被标记了芳香性，这里的判定将能准确找到生成的芳香环
#                         if all(largest_frag.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
#                             aromatic_rings += 1
#
#                     cand_score = (c_qed * 15.0) - c_sa - logp_penalty
#                     # 加大对于优质芳香拓扑特征的奖赏，强行引导模型选出带芳香体系的构象
#                     cand_score += (aromatic_rings * 5.0) + (good_rings * 2.0) - (bad_rings * 4.0)
#
#                     if cand_score > best_score:
#                         best_score = cand_score
#                         best_mol_candidate = largest_frag
#
#                         conf_new = best_mol_candidate.GetConformer()
#                         best_coords_candidate = [conf_new.GetAtomPosition(i) for i in
#                                                  range(best_mol_candidate.GetNumAtoms())]
#                 except:
#                     continue
#
#             # ====================== 重采样结束 ======================
#             if best_mol_candidate is None:
#                 batch_mols.append((None, None))
#                 continue
#
#             # 🌟 绝杀4：力场-芳香共振机制 (Force Field Resonance)
#             # 我们通过代码预先写死了芳香拓扑，现在召唤 MMFF94 力场。
#             # 力场会读取这些键的理想键长与键角，以雷霆万钧之势把扩散模型生成的扭曲 3D 坐标
#             # 暴力压平，还原出一个完美的平面芳香环三维坐标！
#             try:
#                 # 此时由于构象将被MMFF强行校正，我们可以安全地执行完全Sanitize
#                 Chem.SanitizeMol(best_mol_candidate)
#                 rdForceFieldHelpers.MMFFOptimizeMolecule(best_mol_candidate, maxIters=1000, nonBondedThresh=100.0)
#             except:
#                 pass
#
#             try:
#                 conf_new = best_mol_candidate.GetConformer()
#                 final_coords_cleaned = [conf_new.GetAtomPosition(i) for i in range(best_mol_candidate.GetNumAtoms())]
#             except:
#                 final_coords_cleaned = best_coords_candidate
#
#             batch_mols.append((best_mol_candidate, final_coords_cleaned))
#
#         return batch_mols
#
#
# # =========================================================================
# # 后续的数据处理和运行逻辑保持不变
# # =========================================================================
# def _ensure_3d_mask(m: torch.Tensor, B: int, N: int, device, default_val: float = 1.0):
#     if m is None: return torch.full((B, N, 1), float(default_val), device=device)
#     if not isinstance(m, torch.Tensor): m = torch.tensor(m, dtype=torch.float32, device=device)
#     if m.dim() == 1:
#         m = m.view(1, -1, 1).repeat(B, 1, 1)
#     elif m.dim() == 2:
#         m = m.unsqueeze(-1)
#     elif m.dim() == 3:
#         m = m[:, :, :1]
#     return m.float().to(device)
#
#
# def _xyz_only(pos: torch.Tensor):
#     if pos.dim() == 2: pos = pos.unsqueeze(0)
#     if pos.size(-1) >= 3: return pos[..., :3].contiguous()
#     raise RuntimeError(f"Position tensor last dim must be >= 3, got {pos.shape}")
#
#
# def read_molecule(path):
#     mol = None
#     if path.endswith('.pdb'):
#         mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
#     elif path.endswith('.mol') or path.endswith('.mol2'):
#         mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
#     elif path.endswith('.sdf'):
#         suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=True)
#         mol = suppl[0] if len(suppl) > 0 else None
#     if mol is None: raise ValueError(f"Failed to read molecule from {path}")
#     return mol
#
#
# def bridge_rdkit_to_openfold_tensor(mol, device='cpu'):
#     residues = list(mol.GetResidues())
#     n_res = len(residues)
#     all_atom_positions = np.zeros((n_res, 37, 3))
#     all_atom_mask = np.zeros((n_res, 37))
#     aatype = np.zeros(n_res, dtype=np.int64)
#     for i, res in enumerate(residues):
#         res_name = res.GetResidueName().strip()
#         aatype[i] = rc.restype_order.get(rc.restype_3to1.get(res_name, 'X'), 20)
#         for atom in res.GetAtoms():
#             p_info = atom.GetPDBResidueInfo()
#             if p_info is None: continue
#             atom_name = p_info.GetName().strip()
#             if atom_name in rc.atom_order:
#                 idx = rc.atom_order[atom_name]
#                 pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
#                 all_atom_positions[i, idx] = [pos.x, pos.y, pos.z]
#                 all_atom_mask[i, idx] = 1.0
#     return {
#         "aatype": torch.tensor(aatype, device=device),
#         "all_atom_positions": torch.tensor(all_atom_positions, dtype=torch.float32, device=device),
#         "all_atom_mask": torch.tensor(all_atom_mask, dtype=torch.float32, device=device)
#     }
#
#
# def atom37_to_frames(protein, eps=1e-8):
#     aatype = protein["aatype"]
#     all_atom_positions = protein["all_atom_positions"]
#     all_atom_mask = protein["all_atom_mask"]
#     batch_dims = len(aatype.shape[:-1])
#
#     restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
#     restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
#     restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]
#
#     for restype, restype_letter in enumerate(rc.restypes):
#         resname = rc.restype_1to3[restype_letter]
#         for chi_idx in range(4):
#             if rc.chi_angles_mask[restype][chi_idx]:
#                 names = rc.chi_angles_atoms[resname][chi_idx]
#                 restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]
#
#     restype_rigidgroup_mask = all_atom_mask.new_zeros((*aatype.shape[:-1], 21, 8))
#     restype_rigidgroup_mask[..., 0] = 1
#     restype_rigidgroup_mask[..., 3] = 1
#     restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)
#
#     lookuptable = rc.atom_order.copy()
#     lookuptable[""] = 0
#     lookup = np.vectorize(lambda x: lookuptable[x])
#     restype_rigidgroup_base_atom37_idx = lookup(restype_rigidgroup_base_atom_names)
#     restype_rigidgroup_base_atom37_idx = aatype.new_tensor(restype_rigidgroup_base_atom37_idx)
#     restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(*((1,) * batch_dims),
#                                                                                  *restype_rigidgroup_base_atom37_idx.shape)
#
#     residx_rigidgroup_base_atom37_idx = batched_gather(restype_rigidgroup_base_atom37_idx, aatype, dim=-3,
#                                                        no_batch_dims=batch_dims)
#     base_atom_pos = batched_gather(all_atom_positions, residx_rigidgroup_base_atom37_idx, dim=-2,
#                                    no_batch_dims=len(all_atom_positions.shape[:-2]))
#
#     gt_frames = Rigid.from_3_points(p_neg_x_axis=base_atom_pos[..., 0, :], origin=base_atom_pos[..., 1, :],
#                                     p_xy_plane=base_atom_pos[..., 2, :], eps=eps)
#
#     group_exists = batched_gather(restype_rigidgroup_mask, aatype, dim=-2, no_batch_dims=batch_dims)
#     gt_atoms_exist = batched_gather(all_atom_mask, residx_rigidgroup_base_atom37_idx, dim=-1,
#                                     no_batch_dims=len(all_atom_mask.shape[:-1]))
#     gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists
#
#     rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
#     rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
#     rots[..., 0, 0, 0] = -1
#     rots[..., 0, 2, 2] = -1
#     rots = Rotation(rot_mats=rots)
#     gt_frames = gt_frames.compose(Rigid(rots, None))
#
#     gt_frames_tensor = gt_frames.to_tensor_4x4()
#     protein["rigidgroups_gt_frames"] = gt_frames_tensor
#     protein["rigidgroups_gt_exists"] = gt_exists
#     return protein
#
#
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output, exist_ok=True)
#
#     print(f"🚀 Loading FragDiff model from {args.model}...")
#     ddpm = DDPM.load_from_checkpoint(args.model, map_location=device).eval().to(device)
#     if args.n_steps is not None: ddpm.edm.T = args.n_steps
#     ddpm.reconstruct_and_save = lambda *args, **kwargs: []
#
#     if args.linker_size.isdigit():
#         fixed_size = int(args.linker_size)
#
#         def sample_fn(_data):
#             return torch.full((_data['positions'].shape[0],), fixed_size, device=device, dtype=const.TORCH_INT)
#     else:
#         bounds = [int(x.strip()) for x in args.linker_size.split(',')]
#         left, right = bounds[0], bounds[1]
#
#         def sample_fn(_data):
#             return torch.randint(left, right + 1, (len(_data['positions']),), device=device, dtype=const.TORCH_INT)
#
#     dataset_list = []
#
#     if args.use_dataset:
#         print("📁 Running in DATASET MODE")
#         split_path = args.split_file or os.path.join(args.data_dir, f"{args.dataset_prefix}_split.pt")
#         full_dataset = PDBbindDataset(data_path=args.data_dir, prefix=args.dataset_prefix, device=device)
#         split_data = torch.load(split_path)
#         test_indices = split_data.get('test', []) or split_data.get('val', [])
#         if args.limit > 0: test_indices = test_indices[:args.limit]
#
#         for idx in tqdm(test_indices, desc="Pre-loading Targets"):
#             try:
#                 data_item = full_dataset[idx]
#                 if 'positions' not in data_item: data_item['positions'] = data_item['ligand_pos']
#                 if 'one_hot' not in data_item:
#                     elem_map = {1: 0, 6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
#                     elements = data_item['ligand_element'].view(-1).tolist()
#                     one_hot = torch.zeros((len(elements), 9), dtype=torch.float32)
#                     for i, e in enumerate(elements): one_hot[i, elem_map.get(e, 0)] = 1.0
#                     data_item['one_hot'] = one_hot
#                 if 'fragment_mask' not in data_item:
#                     n_atoms = len(data_item['positions'])
#                     import random
#                     n_linker = random.randint(1, 3)
#                     n_frag = max(1, n_atoms - n_linker)
#                     frag_mask = torch.zeros(n_atoms, dtype=torch.float32)
#                     frag_mask[:n_frag] = 1.0
#                     link_mask = torch.zeros(n_atoms, dtype=torch.float32)
#                     link_mask[n_frag:] = 1.0
#                     data_item['fragment_mask'] = frag_mask
#                     data_item['linker_mask'] = link_mask
#                 if 'anchors' not in data_item: data_item['anchors'] = torch.zeros(len(data_item['positions']),
#                                                                                   dtype=torch.float32)
#                 for _ in range(args.n_samples): dataset_list.append({k: v for k, v in data_item.items()})
#             except Exception:
#                 pass
#     else:
#         print("🧪 Running in SINGLE FILE MODE")
#         pocket_mol = read_molecule(args.pocket)
#         pocket_tensor = bridge_rdkit_to_openfold_tensor(pocket_mol, device=device)
#         pocket_with_frames = atom37_to_frames(pocket_tensor)
#         pocket_frames = pocket_with_frames["rigidgroups_gt_frames"][:, 0, :, :]
#         pocket_mask = pocket_with_frames["rigidgroups_gt_exists"][:, 0]
#
#         molecule = read_molecule(args.fragments)
#         positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
#         n_atoms = len(positions)
#
#         anchor_flags = np.zeros(n_atoms)
#         if args.anchors:
#             for a in args.anchors.split(','): anchor_flags[int(a.strip()) - 1] = 1
#
#         single_data = {
#             'uuid': '0', 'name': os.path.basename(args.fragments).split('.')[0],
#             'positions': torch.tensor(positions, dtype=torch.float32, device=device),
#             'one_hot': torch.tensor(one_hot, dtype=torch.float32, device=device),
#             'charges': torch.tensor(charges, dtype=torch.float32, device=device),
#             'anchors': torch.tensor(anchor_flags, dtype=torch.float32, device=device),
#             'fragment_mask': torch.ones(n_atoms, dtype=torch.float32, device=device),
#             'linker_mask': torch.zeros(n_atoms, dtype=torch.float32, device=device),
#             'num_atoms': n_atoms,
#             'pocket_frames': pocket_frames.to(device),
#             'pocket_mask': pocket_mask.to(device)
#         }
#         dataset_list = [single_data] * args.n_samples
#
#     if not dataset_list: return
#     batch_size = min(len(dataset_list), 16)
#     dataloader = get_dataloader(dataset_list, batch_size=batch_size, collate_fn=collate_with_fragment_edges)
#
#     for batch_i, data in enumerate(tqdm(dataloader, desc="Denoising Samples")):
#         for k, v in data.items():
#             if isinstance(v, torch.Tensor): data[k] = v.to(device)
#
#         bs, n_nodes = data['positions'].shape[0], data['positions'].shape[1]
#         curr_node_mask = _ensure_3d_mask(data.get('atom_mask', None), bs, n_nodes, device)
#         curr_edge_mask = None
#         curr_context = data.get('context', None)
#
#         if 'linker_mask' in data and 'fragment_mask' in data:
#             real_pos_temp = data['positions'].clone()
#             linker_mask_3d = _ensure_3d_mask(data['linker_mask'], bs, n_nodes, device)
#             fragment_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
#
#             raw_noise = torch.randn_like(real_pos_temp)
#             sum_pos = (real_pos_temp * fragment_mask_3d).sum(dim=1, keepdim=True)
#             count_pos = fragment_mask_3d.sum(dim=1, keepdim=True).clamp(min=1.0)
#             center_of_mass = sum_pos / count_pos
#             aligned_noise = raw_noise + center_of_mass
#             data['positions'] = torch.where(linker_mask_3d > 0.5, aligned_noise, real_pos_temp)
#
#         sample_results = None
#         for attempt in range(5):
#             try:
#                 decoder_instance = RigidAssemblyDecoder()
#                 my_guidance_fn = lambda x_t, t, base_pred: svdd_qed_guidance(
#                     x_t, t, base_pred, data, decoder_instance, curr_node_mask, num_samples=15, scale=3.5
#                 )
#                 sample_results = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1, edge_mask=curr_edge_mask,
#                                                    context=curr_context, save_path=None, guidance_fn=my_guidance_fn)
#                 break
#             except FoundNaNException:
#                 continue
#             except Exception as e:
#                 import traceback
#                 print(f"\n❌ 采样过程发生致命错误: {e}")
#                 traceback.print_exc()
#                 break
#
#         if sample_results is None: continue
#
#         chain_data, node_mask = sample_results
#         while isinstance(chain_data, (tuple, list)): chain_data = chain_data[0]
#
#         if chain_data.dim() == 4:
#             current_chain = chain_data[-1]
#         else:
#             current_chain = chain_data
#
#         x_gen = current_chain[..., :3]
#
#         if 'fragment_mask' in data:
#             frag_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
#             real_pos = _xyz_only(data['positions'])
#             x_gen = torch.where(frag_mask_3d > 0.5, real_pos, x_gen)
#
#         final_x = x_gen
#
#         try:
#             rot_flat = current_chain[..., 3:12]
#             rot_gen = rot_flat.view(bs, n_nodes, 3, 3)
#
#             if 'frag_rotations' in data and 'fragment_mask' in data:
#                 frag_mask_3d_rot = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device).unsqueeze(-1)
#                 real_rot = data['frag_rotations']
#                 rot_gen = torch.where(frag_mask_3d_rot > 0.5, real_rot, rot_gen)
#
#             raw_h_gen = current_chain[..., 12:]
#             frag_types_gen = raw_h_gen[..., :len(const.NUM2AA)]
#
#             decoder = RigidAssemblyDecoder()
#             mols_and_coords = decoder.assemble(final_x, rot_gen, frag_types_gen, node_mask)
#
#             raw_names = data.get('name', [f"target_{batch_i}" for _ in range(bs)])
#             if isinstance(raw_names, str): raw_names = [raw_names] * bs
#
#             for j, (mol, coords) in enumerate(mols_and_coords):
#                 if mol is None or coords is None: continue
#                 target_name = raw_names[j]
#                 sample_idx = batch_i * batch_size + j
#                 sdf_path = os.path.join(args.output, f"{target_name}_sample_{sample_idx}.sdf")
#
#                 final_mol = Chem.Mol(mol)
#                 try:
#                     conf = Chem.Conformer(final_mol.GetNumAtoms())
#                     for idx, coord in enumerate(coords):
#                         conf.SetAtomPosition(idx, np.array(coord, dtype=np.float64))
#                     final_mol.RemoveAllConformers()
#                     final_mol.AddConformer(conf)
#                     Chem.Kekulize(final_mol, clearAromaticFlags=True)
#                 except Exception as e:
#                     pass
#
#
#                 writer = Chem.SDWriter(sdf_path)
#                 writer.SetKekulize(False)
#                 writer.write(final_mol)
#                 writer.close()
#
#         except Exception as e:
#             print(f"❌ Batch {batch_i} 组装/保存过程发生错误: {e}")
#
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)
#pdb 16
# import argparse
# import os
# import numpy as np
# import torch
# import copy
# from rdkit import Chem
# from rdkit.Chem import rdForceFieldHelpers
# from rdkit.Chem import QED
# from rdkit.Chem import Descriptors
# from tqdm import tqdm
# import sys
# from rdkit import RDConfig
#
# # 引入合成可及性打分
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# try:
#     from src import sascorer
# except ImportError:
#     import sascorer
#
# from src import const
# from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule, PDBbindDataset
# from src.lightning import DDPM
# from src.utils import FoundNaNException
# from src.visualizer import save_xyz_file
#
# from openfold.np import residue_constants as rc
# from openfold.utils.rigid_utils import Rotation, Rigid
# from openfold.utils.tensor_utils import batched_gather
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--output', type=str, default='./results')
# parser.add_argument('--n_samples', type=int, default=5)
# parser.add_argument('--n_steps', type=int, default=None)
# parser.add_argument('--linker_size', type=str, default='5')
# parser.add_argument('--use_dataset', action='store_true')
# parser.add_argument('--data_dir', type=str, default='data')
# parser.add_argument('--dataset_prefix', type=str, default='PDBbind_pocket10')
# parser.add_argument('--split_file', type=str, default=None)
# parser.add_argument('--limit', type=int, default=-1)
# parser.add_argument('--fragments', type=str, default=None)
# parser.add_argument('--pocket', type=str, default=None)
# parser.add_argument('--anchors', type=str, default=None)
#
#
# def svdd_qed_guidance(x_t, t, base_prediction, data, decoder, node_mask, num_samples=3, scale=1.5):
#     """
#     多目标优化版 SVDD 引导：优化 QED，惩罚 SA Score，并将 LogP 锁定在 [1.0, 3.0] 黄金区间。
#     """
#     device = x_t.device
#     B = x_t.shape[0]
#     guided_prediction = base_prediction.clone()
#
#     if isinstance(t, torch.Tensor):
#         t_val = t[0].item() if t.numel() > 1 else t.item()
#     else:
#         t_val = t
#
#     if t_val < 20:
#         with torch.no_grad():
#             for b in range(B):
#                 best_eps = base_prediction[b]
#                 best_score = -100.0
#                 mask = node_mask[b].bool().cpu().view(-1)
#
#                 if not mask.any(): continue
#
#                 for _ in range(num_samples):
#                     cand_eps = base_prediction[b] + torch.randn_like(base_prediction[b]) * 0.05
#                     cand_pos = (x_t[b] - cand_eps)[mask].detach().cpu().numpy()
#
#                     if 'one_hot' in data:
#                         probs = torch.softmax(data['one_hot'][b][mask] / 0.5, dim=-1)
#                         b_type_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
#                     else:
#                         b_type_idx = np.zeros(len(cand_pos), dtype=np.int64)
#
#                     try:
#                         rw_mol = Chem.RWMol()
#                         valid_coords = []
#                         for i in range(len(cand_pos)):
#                             template = decoder.ideal_coords[b_type_idx[i]]
#                             atom_info = template[0][0]
#                             elem_symbol = "".join([c for c in atom_info.strip() if c.isalpha()]).capitalize()
#                             if elem_symbol == 'H': continue
#                             rw_mol.AddAtom(Chem.Atom(elem_symbol))
#                             valid_coords.append(cand_pos[i])
#
#                         num_atoms = rw_mol.GetNumAtoms()
#                         for idx_i in range(num_atoms):
#                             for idx_j in range(idx_i + 1, num_atoms):
#                                 dist = np.linalg.norm(valid_coords[idx_i] - valid_coords[idx_j])
#                                 if 1.0 < dist < 4.0:
#                                     rw_mol.AddBond(idx_i, idx_j, Chem.rdchem.BondType.SINGLE)
#
#                         mol = rw_mol.GetMol()
#                         Chem.SanitizeMol(mol)
#
#                         c_qed = QED.qed(mol)
#                         c_sa = sascorer.calculateScore(mol)
#                         c_logp = Descriptors.MolLogP(mol)
#
#                         # 🌟 平底锅 LogP 惩罚：严格锁定在 1.0 ~ 3.0 之间
#                         if 1.0 <= c_logp <= 3.0:
#                             logp_penalty = 0.0
#                         elif c_logp < 1.0:
#                             logp_penalty = (1.0 - c_logp) * 4.0
#                         else:
#                             logp_penalty = (c_logp - 3.0) * 4.0
#
#                         cand_score = (c_qed * 10.0) - c_sa - logp_penalty
#                     except Exception:
#                         cand_score = -100.0
#
#                     if cand_score > best_score:
#                         best_score = cand_score
#                         best_eps = cand_eps
#
#                 if best_score > -100.0:
#                     guided_prediction[b] = base_prediction[b] + scale * (best_eps - base_prediction[b])
#
#     return guided_prediction
#
#
# class RigidAssemblyDecoder:
#     def __init__(self):
#         self.num2aa = const.NUM2AA
#         self.ideal_coords = const.IDEAL_COORDS
#         self.aa_bonds = const.AA_BONDS
#         self.frag_bond_cache = {}  # 缓存片段的真实化学键类型
#
#     def get_template_bond_type(self, f_type, start_name, end_name):
#         """
#         🌟 核心魔法 1：模板基因印刻。
#         提取极其完美的片段底层键属性，确保苯环等自带真正的“芳香键”！
#         """
#         if f_type not in self.frag_bond_cache:
#             try:
#                 from rdkit.Chem import rdDetermineBonds
#                 temp_mol = Chem.RWMol()
#                 idx_map = {}
#                 conf = Chem.Conformer(len(self.ideal_coords[f_type]))
#                 for i, atom_info in enumerate(self.ideal_coords[f_type]):
#                     raw_name = atom_info[0].strip()
#                     elem_symbol = "".join([c for c in raw_name if c.isalpha()])
#                     if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
#                         elem_symbol = elem_symbol[0]
#                     # 我们需要H来帮助判断真实价态，不跳过H
#                     idx = temp_mol.AddAtom(Chem.Atom(elem_symbol.capitalize()))
#                     idx_map[raw_name] = idx
#                     conf.SetAtomPosition(idx, np.array(atom_info[2]).astype(float))
#                 temp_mol.GetMol().AddConformer(conf)
#
#                 for st, en in self.aa_bonds[f_type]:
#                     u = idx_map.get(st.strip())
#                     v = idx_map.get(en.strip())
#                     if u is not None and v is not None:
#                         temp_mol.AddBond(u, v, Chem.rdchem.BondType.SINGLE)
#
#                 Chem.SanitizeMol(temp_mol)
#                 rdDetermineBonds.DetermineBondOrders(temp_mol.GetMol(), charge=0)
#
#                 b_dict = {}
#                 for bond in temp_mol.GetBonds():
#                     idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#                     n1 = [k for k, v in idx_map.items() if v == idx1][0]
#                     n2 = [k for k, v in idx_map.items() if v == idx2][0]
#                     b_dict[tuple(sorted((n1, n2)))] = bond.GetBondType()
#                 self.frag_bond_cache[f_type] = b_dict
#             except Exception:
#                 self.frag_bond_cache[f_type] = {}
#
#         pair = tuple(sorted((start_name.strip(), end_name.strip())))
#         return self.frag_bond_cache[f_type].get(pair, Chem.rdchem.BondType.SINGLE)
#
#     def assemble(self, frag_pos, frag_rot, frag_types, node_mask):
#         batch_mols = []
#         B = frag_pos.shape[0]
#
#         for b in range(B):
#             mask = node_mask[b].bool().cpu().view(-1)
#             b_pos = frag_pos[b][mask].detach().cpu().numpy()
#             b_rot = frag_rot[b][mask].detach().cpu().numpy()
#             temp = 0.25
#             probs = torch.softmax(frag_types[b][mask] / temp, dim=-1)
#
#             best_score = -1000.0
#             best_mol_candidate = None
#             best_coords_candidate = None
#
#             # 重采样抽卡 10 次
#             for attempt in range(10):
#                 b_type_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
#
#                 rw_mol = Chem.RWMol()
#                 all_final_coords = []
#                 centers = []
#                 atom_to_frag = {}
#
#                 for i in range(len(b_pos)):
#                     f_type = b_type_idx[i]
#                     t = b_pos[i].copy()
#                     R_raw = b_rot[i]
#
#                     # 宏观碰撞排斥
#                     for prev_t in centers:
#                         dist = np.linalg.norm(t - prev_t)
#                         if dist < 3.8:
#                             direction = t - prev_t
#                             norm_dir = np.linalg.norm(direction)
#                             if norm_dir < 1e-4: direction = np.random.randn(3)
#                             norm_dir = np.linalg.norm(direction)
#                             direction = direction / norm_dir
#                             t += direction * (4.0 - dist)
#                     centers.append(t.copy())
#
#                     try:
#                         U, _, Vt = np.linalg.svd(R_raw)
#                         R = np.dot(U, Vt)
#                         if np.linalg.det(R) < 0:
#                             Vt[-1, :] *= -1
#                             R = np.dot(U, Vt)
#                     except:
#                         R = np.eye(3)
#
#                     template = self.ideal_coords[f_type]
#                     f_bonds = self.aa_bonds[f_type]
#                     local_atom_map = {}
#
#                     for atom_info in template:
#                         raw_name = atom_info[0].strip()
#                         elem_symbol = "".join([c for c in raw_name if c.isalpha()])
#                         if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
#                             elem_symbol = elem_symbol[0]
#                         if elem_symbol.upper() == 'H': continue
#
#                         local_pos = np.array(atom_info[2])
#                         global_pos = np.dot(R, local_pos) + t
#
#                         try:
#                             rd_idx = rw_mol.AddAtom(Chem.Atom(elem_symbol.capitalize()))
#                             all_final_coords.append(global_pos)
#                             local_atom_map[raw_name] = rd_idx
#                             atom_to_frag[rd_idx] = i
#                         except:
#                             continue
#
#                     for start_name, end_name in f_bonds:
#                         u = local_atom_map.get(start_name.strip())
#                         v = local_atom_map.get(end_name.strip())
#                         if u is not None and v is not None:
#                             if not rw_mol.GetBondBetweenAtoms(u, v):
#                                 # 🌟 调用核心魔法：直接打上芳香键和双键的烙印！
#                                 correct_bond_type = self.get_template_bond_type(f_type, start_name, end_name)
#                                 rw_mol.AddBond(u, v, correct_bond_type)
#
#                 if rw_mol.GetNumAtoms() == 0: continue
#
#                 final_mol = rw_mol.GetMol()
#                 conf = Chem.Conformer(final_mol.GetNumAtoms())
#                 for idx, coord in enumerate(all_final_coords):
#                     conf.SetAtomPosition(idx, coord.astype(float))
#                 final_mol.AddConformer(conf)
#                 fallback_mol = Chem.Mol(final_mol)
#
#                 rw_mol_constrained = Chem.RWMol(fallback_mol)
#                 num_atoms = rw_mol_constrained.GetNumAtoms()
#                 max_valence = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}
#
#                 connected_frags = set()
#                 frag_connection_count = {}
#
#                 # 🌟 片段之间的拼接，严格保持单键（符合真实化学规律）
#                 for idx_i in range(num_atoms):
#                     for idx_j in range(idx_i + 1, num_atoms):
#                         if not rw_mol_constrained.GetBondBetweenAtoms(idx_i, idx_j):
#                             p1 = np.array(conf.GetAtomPosition(idx_i))
#                             p2 = np.array(conf.GetAtomPosition(idx_j))
#                             dist = np.linalg.norm(p1 - p2)
#
#                             if 0.4 < dist < 1.65:
#                                 frag_i = atom_to_frag.get(idx_i, -1)
#                                 frag_j = atom_to_frag.get(idx_j, -2)
#
#                                 if frag_i != frag_j:
#                                     pair = tuple(sorted((frag_i, frag_j)))
#                                     if pair in connected_frags: continue
#
#                                     MAX_FRAG_DEGREE = 2
#                                     if frag_connection_count.get(frag_i, 0) >= MAX_FRAG_DEGREE or \
#                                             frag_connection_count.get(frag_j, 0) >= MAX_FRAG_DEGREE:
#                                         continue
#
#                                 atom_i = rw_mol_constrained.GetAtomWithIdx(idx_i)
#                                 atom_j = rw_mol_constrained.GetAtomWithIdx(idx_j)
#                                 limit_i = max_valence.get(atom_i.GetSymbol(), 4)
#                                 limit_j = max_valence.get(atom_j.GetSymbol(), 4)
#
#                                 if atom_i.GetDegree() < limit_i and atom_j.GetDegree() < limit_j:
#                                     rw_mol_constrained.AddBond(idx_i, idx_j, Chem.rdchem.BondType.SINGLE)
#
#                                     if frag_i != frag_j:
#                                         connected_frags.add(pair)
#                                         frag_connection_count[frag_i] = frag_connection_count.get(frag_i, 0) + 1
#                                         frag_connection_count[frag_j] = frag_connection_count.get(frag_j, 0) + 1
#
#                 try:
#                     Chem.SanitizeMol(rw_mol_constrained, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
#                     final_mol = rw_mol_constrained.GetMol()
#                 except:
#                     continue
#
#                 try:
#                     frags = Chem.GetMolFrags(final_mol, asMols=True)
#                     if not frags: continue
#                     largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
#                     if largest_frag.GetNumAtoms() < 5: continue
#                 except:
#                     continue
#
#                 # =========================================================
#                 # 🌟 绝杀2：多目标打分，锁定黄金 LogP 与环结构
#                 # =========================================================
#                 try:
#                     c_qed = QED.qed(largest_frag)
#                     c_sa = sascorer.calculateScore(largest_frag)
#                     c_logp = Descriptors.MolLogP(largest_frag)
#
#                     if 1.0 <= c_logp <= 3.0:
#                         logp_penalty = 0.0
#                     elif c_logp < 1.0:
#                         logp_penalty = (1.0 - c_logp) * 6.0  # 极度重罚
#                     else:
#                         logp_penalty = (c_logp - 3.0) * 6.0  # 极度重罚
#
#                     ring_info = largest_frag.GetRingInfo()
#                     atom_rings = ring_info.AtomRings()
#
#                     good_rings = 0
#                     bad_rings = 0
#                     aromatic_rings = 0
#
#                     for ring in atom_rings:
#                         # 奖励最稳定的 5/6 元环，惩罚奇葩环
#                         if len(ring) in [5, 6]:
#                             good_rings += 1
#                         else:
#                             bad_rings += 1
#                         # 识别真正的芳香环
#                         if all(largest_frag.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
#                             aromatic_rings += 1
#
#                     cand_score = (c_qed * 10.0) - c_sa - logp_penalty
#                     # 芳香环加分，5/6元环加分，奇葩环扣大分
#                     cand_score += (aromatic_rings * 3.0) + (good_rings * 1.5) - (bad_rings * 3.0)
#
#                     if cand_score > best_score:
#                         best_score = cand_score
#                         best_mol_candidate = largest_frag
#
#                         conf_new = best_mol_candidate.GetConformer()
#                         best_coords_candidate = [conf_new.GetAtomPosition(i) for i in
#                                                  range(best_mol_candidate.GetNumAtoms())]
#                 except:
#                     continue
#
#             # ====================== 10次重采样结束 ======================
#             if best_mol_candidate is None:
#                 batch_mols.append((None, None))
#                 continue
#
#             # 🌟 绝杀3：力场-芳香共振
#             # 因为现在有真实的芳香键了，MMFF 会自动把苯环/吡啶压成完美的平面！
#             try:
#                 rdForceFieldHelpers.MMFFOptimizeMolecule(best_mol_candidate, maxIters=1000, nonBondedThresh=100.0)
#             except:
#                 pass
#
#             try:
#                 conf_new = best_mol_candidate.GetConformer()
#                 final_coords_cleaned = [conf_new.GetAtomPosition(i) for i in range(best_mol_candidate.GetNumAtoms())]
#             except:
#                 final_coords_cleaned = best_coords_candidate
#
#             batch_mols.append((best_mol_candidate, final_coords_cleaned))
#
#         return batch_mols
#
#
# # =========================================================================
# # 后续的数据处理和运行逻辑保持不变
# # =========================================================================
# def _ensure_3d_mask(m: torch.Tensor, B: int, N: int, device, default_val: float = 1.0):
#     if m is None: return torch.full((B, N, 1), float(default_val), device=device)
#     if not isinstance(m, torch.Tensor): m = torch.tensor(m, dtype=torch.float32, device=device)
#     if m.dim() == 1:
#         m = m.view(1, -1, 1).repeat(B, 1, 1)
#     elif m.dim() == 2:
#         m = m.unsqueeze(-1)
#     elif m.dim() == 3:
#         m = m[:, :, :1]
#     return m.float().to(device)
#
#
# def _xyz_only(pos: torch.Tensor):
#     if pos.dim() == 2: pos = pos.unsqueeze(0)
#     if pos.size(-1) >= 3: return pos[..., :3].contiguous()
#     raise RuntimeError(f"Position tensor last dim must be >= 3, got {pos.shape}")
#
#
# def read_molecule(path):
#     mol = None
#     if path.endswith('.pdb'):
#         mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
#     elif path.endswith('.mol') or path.endswith('.mol2'):
#         mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
#     elif path.endswith('.sdf'):
#         suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=True)
#         mol = suppl[0] if len(suppl) > 0 else None
#     if mol is None: raise ValueError(f"Failed to read molecule from {path}")
#     return mol
#
#
# def bridge_rdkit_to_openfold_tensor(mol, device='cpu'):
#     residues = list(mol.GetResidues())
#     n_res = len(residues)
#     all_atom_positions = np.zeros((n_res, 37, 3))
#     all_atom_mask = np.zeros((n_res, 37))
#     aatype = np.zeros(n_res, dtype=np.int64)
#     for i, res in enumerate(residues):
#         res_name = res.GetResidueName().strip()
#         aatype[i] = rc.restype_order.get(rc.restype_3to1.get(res_name, 'X'), 20)
#         for atom in res.GetAtoms():
#             p_info = atom.GetPDBResidueInfo()
#             if p_info is None: continue
#             atom_name = p_info.GetName().strip()
#             if atom_name in rc.atom_order:
#                 idx = rc.atom_order[atom_name]
#                 pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
#                 all_atom_positions[i, idx] = [pos.x, pos.y, pos.z]
#                 all_atom_mask[i, idx] = 1.0
#     return {
#         "aatype": torch.tensor(aatype, device=device),
#         "all_atom_positions": torch.tensor(all_atom_positions, dtype=torch.float32, device=device),
#         "all_atom_mask": torch.tensor(all_atom_mask, dtype=torch.float32, device=device)
#     }
#
#
# def atom37_to_frames(protein, eps=1e-8):
#     aatype = protein["aatype"]
#     all_atom_positions = protein["all_atom_positions"]
#     all_atom_mask = protein["all_atom_mask"]
#     batch_dims = len(aatype.shape[:-1])
#
#     restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
#     restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
#     restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]
#
#     for restype, restype_letter in enumerate(rc.restypes):
#         resname = rc.restype_1to3[restype_letter]
#         for chi_idx in range(4):
#             if rc.chi_angles_mask[restype][chi_idx]:
#                 names = rc.chi_angles_atoms[resname][chi_idx]
#                 restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]
#
#     restype_rigidgroup_mask = all_atom_mask.new_zeros((*aatype.shape[:-1], 21, 8))
#     restype_rigidgroup_mask[..., 0] = 1
#     restype_rigidgroup_mask[..., 3] = 1
#     restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)
#
#     lookuptable = rc.atom_order.copy()
#     lookuptable[""] = 0
#     lookup = np.vectorize(lambda x: lookuptable[x])
#     restype_rigidgroup_base_atom37_idx = lookup(restype_rigidgroup_base_atom_names)
#     restype_rigidgroup_base_atom37_idx = aatype.new_tensor(restype_rigidgroup_base_atom37_idx)
#     restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(*((1,) * batch_dims),
#                                                                                  *restype_rigidgroup_base_atom37_idx.shape)
#
#     residx_rigidgroup_base_atom37_idx = batched_gather(restype_rigidgroup_base_atom37_idx, aatype, dim=-3,
#                                                        no_batch_dims=batch_dims)
#     base_atom_pos = batched_gather(all_atom_positions, residx_rigidgroup_base_atom37_idx, dim=-2,
#                                    no_batch_dims=len(all_atom_positions.shape[:-2]))
#
#     gt_frames = Rigid.from_3_points(p_neg_x_axis=base_atom_pos[..., 0, :], origin=base_atom_pos[..., 1, :],
#                                     p_xy_plane=base_atom_pos[..., 2, :], eps=eps)
#
#     group_exists = batched_gather(restype_rigidgroup_mask, aatype, dim=-2, no_batch_dims=batch_dims)
#     gt_atoms_exist = batched_gather(all_atom_mask, residx_rigidgroup_base_atom37_idx, dim=-1,
#                                     no_batch_dims=len(all_atom_mask.shape[:-1]))
#     gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists
#
#     rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
#     rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
#     rots[..., 0, 0, 0] = -1
#     rots[..., 0, 2, 2] = -1
#     rots = Rotation(rot_mats=rots)
#     gt_frames = gt_frames.compose(Rigid(rots, None))
#
#     gt_frames_tensor = gt_frames.to_tensor_4x4()
#     protein["rigidgroups_gt_frames"] = gt_frames_tensor
#     protein["rigidgroups_gt_exists"] = gt_exists
#     return protein
#
#
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output, exist_ok=True)
#
#     print(f"🚀 Loading FragDiff model from {args.model}...")
#     ddpm = DDPM.load_from_checkpoint(args.model, map_location=device).eval().to(device)
#     if args.n_steps is not None: ddpm.edm.T = args.n_steps
#     ddpm.reconstruct_and_save = lambda *args, **kwargs: []
#
#     if args.linker_size.isdigit():
#         fixed_size = int(args.linker_size)
#
#         def sample_fn(_data):
#             return torch.full((_data['positions'].shape[0],), fixed_size, device=device, dtype=const.TORCH_INT)
#     else:
#         bounds = [int(x.strip()) for x in args.linker_size.split(',')]
#         left, right = bounds[0], bounds[1]
#
#         def sample_fn(_data):
#             return torch.randint(left, right + 1, (len(_data['positions']),), device=device, dtype=const.TORCH_INT)
#
#     dataset_list = []
#
#     if args.use_dataset:
#         print("📁 Running in DATASET MODE")
#         split_path = args.split_file or os.path.join(args.data_dir, f"{args.dataset_prefix}_split.pt")
#         full_dataset = PDBbindDataset(data_path=args.data_dir, prefix=args.dataset_prefix, device=device)
#         split_data = torch.load(split_path)
#         test_indices = split_data.get('test', []) or split_data.get('val', [])
#         if args.limit > 0: test_indices = test_indices[:args.limit]
#
#         for idx in tqdm(test_indices, desc="Pre-loading Targets"):
#             try:
#                 data_item = full_dataset[idx]
#                 if 'positions' not in data_item: data_item['positions'] = data_item['ligand_pos']
#                 if 'one_hot' not in data_item:
#                     elem_map = {1: 0, 6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
#                     elements = data_item['ligand_element'].view(-1).tolist()
#                     one_hot = torch.zeros((len(elements), 9), dtype=torch.float32)
#                     for i, e in enumerate(elements): one_hot[i, elem_map.get(e, 0)] = 1.0
#                     data_item['one_hot'] = one_hot
#                 if 'fragment_mask' not in data_item:
#                     n_atoms = len(data_item['positions'])
#                     import random
#                     n_linker = random.randint(1, 3)
#                     n_frag = max(1, n_atoms - n_linker)
#                     frag_mask = torch.zeros(n_atoms, dtype=torch.float32)
#                     frag_mask[:n_frag] = 1.0
#                     link_mask = torch.zeros(n_atoms, dtype=torch.float32)
#                     link_mask[n_frag:] = 1.0
#                     data_item['fragment_mask'] = frag_mask
#                     data_item['linker_mask'] = link_mask
#                 if 'anchors' not in data_item: data_item['anchors'] = torch.zeros(len(data_item['positions']),
#                                                                                   dtype=torch.float32)
#                 for _ in range(args.n_samples): dataset_list.append({k: v for k, v in data_item.items()})
#             except Exception:
#                 pass
#     else:
#         print("🧪 Running in SINGLE FILE MODE")
#         pocket_mol = read_molecule(args.pocket)
#         pocket_tensor = bridge_rdkit_to_openfold_tensor(pocket_mol, device=device)
#         pocket_with_frames = atom37_to_frames(pocket_tensor)
#         pocket_frames = pocket_with_frames["rigidgroups_gt_frames"][:, 0, :, :]
#         pocket_mask = pocket_with_frames["rigidgroups_gt_exists"][:, 0]
#
#         molecule = read_molecule(args.fragments)
#         positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
#         n_atoms = len(positions)
#
#         anchor_flags = np.zeros(n_atoms)
#         if args.anchors:
#             for a in args.anchors.split(','): anchor_flags[int(a.strip()) - 1] = 1
#
#         single_data = {
#             'uuid': '0', 'name': os.path.basename(args.fragments).split('.')[0],
#             'positions': torch.tensor(positions, dtype=torch.float32, device=device),
#             'one_hot': torch.tensor(one_hot, dtype=torch.float32, device=device),
#             'charges': torch.tensor(charges, dtype=torch.float32, device=device),
#             'anchors': torch.tensor(anchor_flags, dtype=torch.float32, device=device),
#             'fragment_mask': torch.ones(n_atoms, dtype=torch.float32, device=device),
#             'linker_mask': torch.zeros(n_atoms, dtype=torch.float32, device=device),
#             'num_atoms': n_atoms,
#             'pocket_frames': pocket_frames.to(device),
#             'pocket_mask': pocket_mask.to(device)
#         }
#         dataset_list = [single_data] * args.n_samples
#
#     if not dataset_list: return
#     batch_size = min(len(dataset_list), 16)
#     dataloader = get_dataloader(dataset_list, batch_size=batch_size, collate_fn=collate_with_fragment_edges)
#
#     for batch_i, data in enumerate(tqdm(dataloader, desc="Denoising Samples")):
#         for k, v in data.items():
#             if isinstance(v, torch.Tensor): data[k] = v.to(device)
#
#         bs, n_nodes = data['positions'].shape[0], data['positions'].shape[1]
#         curr_node_mask = _ensure_3d_mask(data.get('atom_mask', None), bs, n_nodes, device)
#         curr_edge_mask = None
#         curr_context = data.get('context', None)
#
#         if 'linker_mask' in data and 'fragment_mask' in data:
#             real_pos_temp = data['positions'].clone()
#             linker_mask_3d = _ensure_3d_mask(data['linker_mask'], bs, n_nodes, device)
#             fragment_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
#
#             raw_noise = torch.randn_like(real_pos_temp)
#             sum_pos = (real_pos_temp * fragment_mask_3d).sum(dim=1, keepdim=True)
#             count_pos = fragment_mask_3d.sum(dim=1, keepdim=True).clamp(min=1.0)
#             center_of_mass = sum_pos / count_pos
#             aligned_noise = raw_noise + center_of_mass
#             data['positions'] = torch.where(linker_mask_3d > 0.5, aligned_noise, real_pos_temp)
#
#         sample_results = None
#         for attempt in range(5):
#             try:
#                 decoder_instance = RigidAssemblyDecoder()
#                 my_guidance_fn = lambda x_t, t, base_pred: svdd_qed_guidance(
#                     x_t, t, base_pred, data, decoder_instance, curr_node_mask, num_samples=3, scale=2.5
#                 )
#                 sample_results = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1, edge_mask=curr_edge_mask,
#                                                    context=curr_context, save_path=None, guidance_fn=my_guidance_fn)
#                 break
#             except FoundNaNException:
#                 continue
#             except Exception as e:
#                 import traceback
#                 print(f"\n❌ 采样过程发生致命错误: {e}")
#                 traceback.print_exc()
#                 break
#
#         if sample_results is None: continue
#
#         chain_data, node_mask = sample_results
#         while isinstance(chain_data, (tuple, list)): chain_data = chain_data[0]
#
#         if chain_data.dim() == 4:
#             current_chain = chain_data[-1]
#         else:
#             current_chain = chain_data
#
#         x_gen = current_chain[..., :3]
#
#         if 'fragment_mask' in data:
#             frag_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
#             real_pos = _xyz_only(data['positions'])
#             x_gen = torch.where(frag_mask_3d > 0.5, real_pos, x_gen)
#
#         final_x = x_gen
#
#         try:
#             rot_flat = current_chain[..., 3:12]
#             rot_gen = rot_flat.view(bs, n_nodes, 3, 3)
#
#             if 'frag_rotations' in data and 'fragment_mask' in data:
#                 frag_mask_3d_rot = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device).unsqueeze(-1)
#                 real_rot = data['frag_rotations']
#                 rot_gen = torch.where(frag_mask_3d_rot > 0.5, real_rot, rot_gen)
#
#             raw_h_gen = current_chain[..., 12:]
#             frag_types_gen = raw_h_gen[..., :len(const.NUM2AA)]
#
#             decoder = RigidAssemblyDecoder()
#             mols_and_coords = decoder.assemble(final_x, rot_gen, frag_types_gen, node_mask)
#
#             raw_names = data.get('name', [f"target_{batch_i}" for _ in range(bs)])
#             if isinstance(raw_names, str): raw_names = [raw_names] * bs
#
#             for j, (mol, coords) in enumerate(mols_and_coords):
#                 if mol is None or coords is None: continue
#                 target_name = raw_names[j]
#                 sample_idx = batch_i * batch_size + j
#                 sdf_path = os.path.join(args.output, f"{target_name}_sample_{sample_idx}.sdf")
#
#                 final_mol = Chem.Mol(mol)
#                 try:
#                     conf = Chem.Conformer(final_mol.GetNumAtoms())
#                     for idx, coord in enumerate(coords):
#                         conf.SetAtomPosition(idx, np.array(coord, dtype=np.float64))
#                     final_mol.RemoveAllConformers()
#                     final_mol.AddConformer(conf)
#                 except Exception as e:
#                     continue
#
#                 writer = Chem.SDWriter(sdf_path)
#                 writer.write(final_mol)
#                 writer.close()
#
#         except Exception as e:
#             print(f"❌ Batch {batch_i} 组装/保存过程发生错误: {e}")
#
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)
import argparse
import csv
import os
import numpy as np
import torch
import copy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from tqdm import tqdm
import sys
from rdkit import RDConfig

# 引入合成可及性打分
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    from src import sascorer
except ImportError:
    import sascorer

from src import const
from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule, PDBbindDataset, CrossDockDataset
from src.edm import ValueNetworkGuidance
from src.lightning import DDPM
from src.utils import FoundNaNException
from src.visualizer import save_xyz_file

from openfold.np import residue_constants as rc
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import batched_gather
from src.visualizer import save_xyz_file
from src.value_net import ValueNetwork
from src.generation_utils import (
    FragmentCountSampler,
    attachment_constrained_tree_plans,
    rotate_points_about_axis,
    rotation_matrix_align_vectors,
)
from src.pocket_pose_value_net import PocketPoseValueGuidance, PocketPoseValueNetwork
from src.pocket_geometry import vdw_radii_from_atomic_numbers

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--output', type=str, default='./results')
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--n_steps', type=int, default=None)
parser.add_argument('--linker_size', type=str, default='5')
parser.add_argument('--fragment_count_mode', choices=['empirical', 'fixed', 'range'], default='empirical')
parser.add_argument('--fragment_count_hist', type=str, default=None)
parser.add_argument('--fragment_count_max', type=int, default=10)
parser.add_argument('--fragment_count_seed', type=int, default=2025)
parser.add_argument('--decoder_attempts', type=int, default=10)
parser.add_argument('--decoder_tree_beam_width', type=int, default=128)
parser.add_argument('--decoder_tree_plans', type=int, default=8)
parser.add_argument('--clash_guidance_scale', type=float, default=0.0)
parser.add_argument('--clash_guidance_start', type=int, default=0)
parser.add_argument('--clash_guidance_end', type=int, default=100)
parser.add_argument('--clash_guidance_max_norm', type=float, default=0.2)
parser.add_argument('--clash_guidance_global_fraction', type=float, default=0.25)
parser.add_argument('--clash_guidance_max_com_shift', type=float, default=1.0)
parser.add_argument('--pocket_value_checkpoint', type=str, default=None)
parser.add_argument('--pocket_value_scale', type=float, default=0.0)
parser.add_argument('--pocket_value_start', type=int, default=0)
parser.add_argument('--pocket_value_end', type=int, default=100)
parser.add_argument('--use_dataset', action='store_true')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset_prefix', type=str, default='PDBbind_pocket10')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--fragment_vocab_path', type=str, default=None)
parser.add_argument('--strict_targets', action='store_true')
parser.add_argument('--dataset_split', choices=['train', 'val', 'test'], default='test')
parser.add_argument('--limit', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--fragments', type=str, default=None)
parser.add_argument('--pocket', type=str, default=None)
parser.add_argument('--anchors', type=str, default=None)
parser.add_argument('--guidance_scale', type=float, default=0.4)
parser.add_argument('--guidance_start', type=int, default=50)
parser.add_argument('--guidance_end', type=int, default=700)
parser.add_argument('--temperature', type=float, default=1.2)
parser.add_argument('--top_k', type=int, default=None)
parser.add_argument('--decoder_type_temperature', type=float, default=0.7)
parser.add_argument('--oversample_factor', type=int, default=8)
parser.add_argument('--diversity_threshold', type=float, default=0.75)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--save_latent_features', action='store_true')


def get_fc_edges(node_mask):
    """为 ValueNetwork 构建全连接边索引"""
    B, N, _ = node_mask.shape
    device = node_mask.device
    rows, cols = [], []
    for b in range(B):
        valid_idx = (node_mask[b, :, 0] > 0.5).nonzero().view(-1)
        if len(valid_idx) == 0: continue
        r = valid_idx.repeat_interleave(len(valid_idx))
        c = valid_idx.repeat(len(valid_idx))
        rows.append(r + b * N)
        cols.append(c + b * N)
    if len(rows) == 0:
        return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(rows), torch.cat(cols)


def value_network_guidance(x_t, t, base_eps, rot, h, node_mask, value_net, scale=2.5, max_norm=1.0):
    """纯正解析梯度引导：带数学防火墙与 L2 范数方向保持"""
    with torch.enable_grad():
        x_in = x_t.detach().clone().requires_grad_(True)
        edge_index = get_fc_edges(node_mask)

        # 1. 评估未来得分
        expected_score = value_net(x_in, rot, h, t, node_mask, edge_index)

        # 2. 安全求导：allow_unused=True 防止孤立节点导致计算图崩溃
        grad_x = torch.autograd.grad(outputs=expected_score.sum(), inputs=x_in, allow_unused=True)[0]

    # 3. 兜底保护：如果整个图断开返回 None，则给零梯度
    if grad_x is None:
        grad_x = torch.zeros_like(x_in)

    # 4. 张量净化：数学未定义值置零
    grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. 高阶优化：放弃 Clamp，采用 L2 范数缩放 (保持空间方向向量不变)
    # 计算每个原子的梯度向量长度
    norm = torch.linalg.norm(grad_x, dim=-1, keepdim=True)
    # 防止除以 0 的安全缩放系数
    scale_factor = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    grad_x = grad_x * scale_factor

    # 6. 计算引导后的噪声并再次净化
    guided_eps = base_eps - scale * grad_x
    guided_eps = torch.nan_to_num(guided_eps, nan=0.0, posinf=0.0, neginf=0.0)

    return guided_eps * node_mask


def calculate_composite_metrics(mol):
    try:
        qed = float(QED.qed(mol))
        sa = float(sascorer.calculateScore(mol))
        logp = float(Descriptors.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        return None

    sa_norm = max(0.0, min(1.0, 1.0 - (sa - 2.5) / 3.5))

    if 1.0 <= logp <= 3.0:
        logp_score = 1.0
    elif logp < 1.0:
        logp_score = float(np.exp(-2.0 * (1.0 - logp) ** 2))
    else:
        logp_score = float(np.exp(-2.0 * (logp - 3.0) ** 2))

    if 300.0 <= mw <= 500.0:
        mw_score = 1.0
    elif mw < 300.0:
        mw_score = float(np.exp(-0.0005 * (300.0 - mw) ** 2))
    else:
        mw_score = float(np.exp(-0.0005 * (mw - 500.0) ** 2))

    composite_score = 0.50 * qed + 0.35 * sa_norm + 0.10 * logp_score + 0.05 * mw_score
    return {
        "qed": qed,
        "sa": sa,
        "logp": logp,
        "mw": mw,
        "sa_norm": sa_norm,
        "logp_score": logp_score,
        "mw_score": mw_score,
        "composite_score": float(composite_score),
        "passes_property_filter": bool(1.0 <= logp <= 3.0 and 300.0 <= mw <= 500.0),
    }


def safe_mol_to_smiles(mol):
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return ""


def mol_fingerprint(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def sanitize_filename(name):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))
    return safe or "sample"


def write_sdf_file(mol, sdf_path):
    os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
    writer = None
    try:
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        return True
    except Exception as exc:
        print(f"Failed to write SDF {sdf_path}: {exc}")
        return False
    finally:
        if writer is not None:
            writer.close()


def select_diverse_candidates(candidates, max_outputs, diversity_threshold):
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            c["metrics"]["passes_property_filter"],
            c["metrics"]["composite_score"],
            c["metrics"]["qed"],
            -c["metrics"]["sa"],
        ),
        reverse=True,
    )

    best_by_smiles = {}
    for cand in sorted_candidates:
        smiles = cand["smiles"]
        if not smiles:
            cand["duplicate_status"] = "invalid_smiles"
            continue
        if smiles not in best_by_smiles:
            best_by_smiles[smiles] = cand
            cand["duplicate_status"] = "unique"
        else:
            cand["duplicate_status"] = "duplicate_smiles"

    unique_candidates = sorted(
        best_by_smiles.values(),
        key=lambda c: (
            c["metrics"]["passes_property_filter"],
            c["metrics"]["composite_score"],
            c["metrics"]["qed"],
            -c["metrics"]["sa"],
        ),
        reverse=True,
    )

    selected = []
    selected_fps = []
    for cand in unique_candidates:
        if len(selected) >= max_outputs:
            break
        fp = cand.get("fingerprint")
        if fp is None:
            continue
        if all(DataStructs.TanimotoSimilarity(fp, old_fp) < diversity_threshold for old_fp in selected_fps):
            selected.append(cand)
            selected_fps.append(fp)

    selected_ids = {id(cand) for cand in selected}
    for cand in candidates:
        cand["diversity_selected"] = id(cand) in selected_ids
    return selected


def write_metrics_csv(candidates, output_dir):
    metrics_path = os.path.join(output_dir, "metrics.csv")
    fieldnames = [
        "sdf_path", "target_name", "source_sample_idx", "smiles", "qed", "sa", "logp", "mw",
        "composite_score", "passes_property_filter", "duplicate_status", "diversity_selected",
        "selected_sdf_path", "lmdb_key", "dataset_target_index", "generation_index_within_target",
        "fragment_count", "assembly_success", "assembly_error", "connected_components", "mmff_rmsd",
        "reference_center_distance", "min_protein_ligand_distance",
        "clash_count_0.8", "clash_count_1.2", "clash_count_1.5",
        "vdw_clash_count", "vdw_overlap_max", "vdw_overlap_mean",
        "minimum_relative_distance", "posebusters_pass", "out_of_box", "latent_path",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            metrics = cand["metrics"]
            writer.writerow({
                "sdf_path": cand.get("sdf_path", ""),
                "target_name": cand.get("target_name", ""),
                "source_sample_idx": cand.get("source_sample_idx", ""),
                "smiles": cand.get("smiles", ""),
                "qed": metrics.get("qed", ""),
                "sa": metrics.get("sa", ""),
                "logp": metrics.get("logp", ""),
                "mw": metrics.get("mw", ""),
                "composite_score": metrics.get("composite_score", ""),
                "passes_property_filter": metrics.get("passes_property_filter", False),
                "duplicate_status": cand.get("duplicate_status", ""),
                "diversity_selected": cand.get("diversity_selected", False),
                "selected_sdf_path": cand.get("selected_sdf_path", ""),
                "lmdb_key": cand.get("lmdb_key", ""),
                "dataset_target_index": cand.get("dataset_target_index", ""),
                "generation_index_within_target": cand.get("generation_index_within_target", ""),
                "fragment_count": cand.get("fragment_count", ""),
                "assembly_success": cand.get("assembly_success", False),
                "assembly_error": cand.get("assembly_error", ""),
                "connected_components": cand.get("connected_components", ""),
                "mmff_rmsd": cand.get("mmff_rmsd", ""),
                "reference_center_distance": cand.get("reference_center_distance", ""),
                "min_protein_ligand_distance": cand.get("min_protein_ligand_distance", ""),
                "clash_count_0.8": cand.get("clash_count_0.8", ""),
                "clash_count_1.2": cand.get("clash_count_1.2", ""),
                "clash_count_1.5": cand.get("clash_count_1.5", ""),
                "vdw_clash_count": cand.get("vdw_clash_count", ""),
                "vdw_overlap_max": cand.get("vdw_overlap_max", ""),
                "vdw_overlap_mean": cand.get("vdw_overlap_mean", ""),
                "minimum_relative_distance": cand.get("minimum_relative_distance", ""),
                "posebusters_pass": cand.get("posebusters_pass", "not_run"),
                "out_of_box": cand.get("out_of_box", ""),
                "latent_path": cand.get("latent_path", ""),
            })
    return metrics_path


def write_generation_attempts_csv(records, output_dir):
    path = os.path.join(output_dir, "generation_attempts.csv")
    fieldnames = [
        "target_name", "lmdb_key", "dataset_target_index", "generation_index_within_target",
        "fragment_count", "assembly_success", "assembly_error", "sdf_path",
        "connected_components", "mmff_rmsd", "reference_center_distance",
        "min_protein_ligand_distance", "clash_count_0.8", "clash_count_1.2",
        "clash_count_1.5", "vdw_clash_count", "vdw_overlap_max",
        "vdw_overlap_mean", "minimum_relative_distance", "posebusters_pass",
        "out_of_box", "latent_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return path


def vdw_overlap_metrics(mol, coords, pocket_coords, pocket_elements, tolerance=0.4):
    coords = np.asarray(coords, dtype=np.float64)
    pocket_coords = np.asarray(pocket_coords, dtype=np.float64)
    result = {
        "vdw_clash_count": 0,
        "vdw_overlap_max": 0.0,
        "vdw_overlap_mean": 0.0,
        "minimum_relative_distance": "",
    }
    if coords.size == 0 or pocket_coords.size == 0:
        return result

    ligand_elements = torch.as_tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
    )
    pocket_elements = torch.as_tensor(pocket_elements, dtype=torch.long)
    ligand_radii = vdw_radii_from_atomic_numbers(
        ligand_elements,
        dtype=torch.float64,
    ).cpu().numpy()
    pocket_radii = vdw_radii_from_atomic_numbers(
        pocket_elements,
        dtype=torch.float64,
    ).cpu().numpy()
    distances = np.linalg.norm(
        coords[:, None, :] - pocket_coords[None, :, :],
        axis=-1,
    )
    sum_radii = ligand_radii[:, None] + pocket_radii[None, :]
    overlap = np.maximum(sum_radii - float(tolerance) - distances, 0.0)
    positive = overlap[overlap > 0]
    result["vdw_clash_count"] = int(positive.size)
    result["vdw_overlap_max"] = float(positive.max()) if positive.size else 0.0
    result["vdw_overlap_mean"] = float(positive.mean()) if positive.size else 0.0
    result["minimum_relative_distance"] = float(
        np.min(distances / np.maximum(sum_radii, 1e-8))
    )
    return result


def pocket_geometry_metrics(mol, coords, batch, batch_index):
    coords = np.asarray(coords, dtype=np.float64)
    result = {
        "reference_center_distance": "",
        "min_protein_ligand_distance": "",
        "clash_count_0.8": 0,
        "clash_count_1.2": 0,
        "clash_count_1.5": 0,
        "vdw_clash_count": 0,
        "vdw_overlap_max": 0.0,
        "vdw_overlap_mean": 0.0,
        "minimum_relative_distance": "",
        "posebusters_pass": "not_run",
        "out_of_box": False,
    }
    if coords.size == 0:
        return result

    if 'reference_box_center' in batch and 'reference_box_size' in batch:
        center = batch['reference_box_center'][batch_index].detach().cpu().numpy().astype(np.float64)
        size = batch['reference_box_size'][batch_index].detach().cpu().numpy().astype(np.float64)
        masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=np.float64)
        ligand_center = np.average(coords, axis=0, weights=masses)
        result["reference_center_distance"] = float(np.linalg.norm(ligand_center - center))
        result["out_of_box"] = bool(np.any(np.abs(coords - center) > size / 2.0))

    if 'pocket_frames' in batch:
        pocket = batch['pocket_frames'][batch_index, :, :3, 3]
        if 'pocket_mask' in batch:
            pocket = pocket[batch['pocket_mask'][batch_index] > 0.5]
        pocket = pocket.detach().cpu().numpy().astype(np.float64)
        if pocket.size > 0:
            distances = np.linalg.norm(coords[:, None, :] - pocket[None, :, :], axis=-1)
            result["min_protein_ligand_distance"] = float(distances.min())
            for threshold in (0.8, 1.2, 1.5):
                result[f"clash_count_{threshold:.1f}"] = int((distances < threshold).sum())
            if 'pocket_element' not in batch:
                raise KeyError("pocket_element is required for vdW diagnostics")
            pocket_elements = batch['pocket_element'][batch_index]
            if 'pocket_mask' in batch:
                pocket_elements = pocket_elements[batch['pocket_mask'][batch_index] > 0.5]
            result.update(
                vdw_overlap_metrics(
                    mol,
                    coords,
                    pocket,
                    pocket_elements.detach().cpu().numpy(),
                )
            )
    return result



class RigidAssemblyDecoder:
    def __init__(
        self,
        decoder_type_temperature=0.7,
        decoder_attempts=10,
        tree_beam_width=128,
        tree_plans=8,
    ):
        self.num2aa = const.NUM2AA
        self.ideal_coords = const.IDEAL_COORDS
        self.aa_bonds = const.AA_BONDS
        self.decoder_type_temperature = decoder_type_temperature
        self.decoder_attempts = max(1, int(decoder_attempts))
        self.tree_beam_width = max(1, int(tree_beam_width))
        self.tree_plans = max(1, int(tree_plans))
        self.frag_bond_types = getattr(const, "FRAGMENT_BOND_TYPES", [])
        self.frag_attachment_points = getattr(const, "FRAGMENT_ATTACHMENT_POINTS", [])
        self.frag_atom_symbols = getattr(const, "FRAGMENT_ATOM_SYMBOLS", [])
        self.frag_bond_cache = {}  # 缓存片段的真实化学键类型

    @staticmethod
    def _bond_order_value(bond_type):
        if bond_type == Chem.rdchem.BondType.SINGLE:
            return 1.0
        if bond_type == Chem.rdchem.BondType.DOUBLE:
            return 2.0
        if bond_type == Chem.rdchem.BondType.TRIPLE:
            return 3.0
        if bond_type == Chem.rdchem.BondType.AROMATIC:
            return 1.5
        return 0.0

    @staticmethod
    def _bond_label_to_rdkit(bond_label):
        bond_type_map = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        return bond_type_map.get(int(bond_label))

    def _template_atom_symbol(self, frag_type, atom_idx, atom_name):
        if frag_type < len(self.frag_atom_symbols) and atom_idx < len(self.frag_atom_symbols[frag_type]):
            return self.frag_atom_symbols[frag_type][atom_idx]

        elem_symbol = "".join([c for c in atom_name if c.isalpha()])
        if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
            elem_symbol = elem_symbol[0]
        return elem_symbol.capitalize()

    def _template_bond_type(self, frag_type, bond_idx):
        if frag_type < len(self.frag_bond_types) and bond_idx < len(self.frag_bond_types[frag_type]):
            return self._bond_label_to_rdkit(self.frag_bond_types[frag_type][bond_idx]) or Chem.rdchem.BondType.SINGLE
        return Chem.rdchem.BondType.SINGLE

    def _attachment_template_indices(self, frag_type):
        if frag_type < len(self.frag_attachment_points) and self.frag_attachment_points[frag_type]:
            return list(self.frag_attachment_points[frag_type])

        name_to_idx = {
            atom_info[0].strip(): idx
            for idx, atom_info in enumerate(self.ideal_coords[frag_type])
        }
        h_bearing_heavy = set()
        for start_name, end_name in self.aa_bonds[frag_type]:
            start_name = start_name.strip()
            end_name = end_name.strip()
            start_idx = name_to_idx.get(start_name)
            end_idx = name_to_idx.get(end_name)
            if start_idx is None or end_idx is None:
                continue
            start_symbol = self._template_atom_symbol(frag_type, start_idx, start_name)
            end_symbol = self._template_atom_symbol(frag_type, end_idx, end_name)
            if start_symbol.upper() == "H" and end_symbol.upper() != "H":
                h_bearing_heavy.add(end_idx)
            elif end_symbol.upper() == "H" and start_symbol.upper() != "H":
                h_bearing_heavy.add(start_idx)

        if not h_bearing_heavy:
            h_bearing_heavy = {
                idx
                for idx, atom_info in enumerate(self.ideal_coords[frag_type])
                if self._template_atom_symbol(frag_type, idx, atom_info[0].strip()).upper() != "H"
            }

        def rank_idx(idx):
            symbol = self._template_atom_symbol(frag_type, idx, self.ideal_coords[frag_type][idx][0].strip())
            priority = {"C": 0, "S": 1, "P": 1, "N": 2, "O": 3}
            return (priority.get(symbol, 4), idx)

        return sorted(h_bearing_heavy, key=rank_idx)

    def _atom_valence_room(self, atom, new_bond_type):
        max_valence = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}
        limit = max_valence.get(atom.GetSymbol(), 4)
        current_order = sum(self._bond_order_value(bond.GetBondType()) for bond in atom.GetBonds())
        return limit - current_order - self._bond_order_value(new_bond_type)

    def _can_add_external_bond(self, mol, idx_i, idx_j, bond_type, used_attachment_atoms=None):
        if idx_i == idx_j:
            return False
        if used_attachment_atoms is not None:
            if idx_i in used_attachment_atoms or idx_j in used_attachment_atoms:
                return False
        if mol.GetBondBetweenAtoms(idx_i, idx_j) is not None:
            return False
        atom_i = mol.GetAtomWithIdx(idx_i)
        atom_j = mol.GetAtomWithIdx(idx_j)
        if atom_i.GetSymbol() == "H" or atom_j.GetSymbol() == "H":
            return False
        if self._atom_valence_room(atom_i, bond_type) < -1e-6:
            return False
        if self._atom_valence_room(atom_j, bond_type) < -1e-6:
            return False
        return True

    @staticmethod
    def _ideal_bond_length(atom_i, atom_j, bond_type):
        periodic_table = Chem.GetPeriodicTable()
        length = (
            float(periodic_table.GetRcovalent(atom_i.GetAtomicNum()))
            + float(periodic_table.GetRcovalent(atom_j.GetAtomicNum()))
        )
        order_scale = {
            Chem.rdchem.BondType.SINGLE: 1.00,
            Chem.rdchem.BondType.DOUBLE: 0.92,
            Chem.rdchem.BondType.TRIPLE: 0.87,
            Chem.rdchem.BondType.AROMATIC: 0.95,
        }
        return float(np.clip(length * order_scale.get(bond_type, 1.0), 1.05, 2.05))

    @staticmethod
    def _conformer_coords(conf):
        return np.asarray(
            [np.asarray(conf.GetAtomPosition(idx), dtype=np.float64) for idx in range(conf.GetNumAtoms())],
            dtype=np.float64,
        )

    @staticmethod
    def _set_conformer_coords(conf, coords, atom_indices=None):
        indices = range(len(coords)) if atom_indices is None else atom_indices
        for atom_idx in indices:
            conf.SetAtomPosition(int(atom_idx), np.asarray(coords[int(atom_idx)], dtype=float))

    @staticmethod
    def _component_geometry_rank(
        mol,
        coords,
        stationary_atoms,
        moving_atoms,
        excluded_pairs,
        coarse_coords,
        pocket_pos=None,
        pocket_elements=None,
        tolerance=0.4,
    ):
        periodic_table = Chem.GetPeriodicTable()
        stationary_atoms = list(stationary_atoms)
        moving_atoms = list(moving_atoms)
        internal_overlaps = []
        excluded_pairs = {tuple(sorted(pair)) for pair in excluded_pairs}
        for idx_i in stationary_atoms:
            radius_i = float(periodic_table.GetRvdw(mol.GetAtomWithIdx(int(idx_i)).GetAtomicNum()))
            for idx_j in moving_atoms:
                if tuple(sorted((int(idx_i), int(idx_j)))) in excluded_pairs:
                    continue
                radius_j = float(periodic_table.GetRvdw(mol.GetAtomWithIdx(int(idx_j)).GetAtomicNum()))
                distance = float(np.linalg.norm(coords[int(idx_i)] - coords[int(idx_j)]))
                overlap = radius_i + radius_j - float(tolerance) - distance
                if overlap > 0.0:
                    internal_overlaps.append(overlap)

        pocket_overlaps = []
        if pocket_pos is not None and pocket_elements is not None and len(pocket_pos) > 0:
            pocket_pos = np.asarray(pocket_pos, dtype=np.float64)
            pocket_elements = np.asarray(pocket_elements, dtype=np.int64)
            pocket_radii = np.asarray(
                [
                    float(periodic_table.GetRvdw(int(z) if 1 <= int(z) <= 118 else 6))
                    for z in pocket_elements
                ],
                dtype=np.float64,
            )
            for atom_idx in range(mol.GetNumAtoms()):
                ligand_radius = float(
                    periodic_table.GetRvdw(mol.GetAtomWithIdx(int(atom_idx)).GetAtomicNum())
                )
                distances = np.linalg.norm(pocket_pos - coords[int(atom_idx)], axis=1)
                overlaps = ligand_radius + pocket_radii - float(tolerance) - distances
                pocket_overlaps.extend(overlaps[overlaps > 0.0].tolist())

        internal_max = max(internal_overlaps, default=0.0)
        pocket_max = max(pocket_overlaps, default=0.0)
        coarse_rmsd = float(
            np.sqrt(np.mean(np.sum((coords[moving_atoms] - coarse_coords[moving_atoms]) ** 2, axis=1)))
        )
        return (
            int(len(internal_overlaps) == 0),
            int(len(pocket_overlaps) == 0),
            -float(internal_max),
            -float(pocket_max),
            -int(len(internal_overlaps)),
            -int(len(pocket_overlaps)),
            -coarse_rmsd,
        )

    def _align_component_for_bond(
        self,
        mol,
        conf,
        stationary_atoms,
        moving_atoms,
        parent_fragment_atoms,
        child_fragment_atoms,
        parent_attachment,
        child_attachment,
        bond_type,
        coarse_coords,
        pocket_pos=None,
        pocket_elements=None,
    ):
        stationary_atoms = list(stationary_atoms)
        moving_atoms = list(moving_atoms)
        if not stationary_atoms or not moving_atoms:
            return False

        coords = self._conformer_coords(conf)
        parent_center = coords[list(parent_fragment_atoms)].mean(axis=0)
        child_center = coords[list(child_fragment_atoms)].mean(axis=0)
        parent_position = coords[int(parent_attachment)]
        child_position = coords[int(child_attachment)]

        outward = parent_position - parent_center
        if np.linalg.norm(outward) < 1e-8:
            outward = child_center - parent_center
        if np.linalg.norm(outward) < 1e-8:
            outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        outward /= np.linalg.norm(outward)

        child_vector = child_position - child_center
        if np.linalg.norm(child_vector) < 1e-8:
            child_vector = -outward
        alignment = rotation_matrix_align_vectors(child_vector, -outward)
        moving_points = coords[moving_atoms]
        aligned_points = (moving_points - child_position) @ alignment.T + child_position

        target_length = self._ideal_bond_length(
            mol.GetAtomWithIdx(int(parent_attachment)),
            mol.GetAtomWithIdx(int(child_attachment)),
            bond_type,
        )
        target_child_position = parent_position + outward * target_length
        aligned_points += target_child_position - child_position

        excluded_pairs = {(int(parent_attachment), int(child_attachment))}
        excluded_pairs.update(
            (neighbor.GetIdx(), int(child_attachment))
            for neighbor in mol.GetAtomWithIdx(int(parent_attachment)).GetNeighbors()
        )
        excluded_pairs.update(
            (int(parent_attachment), neighbor.GetIdx())
            for neighbor in mol.GetAtomWithIdx(int(child_attachment)).GetNeighbors()
        )

        best_rank = None
        best_coords = None
        for angle_degrees in range(0, 360, 30):
            candidate = coords.copy()
            candidate[moving_atoms] = rotate_points_about_axis(
                aligned_points,
                target_child_position,
                outward,
                np.deg2rad(angle_degrees),
            )
            candidate += coarse_coords.mean(axis=0) - candidate.mean(axis=0)
            rank = self._component_geometry_rank(
                mol,
                candidate,
                stationary_atoms,
                moving_atoms,
                excluded_pairs,
                coarse_coords,
                pocket_pos=pocket_pos,
                pocket_elements=pocket_elements,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_coords = candidate

        if best_coords is None:
            return False
        self._set_conformer_coords(conf, best_coords)
        return True

    def _pick_attachment_pair(
        self,
        mol,
        conf,
        frag_attachment_atoms,
        frag_i,
        frag_j,
        bond_label,
        pred_attach_src=None,
        pred_attach_dst=None,
        used_attachment_atoms=None,
        enforce_distance=False,
    ):
        bond_type = self._bond_label_to_rdkit(bond_label)
        if bond_type is None:
            return None

        frag_i_atoms = frag_attachment_atoms.get(frag_i, [])
        frag_j_atoms = frag_attachment_atoms.get(frag_j, [])
        if not frag_i_atoms or not frag_j_atoms:
            return None

        if pred_attach_src is not None and pred_attach_dst is not None:
            if 0 <= pred_attach_src < len(frag_i_atoms) and 0 <= pred_attach_dst < len(frag_j_atoms):
                idx_i = frag_i_atoms[pred_attach_src]
                idx_j = frag_j_atoms[pred_attach_dst]
                if not self._can_add_external_bond(mol, idx_i, idx_j, bond_type, used_attachment_atoms):
                    pass
                else:
                    p1 = np.array(conf.GetAtomPosition(idx_i))
                    p2 = np.array(conf.GetAtomPosition(idx_j))
                    dist = np.linalg.norm(p1 - p2)
                    if not enforce_distance or 0.9 <= dist <= 2.1:
                        return idx_i, idx_j, bond_type
            # If the predicted attachment indices are invalid, fall back to
            # the nearest unused valid attachment pair for this fragment pair.

        best_pair = None
        best_score = float("inf")

        for idx_i in frag_i_atoms:
            for idx_j in frag_j_atoms:
                if not self._can_add_external_bond(mol, idx_i, idx_j, bond_type, used_attachment_atoms):
                    continue
                p1 = np.array(conf.GetAtomPosition(idx_i))
                p2 = np.array(conf.GetAtomPosition(idx_j))
                dist = np.linalg.norm(p1 - p2)
                ideal_length = self._ideal_bond_length(
                    mol.GetAtomWithIdx(idx_i),
                    mol.GetAtomWithIdx(idx_j),
                    bond_type,
                )
                score = abs(dist - ideal_length)
                if score < best_score:
                    best_score = score
                    best_pair = (idx_i, idx_j, bond_type, dist)

        if best_pair is None:
            return None

        idx_i, idx_j, bond_type, dist = best_pair
        if enforce_distance and not 0.9 <= dist <= 2.1:
            return None
        return idx_i, idx_j, bond_type

    def _attachment_pair_candidates(
        self,
        mol,
        conf,
        frag_attachment_atoms,
        frag_i,
        frag_j,
        bond_label,
        pred_attach_src=None,
        pred_attach_dst=None,
        limit=8,
    ):
        """Enumerate deterministic chemically feasible attachment assignments."""
        bond_type = self._bond_label_to_rdkit(bond_label)
        if bond_type is None:
            return []
        frag_i_atoms = frag_attachment_atoms.get(frag_i, [])
        frag_j_atoms = frag_attachment_atoms.get(frag_j, [])
        predicted_pair = None
        if pred_attach_src is not None and pred_attach_dst is not None:
            if 0 <= pred_attach_src < len(frag_i_atoms) and 0 <= pred_attach_dst < len(frag_j_atoms):
                predicted_pair = (
                    int(frag_i_atoms[pred_attach_src]),
                    int(frag_j_atoms[pred_attach_dst]),
                )

        candidates = []
        for idx_i in frag_i_atoms:
            for idx_j in frag_j_atoms:
                if not self._can_add_external_bond(mol, idx_i, idx_j, bond_type):
                    continue
                p1 = np.asarray(conf.GetAtomPosition(idx_i), dtype=np.float64)
                p2 = np.asarray(conf.GetAtomPosition(idx_j), dtype=np.float64)
                distance = float(np.linalg.norm(p1 - p2))
                ideal = self._ideal_bond_length(
                    mol.GetAtomWithIdx(idx_i),
                    mol.GetAtomWithIdx(idx_j),
                    bond_type,
                )
                distance_error = abs(distance - ideal)
                predicted_bonus = 0.10 if predicted_pair == (idx_i, idx_j) else 0.0
                candidates.append({
                    "atom_i": int(idx_i),
                    "atom_j": int(idx_j),
                    "bond_label": int(bond_label),
                    "bond_type": bond_type,
                    "distance_error": float(distance_error),
                    "score": float(predicted_bonus - 0.05 * distance_error),
                })
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                int(item["atom_i"]),
                int(item["atom_j"]),
            )
        )
        return candidates[:max(1, int(limit))]

    def _apply_predicted_bonds(
        self,
        rw_mol,
        conf,
        frag_attachment_atoms,
        frag_count,
        final_bonds_b,
        final_bond_probs_b=None,
        final_attach_src_b=None,
        final_attach_dst_b=None,
        active_frag_mask=None,
        frag_atom_indices=None,
        atom_to_frag=None,
        pocket_pos=None,
        pocket_elements=None,
    ):
        if frag_atom_indices is None or atom_to_frag is None:
            raise ValueError("Fragment atom mappings are required for rigid constrained assembly")
        if frag_count <= 1:
            return True
        used_attachment_atoms = set()
        added_pairs = set()
        coarse_coords = self._conformer_coords(conf)
        edge_options = []
        ranked_edges = []
        for frag_i in range(frag_count):
            for frag_j in range(frag_i + 1, frag_count):
                if active_frag_mask is not None and (
                    not active_frag_mask[frag_i] or not active_frag_mask[frag_j]
                ):
                    continue
                if final_bond_probs_b is not None:
                    pair_probs = 0.5 * (
                        final_bond_probs_b[frag_i, frag_j]
                        + final_bond_probs_b[frag_j, frag_i]
                    )
                    positive_probs = np.asarray(pair_probs[1:3], dtype=np.float64)
                    edge_probability = float(max(0.0, positive_probs.sum()))
                    label_probabilities = [(1, float(positive_probs[0])), (2, float(positive_probs[1]))]
                else:
                    bond_label = int(final_bonds_b[frag_i, frag_j])
                    edge_probability = 1.0 if bond_label > 0 else 0.0
                    if bond_label <= 0:
                        bond_label = 1
                    label_probabilities = [(bond_label, max(edge_probability, 1e-8))]

                attach_src = attach_dst = None
                if final_attach_src_b is not None and final_attach_dst_b is not None:
                    attach_src = int(final_attach_src_b[frag_i, frag_j]) - 1
                    attach_dst = int(final_attach_dst_b[frag_i, frag_j]) - 1

                attachment_choices = []
                for candidate_label, label_probability in sorted(
                    label_probabilities, key=lambda item: (-item[1], item[0])
                ):
                    choices = self._attachment_pair_candidates(
                        rw_mol,
                        conf,
                        frag_attachment_atoms,
                        frag_i,
                        frag_j,
                        candidate_label,
                        pred_attach_src=attach_src,
                        pred_attach_dst=attach_dst,
                        limit=max(8, frag_count * 2),
                    )
                    conditional_probability = label_probability / max(edge_probability, 1e-12)
                    for choice in choices:
                        choice = dict(choice)
                        choice["score"] += float(np.log(max(conditional_probability, 1e-12)))
                        attachment_choices.append(choice)
                if not attachment_choices:
                    continue

                attachment_choices.sort(
                    key=lambda item: (
                        -float(item["score"]),
                        int(item["atom_i"]),
                        int(item["atom_j"]),
                        int(item["bond_label"]),
                    )
                )
                edge_options.append({
                    "frag_i": frag_i,
                    "frag_j": frag_j,
                    "edge_probability": max(edge_probability, 1e-12),
                    "choices": attachment_choices[:max(12, frag_count * 3)],
                })

                primary = attachment_choices[0]
                rank_score = np.log(max(edge_probability, 1e-12)) + float(primary["score"])
                ranked_edges.append(
                    (
                        rank_score,
                        edge_probability,
                        frag_i,
                        frag_j,
                        int(primary["bond_label"]),
                        attach_src,
                        attach_dst,
                    )
                )

        ranked_edges.sort(key=lambda item: item[0], reverse=True)
        tree_plans = attachment_constrained_tree_plans(
            frag_count,
            edge_options,
            beam_width=self.tree_beam_width,
            max_plans=self.tree_plans,
        )
        if not tree_plans:
            return False

        # Plans are grown from one root, so every step attaches one new rigid
        # fragment to the already assembled component.
        tree_plan = tree_plans[0]
        assembled_fragments = {int(tree_plan[0]["parent_frag"])}
        for edge in tree_plan:
            parent_frag = int(edge["parent_frag"])
            child_frag = int(edge["child_frag"])
            idx_i = int(edge["parent_atom"])
            idx_j = int(edge["child_atom"])
            bond_type = edge["bond_type"]
            if parent_frag not in assembled_fragments or child_frag in assembled_fragments:
                return False
            stationary_atoms = [
                atom_idx
                for atom_idx, fragment_idx in atom_to_frag.items()
                if fragment_idx in assembled_fragments
            ]
            if not self._align_component_for_bond(
                rw_mol,
                conf,
                stationary_atoms,
                frag_atom_indices[child_frag],
                frag_atom_indices[parent_frag],
                frag_atom_indices[child_frag],
                idx_i,
                idx_j,
                bond_type,
                coarse_coords,
                pocket_pos=pocket_pos,
                pocket_elements=pocket_elements,
            ):
                return False
            rw_mol.AddBond(idx_i, idx_j, bond_type)
            used_attachment_atoms.update((idx_i, idx_j))
            added_pairs.add(tuple(sorted((parent_frag, child_frag))))
            assembled_fragments.add(child_frag)
        if len(assembled_fragments) != frag_count:
            return False

        # Add only high-confidence extra edges after connectivity is secured.
        for edge in ranked_edges:
            edge_probability, frag_i, frag_j = edge[1], edge[2], edge[3]
            if edge_probability < 0.7 or tuple(sorted((frag_i, frag_j))) in added_pairs:
                continue
            _, _, _, _, bond_label, attach_src, attach_dst = edge
            candidate = self._pick_attachment_pair(
                rw_mol,
                conf,
                frag_attachment_atoms,
                frag_i,
                frag_j,
                bond_label,
                pred_attach_src=attach_src,
                pred_attach_dst=attach_dst,
                used_attachment_atoms=used_attachment_atoms,
                enforce_distance=True,
            )
            if candidate is None:
                continue
            idx_i, idx_j, bond_type = candidate
            rw_mol.AddBond(idx_i, idx_j, bond_type)
            used_attachment_atoms.update((idx_i, idx_j))
            added_pairs.add((frag_i, frag_j))
        return True

    def get_template_bond_type(self, f_type, start_name, end_name):
        """
        ：模板基因印刻。
        提取极其完美的片段底层键属性，确保苯环等自带真正的“芳香键”！
        """
        if f_type not in self.frag_bond_cache:
            try:
                from rdkit.Chem import rdDetermineBonds
                temp_mol = Chem.RWMol()
                idx_map = {}
                conf = Chem.Conformer(len(self.ideal_coords[f_type]))
                for i, atom_info in enumerate(self.ideal_coords[f_type]):
                    raw_name = atom_info[0].strip()
                    elem_symbol = "".join([c for c in raw_name if c.isalpha()])
                    if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
                        elem_symbol = elem_symbol[0]
                    # 我们需要H来帮助判断真实价态，不跳过H
                    idx = temp_mol.AddAtom(Chem.Atom(elem_symbol.capitalize()))
                    idx_map[raw_name] = idx
                    conf.SetAtomPosition(idx, np.array(atom_info[2]).astype(float))
                temp_mol.GetMol().AddConformer(conf)

                for st, en in self.aa_bonds[f_type]:
                    u = idx_map.get(st.strip())
                    v = idx_map.get(en.strip())
                    if u is not None and v is not None:
                        temp_mol.AddBond(u, v, Chem.rdchem.BondType.SINGLE)

                Chem.SanitizeMol(temp_mol)
                rdDetermineBonds.DetermineBondOrders(temp_mol.GetMol(), charge=0)

                b_dict = {}
                for bond in temp_mol.GetBonds():
                    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    n1 = [k for k, v in idx_map.items() if v == idx1][0]
                    n2 = [k for k, v in idx_map.items() if v == idx2][0]
                    b_dict[tuple(sorted((n1, n2)))] = bond.GetBondType()
                self.frag_bond_cache[f_type] = b_dict
            except Exception:
                self.frag_bond_cache[f_type] = {}

        pair = tuple(sorted((start_name.strip(), end_name.strip())))
        return self.frag_bond_cache[f_type].get(pair, Chem.rdchem.BondType.SINGLE)

    def assemble(
        self,
        frag_pos,
        frag_rot,
        frag_types,
        node_mask,
        final_bonds,
        final_attach_src=None,
        final_attach_dst=None,
        final_bond_probs=None,
        pocket_pos=None,
        pocket_elements=None,
        pocket_mask=None,
    ):
        batch_mols = []
        B = frag_pos.shape[0]

        for b in range(B):
            mask = node_mask[b].bool().cpu().view(-1)
            active_indices = np.where(mask.numpy())[0]
            b_pos = frag_pos[b][mask].detach().cpu().numpy()
            b_rot = frag_rot[b][mask].detach().cpu().numpy()
            temp = max(float(self.decoder_type_temperature), 1e-3)
            probs = torch.softmax(frag_types[b][mask] / temp, dim=-1)
            active_pocket = None
            active_pocket_elements = None
            if pocket_pos is not None and pocket_elements is not None:
                active_pocket = pocket_pos[b]
                active_pocket_elements = pocket_elements[b]
                if pocket_mask is not None:
                    valid_pocket = pocket_mask[b].reshape(-1) > 0.5
                    active_pocket = active_pocket[valid_pocket]
                    active_pocket_elements = active_pocket_elements[valid_pocket]
                active_pocket = active_pocket.detach().cpu().numpy()
                active_pocket_elements = active_pocket_elements.detach().cpu().numpy()

            best_score = -1000.0
            best_rank = None
            best_mol_candidate = None
            best_coords_candidate = None
            best_fragment_type_indices = None
            mmff_rmsd = 0.0

            # 重采样抽卡 10 次
            for attempt in range(self.decoder_attempts):
                b_type_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()

                rw_mol = Chem.RWMol()
                all_final_coords = []
                centers = []
                atom_to_frag = {}
                frag_attachment_atoms = {}
                frag_atom_indices = {}

                for i in range(len(b_pos)):
                    f_type = b_type_idx[i]
                    t = b_pos[i].copy()
                    R_raw = b_rot[i]

                    # 宏观碰撞排斥
                    # for prev_t in centers:
                    #     dist = np.linalg.norm(t - prev_t)
                    #     if dist < 3.8:
                    #         direction = t - prev_t
                    #         norm_dir = np.linalg.norm(direction)
                    #         if norm_dir < 1e-4: direction = np.random.randn(3)
                    #         norm_dir = np.linalg.norm(direction)
                    #         direction = direction / norm_dir
                    #         t += direction * (4.0 - dist)
                    # centers.append(t.copy())

                    R = R_raw

                    template = self.ideal_coords[f_type]
                    f_bonds = self.aa_bonds[f_type]
                    local_atom_map = {}
                    local_index_map = {}

                    for atom_idx, atom_info in enumerate(template):
                        raw_name = atom_info[0].strip()
                        elem_symbol = self._template_atom_symbol(f_type, atom_idx, raw_name)
                        if elem_symbol.upper() == 'H':
                            continue

                        local_pos = np.array(atom_info[2])
                        global_pos = np.dot(R, local_pos) + t

                        try:
                            rd_idx = rw_mol.AddAtom(Chem.Atom(elem_symbol))
                            all_final_coords.append(global_pos)
                            local_atom_map[raw_name] = rd_idx
                            local_index_map[atom_idx] = rd_idx
                            atom_to_frag[rd_idx] = i
                        except Exception:
                            continue

                    frag_attachment_atoms[i] = [
                        local_index_map[attach_idx]
                        for attach_idx in self._attachment_template_indices(f_type)
                        if attach_idx in local_index_map
                    ]
                    frag_atom_indices[i] = list(local_index_map.values())

                    for bond_idx, (start_name, end_name) in enumerate(f_bonds):
                        u = local_atom_map.get(start_name.strip())
                        v = local_atom_map.get(end_name.strip())
                        if u is not None and v is not None:
                            if not rw_mol.GetBondBetweenAtoms(u, v):
                                # 直接打上芳香键和双键的烙印！
                                correct_bond_type = self._template_bond_type(f_type, bond_idx)
                                rw_mol.AddBond(u, v, correct_bond_type)
                                if correct_bond_type == Chem.rdchem.BondType.AROMATIC:
                                    rw_mol.GetAtomWithIdx(u).SetIsAromatic(True)
                                    rw_mol.GetAtomWithIdx(v).SetIsAromatic(True)

                if rw_mol.GetNumAtoms() == 0: continue

                final_mol = rw_mol.GetMol()
                conf = Chem.Conformer(final_mol.GetNumAtoms())
                for idx, coord in enumerate(all_final_coords):
                    conf.SetAtomPosition(idx, coord.astype(float))
                final_mol.AddConformer(conf)
                fallback_mol = Chem.Mol(final_mol)

                rw_mol_constrained = Chem.RWMol(fallback_mol)
                constrained_conf = rw_mol_constrained.GetConformer()
                if final_bonds is not None:
                    final_bonds_active = final_bonds[b].detach().cpu().numpy()[np.ix_(active_indices, active_indices)]
                    final_attach_src_active = None
                    final_attach_dst_active = None
                    final_bond_probs_active = None
                    if final_attach_src is not None:
                        final_attach_src_active = final_attach_src[b].detach().cpu().numpy()[np.ix_(active_indices, active_indices)]
                    if final_attach_dst is not None:
                        final_attach_dst_active = final_attach_dst[b].detach().cpu().numpy()[np.ix_(active_indices, active_indices)]
                    if final_bond_probs is not None:
                        final_bond_probs_active = (
                            final_bond_probs[b]
                            .detach()
                            .cpu()
                            .numpy()[active_indices][:, active_indices, :]
                        )
                    connected_topology = self._apply_predicted_bonds(
                        rw_mol_constrained,
                        constrained_conf,
                        frag_attachment_atoms,
                        len(b_pos),
                        final_bonds_active,
                        final_bond_probs_b=final_bond_probs_active,
                        final_attach_src_b=final_attach_src_active,
                        final_attach_dst_b=final_attach_dst_active,
                        frag_atom_indices=frag_atom_indices,
                        atom_to_frag=atom_to_frag,
                        pocket_pos=active_pocket,
                        pocket_elements=active_pocket_elements,
                    )
                    if len(b_pos) > 1 and not connected_topology:
                        continue
                    # NOTE: do NOT null final_bonds here — the distance-based fallback
                    # loop below needs it to decide which fragment pairs to connect.
                num_atoms = rw_mol_constrained.GetNumAtoms()
                max_valence = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}

                connected_frags = set()
                frag_connection_count = {}

                # 片段之间的拼接，严格保持单键（符合真实化学规律）
                for idx_i in range(0):
                    for idx_j in range(idx_i + 1, num_atoms):
                        if not rw_mol_constrained.GetBondBetweenAtoms(idx_i, idx_j):
                            frag_i = atom_to_frag.get(idx_i, -1)
                            frag_j = atom_to_frag.get(idx_j, -2)

                            # 如果是不同片段之间的原子
                            if frag_i != frag_j:
                                # 直接读取神经网络给出的拓扑判定！
                                if final_bonds is not None and final_bonds[b, frag_i, frag_j] == 1:

                                    # 🚨 必须加上物理距离保护！否则远距离连键会导致最后的力场(MMFF)瞬间爆炸
                                    p1 = np.array(conf.GetAtomPosition(idx_i))
                                    p2 = np.array(conf.GetAtomPosition(idx_j))
                                    dist = np.linalg.norm(p1 - p2)

                                    if dist < 2.5:  # 只有物理距离在合理范围内，才真正执行连键！
                                        # 网络钦定这两个片段之间有键！直接连！
                                        # 加入基础的化合价安全锁（防范极小概率的网络幻觉）
                                        atom_i = rw_mol_constrained.GetAtomWithIdx(idx_i)
                                        atom_j = rw_mol_constrained.GetAtomWithIdx(idx_j)
                                        limit_i = max_valence.get(atom_i.GetSymbol(), 4)
                                        limit_j = max_valence.get(atom_j.GetSymbol(), 4)
                                        if atom_i.GetIsAromatic(): limit_i -= 1
                                        if atom_j.GetIsAromatic(): limit_j -= 1

                                        if atom_i.GetDegree() < limit_i and atom_j.GetDegree() < limit_j:
                                            rw_mol_constrained.AddBond(idx_i, idx_j, Chem.rdchem.BondType.SINGLE)

                try:
                    Chem.SanitizeMol(rw_mol_constrained, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                    final_mol = rw_mol_constrained.GetMol()
                except Exception as e:
                    print(f"❌ sanitize failed: {e}")
                    continue

                try:
                    frags = Chem.GetMolFrags(final_mol, asMols=True)
                    if len(frags) != 1:
                        continue
                    connected_mol = final_mol
                    if connected_mol.GetNumAtoms() < 5:
                        continue
                except:
                    continue

                # =========================================================
                # ：多目标打分，锁定黄金 LogP 与环结构
                # =========================================================
                try:
                    # 直接计算 QED (成药性) 作为挑选最优分子的唯一标准
                    metrics = calculate_composite_metrics(connected_mol)
                    if metrics is None:
                        continue
                    cand_score = metrics["composite_score"]
                    candidate_coords = np.array(
                        [
                            connected_mol.GetConformer().GetAtomPosition(atom_idx)
                            for atom_idx in range(connected_mol.GetNumAtoms())
                        ],
                        dtype=np.float64,
                    )
                    if active_pocket is not None and active_pocket.size > 0:
                        vdw_metrics = vdw_overlap_metrics(
                            connected_mol,
                            candidate_coords,
                            active_pocket,
                            active_pocket_elements,
                        )
                    else:
                        vdw_metrics = {
                            "vdw_clash_count": 0,
                            "vdw_overlap_max": 0.0,
                            "vdw_overlap_mean": 0.0,
                        }
                    candidate_rank = (
                        int(vdw_metrics["vdw_clash_count"] == 0),
                        -float(vdw_metrics["vdw_overlap_max"]),
                        -float(vdw_metrics["vdw_overlap_mean"]),
                        float(cand_score),
                    )

                    if best_rank is None or candidate_rank > best_rank:
                        best_rank = candidate_rank
                        best_score = cand_score
                        best_mol_candidate = connected_mol
                        best_fragment_type_indices = b_type_idx.copy()

                        conf_new = best_mol_candidate.GetConformer()
                        best_coords_candidate = [conf_new.GetAtomPosition(i) for i in
                                                 range(best_mol_candidate.GetNumAtoms())]
                except:
                    continue

            # ====================== 10次重采样结束 ======================
            if best_mol_candidate is None:
                batch_mols.append((None, None, {
                    "assembly_success": False,
                    "assembly_error": (
                        "assembly_connectivity_failure"
                        if len(b_pos) > 1
                        else "assembly_disconnected_or_invalid"
                    ),
                    "connected_components": 0,
                    "mmff_rmsd": "",
                }))
                continue

            # 力场-芳香共振
            # 因为现在有真实的芳香键了，MMFF 会自动把苯环/吡啶压成完美的平面
            force_field_success = False

            if best_coords_candidate is not None and len(best_coords_candidate) > 1:
                coords_array = np.array(best_coords_candidate, dtype=np.float64)

                # 1. 终极防线：检测原始坐标是否已经存在 NaN
                if not np.isnan(coords_array).any() and not np.isinf(coords_array).any():
                    from scipy.spatial.distance import pdist

                    # 2. 距离矩阵预检：筛查是否存在极端空间坍缩 (< 0.5 埃)
                    min_dist = np.min(pdist(coords_array))
                    if min_dist > 0.5:
                        try:
                            # 3. 安全调用力场
                            mmff_props = AllChem.MMFFGetMoleculeProperties(best_mol_candidate)
                            if mmff_props is None:
                                raise ValueError("MMFF parameters unavailable")
                            force_field = AllChem.MMFFGetMoleculeForceField(best_mol_candidate, mmff_props)
                            if force_field is None:
                                raise ValueError("MMFF force field construction failed")
                            for atom_idx in range(best_mol_candidate.GetNumAtoms()):
                                force_field.MMFFAddPositionConstraint(atom_idx, 0.3, 100.0)
                            force_field.Initialize()
                            force_field.Minimize(maxIts=200)

                            # 4. 严密校验力场返回值
                            conf_new = best_mol_candidate.GetConformer()
                            opt_coords = np.array(
                                [conf_new.GetAtomPosition(i) for i in range(best_mol_candidate.GetNumAtoms())],
                                dtype=np.float64)

                            # 5. math.isnan 最终检测，防止力场内部将坐标算成 NaN
                            if not np.isnan(opt_coords).any() and not np.isinf(opt_coords).any():
                                displacement = opt_coords - coords_array
                                displacement -= displacement.mean(axis=0, keepdims=True)
                                max_displacement = np.linalg.norm(displacement, axis=1).max()
                                if max_displacement > 0.3:
                                    displacement *= 0.3 / max_displacement
                                opt_coords = coords_array + displacement
                                mmff_rmsd = float(np.sqrt(np.mean(np.sum(displacement ** 2, axis=1))))
                                for atom_idx, coord in enumerate(opt_coords):
                                    conf_new.SetAtomPosition(atom_idx, coord.astype(float))
                                final_coords_cleaned = opt_coords.tolist()
                                force_field_success = True
                        except Exception:
                            pass

            # 6. 安全回退机制 (Fallback)：力场崩溃或预检未通过，强制使用扩散模型原始坐标
            if not force_field_success:
                final_coords_cleaned = best_coords_candidate
                # 确保连保底原始坐标也没有 NaN，如果有，彻底丢弃该分子
                if final_coords_cleaned is None or np.isnan(np.array(final_coords_cleaned, dtype=np.float64)).any():
                    batch_mols.append((None, None, {
                        "assembly_success": False,
                        "assembly_error": "assembly_coordinate_failure",
                        "connected_components": 0,
                        "mmff_rmsd": "",
                    }))
                    continue

            batch_mols.append((best_mol_candidate, final_coords_cleaned, {
                "assembly_success": True,
                "assembly_error": "",
                "connected_components": 1,
                "mmff_rmsd": mmff_rmsd,
                "fragment_type_indices": (
                    best_fragment_type_indices.tolist()
                    if best_fragment_type_indices is not None else []
                ),
            }))

        return batch_mols



# =========================================================================
def _ensure_3d_mask(m: torch.Tensor, B: int, N: int, device, default_val: float = 1.0):
    if m is None: return torch.full((B, N, 1), float(default_val), device=device)
    if not isinstance(m, torch.Tensor): m = torch.tensor(m, dtype=torch.float32, device=device)
    if m.dim() == 1:
        m = m.view(1, -1, 1).repeat(B, 1, 1)
    elif m.dim() == 2:
        m = m.unsqueeze(-1)
    elif m.dim() == 3:
        m = m[:, :, :1]
    return m.float().to(device)


def _xyz_only(pos: torch.Tensor):
    if pos.dim() == 2: pos = pos.unsqueeze(0)
    if pos.size(-1) >= 3: return pos[..., :3].contiguous()
    raise RuntimeError(f"Position tensor last dim must be >= 3, got {pos.shape}")


def read_molecule(path):
    mol = None
    if path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol') or path.endswith('.mol2'):
        mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=True)
        mol = suppl[0] if len(suppl) > 0 else None
    if mol is None: raise ValueError(f"Failed to read molecule from {path}")
    return mol


def bridge_rdkit_to_openfold_tensor(mol, device='cpu'):
    residues = list(mol.GetResidues())
    n_res = len(residues)
    all_atom_positions = np.zeros((n_res, 37, 3))
    all_atom_mask = np.zeros((n_res, 37))
    aatype = np.zeros(n_res, dtype=np.int64)
    for i, res in enumerate(residues):
        res_name = res.GetResidueName().strip()
        aatype[i] = rc.restype_order.get(rc.restype_3to1.get(res_name, 'X'), 20)
        for atom in res.GetAtoms():
            p_info = atom.GetPDBResidueInfo()
            if p_info is None: continue
            atom_name = p_info.GetName().strip()
            if atom_name in rc.atom_order:
                idx = rc.atom_order[atom_name]
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                all_atom_positions[i, idx] = [pos.x, pos.y, pos.z]
                all_atom_mask[i, idx] = 1.0
    return {
        "aatype": torch.tensor(aatype, device=device),
        "all_atom_positions": torch.tensor(all_atom_positions, dtype=torch.float32, device=device),
        "all_atom_mask": torch.tensor(all_atom_mask, dtype=torch.float32, device=device)
    }


def atom37_to_frames(protein, eps=1e-8):
    aatype = protein["aatype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]
    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros((*aatype.shape[:-1], 21, 8))
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(restype_rigidgroup_base_atom_names)
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(restype_rigidgroup_base_atom37_idx)
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(*((1,) * batch_dims),
                                                                                 *restype_rigidgroup_base_atom37_idx.shape)

    residx_rigidgroup_base_atom37_idx = batched_gather(restype_rigidgroup_base_atom37_idx, aatype, dim=-3,
                                                       no_batch_dims=batch_dims)
    base_atom_pos = batched_gather(all_atom_positions, residx_rigidgroup_base_atom37_idx, dim=-2,
                                   no_batch_dims=len(all_atom_positions.shape[:-2]))

    gt_frames = Rigid.from_3_points(p_neg_x_axis=base_atom_pos[..., 0, :], origin=base_atom_pos[..., 1, :],
                                    p_xy_plane=base_atom_pos[..., 2, :], eps=eps)

    group_exists = batched_gather(restype_rigidgroup_mask, aatype, dim=-2, no_batch_dims=batch_dims)
    gt_atoms_exist = batched_gather(all_atom_mask, residx_rigidgroup_base_atom37_idx, dim=-1,
                                    no_batch_dims=len(all_atom_mask.shape[:-1]))
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)
    gt_frames = gt_frames.compose(Rigid(rots, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    protein["rigidgroups_gt_frames"] = gt_frames_tensor
    protein["rigidgroups_gt_exists"] = gt_exists
    return protein


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    oversample_factor = max(1, int(args.oversample_factor))
    target_output_count = 0
    all_candidates = []
    generation_records = []
    candidate_output_dir = os.path.join(args.output, "candidates")
    selected_output_dir = os.path.join(args.output, "selected")
    os.makedirs(candidate_output_dir, exist_ok=True)
    os.makedirs(selected_output_dir, exist_ok=True)
    latent_output_dir = os.path.join(args.output, "latents")
    if args.save_latent_features:
        os.makedirs(latent_output_dir, exist_ok=True)

    print(f"🚀 Loading FragDiff model from {args.model}...")
    ddpm = DDPM.load_from_checkpoint(
        args.model,
        map_location=device,
        strict=False,
    ).eval().to(device)
    if ddpm.apply_ema_weights():
        print(
            f"EMA weights activated (decay={ddpm.ema_decay:.6f}, "
            f"updates={ddpm._ema_num_updates})."
        )
    else:
        print("EMA weights not found; using raw checkpoint weights.")
    if args.n_steps is not None: ddpm.edm.T = args.n_steps
    ddpm.reconstruct_and_save = lambda *args, **kwargs: []
    """
    print("🧠 Loading SVDD Value Network...")
    value_net = None
    try:
        value_net = ValueNetwork(in_node_nf=ddpm.num_classes).to(device)
        value_net.load_state_dict(torch.load("checkpoints/value_net_final.pt", map_location=device))
        value_net.eval()
    print("✅ SVDD Value Network loaded successfully!")

    """
    print("Loading SVDD Value Network...")
    value_net = None
    try:
        value_net = ValueNetwork(in_node_nf=ddpm.num_classes).to(device)
        value_net.load_state_dict(torch.load("checkpoints/value_net_crossdock_ep60.pt", map_location=device))
        value_net.eval()
        print("SVDD Value Network loaded successfully!")
    except Exception as exc:
        value_net = None
        print(f"Value network disabled: {exc}")
    value_guidance = None
    if value_net is not None:
        value_guidance = ValueNetworkGuidance(
            value_net=value_net,
            guidance_scale=args.guidance_scale,
            start_step=args.guidance_start,
            end_step=args.guidance_end,
            max_norm=1.0,
        )
    if args.pocket_value_checkpoint:
        payload = torch.load(args.pocket_value_checkpoint, map_location=device)
        if not bool(payload.get("guidance_approved", False)):
            raise RuntimeError(
                "Pocket value checkpoint is not approved: validation requires Spearman >= 0.5 and MAE <= 2.5"
            )
        if int(payload.get("ligand_feature_dim", ddpm.num_classes)) != int(ddpm.num_classes):
            raise RuntimeError("Pocket value checkpoint fragment vocabulary does not match the diffusion model")
        hidden_dim = int(payload.get("hidden_dim", 128))
        pocket_value_net = PocketPoseValueNetwork(ddpm.num_classes, hidden_dim=hidden_dim).to(device)
        pocket_value_net.load_state_dict(payload.get("state_dict", payload))
        value_guidance = PocketPoseValueGuidance(
            pocket_value_net,
            scale=args.pocket_value_scale,
            start_step=args.pocket_value_start,
            end_step=args.pocket_value_end,
            max_norm=0.2,
        )
        print(f"Pocket-conditioned Vina surrogate guidance loaded: {args.pocket_value_checkpoint}")

    fragment_count_sampler = FragmentCountSampler(
        mode=args.fragment_count_mode,
        legacy_value=args.linker_size,
        max_count=args.fragment_count_max,
        seed=args.fragment_count_seed,
        histogram_path=args.fragment_count_hist,
    )

    def sample_fn(_data):
        mask = _data.get('atom_mask')
        if mask is None:
            return torch.full(
                (_data['positions'].shape[0],),
                int(args.fragment_count_max),
                device=device,
                dtype=const.TORCH_INT,
            )
        return mask.reshape(mask.shape[0], -1).sum(dim=1).long()

    dataset_list = []

    if args.use_dataset:
        print("📁 Running in DATASET MODE")
        split_path = args.split_file or os.path.join(args.data_dir, f"{args.dataset_prefix}_split.pt")
        split_data = torch.load(split_path)
        dataset_split = args.dataset_split
        if dataset_split not in split_data:
            raise KeyError(f"Split {dataset_split!r} is absent from {split_path}")
        dataset_cls = CrossDockDataset if "crossdock" in args.dataset_prefix.lower() else PDBbindDataset
        dataset_kwargs = dict(
            data_path=args.data_dir,
            prefix=args.dataset_prefix,
            split=dataset_split,
            device=device,
        )
        if dataset_cls is CrossDockDataset:
            dataset_kwargs["split_file"] = split_path
            dataset_kwargs["fragment_vocab_path"] = args.fragment_vocab_path
            dataset_kwargs["pocket_only"] = True
        full_dataset = dataset_cls(**dataset_kwargs)
        dataset_vocab_size = len(full_dataset.fragment_vocab.get("entries", []))
        if dataset_vocab_size != int(ddpm.num_classes):
            raise RuntimeError(
                "Fragment vocabulary/checkpoint mismatch before sampling: "
                f"dataset_vocab={dataset_vocab_size}, checkpoint_vocab={ddpm.num_classes}. "
                "For a rebuilt LMDB, pass the exact vocabulary used to train the checkpoint "
                "with --fragment_vocab_path."
            )
        test_indices = list(range(len(full_dataset)))
        if args.limit > 0: test_indices = test_indices[:args.limit]

        for idx in tqdm(test_indices, desc="Pre-loading Targets"):
            try:
                data_item = full_dataset[idx]
                base_item = {
                    key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
                    for key, value in data_item.items()
                }
                # 强制 De Novo 生成，彻底抛弃原配体的大小
                if args.linker_size.isdigit():
                    sampled_size = int(args.linker_size)
                else:
                    bounds = [int(x.strip()) for x in args.linker_size.split(',')]
                    sampled_size = bounds[1]

                # 提取口袋中心（或原配体中心）作为新片段的出生点
                if 'ligand_pos' in data_item:
                    center_pos = data_item['ligand_pos'].mean(dim=0)
                elif 'positions' in data_item:
                    center_pos = data_item['positions'].mean(dim=0)
                else:
                    center_pos = torch.zeros(3, device=device)

                # 强行覆盖数据：构造全新的 sampled_size (8~12) 个空白片段节点
                data_item['positions'] = center_pos.unsqueeze(0).repeat(sampled_size, 1)
                data_item['frag_rotations'] = torch.eye(3).unsqueeze(0).repeat(sampled_size, 1, 1)

                # One-hot: use uniform prior for De Novo generation.
                # Zeroing or random-then-zero leaves the model with no node-type signal.
                # Uniform means "uninformative prior over all fragment types".
                data_item['one_hot'] = torch.full(
                    (sampled_size, len(const.NUM2AA)),
                    1.0 / len(const.NUM2AA),
                    dtype=torch.float32,
                )
                data_item['fragment_type_ids'] = torch.full((sampled_size,), -1, dtype=torch.long)

                # All sampled_size nodes are valid fragment slots to generate —
                # atom_mask = 1 for every node so valid_node_mask is not all-zero.
                data_item['linker_mask'] = torch.ones(sampled_size, dtype=torch.float32)
                data_item['fragment_mask'] = torch.zeros(sampled_size, dtype=torch.float32)
                data_item['atom_mask'] = torch.ones(sampled_size, dtype=torch.float32)
                data_item['slot_mask'] = torch.ones(sampled_size, dtype=torch.float32)
                data_item['anchors'] = torch.zeros(sampled_size, dtype=torch.float32)
                if 'true_bonds' in data_item:
                    del data_item['true_bonds']
                if 'true_attach_src' in data_item:
                    del data_item['true_attach_src']
                if 'true_attach_dst' in data_item:
                    del data_item['true_attach_dst']

                target_output_count += args.n_samples
                attempt_count = args.n_samples * oversample_factor
                sampled_counts = fragment_count_sampler.sample(attempt_count)
                for generation_index, sampled_size in enumerate(sampled_counts):
                    generated_item = {
                        key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
                        for key, value in base_item.items()
                    }
                    generated_center = generated_item.get('reference_box_center')
                    if generated_center is None:
                        generated_center = generated_item['positions'].mean(dim=0)
                    generated_item['positions'] = generated_center.unsqueeze(0).repeat(sampled_size, 1)
                    generated_item['frag_rotations'] = torch.eye(
                        3, device=generated_center.device, dtype=generated_center.dtype
                    ).unsqueeze(0).repeat(sampled_size, 1, 1)
                    generated_item['one_hot'] = torch.full(
                        (sampled_size, len(const.NUM2AA)),
                        1.0 / len(const.NUM2AA),
                        dtype=torch.float32,
                        device=generated_center.device,
                    )
                    generated_item['fragment_type_ids'] = torch.full(
                        (sampled_size,), -1, dtype=torch.long, device=generated_center.device
                    )
                    generated_item['linker_mask'] = torch.ones(sampled_size, dtype=torch.float32, device=generated_center.device)
                    generated_item['fragment_mask'] = torch.zeros(sampled_size, dtype=torch.float32, device=generated_center.device)
                    generated_item['atom_mask'] = torch.ones(sampled_size, dtype=torch.float32, device=generated_center.device)
                    generated_item['slot_mask'] = torch.ones(sampled_size, dtype=torch.float32, device=generated_center.device)
                    generated_item['anchors'] = torch.zeros(sampled_size, dtype=torch.float32, device=generated_center.device)
                    generated_item['name'] = f"target_{idx}"
                    generated_item['dataset_target_index'] = idx
                    generated_item['generation_index_within_target'] = generation_index
                    for key in ('true_bonds', 'true_attach_src', 'true_attach_dst'):
                        generated_item.pop(key, None)
                    dataset_list.append(generated_item)
            except Exception as exc:
                print(f"Target {idx} generation input failed without substitution: {exc}")
                if args.strict_targets:
                    raise
    else:
        print("🧪 Running in SINGLE FILE MODE")
        pocket_mol = read_molecule(args.pocket)
        pocket_tensor = bridge_rdkit_to_openfold_tensor(pocket_mol, device=device)
        pocket_with_frames = atom37_to_frames(pocket_tensor)
        pocket_frames = pocket_with_frames["rigidgroups_gt_frames"][:, 0, :, :]
        pocket_mask = pocket_with_frames["rigidgroups_gt_exists"][:, 0]

        molecule = read_molecule(args.fragments)
        positions, one_hot, charges = parse_molecule(molecule, is_geom=ddpm.is_geom)
        n_atoms = len(positions)
        if one_hot.shape[1] != ddpm.num_classes:
            raise ValueError(
                "Single-file mode still provides atom-level features, but the loaded checkpoint expects "
                f"{ddpm.num_classes} rigid fragment classes. Use --use_dataset for from-scratch rigid-fragment generation."
            )

        anchor_flags = np.zeros(n_atoms)
        if args.anchors:
            for a in args.anchors.split(','): anchor_flags[int(a.strip()) - 1] = 1

        single_data = {
            'uuid': '0', 'name': os.path.basename(args.fragments).split('.')[0],
            'positions': torch.tensor(positions, dtype=torch.float32, device=device),
            'one_hot': torch.tensor(one_hot, dtype=torch.float32, device=device),
            'charges': torch.tensor(charges, dtype=torch.float32, device=device),
            'anchors': torch.tensor(anchor_flags, dtype=torch.float32, device=device),
            'fragment_mask': torch.ones(n_atoms, dtype=torch.float32, device=device),
            'linker_mask': torch.zeros(n_atoms, dtype=torch.float32, device=device),
            'slot_mask': torch.ones(n_atoms, dtype=torch.float32, device=device),
            'num_atoms': n_atoms,
            'pocket_frames': pocket_frames.to(device),
            'pocket_mask': pocket_mask.to(device)
        }
        target_output_count = args.n_samples
        dataset_list = [single_data] * (args.n_samples * oversample_factor)

    if not dataset_list: return
    generation_record_lookup = {}
    for raw_attempt_index, item in enumerate(dataset_list):
        lmdb_key = str(item.get('uuid', ''))
        target_index = item.get('dataset_target_index', '')
        generation_index = item.get('generation_index_within_target', raw_attempt_index)
        record_key = (lmdb_key, str(target_index), str(generation_index))
        record = {
            "target_name": str(item.get('name', '')),
            "lmdb_key": lmdb_key,
            "dataset_target_index": target_index,
            "generation_index_within_target": generation_index,
            "fragment_count": int(item['atom_mask'].sum().item()) if 'atom_mask' in item else "",
            "assembly_success": False,
            "assembly_error": "sampling_not_completed",
            "sdf_path": "",
        }
        generation_records.append(record)
        generation_record_lookup[record_key] = record
    if target_output_count <= 0:
        target_output_count = len(dataset_list)
    batch_size = max(1, min(len(dataset_list), int(args.batch_size)))
    dataloader = get_dataloader(dataset_list, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

    for batch_i, data in enumerate(tqdm(dataloader, desc="Denoising Samples")):
        for k, v in data.items():
            if isinstance(v, torch.Tensor): data[k] = v.to(device)

        bs, n_nodes = data['positions'].shape[0], data['positions'].shape[1]

        # =========================================================================
        # 严格掩码保真度与拓扑动态切片 (Strict Mask Fidelity)
        # 废除盲目覆写全局！利用 one_hot 特征提取出当前批次真实的原子数，屏蔽 Padding 虚拟节点
        # =========================================================================
        # 只要特征维度上有激活，就说明是真实节点
        if 'atom_mask' in data:
            valid_node_mask = _ensure_3d_mask(data['atom_mask'], bs, n_nodes, device)
        else:
            valid_node_mask = (data['one_hot'].sum(dim=-1) > 0.5).float().unsqueeze(-1)

        data['atom_mask'] = valid_node_mask
        if 'slot_mask' in data:
            data['slot_mask'] = _ensure_3d_mask(data['slot_mask'], bs, n_nodes, device)
        else:
            data['slot_mask'] = torch.ones_like(valid_node_mask)
        if 'linker_mask' in data:
            data['linker_mask'] = _ensure_3d_mask(data['linker_mask'], bs, n_nodes, device)
        else:
            data['linker_mask'] = data['slot_mask']
        if 'fragment_mask' in data:
            data['fragment_mask'] = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
        else:
            data['fragment_mask'] = torch.zeros_like(data['slot_mask'])
        # =========================================================================

        curr_node_mask = _ensure_3d_mask(data.get('atom_mask', None), bs, n_nodes, device)
        curr_edge_mask = None
        curr_context = data.get('context', None)

        if 'linker_mask' in data and 'fragment_mask' in data:
            real_pos_temp = data['positions'].clone()
            linker_mask_3d = _ensure_3d_mask(data['linker_mask'], bs, n_nodes, device)
            fragment_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)

            raw_noise = torch.randn_like(real_pos_temp)
            # 净化初始噪声
            raw_noise = torch.nan_to_num(raw_noise, nan=0.0, posinf=0.0, neginf=0.0)


            # 1. 检查当前分子是否有预先保留的骨架 (fragment)
            has_frag = (fragment_mask_3d.sum(dim=1, keepdim=True) > 0.5).float()

            # 2. 如果有保留骨架，质心就是骨架的质心
            sum_pos = (real_pos_temp * fragment_mask_3d).sum(dim=1, keepdim=True)
            count_pos = fragment_mask_3d.sum(dim=1, keepdim=True).clamp(min=1.0)
            frag_center = sum_pos / count_pos

            # 3.  (De Novo)，质心必须是蛋白质口袋的中心！
            # 因为我们在 Dataset 阶段已经把 De Novo 的 real_pos_temp 初始化在口袋中心了，所以直接取均值
            valid_nodes_count = valid_node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pocket_center = (real_pos_temp * valid_node_mask).sum(dim=1, keepdim=True) / valid_nodes_count

            # 4. 智能选择：锚点 = 骨架质心 (如果有) or 口袋质心 (如果从头生成)
            center_of_mass = frag_center * has_frag + pocket_center * (1.0 - has_frag)

            # 5. 将高斯噪声精准空投到蛋白质口袋内部！
            aligned_noise = raw_noise + center_of_mass
            data['positions'] = torch.where(linker_mask_3d > 0.5, aligned_noise, real_pos_temp)

        try:
            if False:
                value_network_guidance(
            # 提取初始旋转矩阵 (如果不存在则用单位阵兜底)

                scale=2.0  # 引导强度
            )


            sample_results = ddpm.sample_chain(
                data, sample_fn=sample_fn, keep_frames=1, edge_mask=curr_edge_mask,
                context=curr_context, save_path=None, guidance_fn=value_guidance,
                temperature=args.temperature, top_k=args.top_k,
                clash_guidance_scale=args.clash_guidance_scale,
                clash_guidance_start=args.clash_guidance_start,
                clash_guidance_end=args.clash_guidance_end,
                clash_guidance_max_norm=args.clash_guidance_max_norm,
                clash_guidance_global_fraction=args.clash_guidance_global_fraction,
                clash_guidance_max_com_shift=args.clash_guidance_max_com_shift,
                return_bond_probs=True,
            )

        except Exception as e:
            import traceback
            print(f"\n❌ 采样过程发生致命错误: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if sample_results is None: continue

        (
            chain_data,
            node_mask,
            final_bonds,
            final_attach_src,
            final_attach_dst,
            final_bond_probs,
        ) = sample_results
        while isinstance(chain_data, (tuple, list)): chain_data = chain_data[0]

        if chain_data.dim() == 4:
            current_chain = chain_data[-1]
        else:
            current_chain = chain_data

        x_gen = current_chain[..., :3]

        if 'fragment_mask' in data:
            frag_mask_3d = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device)
            real_pos = _xyz_only(data['positions'])
            x_gen = torch.where(frag_mask_3d > 0.5, real_pos, x_gen)

        final_x = x_gen

        try:
            rot_flat = current_chain[..., 3:12]
            rot_gen = rot_flat.view(bs, n_nodes, 3, 3)

            if 'frag_rotations' in data and 'fragment_mask' in data:
                frag_mask_3d_rot = _ensure_3d_mask(data['fragment_mask'], bs, n_nodes, device).unsqueeze(-1)
                real_rot = data['frag_rotations']
                rot_gen = torch.where(frag_mask_3d_rot > 0.5, real_rot, rot_gen)

            raw_h_gen = current_chain[..., 12:]
            frag_types_gen = raw_h_gen[..., :ddpm.num_classes]

            decoder = RigidAssemblyDecoder(
                decoder_type_temperature=args.decoder_type_temperature,
                decoder_attempts=args.decoder_attempts,
                tree_beam_width=args.decoder_tree_beam_width,
                tree_plans=args.decoder_tree_plans,
            )
            mols_and_coords = decoder.assemble(
                final_x,
                rot_gen,
                frag_types_gen,
                node_mask,
                final_bonds,
                final_attach_src,
                final_attach_dst,
                final_bond_probs=final_bond_probs,
                pocket_pos=data.get('pocket_frames', None)[..., :3, 3]
                if data.get('pocket_frames', None) is not None else None,
                pocket_elements=data.get('pocket_element', None),
                pocket_mask=data.get('pocket_mask', None),
            )

            raw_names = data.get('name', [f"target_{batch_i}" for _ in range(bs)])
            if isinstance(raw_names, str): raw_names = [raw_names] * bs

            raw_lmdb_keys = data.get('uuid', ["" for _ in range(bs)])
            raw_target_indices = data.get('dataset_target_index', ["" for _ in range(bs)])
            raw_generation_indices = data.get('generation_index_within_target', ["" for _ in range(bs)])

            for j, (mol, coords, assembly_meta) in enumerate(mols_and_coords):
                target_name = raw_names[j]
                lmdb_key = raw_lmdb_keys[j] if j < len(raw_lmdb_keys) else ""
                dataset_target_index = raw_target_indices[j] if j < len(raw_target_indices) else ""
                generation_index = raw_generation_indices[j] if j < len(raw_generation_indices) else ""
                fragment_count = int(node_mask[j].sum().item())
                record_key = (str(lmdb_key), str(dataset_target_index), str(generation_index))
                attempt_record = generation_record_lookup.get(record_key)
                if attempt_record is None:
                    attempt_record = {
                        "target_name": str(target_name),
                        "lmdb_key": str(lmdb_key),
                        "dataset_target_index": dataset_target_index,
                        "generation_index_within_target": generation_index,
                        "fragment_count": fragment_count,
                        "sdf_path": "",
                    }
                    generation_records.append(attempt_record)
                attempt_record.update(assembly_meta)
                if mol is None or coords is None:
                    continue
                coords_array = np.array(coords, dtype=np.float64)
                if np.isnan(coords_array).any() or np.isinf(coords_array).any():
                    attempt_record["assembly_success"] = False
                    attempt_record["assembly_error"] = "assembly_coordinate_failure"
                    continue
                sample_idx = batch_i * batch_size + j

                final_mol = Chem.Mol(mol)
                try:
                    conf = Chem.Conformer(final_mol.GetNumAtoms())
                    for idx, coord in enumerate(coords):
                        conf.SetAtomPosition(idx, np.array(coord, dtype=np.float64))
                    final_mol.RemoveAllConformers()
                    final_mol.AddConformer(conf)
                except Exception as e:
                    attempt_record["assembly_success"] = False
                    attempt_record["assembly_error"] = "conformer_write_failure"
                    continue

                geometry = pocket_geometry_metrics(final_mol, coords_array, data, j)
                attempt_record.update(geometry)
                if geometry["out_of_box"]:
                    attempt_record["assembly_success"] = False
                    attempt_record["assembly_error"] = "assembly_out_of_box"
                    continue
                metrics = calculate_composite_metrics(final_mol)
                smiles = safe_mol_to_smiles(final_mol)
                fp = mol_fingerprint(final_mol)
                if metrics is None or not smiles or fp is None:
                    attempt_record["assembly_success"] = False
                    attempt_record["assembly_error"] = "property_or_fingerprint_failure"
                    continue
                candidate_path = os.path.join(
                    candidate_output_dir,
                    f"{sanitize_filename(target_name)}_sample_{sample_idx}.sdf",
                )
                final_mol.SetProp("lmdb_key", str(lmdb_key))
                final_mol.SetProp("dataset_target_index", str(dataset_target_index))
                final_mol.SetProp("generation_index_within_target", str(generation_index))
                if not write_sdf_file(final_mol, candidate_path):
                    attempt_record["assembly_success"] = False
                    attempt_record["assembly_error"] = "sdf_write_failure"
                    continue
                attempt_record["sdf_path"] = candidate_path
                latent_path = ""
                if args.save_latent_features:
                    active_mask = node_mask[j].reshape(-1) > 0.5
                    latent_path = os.path.join(
                        latent_output_dir,
                        f"{sanitize_filename(target_name)}_sample_{sample_idx}.pt",
                    )
                    latent_payload = {
                        "ligand_pos": final_x[j][active_mask].detach().cpu(),
                        "ligand_features": torch.softmax(
                            frag_types_gen[j][active_mask], dim=-1
                        ).detach().cpu(),
                        "pocket_pos": data['pocket_frames'][j, :, :3, 3].detach().cpu(),
                        "pocket_elements": data.get(
                            'pocket_element',
                            torch.zeros(data['pocket_frames'].shape[:2], device=device, dtype=torch.long),
                        )[j].detach().cpu(),
                        "pocket_mask": data['pocket_mask'][j].detach().cpu(),
                        "lmdb_key": str(lmdb_key),
                        "dataset_target_index": dataset_target_index,
                    }
                    torch.save(latent_payload, latent_path)
                    attempt_record["latent_path"] = latent_path
                candidate = {
                    "mol": final_mol,
                    "target_name": str(target_name),
                    "source_sample_idx": sample_idx,
                    "smiles": smiles,
                    "fingerprint": fp,
                    "metrics": metrics,
                    "sdf_path": candidate_path,
                    "selected_sdf_path": "",
                    "duplicate_status": "",
                    "diversity_selected": False,
                    "lmdb_key": str(lmdb_key),
                    "dataset_target_index": dataset_target_index,
                    "generation_index_within_target": generation_index,
                    "fragment_count": fragment_count,
                    **assembly_meta,
                    **geometry,
                    "latent_path": latent_path,
                }
                all_candidates.append(candidate)
                print(f"Saved valid candidate SDF: {candidate_path}")
                write_metrics_csv(all_candidates, args.output)

        except Exception as e:
            print(f"❌ Batch {batch_i} 组装/保存过程发生错误: {e}")


    selected_candidates = select_diverse_candidates(
        all_candidates,
        max_outputs=target_output_count,
        diversity_threshold=args.diversity_threshold,
    )

    for out_idx, cand in enumerate(selected_candidates):
        target_name = sanitize_filename(cand.get("target_name", f"target_{out_idx}"))
        sdf_path = os.path.join(selected_output_dir, f"{target_name}_selected_{out_idx}.sdf")
        if write_sdf_file(cand["mol"], sdf_path):
            cand["selected_sdf_path"] = sdf_path

    metrics_path = write_metrics_csv(all_candidates, args.output)
    attempts_path = write_generation_attempts_csv(generation_records, args.output)
    print(f"Saved {len(all_candidates)} immediate candidate SDF files.")
    print(f"Saved {len(selected_candidates)} diverse selected SDF copies.")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Generation attempts CSV: {attempts_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
