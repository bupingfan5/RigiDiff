import numpy as np
import os
import pytorch_lightning as pl
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from src import const
from src.const import IDEAL_COORDS, NUM2AA, AA_BONDS, AA_NAME2IDX
from rdkit import Chem
from src import utils
import time
from src.egnn import Dynamics, DynamicsWithPockets
from src.edm import EDM, InpaintingEDM
from src.fragment_vocab import ensure_fragment_vocabulary
from src.pocket_geometry import center_ligand_and_pocket
from src.datasets import (
    ZincDataset, MOADDataset, get_dataloader, collate_with_fragment_edges
)

try:
    from src.datasets import PDBbindDataset, CrossDockDataset
except Exception:
    PDBbindDataset = None
    CrossDockDataset = None

from src.linker_size import DistributionNodes
from src.visualizer import save_xyz_file, visualize_chain


# ==============================================================================
# ==============================================================================
def compute_repulsive_force(x, node_mask, r_vdw=1.0, strength=0.1):
    'Compute repulsive gradients for close atom pairs.'
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        dist_sq = torch.sum((x.unsqueeze(1) - x.unsqueeze(2)) ** 2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8)

        mask = node_mask @ node_mask.transpose(1, 2)
        mask.diagonal(dim1=1, dim2=2).fill_(0)

        safe_dist = torch.clamp(dist, min=0.1)
        energy = torch.where(
            (dist < r_vdw) & (mask > 0.5),
            (1.0 / (safe_dist ** 6)) - (1.0 / (r_vdw ** 6)),
            torch.zeros_like(dist)
        )
        total_energy = energy.sum()
        if total_energy > 0:
            grad = torch.autograd.grad(total_energy, x)[0]
            return torch.clamp(grad * strength, -1.0, 1.0)
        return torch.zeros_like(x)


def prevent_z_collapse(x, node_mask, eps=1e-2):
    'Repair degenerate coordinate variance along the z-axis.'
    B = x.shape[0]  
    x_out = x.clone()
    for b in range(B):
        valid_mask = node_mask[b].squeeze(-1) > 0.5
        valid_x = x_out[b][valid_mask]

        if valid_x.shape[0] < 3:  
            continue

        var_3d = torch.var(valid_x, dim=0)
        for dim_idx in range(3):
            if var_3d[dim_idx] < eps:
                noise = torch.randn_like(x_out[b, :, dim_idx]) * 0.5
                x_out[b, :, dim_idx] += noise * node_mask[b, :, 0]
    return x_out


# ==============================================================================

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")


class DDPM(pl.LightningModule):
    def __init__(self, in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention,
                 norm_constant, inv_sublayers, sin_embedding, normalization_factor, aggregation_method, diffusion_steps,
                 diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type, normalize_factors,
                 include_charges, model, data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device,
                 test_epochs, n_stability_samples, normalization=None, log_iterations=None, samples_dir=None,
                 data_augmentation=False, center_of_mass='fragments', inpainting=False, anchors_context=True,
                 graph_type=None, dataset: str = None,
                 lambda_pocket_clash: float = 1.0, lambda_contact: float = 0.1,
                 lambda_attach: float = 1.0, lambda_connectivity: float = 0.2,
                 lambda_gf: float = 0.5,
                 bond_class_weights=(0.2, 1.0, 1.0, 1.0, 1.0),
                 corrective_freeze_epochs: int = 3,
                 pocket_encoder_lr_multiplier: float = 10.0,
                 bond_head_lr_multiplier: float = 3.0,
                 rigid_head_lr_multiplier: float = 20.0,
                 clash_warmup_epochs: int = 10):
        super(DDPM, self).__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type
        self.n_dims = n_dims
        self.anchors_context = anchors_context
        self.dataset_name = (dataset or "").lower().strip()
        self.clash_warmup_epochs = int(clash_warmup_epochs)
        self.corrective_freeze_epochs = int(corrective_freeze_epochs)
        self.pocket_encoder_lr_multiplier = float(pocket_encoder_lr_multiplier)
        self.bond_head_lr_multiplier = float(bond_head_lr_multiplier)
        self.rigid_head_lr_multiplier = float(rigid_head_lr_multiplier)
        train_prefix_lower = self.train_data_prefix.lower()
        val_prefix_lower = self.val_data_prefix.lower()
        self.is_crossdock = (
            self.dataset_name in {"crossdock", "crossdocked"}
            or "crossdock" in train_prefix_lower
            or "crossdock" in val_prefix_lower
        )
        has_pocket_mode = "." in train_data_prefix and not self.is_crossdock
        self.metrics = {}
        self.is_geom = ('geom' in self.train_data_prefix) or ('MOAD' in self.train_data_prefix) or self.is_crossdock
        self.fragment_vocab = None

        if graph_type is None:
            graph_type = '4A' if has_pocket_mode else 'FC'

        if isinstance(activation, str):
            activation = get_activation(activation)

        effective_in_node_nf = in_node_nf
        fragment_prefix = train_data_prefix or val_data_prefix or ""
        uses_fragment_vocab = (
            self.dataset_name in {"pdbbind", "moad", "crossdock", "crossdocked"}
            or ("PDBbind" in fragment_prefix)
            or ("MOAD" in fragment_prefix)
            or ("crossdock" in fragment_prefix.lower())
        )
        if uses_fragment_vocab:
            try:
                self.fragment_vocab = ensure_fragment_vocabulary(data_path, fragment_prefix.split('.')[0])
                effective_in_node_nf = const.GEOM_NUMBER_OF_FRAGMENT_TYPES
            except Exception as exc:
                print(f"[DDPM] fragment vocabulary bootstrap failed, falling back to in_node_nf={in_node_nf}: {exc}")

        self.num_classes = effective_in_node_nf

        dynamics_class = DynamicsWithPockets if has_pocket_mode else Dynamics
        dynamics = dynamics_class(
            in_node_nf=effective_in_node_nf, n_dims=n_dims, context_node_nf=context_node_nf,
            device=torch_device, hidden_nf=hidden_nf, activation=activation, n_layers=n_layers,
            attention=attention, tanh=tanh, norm_constant=norm_constant, inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding, normalization_factor=normalization_factor,
            aggregation_method=aggregation_method, model=model, normalization=normalization,
            centering=inpainting, graph_type=graph_type,
        )

        edm_class = InpaintingEDM if inpainting else EDM
        self.edm = edm_class(
            dynamics=dynamics, in_node_nf=effective_in_node_nf, n_dims=n_dims,
            timesteps=diffusion_steps, noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision, loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
            lambda_pocket_clash=lambda_pocket_clash,
            lambda_contact=lambda_contact,
            lambda_attach=lambda_attach,
            lambda_connectivity=lambda_connectivity,
            lambda_gf=lambda_gf,
            bond_class_weights=bond_class_weights,
        )

    def _select_dataset_class(self):
        if self.dataset_name in {'crossdock', 'crossdocked'}:
            if CrossDockDataset is None: raise RuntimeError("CrossDockDataset not found.")
            return CrossDockDataset
        if self.dataset_name in {'pdbbind'}:
            if PDBbindDataset is None: raise RuntimeError("PDBbindDataset not found.")
            return PDBbindDataset
        if ('crossdock' in self.train_data_prefix.lower()) or ('crossdock' in self.val_data_prefix.lower()):
            if CrossDockDataset is not None: return CrossDockDataset
        if ('PDBbind' in self.train_data_prefix) or ('PDBbind' in self.val_data_prefix):
            if PDBbindDataset is not None: return PDBbindDataset
            return MOADDataset
        if '.' in self.train_data_prefix: return MOADDataset
        return ZincDataset

    def _build_dataset(self, dataset_type, prefix: str, split: Optional[str] = None):
        kwargs = {
            'data_path': self.data_path,
            'prefix': prefix,
            'device': self.torch_device,
        }
        if (
            (PDBbindDataset is not None and dataset_type is PDBbindDataset)
            or (CrossDockDataset is not None and dataset_type is CrossDockDataset)
        ):
            kwargs['split'] = split or 'train'
        return dataset_type(**kwargs)

    def setup(self, stage: Optional[str] = None):
        dataset_type = self._select_dataset_class()
        if stage == 'fit':
            self.train_dataset = self._build_dataset(dataset_type, self.train_data_prefix, split='train')
            self.val_dataset = self._build_dataset(dataset_type, self.val_data_prefix, split='val')
        elif stage == 'val':
            self.val_dataset = self._build_dataset(dataset_type, self.val_data_prefix, split='val')

    def train_dataloader(self, collate_fn=collate_with_fragment_edges):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self, collate_fn=collate_with_fragment_edges):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=collate_with_fragment_edges):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn)

    @staticmethod
    def _ensure_3d_mask(x):
        if x is None: return None
        if x.dim() == 2: return x.unsqueeze(-1)
        if x.dim() == 3: return x
        return x.view(x.size(0), x.size(1), -1)[:, :, :1]

    @staticmethod
    def _get_rot_from_batch(data, bs, n_nodes, device):
        rot = data.get('frag_rotations', None)
        if rot is None: rot = data.get('rotations', None)
        if rot is None: rot = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(bs, n_nodes, 1, 1)
        return rot

    @staticmethod
    def _get_pocket_pose_from_batch(data):
        pocket_pos = data.get('pocket_pos', None)
        pocket_rot = data.get('pocket_rot', None)
        if pocket_pos is None and 'pocket_frames' in data:
            pocket_frames = data['pocket_frames']
            pocket_pos = pocket_frames[..., :3, 3]
            pocket_rot = pocket_frames[..., :3, :3]
        return pocket_pos, pocket_rot

    def forward(self, data, training: bool):
        frag_pos = data['positions']
        bs, n_nodes = frag_pos.shape[0], frag_pos.shape[1]
        device = frag_pos.device
        # if training and (self.global_step < 5 or self.global_step % 100 == 0):
        #     print(f"\n--- 🔍 DEBUG (Step {self.global_step}) ---")
        #     print(f"  positions(ligand): {frag_pos.shape}")
        #     print(f"  keys: {list(data.keys())[:20]} ...")
        #     if 'pocket_frames' in data:
        #         print(f"  pocket_frames: {data['pocket_frames'].shape}")
        #     print("----------------------------------")

        node_mask = self._ensure_3d_mask(data.get('atom_mask', None))
        if node_mask is None:
            node_mask = torch.ones(bs, n_nodes, 1, device=device)
        slot_mask = self._ensure_3d_mask(data.get('slot_mask', None))
        if slot_mask is None:
            slot_mask = torch.ones_like(node_mask)
        fragment_mask = self._ensure_3d_mask(data.get('fragment_mask', None))
        linker_mask = data.get('linker_mask', None)
        if linker_mask is None:
            linker_mask = node_mask
        else:
            linker_mask = self._ensure_3d_mask(linker_mask)
        edge_mask = (slot_mask @ slot_mask.transpose(1, 2)).unsqueeze(-1)

        frag_rot = data.get('frag_rotations', None)
        if frag_rot is None:
            frag_rot = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(bs, n_nodes, 1, 1)

        frag_feat = data['one_hot']

        pocket_pos, pocket_rot = self._get_pocket_pose_from_batch(data)
        pocket_mask = self._ensure_3d_mask(data.get('pocket_mask', None))
        pocket_element = data.get('pocket_element', None)
        pocket_aa_type = data.get('pocket_aa_type', None)
        pocket_is_backbone = data.get('pocket_is_backbone', None)
        true_bonds = data.get('true_bonds', None)
        true_attach_src = data.get('true_attach_src', None)
        true_attach_dst = data.get('true_attach_dst', None)
        true_fragment_types = data.get('fragment_type_ids', None)

        center_of_mass_mask = fragment_mask if self.center_of_mass == 'fragments' else node_mask
        if center_of_mass_mask.sum() < 1e-6:
            center_of_mass_mask = node_mask
        x, pocket_pos, _ = center_ligand_and_pocket(
            frag_pos,
            center_of_mass_mask,
            pocket_pos=pocket_pos,
            pocket_mask=pocket_mask,
        )
        x = x * node_mask
        if self.clash_warmup_epochs <= 0:
            clash_weight = 1.0
        else:
            clash_weight = min(1.0, float(self.current_epoch + 1) / float(self.clash_warmup_epochs))

        return self.edm.forward(
            x=x, rot=frag_rot, h=frag_feat, node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, pocket_pos=pocket_pos, pocket_rot=pocket_rot,
            pocket_mask=pocket_mask, pocket_element=pocket_element,
            pocket_aa_type=pocket_aa_type,
            pocket_is_backbone=pocket_is_backbone,
            true_bonds=true_bonds,
            true_attach_src=true_attach_src,
            true_attach_dst=true_attach_dst,
            true_fragment_types=true_fragment_types,
            slot_mask=slot_mask,
            clash_weight=clash_weight,
        )

    def training_step(self, data, *args):
        if data is None: return None
        try:
            bs = data['positions'].shape[0] if 'positions' in data else 0

            
            is_debug_step = (self.global_step < 5 or self.global_step % 100 == 0)

            # ==============================================================
            
            # ==============================================================
            if is_debug_step and 'positions' in data:
                print(f"\n🚀 [Training] Step {self.global_step} ... Batch={bs}")
                print(f"--- 🔍 DEBUG (Step {self.global_step}) ---")
                print(f"  positions(ligand): {data['positions'].shape}")
                print(f"  keys: {list(data.keys())[:20]} ...")
                if 'pocket_frames' in data:
                    print(f"  pocket_frames: {data['pocket_frames'].shape}")
                print("----------------------------------")

            
            loss, loss_dict = self.forward(data, training=True)
            if not torch.isfinite(loss):
                bad_metrics = [
                    name
                    for name, value in loss_dict.items()
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                ]
                uuids = data.get('uuid', [])
                raise FloatingPointError(
                    f"Non-finite training loss at global_step={self.global_step}; "
                    f"bad_metrics={bad_metrics}; lmdb_keys={uuids}"
                )

            # ==============================================================
            
            # ==============================================================
            loss_total = float(loss_dict.get('loss', 0))
            loss_x = float(loss_dict.get('loss_x', 0))
            loss_rot = float(loss_dict.get('loss_rot', 0))
            loss_bond = float(loss_dict.get('loss_bond', 0))
            loss_gf = float(loss_dict.get('loss_gf', 0))
            loss_type = float(loss_dict.get('loss_type', 0))
            loss_attach = float(loss_dict.get('loss_attach', 0))
            loss_n = float(loss_dict.get('loss_n', 0))

            log_str = f"🚀 [Training] Step {self.global_step:05d} ... Batch={bs} | Tot: {loss_total:.3f} | X: {loss_x:.3f} | Rot: {loss_rot:.3f} | Type: {loss_type:.3f} | Bond: {loss_bond:.3f} | GF: {loss_gf:.3f}     "

            if is_debug_step:
                
                print(log_str)
            else:
                
                print(f"\r{log_str}", end='', flush=True)

            
            if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
                for metric_name, metric in loss_dict.items():
                    if isinstance(metric, torch.Tensor):
                        self.log(f'{metric_name}/train', float(metric.detach().cpu()), prog_bar=True)
                    else:
                        self.log(f'{metric_name}/train', float(metric), prog_bar=True)

            return loss

        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nWarning: CUDA out of memory; clearing the cache and skipping this batch...")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                return None
            else:
                import traceback
                print("\n" + "=" * 50)
                print("Unhandled RuntimeError:")
                traceback.print_exc()
                print("=" * 50 + "\n")
                raise e

    def validation_step(self, data, *args):
        if data is None: return None
        try:
            loss, loss_dict = self.forward(data, training=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at global_step={self.global_step}"
                )
            self.log('loss/val', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            return loss_dict
        except FloatingPointError:
            raise
        except Exception:
            return None

    def test_step(self, data, *args):
        if data is None: return None
        try:
            loss, loss_dict = self.forward(data, training=False)
            return loss_dict
        except Exception:
            return None

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        if (self.current_epoch + 1) % self.test_epochs == 0:
            self.sample_and_analyze(self.val_dataloader())
            self.log('val_check', 1.0, prog_bar=True)

    def test_epoch_end(self, test_step_outputs):
        pass

    def sample_and_analyze(self, dataloader):
        max_sampling_batches = 2
        for b, data in tqdm(enumerate(dataloader), total=max_sampling_batches, desc='Sampling'):
            if b >= max_sampling_batches: break
            if data is None: continue
            try:
                chain_batch, node_mask, _, _, _ = self.sample_chain(data, keep_frames=self.FRAMES)
            except Exception as e:
                continue
        return {}

    def sample_chain(self, data, sample_fn=None, keep_frames=None, edge_mask=None, context=None, save_path=None,
                     guidance_fn=None, temperature=1.0, top_k=None,
                     clash_guidance_scale=0.0, clash_guidance_start=0,
                     clash_guidance_end=100, clash_guidance_max_norm=0.2,
                     clash_guidance_global_fraction=0.25,
                     clash_guidance_max_com_shift=1.0,
                     return_bond_probs=False):
        x = data['positions']
        bs, n_nodes = x.shape[0], x.shape[1]
        device = x.device

        node_mask = self._ensure_3d_mask(data.get('atom_mask', None))
        if node_mask is None: node_mask = torch.ones(bs, n_nodes, 1, device=device)
        slot_mask = self._ensure_3d_mask(data.get('slot_mask', None))
        if slot_mask is None:
            slot_mask = torch.ones_like(node_mask)

        fragment_mask = data.get('fragment_mask', None)
        if fragment_mask is None:
            fragment_mask = torch.zeros_like(node_mask)
        else:
            fragment_mask = self._ensure_3d_mask(fragment_mask)

        linker_mask = data.get('linker_mask', None)
        if linker_mask is None:
            linker_mask = node_mask - fragment_mask
        else:
            linker_mask = self._ensure_3d_mask(linker_mask)

        if edge_mask is None:
            edge_mask = (slot_mask @ slot_mask.transpose(1, 2)).unsqueeze(-1)

        h = data['one_hot']

        rot = self._get_rot_from_batch(data, bs, n_nodes, device=device)
        pocket_pos, pocket_rot = self._get_pocket_pose_from_batch(data)
        pocket_mask = self._ensure_3d_mask(data.get('pocket_mask', None))
        pocket_element = data.get('pocket_element', None)
        pocket_aa_type = data.get('pocket_aa_type', None)
        pocket_is_backbone = data.get('pocket_is_backbone', None)

        center_of_mass_mask = fragment_mask if self.center_of_mass == 'fragments' else node_mask
        if center_of_mass_mask.sum() < 1e-6:
            center_of_mass_mask = node_mask
        x_centered, pocket_pos, mean_pos = center_ligand_and_pocket(
            x,
            center_of_mass_mask,
            pocket_pos=pocket_pos,
            pocket_mask=pocket_mask,
        )
        x_centered = x_centered * node_mask

        sample_result = self.edm.sample_chain(
            x=x_centered, h=h, rot=rot, node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context,
            pocket_pos=pocket_pos, pocket_rot=pocket_rot, pocket_mask=pocket_mask,
            pocket_element=pocket_element,
            pocket_aa_type=pocket_aa_type,
            pocket_is_backbone=pocket_is_backbone,
            keep_frames=keep_frames,
            guidance_fn=guidance_fn, temperature=temperature, top_k=top_k, slot_mask=slot_mask,
            clash_guidance_scale=clash_guidance_scale,
            clash_guidance_start=clash_guidance_start,
            clash_guidance_end=clash_guidance_end,
            clash_guidance_max_norm=clash_guidance_max_norm,
            clash_guidance_global_fraction=clash_guidance_global_fraction,
            clash_guidance_max_com_shift=clash_guidance_max_com_shift,
            return_bond_probs=return_bond_probs,
        )
        chain = sample_result[0]

        if chain is not None:
            chain[..., :3] = chain[..., :3] + mean_pos.unsqueeze(0)

        return (chain,) + sample_result[1:]

    def reconstruct_and_save(self, states, mask, save_path=None, batch_name="sample"):
        from src.const import IDEAL_COORDS, AA_BONDS, AA_NAME2IDX
        from rdkit import Chem
        import os
        import torch

        bs, n_nodes, _ = states.shape
        generated_mols =[]

        for b in range(bs):
            combined_mol = Chem.RWMol()
            node_atom_indices =[]
            all_final_coords =[]

            valid_node_count = int(mask[b].sum().item())
            if valid_node_count == 0: continue

            for i in range(n_nodes):
                if mask[b, i] == 0: continue
                t = states[b, i, 0:3].cpu()
                R = states[b, i, 3:12].reshape(3, 3).cpu()
                type_id = torch.argmax(states[b, i, 12:]).item()

                template = IDEAL_COORDS[type_id]
                bonds_info = AA_BONDS[type_id]
                name2idx = AA_NAME2IDX[type_id]
                current_node_atoms =[]

                for atom_info in template:
                    raw_name = atom_info[0].strip()  
                    elem_symbol = "".join([c for c in raw_name if c.isalpha()])
                    
                    if len(elem_symbol) > 1 and elem_symbol.upper() not in ["CL", "BR"]:
                        elem_symbol = elem_symbol[0]
                    if elem_symbol == 'H': continue

                    local_pos = torch.tensor(atom_info[2], dtype=torch.float32)
                    global_pos = torch.matmul(R, local_pos) + t

                    try:
                        rd_idx = combined_mol.AddAtom(Chem.Atom(elem_symbol))
                        current_node_atoms.append(rd_idx)
                        all_final_coords.append(global_pos.numpy())
                    except:
                        continue

                if current_node_atoms: node_atom_indices.append(current_node_atoms)
                for start_name, end_name in bonds_info:
                    try:
                        if start_name in name2idx and end_name in name2idx: pass
                    except:
                        continue

            num_atoms = combined_mol.GetNumAtoms()
            if num_atoms == 0: continue

            final_mol = combined_mol.GetMol()
            conf = Chem.Conformer(num_atoms)
            for idx, coord in enumerate(all_final_coords):
                conf.SetAtomPosition(idx, coord.astype(float))
            final_mol.AddConformer(conf)

            try:
                final_mol = Chem.RemoveHs(final_mol)
                Chem.SanitizeMol(final_mol)
                final_mol = Chem.AddHs(final_mol, addCoords=True)
            except Exception as e:
                print(f"  Chemical validation failed: {e}; saving the original coordinates.")

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                file_name = f"{batch_name}_b{b}.sdf"
                out_file = os.path.join(save_path, file_name)
                writer = Chem.SDWriter(out_file)
                writer.write(final_mol)
                writer.close()
            generated_mols.append(final_mol)
        return generated_mols

    def configure_optimizers(self):
        pocket_parameters = []
        rigid_parameters = []
        head_parameters = []
        backbone_parameters = []
        for name, parameter in self.edm.named_parameters():
            if "pocket_feature_encoder" in name or "node_role_embedding" in name:
                pocket_parameters.append(parameter)
            elif "gcl_equiv.rigid_mlp" in name:
                rigid_parameters.append(parameter)
            elif any(
                token in name
                for token in (
                    "edge_predictor",
                    "attach_src_predictor",
                    "attach_dst_predictor",
                    "presence_predictor",
                )
            ):
                head_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": self.lr},
                {
                    "params": head_parameters,
                    "lr": self.lr * self.bond_head_lr_multiplier,
                },
                {
                    "params": rigid_parameters,
                    "lr": self.lr * self.rigid_head_lr_multiplier,
                },
                {
                    "params": pocket_parameters,
                    "lr": self.lr * self.pocket_encoder_lr_multiplier,
                },
            ],
            amsgrad=True,
            weight_decay=1e-12,
        )

    def on_train_epoch_start(self):
        freeze_backbone = self.current_epoch < self.corrective_freeze_epochs
        trainable_tokens = (
            "pocket_feature_encoder",
            "node_role_embedding",
            "edge_predictor",
            "attach_src_predictor",
            "attach_dst_predictor",
            "presence_predictor",
            "gcl_equiv.rigid_mlp",
        )
        for name, parameter in self.edm.named_parameters():
            parameter.requires_grad = (
                any(token in name for token in trainable_tokens)
                if freeze_backbone
                else True
            )

    def compute_best_validation_metrics(self):
        return {}, 0

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
