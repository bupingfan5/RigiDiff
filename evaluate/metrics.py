import os
import glob
import re
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, AllChem

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RDLogger.DisableLog("rdApp.*")

# 引入合成可及性打分
try:
    from src import sascorer
except ImportError:
    sascorer = None


def calculate_sa_score(mol):
    """
    DiffGui 论文评估口径的 SA：将原始 Ertl SA Score(1~10，越低越好)
    转换为 0~1 区间的合成可及性指标（越高越好）。
    """
    if sascorer is None:
        return np.nan

    try:
        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if canonical_mol is None:
            canonical_mol = mol

        raw_sa = float(sascorer.calculateScore(canonical_mol))
        sa_score = (10.0 - raw_sa) / 9.0
        return round(float(np.clip(sa_score, 0.0, 1.0)), 2)
    except Exception:
        return np.nan


def count_lipinski_rules(mol):
    """
    计算符合 Lipinski 五规则的条数。
    为了对齐 TargetDiff / Pocket2Mol 等顶会论文的评测标准，
    此处加入第5条规则（可旋转键<=10），使满分达到 5.0。
    """
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        # 🌟 顶会评测脚本里用来凑满 5 分的潜规则
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        score = 0
        if mw <= 500: score += 1
        if logp <= 5: score += 1
        if hbd <= 5: score += 1
        if hba <= 10: score += 1
        if rot_bonds <= 10: score += 1

        return score
    except:
        return 0


def calculate_diversity(fps):
    """
    Calculate mean pairwise Tanimoto dissimilarity within one pocket.

    This matches the TargetDiff, PMDM, and GCDM evaluation protocol:
    default RDKit topological fingerprints and all unordered molecule pairs.
    """
    if len(fps) < 2:
        return 0.0

    dissimilarities = []
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        dissimilarities.extend(1.0 - sim for sim in sims)

    return float(np.mean(dissimilarities))


def extract_pocket_id(file_name):
    """Extract the pocket identifier from target_<id>_sample_<id>.sdf."""
    match = re.match(r"^target_(.+?)_sample_", file_name)
    if match is None:
        raise ValueError(
            "Cannot infer pocket ID from SDF filename "
            f"'{file_name}'. Expected target_<id>_sample_<id>.sdf."
        )
    return match.group(1)


def evaluate_folder(sdf_dir, output_csv="metrics_standard_sa.csv"):
    # 检查路径
    if not os.path.exists(sdf_dir):
        print(f"[Error] 路径不存在: {sdf_dir}")
        return

    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    print(f"[Start] 找到 {len(sdf_files)} 个分子文件，开始标准评估...")

    results = []
    fps_by_pocket = defaultdict(list)
    result_indices_by_pocket = defaultdict(list)

    # ==========================================
    # 第一阶段：提取基础理化属性、拓扑信息和 RDKit 拓扑指纹
    # ==========================================
    for f in tqdm(sdf_files, desc="1/2 评估基础理化属性", file=sys.stdout):
        file_name = os.path.basename(f)
        pocket_id = extract_pocket_id(file_name)

        suppl = Chem.SDMolSupplier(f, sanitize=False)

        for mol in suppl:
            if mol is None: continue

            try:

                Chem.SanitizeMol(mol)

                # 计算 DiffGui 论文口径的 SA：0~1，越高越好
                sa_val = calculate_sa_score(mol)

                # 计算常规药效学指标
                qed_val = QED.qed(mol)
                lipinski_score = count_lipinski_rules(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)

                # 🌟 新增：统计成环个数
                num_rings = mol.GetRingInfo().NumRings()

                # 🌟 新增：统计化学键及其类型
                num_bonds = mol.GetNumBonds()
                bond_single = 0
                bond_double = 0
                bond_triple = 0
                bond_aromatic = 0

                for bond in mol.GetBonds():
                    bt = bond.GetBondType()
                    if bt == Chem.rdchem.BondType.SINGLE:
                        bond_single += 1
                    elif bt == Chem.rdchem.BondType.DOUBLE:
                        bond_double += 1
                    elif bt == Chem.rdchem.BondType.TRIPLE:
                        bond_triple += 1
                    elif bt == Chem.rdchem.BondType.AROMATIC:
                        bond_aromatic += 1

                # 使用与 TargetDiff、PMDM 和 GCDM 一致的默认 RDKit 拓扑指纹
                fp = Chem.RDKFingerprint(mol)
                result_index = len(results)
                fps_by_pocket[pocket_id].append(fp)
                result_indices_by_pocket[pocket_id].append(result_index)

                results.append({
                    "file": file_name,
                    "QED": qed_val,
                    "SA_Score": sa_val,
                    "Lipinski": lipinski_score,
                    "MW": mw,
                    "LogP": logp,
                    "Atoms": mol.GetNumAtoms(),
                    "Rings": num_rings,  # 新增写入表格
                    "Total_Bonds": num_bonds,  # 新增写入表格
                    "Single_Bonds": bond_single,  # 新增写入表格
                    "Double_Bonds": bond_double,  # 新增写入表格
                    "Triple_Bonds": bond_triple,  # 新增写入表格
                    "Aromatic_Bonds": bond_aromatic  # 新增写入表格
                })
            except Exception as e:
                continue

    if not results:
        print("[Warning] 未成功评估任何分子，请检查 SDF 文件是否有效。")
        return

    # ==========================================
    # 第二阶段：逐口袋计算平均两两 Tanimoto 距离并写入表格
    # ==========================================
    print("\n[Progress] 2/2 正在计算逐口袋结构多样性 (Diversity)...")
    per_pocket_diversity = {}
    for pocket_id, pocket_fps in tqdm(
        fps_by_pocket.items(),
        desc="Calculating Diversity",
        file=sys.stdout,
    ):
        diversity = calculate_diversity(pocket_fps)
        per_pocket_diversity[pocket_id] = diversity
        for result_index in result_indices_by_pocket[pocket_id]:
            results[result_index]["Diversity"] = diversity

    # 存入 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # ==========================================
    # 第三阶段：打印终端统计报表
    # ==========================================
    diversity_values = np.asarray(list(per_pocket_diversity.values()), dtype=float)
    macro_diversity = float(np.mean(diversity_values))
    median_diversity = float(np.median(diversity_values))
    std_diversity = float(np.std(diversity_values))

    print("\n" + "=" * 55)
    print("论文统计报表 (Standard Metrics)")
    print("=" * 55)
    print(f"总成功样本数:     {len(results)}")
    print(f"平均 QED (类药性):   {df['QED'].mean():.4f}")
    print(f"平均 SA (0~1 合成可及性): {df['SA_Score'].mean():.4f} (越高越好)")
    print(f"平均 Lipinski 得分:   {df['Lipinski'].mean():.2f} / 5.0")
    print(f"平均分子量 (MW):      {df['MW'].mean():.2f}")
    print(f"平均成环数 (Rings):   {df['Rings'].mean():.2f}")
    print(f"平均化学键数:        {df['Total_Bonds'].mean():.2f}")
    print(f"逐口袋结构多样性（宏平均）: {macro_diversity:.4f} (接近1越好)")
    print(f"逐口袋结构多样性（中位数）: {median_diversity:.4f}")
    print(f"逐口袋结构多样性（标准差）: {std_diversity:.4f}")
    print(f"参与 Diversity 汇总的口袋数: {len(per_pocket_diversity)}")
    print("=" * 55)
    print(f"完整数据已保存至: {output_csv}")


if __name__ == "__main__":
    # 确保此路径下有你刚才生成的 SDF 文件
    SDF_PATH = "samples/final run1/candidates"
    evaluate_folder(SDF_PATH)
