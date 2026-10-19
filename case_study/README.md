# Case study：run12 / epoch 192 / seed 3407

所有命令在 Linux/WSL 的 RigiDiff 仓库根目录执行。配置入口为
`case_study/config.json`。`project_root: "."` 相对于仓库根目录解析，
不依赖终端当前目录，不再使用旧的 `E:/fragdiff1/fragdiff`。

## 输入文件

配置与本次用户提供的生成和全蛋白对接命令对应：

```text
samples/run12_epoch192_formal_test99_100perpocket_seed3407/
vina_openbabel_test99_seed3407.csv
vina_openbabel_test99_seed3407_per_target.csv
openbabel_test99_seed3407_manifest.csv
work_vina_openbabel_test99_seed3407/
  prepared_receptors/
  ligands_pdbqt/
  vina_outputs/
data/crossdocked_pocket10/
```

如果结果保存在其他位置，修改配置中的 `vina_csv`、`vina_work_dir`、
`data_root` 等对应路径；CSV 和工作目录必须来自同一次实验。
只复制 CSV 不足以完成构象与相互作用分析。不自动回退到 seed 2025。
对接脚本的旧 test99 划分与新版 val1000_fixed 划分具有相同测试条目和顺序。

`target_selection/pdb_metadata.csv` 沿用原工作区已有的三个靶点注释，
不是新生成结果，也不是针对所有 99 个靶点的完整注释库。
主要 DPP4 分析目录改为 `DPP4_2OQV_target48_seed3407`，避免复用旧分析结果。

## 运行顺序

```bash
conda activate fragdiff
python -m unittest discover -s tests -p 'test_case_study_config.py'

# 从同一批新分子重新计算性质，不使用旧实验的同名指标文件。
python -c "from evaluate.metrics import evaluate_folder; evaluate_folder('samples/run12_epoch192_formal_test99_100perpocket_seed3407/candidates', 'metrics_standard_sa_seed3407.csv')"

python case_study/select_case_target.py

# 以下继续分析原有 DPP4 靶点；先检查新数据的靶点选择结果。
python case_study/prepare_case_poses.py --target-id 48
python case_study/redock_case_target.py --target-id 48
python case_study/analyze_clashes.py --target-id 48
python case_study/select_case_molecules.py --target-id 48
python case_study/analyze_interactions.py --target-id 48
python case_study/case_property_table.py --target-id 48
python case_study/generate_pymol_case_study.py --target-id 48
python case_study/validate_case_study.py --target-id 48
```

性质计算的 candidates 路径必须是完整新实验分子目录，不是仅含少量分子的测试目录。
`select_case_target.py` 可以在没有标准性质 CSV 时用对接表内的性质进行部分汇总；
后续 `select_case_molecules.py` 需要 `metrics_standard_sa_seed3407.csv`。

新颖性计算还需要 `training_smiles` 指向新训练划分对应的规范 SMILES 文件。
不要将旧训练集的 SMILES 当作新划分使用；尚未准备时可显式给
`select_case_molecules.py` 加 `--skip-training-novelty`，但不能报告训练集新颖性结论。

## 工具与评价范围

`obabel`、`vina` 从激活环境的 PATH 查找，也可在配置中填写可执行文件路径。
PyMOL 脚本生成后，在安装 PyMOL 的环境中打开；不会自动启动 GUI。

全蛋白主实验的 Vina 分数直接读取 `vina_openbabel_test99_seed3407.csv`。
现有 case-study 管线另有**局部口袋重对接**步骤，仍使用原来的局部设置
（exhaustiveness 8、参考配体盒 padding 5、每任务 CPU 1），本次仅将其 seed
改为 3407 并修复工具路径。它不是主实验的全蛋白盲对接（exhaustiveness 64、
padding 6、CPU 8），两种分数不能混写。本次未改变候选分子的筛选公式或阈值。

checkpoint、split、generation_dir、generation_seed 等配置项记录实验来源；
这些 case-study 脚本不会重新训练或生成分子。必须先完成相应的生成和对接。
