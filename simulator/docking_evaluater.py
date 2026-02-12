#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

def read_inputs(path: str) -> List[Dict[str, str]]:

    rows = []
    if path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            assert "id" in r.fieldnames and "smiles" in r.fieldnames, "CSV需包含表头 id,smiles"
            for row in r:
                rows.append({"id": row["id"], "smiles": row["smiles"]})
    elif path.endswith('.smi'):
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    rows.append({"id": f"mol_{i+1}", "smiles": line})
    else:
        raise ValueError("输入文件必须是.csv或.smi格式")
    return rows

def smiles_to_3d_sdf(smiles: str, out_sdf: str) -> bool:

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成3D构象
        conf = AllChem.EmbedMolecule(mol, randomSeed=42)
        if conf != 0:
            return False
        
        # 优化构象
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 保存为SDF
        writer = Chem.SDWriter(out_sdf)
        writer.write(mol)
        writer.close()
        
        return True
    except Exception as e:
        print(f"RDKit转换失败: {e}")
        return False

def sdf_to_pdbqt(in_sdf: str, out_pdbqt: str) -> Tuple[bool, str]:

    cmd = ["obabel", in_sdf, "-O", out_pdbqt, "--partialcharge", "gasteiger"]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0, r.stderr
    except FileNotFoundError:
        return False, "obabel命令未找到，请安装OpenBabel"

def run_vina(receptor_pdbqt: str, ligand_pdbqt: str, box_json: str, 
             exhaustiveness: int = 8, num_modes: int = 9, 
             vina_bin: str = "vina") -> Tuple[bool, float, int, str]:

    # 读取box配置
    with open(box_json, 'r') as f:
        box = json.load(f)
    
    # 创建临时输出文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
        out_pose = f.name
    
    try:
        # 构建vina命令
        cmd = [
            vina_bin,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--out", out_pose,
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes)
        ]
        
        # 运行vina
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return False, 0.0, 0, f"Vina运行失败: {result.stderr}"
        
        # 解析结合能
        affinity, rank = _parse_affinity_from_pose(out_pose)
        if affinity is None:
            affinity, rank = _parse_affinity_from_text(result.stdout + result.stderr)
        
        return True, affinity or 0.0, rank or 0, ""
        
    except subprocess.TimeoutExpired:
        return False, 0.0, 0, "Vina运行超时"
    except Exception as e:
        return False, 0.0, 0, f"Vina运行异常: {e}"
    finally:
        # 清理临时文件
        if os.path.exists(out_pose):
            os.unlink(out_pose)

def _parse_affinity_from_pose(pose_path: str) -> Tuple[Optional[float], Optional[int]]:

    try:
        with open(pose_path, 'r') as f:
            for line in f:
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    # 格式: REMARK VINA RESULT: -9.156 0.000 0.000
                    # 第一个数字是结合能，第二个是RMSD
                    if len(parts) >= 4:
                        try:
                            affinity = float(parts[3])  # 第4个元素是结合能
                            rank = 1  # 第一个结果排名为1
                            return affinity, rank
                        except (ValueError, IndexError):
                            continue
    except Exception:
        pass
    return None, None

def _parse_affinity_from_text(text: str) -> Tuple[Optional[float], Optional[int]]:

    lines = text.split('\n')
    for line in lines:
        if 'REMARK VINA RESULT:' in line:
            parts = line.split()
            # 格式: REMARK VINA RESULT: -9.156 0.000 0.000
            # 第一个数字是结合能，第二个是RMSD
            if len(parts) >= 4:
                try:
                    affinity = float(parts[3])  # 第4个元素是结合能
                    rank = 1  # 第一个结果排名为1
                    return affinity, rank
                except (ValueError, IndexError):
                    continue
    return None, None

def main():
    parser = argparse.ArgumentParser(description="分子对接脚本（带进度条）")
    parser.add_argument("--input", required=True, help=".smi 或包含 id,smiles 的 .csv")
    parser.add_argument("--out_csv", required=True, help="输出CSV文件")
    parser.add_argument("--targets", nargs="+", required=True, 
                       help="格式 PDBID:receptor.pdbqt:box.json，多靶点空格分隔")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="搜索强度")
    parser.add_argument("--num_modes", type=int, default=9, help="生成构象数")
    parser.add_argument("--tmpdir", default=None, help="临时目录")
    parser.add_argument("--vina_bin", default="vina", help="Vina可执行文件路径")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 检查靶点文件
    targets = []
    for target_str in args.targets:
        parts = target_str.split(":")
        if len(parts) != 3:
            print(f"错误：靶点格式错误: {target_str}")
            sys.exit(1)
        
        pdbid, receptor, box = parts
        if not os.path.exists(receptor):
            print(f"错误：受体文件不存在: {receptor}")
            sys.exit(1)
        if not os.path.exists(box):
            print(f"错误：box文件不存在: {box}")
            sys.exit(1)
        
        targets.append((pdbid, receptor, box))
    
    # 读取输入分子
    print("读取输入分子...")
    rows = read_inputs(args.input)
    print(f"共读取 {len(rows)} 个分子")
    
    # 创建临时目录
    if args.tmpdir:
        tmpdir = Path(args.tmpdir)
        tmpdir.mkdir(exist_ok=True)
    else:
        tmpdir = Path(tempfile.mkdtemp())
    
    print(f"临时目录: {tmpdir}")
    
    # 准备输出CSV
    fieldnames = ["id", "smiles"]
    for pdbid, _, _ in targets:
        fieldnames.extend([f"{pdbid}_vina_kcalmol", f"{pdbid}_rank"])
    
    # 添加选择性列
    if len(targets) >= 2:
        fieldnames.append("COX2_minus_COX1_kcalmol")
    
    # 写入表头
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # 处理每个分子
    print("开始分子对接...")
    successful_dockings = 0
    failed_dockings = 0
    
    with tqdm(total=len(rows), desc="分子对接进度") as pbar:
        for i, row in enumerate(rows):
            mol_id = row["id"]
            smiles = row["smiles"]
            
            # 创建临时文件
            lig_sdf = tmpdir / f"{mol_id}.sdf"
            lig_pdbqt = tmpdir / f"{mol_id}.pdbqt"
            
            # 转换SMILES到3D SDF
            if not smiles_to_3d_sdf(smiles, str(lig_sdf)):
                print(f"警告：分子 {mol_id} SMILES转换失败")
                failed_dockings += 1
                pbar.update(1)
                continue
            
            # 转换SDF到PDBQT
            success, error_msg = sdf_to_pdbqt(str(lig_sdf), str(lig_pdbqt))
            if not success:
                print(f"警告：分子 {mol_id} PDBQT转换失败: {error_msg}")
                failed_dockings += 1
                pbar.update(1)
                continue
            
            # 对每个靶点进行对接
            result_row = {"id": mol_id, "smiles": smiles}
            target_results = []
            
            for pdbid, receptor, box in targets:
                success, affinity, rank, error_msg = run_vina(
                    receptor, str(lig_pdbqt), box, 
                    args.exhaustiveness, args.num_modes, args.vina_bin
                )
                
                if success:
                    result_row[f"{pdbid}_vina_kcalmol"] = affinity
                    result_row[f"{pdbid}_rank"] = rank
                    target_results.append(affinity)
                else:
                    result_row[f"{pdbid}_vina_kcalmol"] = ""
                    result_row[f"{pdbid}_rank"] = ""
                    print(f"警告：分子 {mol_id} 与 {pdbid} 对接失败: {error_msg}")
            
            # 计算选择性
            if len(target_results) >= 2:
                result_row["COX2_minus_COX1_kcalmol"] = target_results[1] - target_results[0] if len(target_results) > 1 else ""
            
            # 写入结果
            with open(args.out_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result_row)
            
            successful_dockings += 1
            pbar.update(1)
            
            # 清理临时文件
            if lig_sdf.exists():
                lig_sdf.unlink()
            if lig_pdbqt.exists():
                lig_pdbqt.unlink()
    
    # 清理临时目录
    if not args.tmpdir:
        import shutil
        shutil.rmtree(tmpdir)
    
    print(f"\n对接完成！")
    print(f"成功对接: {successful_dockings}")
    print(f"失败对接: {failed_dockings}")
    print(f"结果保存到: {args.out_csv}")

if __name__ == "__main__":
    main()
