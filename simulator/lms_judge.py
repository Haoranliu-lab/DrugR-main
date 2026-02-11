#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
    LMS_JUDGE_AVAILABLE = True
except ImportError:
    LMS_JUDGE_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class LMSJudgeV2Config:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 800
    timeout: int = 60
    base_url: str = ""
    api_key: str = ""


class ChemicalCorrectnessChecker:
    
    def __init__(self):
        self.common_errors = {
            "physicochemical": ["solubility", "lipophilicity", "logP", "pKa", "polarity"],
            "pharmacokinetics": ["ADMET", "CYP", "P-gp", "BSEP", "metabolism", "absorption", "distribution", "excretion"],
            "molecular_properties": ["molecular_weight", "hydrogen_bonding", "rotatable_bonds", "TPSA", "aromaticity"],
            "toxicity": ["hERG", "DILI", "hepatotoxicity", "cardiotoxicity", "genotoxicity"]
        }
    
    def check_chemical_correctness(self, reasoning: str, orig_smiles: str, 
                                   opt_smiles: str, client: OpenAI) -> Dict[str, Any]:

        # 提取原始分子和优化分子的关键特征（用于交叉验证）
        orig_features = self._extract_molecular_features(orig_smiles)
        opt_features = self._extract_molecular_features(opt_smiles)
        
        # 格式化分子特征（安全处理）
        def format_feature(val, fmt=':.2f'):
            if val == 'N/A' or val is None:
                return 'N/A'
            try:
                if fmt == ':.2f':
                    return f"{float(val):.2f}"
                else:
                    return str(val)
            except:
                return 'N/A'
        
      
        try:
            response = client.chat.completions.create(
                model=self.config.model if hasattr(self, 'config') else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
                timeout=60
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logging.warning(f"化学正确性检查失败: {e}")
        
        return {
            "physical_chemistry_correctness": 0.5,
            "pharmacokinetics_correctness": 0.5,
            "structure_property_correctness": 0.5,
            "feature_consistency": 0.5,
            "overall_scientific_rigor": 0.5,
            "has_major_error": False,
            "error_examples": [],
            "brief_assessment": "评估失败"
        }
    
    def _extract_molecular_features(self, smiles: str) -> Dict[str, float]:
        """提取分子特征用于验证"""
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "rings": Descriptors.RingCount(mol)
            }
        except:
            return {}


class LogicalCoherenceChecker:
    """逻辑连贯性检查器"""
    
    def check_logical_coherence(self, reasoning: str, orig_smiles: str, 
                               opt_smiles: str, client: OpenAI) -> Dict[str, Any]:
        """
        检查逻辑连贯性
        核心问题：问题识别→解决方案→SMILES修改是否对应一致？
        """
        # 尝试解析SMILES差异
        structural_diff = self._analyze_structural_changes(orig_smiles, opt_smiles)
        
        prompt = f"""
你是一位逻辑分析专家。请评估以下分子优化推理的**逻辑连贯性**。

**原始分子**: {orig_smiles}
**优化后分子**: {opt_smiles}

**结构变化**:
{structural_diff}

**推理过程**:
{reasoning}

---

请评估以下三个核心问题：

### 1. 问题识别 → 解决方案的对应性 (0-1分，1分=完全对应)
- 识别的问题是否清晰？
- 提出的解决方案是否针对这些问题？
- 是否存在"提出问题A，但解决方案针对问题B"的情况？

**举例**：
✓ 好："识别hERG风险 → 减少芳香性降低π-π作用"
✗ 差："识别hERG风险 → 增加分子量"（方向错误）

### 2. 解决方案 → SMILES修改的对应性 (0-1分，1分=完全对应)
- 提出的修改策略是否在SMILES中体现？
- SMILES的实际修改是否符合策略描述？
- 是否存在"说要加基团，实际却删除基团"的情况？

**举例**：
✓ 好："移除氟原子降低亲脂性 → 实际SMILES中F被H替换"
✗ 差："增加极性基团 → 实际SMILES中删除了羟基"

### 3. 整体逻辑链完整性 (0-1分，1分=完整)
问题 → 策略 → 修改 → 预期效果，整个链条是否连贯？

### 4. 因果关系准确性 (0-1分，1分=准确)
"因为X所以Y"的推理是否合理？是否存在倒因为果？

### 5. 自相矛盾检测 (0-1分，1分=无矛盾)
推理过程中是否存在自相矛盾的表述？

---

**重要**：如果发现逻辑问题，请在`logic_issues`字段中给出**具体例子**。

请以JSON格式返回：
{{
    "problem_solution_alignment": 0.9,
    "solution_smiles_alignment": 0.8,
    "overall_logic_chain": 0.85,
    "causal_accuracy": 0.9,
    "contradiction_score": 0.95,
    "has_logic_flaw": false,
    "logic_issues": ["具体问题1", "具体问题2"],
    "brief_assessment": "简短总结（1-2句话）"
}}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
                timeout=60
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logging.warning(f"逻辑连贯性检查失败: {e}")
        
        return {
            "problem_solution_alignment": 0.5,
            "solution_smiles_alignment": 0.5,
            "overall_logic_chain": 0.5,
            "causal_accuracy": 0.5,
            "contradiction_score": 0.5,
            "has_logic_flaw": False,
            "logic_issues": [],
            "brief_assessment": "评估失败"
        }
    
    def _analyze_structural_changes(self, orig_smiles: str, opt_smiles: str) -> str:
        """分析结构变化（简化版）"""
        if not RDKIT_AVAILABLE:
            return "无法分析（RDKit未安装）"
        
        try:
            mol1 = Chem.MolFromSmiles(orig_smiles)
            mol2 = Chem.MolFromSmiles(opt_smiles)
            
            if mol1 is None or mol2 is None:
                return "SMILES无效"
            
            # 简单的特征对比
            changes = []
            
            mw1, mw2 = Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)
            if abs(mw1 - mw2) > 1:
                changes.append(f"分子量: {mw1:.1f} → {mw2:.1f}")
            
            logp1, logp2 = Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)
            if abs(logp1 - logp2) > 0.1:
                changes.append(f"logP: {logp1:.2f} → {logp2:.2f}")
            
            hbd1, hbd2 = Descriptors.NumHDonors(mol1), Descriptors.NumHDonors(mol2)
            if hbd1 != hbd2:
                changes.append(f"氢键供体: {hbd1} → {hbd2}")
            
            rings1, rings2 = Descriptors.NumAromaticRings(mol1), Descriptors.NumAromaticRings(mol2)
            if rings1 != rings2:
                changes.append(f"芳香环: {rings1} → {rings2}")
            
            return "\n".join(changes) if changes else "未检测到明显变化"
        except:
            return "分析失败"


class LMSJudgeV2:
    """LMS Judge V2主类 - 聚焦版"""
    
    def __init__(self, config: LMSJudgeV2Config = None):
        self.config = config or LMSJudgeV2Config()
        self.correctness_checker = ChemicalCorrectnessChecker()
        self.coherence_checker = LogicalCoherenceChecker()
        
        if not LMS_JUDGE_AVAILABLE:
            raise ImportError("OpenAI库未安装")
        
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
    
    def evaluate(self, reasoning: str, orig_smiles: str, opt_smiles: str) -> Dict[str, Any]:
        """
        聚焦评估：只关注两个核心问题
        1. 科学正确性
        2. 逻辑连贯性
        """
        # 1. 检查化学正确性
        correctness_result = self.correctness_checker.check_chemical_correctness(
            reasoning, orig_smiles, opt_smiles, self.client
        )
        
        # 2. 检查逻辑连贯性
        coherence_result = self.coherence_checker.check_logical_coherence(
            reasoning, orig_smiles, opt_smiles, self.client
        )
        
        # 3. 计算总分
        # 调整权重：逻辑连贯性更重要（可以区分模板 vs 真实推理）
        # 科学正确性 (40%) + 逻辑连贯性 (60%)
        correctness_score = correctness_result.get("overall_scientific_rigor", 0.5)
        coherence_score = coherence_result.get("overall_logic_chain", 0.5)
        
        overall_score = 0.40 * correctness_score + 0.60 * coherence_score
        
        # 4. 判断是否有严重问题
        has_critical_issue = (
            correctness_result.get("has_major_error", False) or
            coherence_result.get("has_logic_flaw", False) or
            correctness_score < 0.4 or
            coherence_score < 0.4
        )
        
        return {
            "overall_score": overall_score,
            "correctness_score": correctness_score,
            "coherence_score": coherence_score,
            "has_critical_issue": has_critical_issue,
            "correctness_details": correctness_result,
            "coherence_details": coherence_result,
            "summary": {
                "scientific_errors": correctness_result.get("error_examples", []),
                "logic_issues": coherence_result.get("logic_issues", []),
                "correctness_assessment": correctness_result.get("brief_assessment", ""),
                "coherence_assessment": coherence_result.get("brief_assessment", "")
            }
        }


def create_lms_judge_v2(config: LMSJudgeV2Config = None) -> LMSJudgeV2:
    """创建LMS Judge V2实例"""
    return LMSJudgeV2(config)


if __name__ == "__main__":
    # 测试用例
    test_reasoning = """
原始分子存在以下问题：
1. 分子量过大(450Da)，可能影响口服吸收
2. logP值过高(5.2)，亲脂性太强，水溶性差
3. 氢键供体数量过多(5个)，膜渗透性低

优化策略：
1. 移除外围甲基，降低分子量
2. 引入羟基增加极性，改善水溶性
3. 减少氢键供体，提高膜渗透性

最终优化：移除2个甲基基团，预期分子量降至420Da，logP降至4.5。
"""
    
    test_orig = "CC(C)CC1=CC=C(C=C1)C(C)CC(C)C(C)C"
    test_opt = "CC1=CC=C(C=C1)C(C)CC"
    
    if LMS_JUDGE_AVAILABLE:
        judge = create_lms_judge_v2()
        result = judge.evaluate(test_reasoning, test_orig, test_opt)
        
        print("=" * 60)
        print("LMS Judge V2 - 聚焦版评估结果")
        print("=" * 60)
        print(f"\n总分: {result['overall_score']:.4f}")
        print(f"  - 科学正确性: {result['correctness_score']:.4f}")
        print(f"  - 逻辑连贯性: {result['coherence_score']:.4f}")
        print(f"\n严重问题: {'是' if result['has_critical_issue'] else '否'}")
        
        if result['summary']['scientific_errors']:
            print("\n科学错误:")
            for err in result['summary']['scientific_errors']:
                print(f"  - {err}")
        
        if result['summary']['logic_issues']:
            print("\n逻辑问题:")
            for issue in result['summary']['logic_issues']:
                print(f"  - {issue}")
    else:
        print("LMS Judge V2不可用，请安装OpenAI库")
