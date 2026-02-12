# -*- coding: utf-8 -*-
# from sklearn.metrics import f1_score
try:
    from sklearn.metrics import f1_score
except ImportError:
    def f1_score(y_true, y_pred):

        if len(y_true) != len(y_pred):
            return 0.0
        
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            return 0.0
        else:
            return 2 * (precision * recall) / (precision + recall)
import json
import time
import re
import os
from typing import Dict, Any, List, Tuple, Set, Optional
from test2 import search_admet
from filter import checks, check_item



def simple_reasoning_quality_evaluation(reasoning: str) -> Dict[str, float]:

    if not reasoning or not reasoning.strip():
        return {
            "problem_identification": 0.0,
            "solution_quality": 0.0,
            "chemical_knowledge": 0.0,
            "optimization_strategy": 0.0,
            "overall_quality": 0.0,
            "lms_judge_available": False
        }
    
    reasoning_lower = reasoning.lower()
    

    problem_keywords = [
        "问题", "issue", "problem", "缺陷", "deficiency", "不足", "lack",
        "毒性", "toxicity", "毒性", "toxic", "副作用", "side effect",
        "代谢", "metabolism", "代谢稳定性", "metabolic stability",
        "渗透性", "permeability", "溶解度", "solubility",
        "清除率", "clearance", "半衰期", "half-life"
    ]
    

    solution_keywords = [
        "优化", "optimize", "改进", "improve", "增强", "enhance",
        "降低", "reduce", "减少", "decrease", "提高", "increase",
        "修饰", "modify", "替换", "replace", "添加", "add",
        "去除", "remove", "避免", "avoid", "防止", "prevent"
    ]
    

    chemical_keywords = [
        "官能团", "functional group", "基团", "group", "原子", "atom",
        "键", "bond", "环", "ring", "芳香", "aromatic", "脂肪", "aliphatic",
        "极性", "polar", "非极性", "nonpolar", "氢键", "hydrogen bond",
        "分子量", "molecular weight", "logp", "logd", "tpsa"
    ]
    

    strategy_keywords = [
        "策略", "strategy", "方法", "method", "途径", "approach",
        "设计", "design", "合成", "synthesis", "修饰", "modification",
        "结构", "structure", "构效关系", "structure-activity",
        "药效团", "pharmacophore", "先导化合物", "lead compound"
    ]
    
    def calculate_score(text, keywords):

        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / len(keywords) * 2, 1.0)
    

    problem_score = calculate_score(reasoning_lower, problem_keywords)
    solution_score = calculate_score(reasoning_lower, solution_keywords)
    chemical_score = calculate_score(reasoning_lower, chemical_keywords)
    strategy_score = calculate_score(reasoning_lower, strategy_keywords)
    

    length_bonus = min(len(reasoning) / 500, 0.2)
    

    overall_score = (problem_score + solution_score + chemical_score + strategy_score) / 4 + length_bonus
    overall_score = min(overall_score, 1.0)
    
    return {
        "problem_identification": problem_score,
        "solution_quality": solution_score,
        "chemical_knowledge": chemical_score,
        "optimization_strategy": strategy_score,
        "overall_quality": overall_score,
        "lms_judge_available": False,
        "evaluation_method": "simple_keyword_analysis"
    }

############################################################

############################################################


FEATURES = [
    "10肽", "Alarm_NMR Rule", "Acute Toxicity Rule", "AMES Toxicity", "Aquatic Toxicity Rule",
    "A549 Cytotoxicity", "BCF", "BCRP inhibitor", "BMS Rule", "BBB", "Boiling point",
    "Caco-2 Permeability", "Carcinogenicity", "Chelating Rule", "CL_plasma",
    "CYP1A2 inhibitor", "CYP1A2 substrate", "CYP2B6 inhibitor", "CYP2B6 substrate",
    "CYP2C19 inhibitor", "CYP2C19 substrate", "CYP2C8 inhibitor", "CYP2C9 inhibitor",
    "CYP2C9 substrate", "CYP2D6 inhibitor", "CYP2D6 substrate", "CYP3A4 inhibitor",
    "CYP3A4 substrate", "DILI", "Density", "Eye Corrosion", "Eye Irritation",
    "FDAMDD", "FLuc inhibitors", "Flexibility", "Fsp3", "F20%", "F30%", "F50%",
    "Fu", "GASA", "GSK Rule", "GoldenTriangle", "Genotoxic Carcinogenicity Mutagenicity Rule",
    "Genotoxicity", "Green fluorescence", "HBD (nHD)", "HBA (nHA)",
    "Hek293 Cytotoxicity", "HIA", "HLM Stability", "Human Hepatotoxicity",
    "IG C50", "IGC50", "LC50DM", "LC50FM", "Lipinski Rule", "logD7.4", "logP",
    "logS", "MCE-18", "MDCK Permeability", "Melting point", "Molecular Weight (MW)",
    "NPscore", "NonBiodegradable", "NonGenotoxic Carcinogenicity Rule", "nHet",
    "nHD", "nHA", "nRing", "nRig", "nRot", "NR-AR", "NR-AR-LBD", "NR-AhR",
    "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "PAINS", "PAMPA",
    "Pfizer Rule", "PPB", "Promiscuous compounds", "QED", "Rat Oral Acute Toxicity",
    "Respiratory", "RPMI-8226 Immunitoxicity", "Reactive compounds", "Roam (nRot)",
    "SAscore", "Skin Sensitization", "Skin Sensitization Rule", "Stereo Centers",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53", "Topological Polar Surface Area (TPSA)",
    "T1/2", "Toxicity: Eye Corrosion", "Toxicity: Eye Irritation",
    "Toxicity: Human Hepatotoxicity", "Toxicity: Drug-induced Nephrotoxicity",
    "Toxicity: Drug-induced Neurotoxicity", "Toxicity: Ototoxicity",
    "Toxicity: Hematotoxicity", "Toxicity: Genotoxicity", "Toxicity: Carcinogenicity",
    "Toxicity: Skin Sensitization", "Toxicity: DILI", "Toxicity: AMES Toxicity",
    "Toxicity: Rats Oral Acute Toxicity", "Toxicity: FDAMDD",
    "Toxicity: RPMI-8226 Immunitoxicity", "Toxicity: A549 Cytotoxicity",
    "Toxicity: Hek293 Cytotoxicity", "Toxicity: BCF", "Toxicity: IGC50",
    "Toxicity: LC50DM", "Toxicity: LC50FM"
]


ALIAS: Dict[str, Set[str]] = {
    "10肽": {"10肽", "ten peptides"},
    "Alarm_NMR Rule": {"alarm_nmr rule", "警报 nmr 规则"},
    "Acute Toxicity Rule": {"acute toxicity rule", "急性毒性 规则"},
    "AMES Toxicity": {"ames toxicity", "ames", "艾姆斯 毒性", "mutagenicity"},
    "Aquatic Toxicity Rule": {"aquatic toxicity rule", "水生毒性 规则"},
    "A549 Cytotoxicity": {"a549 cytotoxicity", "a549 细胞毒性"},
    "BCF": {"bcf", "生物富集因子"},
    "BCRP inhibitor": {"bcrp inhibitor", "bcrp 抑制剂", "bcrp"},
    "BMS Rule": {"bms rule", "bms 规则"},
    "BBB": {"bbb", "blood brain barrier", "血脑屏障"},
    "Boiling point": {"boiling point", "沸点"},
    "Caco-2 Permeability": {"caco-2 permeability", "caco2", "caco-2", "caco2 permeability"},
    "Carcinogenicity": {"carcinogenicity", "致癌性"},
    "Chelating Rule": {"chelating rule", "螯合 规则"},
    "CL_plasma": {"cl_plasma", "cl plasma", "cl<sub>plasma</sub>", "血浆 清除率"},
    "CYP1A2 inhibitor": {"cyp1a2 inhibitor", "cyp1a2 抑制剂", "抑制 cyp1a2"},
    "CYP1A2 substrate": {"cyp1a2 substrate", "cyp1a2 底物"},
    "CYP2B6 inhibitor": {"cyp2b6 inhibitor", "cyp2b6 抑制剂"},
    "CYP2B6 substrate": {"cyp2b6 substrate", "cyp2b6 底物"},
    "CYP2C19 inhibitor": {"cyp2c19 inhibitor", "cyp2c19 抑制剂"},
    "CYP2C19 substrate": {"cyp2c19 substrate", "cyp2c19 底物"},
    "CYP2C8 inhibitor": {"cyp2c8 inhibitor", "cyp2c8 抑制剂"},
    "CYP2C9 inhibitor": {"cyp2c9 inhibitor", "cyp2c9 抑制剂"},
    "CYP2C9 substrate": {"cyp2c9 substrate", "cyp2c9 底物"},
    "CYP2D6 inhibitor": {"cyp2d6 inhibitor", "cyp2d6 抑制剂", "抑制 cyp2d6", "cyp2d6/抑制"},
    "CYP2D6 substrate": {"cyp2d6 substrate", "cyp2d6 底物"},
    "CYP3A4 inhibitor": {"cyp3a4 inhibitor", "cyp3a4 抑制剂", "抑制 cyp3a4", "cyp3a4/抑制"},
    "CYP3A4 substrate": {"cyp3a4 substrate", "cyp3a4 底物"},
    "DILI": {"dili", "drug-induced liver injury", "药物性 肝损伤", "药物 诱导 肝损伤"},
    "Density": {"density", "密度"},
    "Eye Corrosion": {"eye corrosion", "眼腐蚀"},
    "Eye Irritation": {"eye irritation", "眼刺激"},
    "FDAMDD": {"fdamdd", "fda 最大推荐剂量", "fda mdd"},
    "FLuc inhibitors": {"fluc inhibitors", "荧光素酶 抑制剂"},
    "Flexibility": {"flexibility", "柔性"},
    "Fsp3": {"fsp3", "fsp³", "fsp 3", "fsp<sup>3</sup>"},
    "F20%": {"f20%"},
    "F30%": {"f30%"},
    "F50%": {"f50%"},
    "Fu": {"fu", "血浆 游离分数"},
    "GASA": {"gasa"},
    "GSK Rule": {"gsk rule", "gsk 规则"},
    "GoldenTriangle": {"goldentriangle", "金三角规则"},
    "Genotoxic Carcinogenicity Mutagenicity Rule": {
        "genotoxic carcinogenicity mutagenicity rule", "遗传毒性 规则"
    },
    "Genotoxicity": {"genotoxicity", "遗传毒性", "genotox"},
    "Green fluorescence": {"green fluorescence", "绿色荧光"},
    "HBD (nHD)": {"hbd (nhd)", "hbd", "nhd", "氢供体"},
    "HBA (nHA)": {"hba (nha)", "hba", "nha", "氢受体"},
    "Hek293 Cytotoxicity": {"hek293 cytotoxicity", "hek293 细胞毒性"},
    "HIA": {"hia", "human intestinal absorption", "人肠道吸收"},
    "HLM Stability": {"hlm stability", "代谢 稳定性", "人肝微粒体 稳定性", "代谢 不稳定性", "分子 不稳定性"},
    "Human Hepatotoxicity": {"human hepatotoxicity", "人类 肝毒性", "liver toxicity"},
    "IG C50": {"ig c50", "igc50"},
    "IGC50": {"igc50"},
    "LC50DM": {"lc50dm"},
    "LC50FM": {"lc50fm"},
    "Lipinski Rule": {"lipinski rule", "lipinski 规则"},
    "logD7.4": {"logd7.4", "logd 7.4", "logd_7_4"},
    "logP": {"logp", "clogp"},
    "logS": {"logs", "溶解度"},
    "MCE-18": {"mce-18"},
    "MDCK Permeability": {"mdck permeability", "mdck"},
    "Melting point": {"melting point", "熔点"},
    "Molecular Weight (MW)": {"molecular weight (mw)", "molecular weight", "mw", "分子量"},
    "NPscore": {"npscore", "np 分数"},
    "NonBiodegradable": {"nonbiodegradable", "非 生物 降解"},
    "NonGenotoxic Carcinogenicity Rule": {"nongenotoxic carcinogenicity rule", "非 遗传毒性 规则"},
    "nHet": {"nhet", "杂原子数"},
    "nHD": {"nhd", "氢供体 数"},
    "nHA": {"nha", "氢受体 数"},
    "nRing": {"nring", "环数"},
    "nRig": {"nrig", "刚性 数"},
    "nRot": {"nrot", "旋转键 数", "roam"},
    "NR-AR": {"nr-ar"},
    "NR-AR-LBD": {"nr-ar-lbd"},
    "NR-AhR": {"nr-ahr"},
    "NR-Aromatase": {"nr-aromatase", "nr-芳构酶"},
    "NR-ER": {"nr-er"},
    "NR-ER-LBD": {"nr-er-lbd"},
    "NR-PPAR-gamma": {"nr-ppar-gamma", "nr-ppar-γ"},
    "PAINS": {"pains"},
    "PAMPA": {"pampa"},
    "Pfizer Rule": {"pfizer rule", "pfizer 规则"},
    "PPB": {"ppb", "plasma protein binding", "血浆 蛋白 结合率"},
    "Promiscuous compounds": {"promiscuous compounds", "非特异性 配体"},
    "QED": {"qed", "药物 相似度"},
    "Rat Oral Acute Toxicity": {"rat oral acute toxicity", "大鼠 口服 急性毒性"},
    "Respiratory": {"respiratory", "呼吸毒性"},
    "RPMI-8226 Immunitoxicity": {"rpmi-8226 immunitoxicity", "rpmi-8226 免疫毒性"},
    "Reactive compounds": {"reactive compounds", "活性化合物"},
    "SAscore": {"sascore", "合成可及性 分数", "synthesizability"},
    "Skin Sensitization": {"skin sensitization", "皮肤 致敏性"},
    "Skin Sensitization Rule": {"skin sensitization rule", "皮肤 致敏性 规则"},
    "Stereo Centers": {"stereo centers", "立体 中心"},
    "SR-ARE": {"sr-are"},
    "SR-ATAD5": {"sr-atad5"},
    "SR-HSE": {"sr-hse"},
    "SR-MMP": {"sr-mmp"},
    "SR-p53": {"sr-p53"},
    "Topological Polar Surface Area (TPSA)": {"topological polar surface area (tpsa)", "tpsa", "极性表面积"},
    "T1/2": {"t1/2", "半衰期"},

    "Toxicity: Eye Corrosion": {"toxicity: eye corrosion", "眼腐蚀"},
    "Toxicity: Eye Irritation": {"toxicity: eye irritation", "眼刺激"},
    "Toxicity: Human Hepatotoxicity": {"toxicity: human hepatotoxicity", "人类 肝毒性"},
    "Toxicity: Drug-induced Nephrotoxicity": {"toxicity: drug-induced nephrotoxicity", "药物 诱导 肾毒性"},
    "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", "药物 诱导 神经毒性"},
    "Toxicity: Ototoxicity": {"toxicity: ototoxicity", "耳毒性"},
    "Toxicity: Hematotoxicity": {"toxicity: hematotoxicity", "血液 毒性"},
    "Toxicity: Genotoxicity": {"toxicity: genotoxicity", "基因 毒性"},
    "Toxicity: Carcinogenicity": {"toxicity: carcinogenicity", "致癌性"},
    "Toxicity: Skin Sensitization": {"toxicity: skin sensitization", "皮肤 致敏性"},
    "Toxicity: DILI": {"toxicity: dili", "药物性 肝损伤"},
    "Toxicity: AMES Toxicity": {"toxicity: ames toxicity", "ames 毒性"},
    "Toxicity: Rats Oral Acute Toxicity": {"toxicity: rats oral acute toxicity", "大鼠 口服 急性毒性"},
    "Toxicity: FDAMDD": {"toxicity: fdamdd", "fda mdd"},
    "Toxicity: RPMI-8226 Immunitoxicity": {"toxicity: rpmi-8226 immunitoxicity", "rpmi-8226 免疫毒性"},
    "Toxicity: A549 Cytotoxicity": {"toxicity: a549 cytotoxicity", "a549 细胞毒性"},
    "Toxicity: Hek293 Cytotoxicity": {"toxicity: hek293 cytotoxicity", "hek293 细胞毒性"},
    "Toxicity: BCF": {"toxicity: bcf"},
    "Toxicity: IGC50": {"toxicity: igc50"},
    "Toxicity: LC50DM": {"toxicity: lc50dm"},
    "Toxicity: LC50FM": {"toxicity: lc50fm"},
}

ALIAS.setdefault("CYP3A4 inhibitor", set()).update({"cyp3a inhibitor", "cyp3a4 inhibitor", "cyp3a 抑制剂"})
ALIAS.setdefault("CYP1A2 inhibitor", set()).update({"cyp1a inhibitor", "cyp1a2 inhibitor", "cyp1a 抑制剂"})


ALIAS.setdefault("logP", set()).update({"clogp", "xlogp", "xlogp3", "log p", "ilogp"})


ALIAS.setdefault("Fu", set()).update({
    "fraction unbound", "unbound fraction", "fu_plasma", "fu (plasma)", "fu%", "fu - plasma"
})


ALIAS.setdefault("hERG Blockers", set()).update({
    "herg", "hERG", "hERG blockade", "hERG inhibition", "hERG blocker", "herg blockers",
    "hERG channel", "cardiac ion channel"
})
ALIAS.setdefault("Pgp inhibitor", set()).update({
    "pgp inhibitor", "p-gp inhibitor", "p-glycoprotein inhibitor", "p-gp inhibition",
    "efflux transporter inhibition", "efflux pump inhibition"
})
ALIAS.setdefault("Pgp substrate", set()).update({
    "pgp substrate", "p-gp substrate", "p-glycoprotein substrate", "efflux transporter substrate"
})
ALIAS.setdefault("BCRP inhibitor", set()).update({
    "bcrp inhibition", "bcrp inhibitor", "bcrp", "breast cancer resistance protein inhibitor"
})


ALIAS.setdefault("Flexibility", set()).update({
    "rotatable bonds", "num rotatable bonds", "nrot", "roam", "旋转键 数"
})

ALIAS.update({

    "DILI": {"dili", "drug-induced liver injury", "药物性 肝损伤", "药物 诱导 肝损伤", 
             "hepatotoxicity", "liver injury", "liver toxicity", "hepatic toxicity", 
             "liver damage", "hepatotoxic", "liver safety", "hepatic injury"},
    
    "Human Hepatotoxicity": {"human hepatotoxicity", "人类 肝毒性", "liver toxicity",
                            "hepatotoxicity", "hepatic toxicity", "liver injury", 
                            "liver damage", "hepatotoxic", "liver safety"},
    

    "Genotoxicity": {"genotoxicity", "遗传毒性", "genotox", "genotoxic", "mutagenicity",
                    "mutagenic", "genetic toxicity", "DNA damage", "chromosomal damage"},
    
    "AMES Toxicity": {"ames toxicity", "ames", "艾姆斯 毒性", "mutagenicity", "mutagenic",
                     "ames test", "bacterial mutagenicity", "in vitro mutagenicity"},
    

    "HLM Stability": {"hlm stability", "代谢 稳定性", "人肝微粒体 稳定性", "代谢 不稳定性", 
                     "分子 不稳定性", "metabolic stability", "metabolism stability",
                     "hepatic metabolism", "liver metabolism", "metabolic clearance",
                     "oxidative metabolism", "metabolic activation"},
    

    "Caco-2 Permeability": {"caco-2 permeability", "caco2", "caco-2", "caco2 permeability",
                           "intestinal permeability", "permeability", "membrane permeability",
                           "passive permeability", "absorption", "bioavailability"},
    

    "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", 
                                            "药物 诱导 神经毒性", "neurotoxicity", "neurotoxic",
                                            "CNS toxicity", "brain toxicity", "neural toxicity",
                                            "drug-induced neurotoxicity", "nervous system toxicity"},
    

    "Carcinogenicity": {"carcinogenicity", "致癌性", "carcinogenic", "cancer risk",
                       "tumorigenic", "oncogenic", "cancer-causing"},
    

    "Skin Sensitization": {"skin sensitization", "皮肤 致敏性", "skin allergy", "dermatitis",
                          "contact sensitization", "allergic reaction", "hypersensitivity"},
    

    "Eye Irritation": {"eye irritation", "眼刺激", "ocular irritation", "eye toxicity",
                      "ocular toxicity", "eye damage", "corneal irritation"},
    

    "Respiratory": {"respiratory", "呼吸毒性", "lung toxicity", "pulmonary toxicity",
                   "respiratory toxicity", "breathing problems", "lung damage"},
    

    "SAscore": {"sascore", "合成可及性 分数", "synthesizability", "synthetic accessibility",
               "synthesis difficulty", "hard to synthesize", "synthetic challenge"},
    

    "QED": {"qed", "药物 相似度", "drug-likeness", "drug like", "desirability",
           "drug similarity", "pharmaceutical properties"},
    

    "logS": {"logs", "溶解度", "solubility", "aqueous solubility", "water solubility",
            "dissolution", "solubility profile"},
    

    "logP": {"logp", "clogp", "lipophilicity", "hydrophobicity", "partition coefficient",
            "octanol-water partition", "lipid solubility"},
    

    "Molecular Weight (MW)": {"molecular weight (mw)", "molecular weight", "mw", "分子量",
                             "molecular mass", "compound weight", "molecule size"},
    

    "Topological Polar Surface Area (TPSA)": {"topological polar surface area (tpsa)", 
                                             "tpsa", "极性表面积", "polar surface area",
                                             "PSA", "polar area", "surface polarity"},
    

    "HBD (nHD)": {"hbd (nhd)", "hbd", "nhd", "氢供体", "hydrogen bond donor", "HBD",
                 "hydrogen donors", "donor groups"},
    
    "HBA (nHA)": {"hba (nha)", "hba", "nha", "氢受体", "hydrogen bond acceptor", "HBA",
                 "hydrogen acceptors", "acceptor groups"},
    

    "Flexibility": {"flexibility", "柔性", "rotatable bonds", "num rotatable bonds", 
                   "nrot", "roam", "旋转键 数", "molecular flexibility", "conformational flexibility"},
    

    "BBB": {"bbb", "blood brain barrier", "血脑屏障", "brain barrier", "CNS penetration",
           "blood-brain barrier", "brain access", "CNS access"},
    

    "PPB": {"ppb", "plasma protein binding", "血浆 蛋白 结合率", "protein binding",
           "plasma binding", "serum protein binding", "albumin binding"},
    

    "Fu": {"fu", "血浆 游离分数", "fraction unbound", "unbound fraction", "fu_plasma",
          "fu (plasma)", "fu%", "fu - plasma", "free fraction", "unbound drug"},
    

    "T1/2": {"t1/2", "半衰期", "half-life", "elimination half-life", "plasma half-life",
            "clearance half-life", "metabolic half-life"},
    

    "CL_plasma": {"cl_plasma", "cl plasma", "cl<sub>plasma</sub>", "血浆 清除率",
                 "plasma clearance", "systemic clearance", "total clearance", "CL"},
    

    "HIA": {"hia", "human intestinal absorption", "人肠道吸收", "intestinal absorption",
           "oral absorption", "GI absorption", "gastrointestinal absorption"},
    

    "Rat Oral Acute Toxicity": {"rat oral acute toxicity", "大鼠 口服 急性毒性",
                               "acute oral toxicity", "LD50", "lethal dose", "acute lethality"},
    

    "Aquatic Toxicity Rule": {"aquatic toxicity rule", "水生毒性 规则", "water toxicity",
                             "aquatic life toxicity", "environmental toxicity"},
    

    "BCF": {"bcf", "生物富集因子", "bioconcentration factor", "bioaccumulation",
           "tissue accumulation", "biomagnification"},
    

    "IGC50": {"igc50", "inhibitory concentration", "growth inhibition", "cell viability"},
    "LC50DM": {"lc50dm", "lethal concentration", "mortality concentration"},
    "LC50FM": {"lc50fm", "fish mortality", "aquatic lethality"},
    

    "Lipinski Rule": {"lipinski rule", "lipinski 规则", "rule of five", "drug-like properties",
                     "Lipinski's rule", "molecular properties rule"},
    
    "GoldenTriangle": {"goldentriangle", "金三角规则", "golden triangle", "ADMET triangle",
                      "property triangle", "drug optimization triangle"},
    

    "Fsp3": {"fsp3", "fsp³", "fsp 3", "fsp<sup>3</sup>", "fraction sp3", "sp3 carbon fraction",
            "saturation", "molecular saturation"},
    
    "NPscore": {"npscore", "np 分数", "natural product score", "natural product likeness",
               "NP likeness", "natural product similarity"},
    

    "nRing": {"nring", "环数", "number of rings", "ring count", "aromatic rings",
             "cyclic structures", "ring systems"},
    
    "nHet": {"nhet", "杂原子数", "heteroatoms", "heteroatom count", "non-carbon atoms",
            "hetero atoms", "heteroatom number"},
    
    "nRig": {"nrig", "刚性 数", "rigid bonds", "rigid atom count", "rigidity"},
    

    "Stereo Centers": {"stereo centers", "立体 中心", "chiral centers", "stereocenters",
                      "asymmetric centers", "chirality", "stereochemistry"},
    

    "Melting point": {"melting point", "熔点", "mp", "fusion point", "liquefaction point"},
    "Boiling point": {"boiling point", "沸点", "bp", "vaporization point", "evaporation point"},
    "Density": {"density", "密度", "specific gravity", "mass density", "bulk density"},
    

    "F20%": {"f20%", "fraction at 20%", "20% fraction"},
    "F30%": {"f30%", "fraction at 30%", "30% fraction"},
    "F50%": {"f50%", "fraction at 50%", "50% fraction"},
    

    "logD7.4": {"logd7.4", "logd 7.4", "logd_7_4", "distribution coefficient", "logD at pH 7.4"},
    

    "MCE-18": {"mce-18", "MCE rule", "molecular complexity", "complexity rule"},
    "GASA": {"gasa", "GASA score", "synthetic accessibility", "synthesis difficulty"},
    "FDAMDD": {"fdamdd", "fda 最大推荐剂量", "fda mdd", "FDA maximum daily dose", "maximum dose"},
    "IG C50": {"ig c50", "igc50", "inhibitory concentration", "growth inhibition"},
    

    "BCRP inhibitor": {"bcrp inhibitor", "bcrp 抑制剂", "bcrp", "breast cancer resistance protein"},
    "FLuc inhibitors": {"fluc inhibitors", "荧光素酶 抑制剂", "firefly luciferase inhibitors"},
    

    "CYP1A2 substrate": {"cyp1a2 substrate", "cyp1a2 底物", "1A2 substrate", "CYP1A2 metabolism"},
    "CYP2B6 substrate": {"cyp2b6 substrate", "cyp2b6 底物", "2B6 substrate", "CYP2B6 metabolism"},
    "CYP2C19 substrate": {"cyp2c19 substrate", "cyp2c19 底物", "2C19 substrate", "CYP2C19 metabolism"},
    "CYP2C9 substrate": {"cyp2c9 substrate", "cyp2c9 底物", "2C9 substrate", "CYP2C9 metabolism"},
    "CYP2D6 substrate": {"cyp2d6 substrate", "cyp2d6 底物", "2D6 substrate", "CYP2D6 metabolism"},
    "CYP3A4 substrate": {"cyp3a4 substrate", "cyp3a4 底物", "3A4 substrate", "CYP3A4 metabolism"},
    

    "CYP1A2 inhibitor": {
        "cyp1a2 inhibitor", "cyp1A2 inhibitor", "CYP1a2 inhibitor", "CYP1A2 inhibitor",
        "cyp1a2 抑制剂", "CYP1a2 抑制剂", "1A2 inhibitor", "1a2 inhibitor"
    },
    "CYP2B6 inhibitor": {"cyp2b6 inhibitor", "cyp2b6 抑制剂", "2B6 inhibitor"},
    "CYP2C19 inhibitor": {
        "cyp2c19 inhibitor", "cyp2C19 inhibitor", "CYP2c19 inhibitor", "CYP2C19 inhibitor",
        "cyp2c19 抑制剂", "CYP2c19 抑制剂", "2C19 inhibitor", "2c19 inhibitor"
    },
    "CYP2C8 inhibitor": {"cyp2c8 inhibitor", "cyp2c8 抑制剂", "2C8 inhibitor"},
    "CYP2C9 inhibitor": {
        "cyp2c9 inhibitor", "cyp2C9 inhibitor", "CYP2c9 inhibitor", "CYP2C9 inhibitor",
        "cyp2c9 抑制剂", "CYP2c9 抑制剂", "2C9 inhibitor", "2c9 inhibitor"
    },
    "CYP2D6 inhibitor": {"cyp2d6 inhibitor", "cyp2d6 抑制剂", "抑制 cyp2d6", "cyp2d6/抑制", "2D6 inhibitor"},
    "CYP3A4 inhibitor": {
        "cyp3a4 inhibitor", "cyp3A4 inhibitor", "CYP3a4 inhibitor", "CYP3A4 inhibitor",
        "cyp3a4 抑制剂", "CYP3a4 抑制剂", "3A4 inhibitor", "3a4 inhibitor"
    },
    

    "Toxicity: Eye Corrosion": {"toxicity: eye corrosion", "眼腐蚀", "ocular corrosion", "eye damage"},
    "Toxicity: Eye Irritation": {"toxicity: eye irritation", "眼刺激", "ocular irritation", "eye irritation"},
    "Toxicity: Human Hepatotoxicity": {"toxicity: human hepatotoxicity", "人类 肝毒性", "human liver toxicity"},
    "Toxicity: Drug-induced Nephrotoxicity": {"toxicity: drug-induced nephrotoxicity", "药物 诱导 肾毒性", 
                                             "nephrotoxicity", "kidney toxicity", "renal toxicity"},
    "Toxicity: Drug-induced Neurotoxicity": {"toxicity: drug-induced neurotoxicity", "药物 诱导 神经毒性",
                                            "neurotoxicity", "CNS toxicity", "brain toxicity"},
    "Toxicity: Ototoxicity": {"toxicity: ototoxicity", "耳毒性", "hearing toxicity", "auditory toxicity"},
    "Toxicity: Hematotoxicity": {"toxicity: hematotoxicity", "血液 毒性", "blood toxicity", "hematologic toxicity"},
    "Toxicity: Genotoxicity": {"toxicity: genotoxicity", "基因 毒性", "genetic toxicity", "DNA toxicity"},
    "Toxicity: Carcinogenicity": {"toxicity: carcinogenicity", "致癌性", "cancer toxicity", "tumorigenic"},
    "Toxicity: Skin Sensitization": {"toxicity: skin sensitization", "皮肤 致敏性", "skin allergy", "dermatitis"},
    "Toxicity: DILI": {"toxicity: dili", "药物性 肝损伤", "drug-induced liver injury"},
    "Toxicity: AMES Toxicity": {"toxicity: ames toxicity", "ames 毒性", "mutagenicity", "AMES test"},
    "Toxicity: Rats Oral Acute Toxicity": {"toxicity: rats oral acute toxicity", "大鼠 口服 急性毒性", 
                                          "acute oral toxicity", "LD50"},
    "Toxicity: FDAMDD": {"toxicity: fdamdd", "fda mdd", "FDA maximum daily dose", "maximum dose"},
    "Toxicity: RPMI-8226 Immunitoxicity": {"toxicity: rpmi-8226 immunitoxicity", "rpmi-8226 免疫毒性",
                                          "immune toxicity", "immunotoxicity"},
    "Toxicity: A549 Cytotoxicity": {"toxicity: a549 cytotoxicity", "a549 细胞毒性", "lung cell toxicity"},
    "Toxicity: Hek293 Cytotoxicity": {"toxicity: hek293 cytotoxicity", "hek293 细胞毒性", "kidney cell toxicity"},
    "Toxicity: BCF": {"toxicity: bcf", "bioconcentration", "bioaccumulation"},
    "Toxicity: IGC50": {"toxicity: igc50", "inhibitory concentration", "growth inhibition"},
    "Toxicity: LC50DM": {"toxicity: lc50dm", "lethal concentration", "mortality"},
    "Toxicity: LC50FM": {"toxicity: lc50fm", "fish mortality", "aquatic lethality"},
    

    "BMS Rule": {"bms rule", "bms 规则", "Bristol-Myers Squibb rule", "BMS criteria"},
    "GSK Rule": {"gsk rule", "gsk 规则", "GlaxoSmithKline rule", "GSK criteria"},
    "Pfizer Rule": {"pfizer rule", "pfizer 规则", "Pfizer criteria", "Pfizer guidelines"},
    

    "Alarm_NMR Rule": {"alarm_nmr rule", "警报 nmr 规则", "NMR alarm", "NMR alerts"},
    "Acute Toxicity Rule": {"acute toxicity rule", "急性毒性 规则", "acute toxicity alerts"},
    "Chelating Rule": {"chelating rule", "螯合 规则", "metal chelation", "chelating agents"},
    "Genotoxic Carcinogenicity Mutagenicity Rule": {"genotoxic carcinogenicity mutagenicity rule", 
                                                   "遗传毒性 规则", "genotoxicity alerts"},
    "NonGenotoxic Carcinogenicity Rule": {"nongenotoxic carcinogenicity rule", "非 遗传毒性 规则",
                                         "non-genotoxic alerts"},
    "Skin Sensitization Rule": {"skin sensitization rule", "皮肤 致敏性 规则", 
                               "skin sensitization alerts"},
    

    "NR-AR": {"nr-ar", "androgen receptor", "AR receptor", "androgen binding"},
    "NR-AR-LBD": {"nr-ar-lbd", "androgen receptor ligand binding", "AR LBD"},
    "NR-AhR": {"nr-ahr", "aryl hydrocarbon receptor", "AhR receptor", "dioxin receptor"},
    "NR-Aromatase": {"nr-aromatase", "nr-芳构酶", "aromatase enzyme", "estrogen synthesis"},
    "NR-ER": {"nr-er", "estrogen receptor", "ER receptor", "estrogen binding"},
    "NR-ER-LBD": {"nr-er-lbd", "estrogen receptor ligand binding", "ER LBD"},
    "NR-PPAR-gamma": {"nr-ppar-gamma", "nr-ppar-γ", "PPAR gamma", "peroxisome proliferator"},
    

    "SR-ARE": {"sr-are", "antioxidant response element", "ARE pathway", "oxidative stress"},
    "SR-ATAD5": {"sr-atad5", "ATAD5 stress response", "DNA replication stress"},
    "SR-HSE": {"sr-hse", "heat shock element", "heat shock response", "thermal stress"},
    "SR-MMP": {"sr-mmp", "mitochondrial membrane potential", "MMP stress", "mitochondrial stress"},
    "SR-p53": {"sr-p53", "p53 stress response", "tumor suppressor", "DNA damage response"},
    

    "A549 Cytotoxicity": {"a549 cytotoxicity", "a549 细胞毒性", "lung cell toxicity",
                         "A549 cell line", "pulmonary cytotoxicity"},
    "Hek293 Cytotoxicity": {"hek293 cytotoxicity", "hek293 细胞毒性", "kidney cell toxicity",
                           "HEK293 cell line", "renal cytotoxicity"},
    

    "RPMI-8226 Immunitoxicity": {"rpmi-8226 immunitoxicity", "rpmi-8226 免疫毒性",
                                "RPMI8226 toxicity", "myeloma cell toxicity", "immune toxicity"},
    

    "NonBiodegradable": {"nonbiodegradable", "非 生物 降解", "non-biodegradable",
                        "persistent", "environmental persistence", "biodegradation resistance"},
    "PAINS": {"pains", "pan-assay interference compounds", "assay interference",
             "false positives", "promiscuous binders"},
    "Promiscuous compounds": {"promiscuous compounds", "非特异性 配体", "promiscuous binders",
                             "non-specific binding", "off-target binding", "polypharmacology"},
    "Reactive compounds": {"reactive compounds", "活性化合物", "reactive metabolites",
                          "electrophilic compounds", "reactive intermediates"},
    "Green fluorescence": {"green fluorescence", "绿色荧光", "fluorescent", "fluorescence",
                          "autofluorescence", "background fluorescence"},
    "PAMPA": {"pampa", "parallel artificial membrane permeability", "artificial membrane",
             "membrane model", "permeability assay"},
    "MDCK Permeability": {"mdck permeability", "mdck", "Madin-Darby canine kidney",
                         "canine kidney permeability", "MDCK model"},
    "CYP1A2 substrate": {
        "cyp1a2 substrate", "cyp1A2 substrate", "CYP1a2 substrate", "CYP1A2 substrate",
        "cyp1a2 底物", "CYP1a2 底物", "1A2 substrate", "1a2 substrate"
    },
    "CYP2C19 substrate": {
        "cyp2c19 substrate", "cyp2C19 substrate", "CYP2c19 substrate", "CYP2C19 substrate",
        "cyp2c19 底物", "CYP2c19 底物", "2C19 substrate", "2c19 substrate"
    },
})



REV_ALIAS: Dict[str, str] = {}
for canon, synonyms in ALIAS.items():
    for s in synonyms | {canon}:
        key = s.lower().strip()
        REV_ALIAS[key] = canon


CANONICAL_MERGE: Dict[str, str] = {
    "nRot": "Flexibility",
    "Roam (nRot)": "Flexibility",
    "nHD": "HBD (nHD)",
    "nHA": "HBA (nHA)",
}

CANONICAL_MERGE.update({
    "Drug induced Neurotoxicity": "Toxicity: Drug-induced Neurotoxicity",
    "Drug-induced Neurotoxicity": "Toxicity: Drug-induced Neurotoxicity",
    "Drug induced Nephrotoxicity": "Toxicity: Drug-induced Nephrotoxicity",
    "Drug-induced Nephrotoxicity": "Toxicity: Drug-induced Nephrotoxicity",
    "Drug induced Ototoxicity": "Toxicity: Ototoxicity",
    "Drug-induced Ototoxicity": "Toxicity: Ototoxicity",
    "Drug induced Hematotoxicity": "Toxicity: Hematotoxicity",
    "Drug-induced Hematotoxicity": "Toxicity: Hematotoxicity",
})

import string
def _flatten_alnum(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", s or "").lower()

REV_ALIAS_FLAT: Dict[str, str] = {}
for k_low, canon in REV_ALIAS.items():
    REV_ALIAS_FLAT[_flatten_alnum(k_low)] = canon

def normalize_key(k: str) -> str:
    if not k:
        return ""
    s = str(k)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("（", "(").replace("）", ")")
    raw_low = s.lower().strip()

    if raw_low in REV_ALIAS:
        s = REV_ALIAS[raw_low]
    else:
        s2 = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", s)
        s2 = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s2)
        s2 = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s2)
        low2 = s2.lower().strip()
        if low2 in REV_ALIAS:
            s = REV_ALIAS[low2]
        else:
            s3 = re.sub(r"[_\-]+", " ", s2)
            s3 = re.sub(r"\s+", " ", s3).strip()
            low3 = s3.lower()
            s = REV_ALIAS.get(low3, s3)

    s = CANONICAL_MERGE.get(s, s)
    flat = _flatten_alnum(s)
    s = REV_ALIAS_FLAT.get(flat, s)
    return s

def normalize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in props.items():
        nk = normalize_key(k)
        out[nk] = v
    return out


UP_KW = ("提升", "增强", "increase", "up", "提高", "更高", "enhance", "improve", "boost", "raise")
DOWN_KW = ["降低", "减少", "decrease", "down", "抑制", "减弱", "reduce", "lower", "mitigate", "cut", "less", "drop"]

############################################################

############################################################

RANGE_TARGETS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "logP": (1.0, 3.0),
    "Topological Polar Surface Area (TPSA)": (20.0, 130.0),  # Å²
    "Molecular Weight (MW)": (150.0, 500.0),                  # Da
}

MORE_IS_BETTER_BASE: Set[str] = {
    "HLM Stability",
    "logS",
    "logD7.4",
    "Flexibility",
    "Fsp3",
    "QED",
    "NPscore",
    "Caco-2 Permeability",
    "MDCK Permeability",
    "PAMPA",
    "Fu",
}

def build_less_is_better(all_features: List[str]) -> Set[str]:
    s: Set[str] = set()
    for f in all_features:
        fn = normalize_key(f)
        if fn.lower().endswith("inhibitor"):
            s.add(fn)
        if fn.lower().startswith("toxicity:"):
            s.add(fn)
    s.update({normalize_key(x) for x in [
        "Human Hepatotoxicity",
        "Genotoxicity",
        "Carcinogenicity",
        "Eye Irritation",
        "Eye Corrosion",
        "Respiratory",
        "Skin Sensitization",
        "DILI",
        "AMES Toxicity",
        "PPB"
    ]})
    return s

ALL_FEATURES_CANON = sorted({normalize_key(f) for f in FEATURES})


CORE_FEATURES = {
    "Caco-2 Permeability", "F50%", "CYP3A4 inhibitor", "CYP2D6 inhibitor",
    "Pgp substrate", "hERG Blockers", "DILI", "Human Hepatotoxicity",
    "AMES Toxicity", "Genotoxicity", "Toxicity: Drug-induced Neurotoxicity",
    "QED", "SAScore", "GASA", "Lipinski Rule", "HLM Stability"
}

LESS_IS_BETTER: Set[str] = build_less_is_better(FEATURES)
MORE_IS_BETTER: Set[str] = set(
    normalize_key(f) for f in MORE_IS_BETTER_BASE
    if normalize_key(f) not in RANGE_TARGETS
)

def is_more_is_better(feat: str) -> Optional[bool]:
    if feat in MORE_IS_BETTER:
        return True
    if feat in LESS_IS_BETTER:
        return False
    if feat in RANGE_TARGETS:
        return None
    return None

############################################################

############################################################

def load_thresholds() -> Dict[str, float]:
    thr: Dict[str, float] = {}
    for _cat, feat_dict in checks.items():
        for raw_feat, (flag, threshold, _msg) in feat_dict.items():
            try:
                t = float(threshold)
            except Exception:
                continue
            canon = normalize_key(raw_feat)
            thr[canon] = t
    return thr

THR_MAP = load_thresholds()

BASE_IMPROVEMENT_BONUS = 2.0
RELATIVE_IMPROVEMENT_WEIGHT = 1.5
THR_BONUS = 2.5
RANGE_BONUS = 3.5
MAX_REWARD_PER_FEAT = 2.0

BASE_DEGRADATION_PENALTY = -1.5
RELATIVE_DEGRADATION_WEIGHT = -1.2
THR_PENALTY = -2.0
MIN_PENALTY_PER_FEAT = -2.0


FEATURE_WEIGHTS = {

    "Genotoxicity": 3.0,
    "DILI": 3.0,
    "Carcinogenicity": 3.0,
    "AMES Toxicity": 2.5,
    "Human Hepatotoxicity": 2.5,
    

    "hERG Blockers": 2.0,
    "hERG Blockers (10 um)": 2.0,
    "Toxicity: Drug-induced Neurotoxicity": 2.0,
    "Toxicity: Drug-induced Nephrotoxicity": 2.0,
    "Hematotoxicity": 1.8,
    

    "Caco-2 Permeability": 1.5,
    "HIA": 1.5,
    "BBB": 1.5,
    "logS": 1.5,
    "PPB": 1.3,
    "Fu": 1.3,
    

    "CYP3A4 inhibitor": 1.0,
    "CYP2D6 inhibitor": 1.0,
    "CYP2C9 inhibitor": 1.0,
    "CYP2C19 inhibitor": 1.0,
    "HLM Stability": 1.0,
    "Pgp substrate": 1.0,
    "Pgp inhibitor": 1.0,
    "QED": 1.0,
    "logP": 1.0,
    "logD7.4": 1.0,
    

}

def get_feature_weight(feat: str) -> float:

    return FEATURE_WEIGHTS.get(feat, 1.0)

DELTA_WEIGHT = 0.3

############################################################

############################################################

_SCI_RE = re.compile(
    r'(?P<sign>[<>]=?|~)?\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<unit>[munp]M|μM|uM|nM|mM|%)?',
    re.IGNORECASE
)

def _convert_unit(val: float, unit: Optional[str]) -> float:
    if not unit:
        return val
    u = unit.strip().lower()
    if u == "mm":
        return val * 1000.0
    if u in ("um", "μm"):
        return val
    if u == "nm":
        return val / 1000.0
    if u == "pm":
        return val / 1_000_000.0
    if u == "%":
        return val
    return val

def parse_numeric_value(raw: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw)
    m = _SCI_RE.search(s)
    if not m:
        return None
    try:
        val = float(m.group("val"))
    except Exception:
        return None
    unit = m.group("unit")
    val = _convert_unit(val, unit)
    return val

def map_admet_value(feature: str, raw: Any) -> Optional[float]:

    raw_s = "" if raw is None else str(raw).strip()


    num = parse_numeric_value(raw_s)
    if num is not None:
        return num


    pluses = raw_s.count("+")
    minuses = raw_s.count("-")
    if pluses == 0 and minuses == 0:
        return None

    score = float(pluses - minuses)
    return score


def extract_numeric(props: Dict[str, Any]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    for feat, raw in props.items():
        v = map_admet_value(feat, raw)
        if v is not None:
            numeric[feat] = float(v)
    return numeric

############################################################

############################################################
def _alias_hits(txt: str, alias: str):
    a = alias.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9 _\-]*[a-z0-9]", a):
        return [m.span() for m in re.finditer(rf"\b{re.escape(a)}\b", txt)]
    i = txt.find(a)
    return [(i, i + len(a))] if i >= 0 else []

def parse_reasoning_enhanced(reasoning: str) -> List[Tuple[str, str]]:

    txt = (reasoning or "").lower()
    pairs: List[Tuple[str, str]] = []
    WINDOW = 100
    


    problem_mappings = {

        "low caco-2 permeability": ("Caco-2 Permeability", "down"),
        "poor permeability": ("Caco-2 Permeability", "down"),
        "permeability issues": ("Caco-2 Permeability", "down"),
        "permeability": ("Caco-2 Permeability", "down"),
        "low solubility": ("logS", "down"),
        "poor solubility": ("logS", "down"),
        "solubility issues": ("logS", "down"),
        "solubility": ("logS", "down"),
        "high clearance": ("CL_plasma", "up"),
        "rapid clearance": ("CL_plasma", "up"),
        "clearance issues": ("CL_plasma", "up"),
        "clearance": ("CL_plasma", "up"),
        "cyp1a2 inhibition": ("CYP1A2 inhibitor", "up"),
        "cyp1a2 inhibitor": ("CYP1A2 inhibitor", "up"),
        "cyp3a4 inhibition": ("CYP3A4 inhibitor", "up"),
        "cyp3a4 inhibitor": ("CYP3A4 inhibitor", "up"),
        "cyp2c19 inhibition": ("CYP2C19 inhibitor", "up"),
        "cyp2c19 inhibitor": ("CYP2C19 inhibitor", "up"),
        "cyp2c9 inhibition": ("CYP2C9 inhibitor", "up"),
        "cyp2c9 inhibitor": ("CYP2C9 inhibitor", "up"),
        "cyp450": ("CYP3A4 inhibitor", "up"),
        "cyp": ("CYP3A4 inhibitor", "up"),
        "herg blockage": ("hERG Blockers", "up"),
        "herg inhibition": ("hERG Blockers", "up"),
        "cardiac issues": ("hERG Blockers", "up"),
        "genotoxicity": ("Genotoxicity", "up"),
        "mutagenicity": ("Genotoxicity", "up"),
        "dna damage": ("Genotoxicity", "up"),
        "hepatotoxicity": ("DILI", "up"),
        "liver toxicity": ("DILI", "up"),
        "liver injury": ("DILI", "up"),
        "dili": ("DILI", "up"),
        "neurotoxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
        "cns toxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
        "brain toxicity": ("Toxicity: Drug-induced Neurotoxicity", "up"),
        "ames toxicity": ("AMES Toxicity", "up"),
        "ames test": ("AMES Toxicity", "up"),
        "mutagenic": ("AMES Toxicity", "up"),
        "carcinogenicity": ("Carcinogenicity", "up"),
        "cancer risk": ("Carcinogenicity", "up"),
        "skin sensitization": ("Skin Sensitization", "up"),
        "allergic reaction": ("Skin Sensitization", "up"),
        "eye irritation": ("Eye Irritation", "up"),
        "ocular toxicity": ("Eye Irritation", "up"),
        "respiratory toxicity": ("Respiratory", "up"),
        "lung toxicity": ("Respiratory", "up"),
        "metabolic instability": ("HLM Stability", "down"),
        "metabolism issues": ("HLM Stability", "down"),
        "metabolism": ("HLM Stability", "down"),
        "synthesis difficulty": ("SAScore", "up"),
        "hard to synthesize": ("SAScore", "up"),
        "drug-likeness": ("QED", "down"),
        "desirability": ("QED", "down"),

        "dili risk": ("DILI", "up"),
        "liver injury risk": ("DILI", "up"),
        "hepatotoxicity risk": ("DILI", "up"),
        "genotoxicity risk": ("Genotoxicity", "up"),
        "mutagenicity risk": ("Genotoxicity", "up"),
        "ames positive": ("AMES Toxicity", "up"),
        "neurotoxicity risk": ("Toxicity: Drug-induced Neurotoxicity", "up"),
        "cns toxicity risk": ("Toxicity: Drug-induced Neurotoxicity", "up"),
        "human hepatotoxicity": ("Human Hepatotoxicity", "up"),
        "liver toxicity": ("Human Hepatotoxicity", "up"),
        "hepatotoxicity risk": ("Human Hepatotoxicity", "up"),
        "liver injury risk": ("Human Hepatotoxicity", "up"),

        "bioactivation": ("DILI", "up"),
        "metabolic soft spot": ("DILI", "up"),
        "metabolically stable": ("HLM Stability", "up"),
        "aromaticity": ("logP", "up"),
        "planarity": ("logP", "up"),
        "polarity": ("logS", "up"),
        "hydrogen bond": ("logS", "up"),
        "hydrogen bonding": ("logS", "up"),

        "肝毒性": ("DILI", "up"),
        "肝损伤": ("DILI", "up"),
        "药物性肝损伤": ("DILI", "up"),
        "药物性肝毒性": ("DILI", "up"),
        "hepatotoxic": ("DILI", "up"),
        "liver damage": ("DILI", "up"),
        "hepatic injury": ("DILI", "up"),
        "hepatic toxicity": ("DILI", "up"),
        "liver safety": ("DILI", "up"),
        "hepatotoxic risk": ("DILI", "up"),
        "liver injury risk": ("DILI", "up"),
        "hepatotoxicity concern": ("DILI", "up"),
        "liver toxicity concern": ("DILI", "up"),
        "hepatotoxic potential": ("DILI", "up"),
        "liver toxicity potential": ("DILI", "up"),

        "代谢稳定性": ("HLM Stability", "down"),
        "代谢不稳定性": ("HLM Stability", "down"),

        "bsep": ("Pgp inhibitor", "up"),
        "oatp": ("Pgp inhibitor", "up"),
        "oatp1b1": ("Pgp inhibitor", "up"),
        "oatp1b3": ("Pgp inhibitor", "up"),
        "efflux transporter": ("Pgp inhibitor", "up"),
        "transporter inhibition": ("Pgp inhibitor", "up"),
        "代谢清除": ("HLM Stability", "down"),
        "metabolic clearance": ("HLM Stability", "down"),
        "metabolic stability": ("HLM Stability", "up"),
        "metabolism stability": ("HLM Stability", "up"),
        "hepatic metabolism": ("HLM Stability", "down"),
        "liver metabolism": ("HLM Stability", "down"),
        "oxidative metabolism": ("HLM Stability", "down"),
        "metabolic activation": ("HLM Stability", "down"),

        "渗透性": ("Caco-2 Permeability", "down"),
        "渗透性不足": ("Caco-2 Permeability", "down"),
        "渗透性问题": ("Caco-2 Permeability", "down"),
        "intestinal permeability": ("Caco-2 Permeability", "down"),
        "membrane permeability": ("Caco-2 Permeability", "down"),
        "passive permeability": ("Caco-2 Permeability", "down"),
        "absorption": ("Caco-2 Permeability", "down"),
        "bioavailability": ("Caco-2 Permeability", "down"),

        "molecular weight": ("Molecular Weight (MW)", "down"),
        "molecular mass": ("Molecular Weight (MW)", "down"),
        "high molecular weight": ("Molecular Weight (MW)", "down"),
        "large molecule": ("Molecular Weight (MW)", "down"),
        "heavy molecule": ("Molecular Weight (MW)", "down"),

        "hydrogen bond donors": ("nHD", "down"),
        "hbd": ("nHD", "down"),
        "hydrogen bond acceptors": ("nHA", "down"),
        "hba": ("nHA", "down"),

        "lipophilicity": ("logP", "down"),
        "hydrophobicity": ("logP", "down"),
        "high logp": ("logP", "down"),
        "low logp": ("logP", "up"),

        "分子量": ("Molecular Weight (MW)", "down"),
        "氢键供体": ("nHD", "down"),
        "氢键受体": ("nHA", "down"),
        "亲脂性": ("logP", "down"),
        "疏水性": ("logP", "down"),
        "水溶性": ("logS", "up"),
        "渗透性": ("Caco-2 Permeability", "down"),
        "代谢稳定性": ("HLM Stability", "up"),
        "代谢不稳定性": ("HLM Stability", "down"),
    }
    

    for problem_text, (feature, direction) in problem_mappings.items():
        if problem_text in txt:
            pairs.append((feature, direction))
    

    context_keywords = ["liabilities", "concerns", "problems", "issues", "need to be addressed", 
                       "key liabilities", "main concerns", "several liabilities"]
    for keyword in context_keywords:
        if keyword in txt:

            keyword_pos = txt.find(keyword)
            context_window = txt[max(0, keyword_pos-200):keyword_pos+200]
            

            for problem_text, (feature, direction) in problem_mappings.items():
                if problem_text in context_window:
                    pairs.append((feature, direction))
    

    unique_pairs = []
    seen_features = set()
    for feature, direction in pairs:
        nf = normalize_key(feature)
        if not nf:
            continue
        if nf not in seen_features:
            unique_pairs.append((nf, direction))
            seen_features.add(nf)
    
    return unique_pairs


def parse_reasoning(reasoning: str) -> List[Tuple[str, str]]:
    return parse_reasoning_enhanced(reasoning)

############################################################

############################################################

def _interval_distance_to_band(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None and x < lo:
        return lo - x
    if hi is not None and x > hi:
        return x - hi
    return 0.0

def reward_for_improvement(o: float, p: float, feat: str) -> float:

    score = 0.0
    

    weight = get_feature_weight(feat)
    

    if feat in RANGE_TARGETS:
        lo, hi = RANGE_TARGETS[feat]
        dist_o = _interval_distance_to_band(o, lo, hi)
        dist_p = _interval_distance_to_band(p, lo, hi)
        delta_eff = (dist_o - dist_p)
        

        if dist_o > 0 and dist_p == 0:
            score += RANGE_BONUS
        

        if dist_o == 0 and dist_p > 0:
            score += -RANGE_BONUS * 0.8
        

        if delta_eff > 0:
            score += RELATIVE_IMPROVEMENT_WEIGHT * min(delta_eff, 2.0)
        elif delta_eff < 0:

            score += RELATIVE_DEGRADATION_WEIGHT * max(delta_eff, -2.0)
        

        weighted_score = score * weight
        return max(min(weighted_score, MAX_REWARD_PER_FEAT * weight), MIN_PENALTY_PER_FEAT * weight)
    

    direction_more = is_more_is_better(feat)
    if direction_more is None:
        return 0.0
    
    delta_eff = (p - o) if direction_more else (o - p)
    

    if delta_eff > 0:
        score += BASE_IMPROVEMENT_BONUS
    elif delta_eff < 0:

        abs_change = abs(delta_eff)
        relative_change = abs_change / max(abs(o), 0.1)
        
        if relative_change < 0.05:

            score += BASE_DEGRADATION_PENALTY * 0.2
        elif relative_change < 0.15:

            score += BASE_DEGRADATION_PENALTY * 0.5
        else:

            score += BASE_DEGRADATION_PENALTY
    

    if feat in THR_MAP:
        T = THR_MAP[feat]
        if direction_more:
            if o < T <= p:
                score += THR_BONUS
            elif o >= T > p:
                score += THR_PENALTY
        else:
            if o > T >= p:
                score += THR_BONUS
            elif o <= T < p:
                score += THR_PENALTY
    

    if delta_eff > 0:

        relative_improvement = delta_eff / max(abs(o), 0.1)
        score += RELATIVE_IMPROVEMENT_WEIGHT * min(relative_improvement, 2.0)
    elif delta_eff < 0:

        relative_degradation = delta_eff / max(abs(o), 0.1)
        score += RELATIVE_DEGRADATION_WEIGHT * max(relative_degradation, -2.0)
    

    weighted_score = score * weight
    return max(min(weighted_score, MAX_REWARD_PER_FEAT * weight), MIN_PENALTY_PER_FEAT * weight)


############################################################

############################################################

def get_bad_features(ro: Dict[str, float]) -> Set[str]:
    bad = set()
    for _cat, feat_dict in checks.items():
        for raw_feat, ft in feat_dict.items():
            feat = normalize_key(raw_feat)
            if feat not in ro:
                continue
            if check_item(ft, str(ro[feat])):
                bad.add(feat)
    return bad

############################################################

############################################################
def _smiles_variants(s: str):
    vs = []
    s0 = (s or "").strip()
    if not s0:
        return vs
    vs.append(s0)
    vs.append(re.sub(r"\s+", "", s0))
    vs.append(re.sub(r"[@/\\]", "", s0))
    vs.append(s0.replace("–", "-").replace("—", "-").replace("−", "-"))
    if "." in s0:
        vs.append(s0.split(".", 1)[0])
    vs.append(re.sub(r"\[[^\]]+\]", "", s0))
    vs.append(re.sub(r"\(\)", "", s0))
    out, seen = [], set()
    for v in vs:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out




_ALLOWED = set("=#()[]+-@\\/0123456789%:.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*")

def _basic_smiles_sanity(s: str) -> bool:
    if not s or any(ch not in _ALLOWED for ch in s):
        return False

    st = 0
    for ch in s:
        if ch == '(':
            st += 1
        elif ch == ')':
            st -= 1
            if st < 0:
                return False
    if st != 0:
        return False


    digits = [c for c in s if c.isdigit()]
    if len(digits) % 2 == 1:
        return False
    return True


try:
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles
    
    def _is_valid_smiles(s: str) -> bool:
       
        try:

            mol = Chem.MolFromSmiles(s, sanitize=True)
            if mol is not None:
                return True
        except Exception:
            pass
        
        try:

            mol = Chem.MolFromSmiles(s, sanitize=False)
            if mol is not None:
                return True
        except Exception:
            pass
        
        return False
        
except Exception:
    def _is_valid_smiles(s: str) -> bool:
        return _basic_smiles_sanity(s)



def _safe_search(smiles: str, retries: int = 2, sample_idx: Optional[int] = None, role: str = "") -> Dict[str, Any]:

    s = (smiles or "").strip()
    s = re.sub(r"\s*<\s*end\s*>\s*$", "", s, flags=re.IGNORECASE)
    if not s:
        if sample_idx is not None:
            print(f"[ADMET-PREDICT][ERROR] Sample {sample_idx} ({role}) empty smiles")
        return {}

    backoff = 0.2
    last_err = ""
    for t in range(retries + 1):
        try:
            r = search_admet(s) or {}
            if isinstance(r, dict) and r:
                return r
            last_err = f"try{t}: empty dict"
        except Exception as e:
            last_err = f"try{t}: {type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff = min(backoff * 2, 1.6)

    tag = f"Sample {sample_idx} ({role})" if sample_idx is not None else role
    print(f"[ADMET-PREDICT][ERROR] {tag} failed. smiles='{s[:120]}', reason={last_err}")
    return {}


class RewardEngine:
    def __init__(self, w_main: float = 1.0, w_bonus: float = 1.0, coverage_weight: float = 0.3):

        self.w_main = w_main
        self.w_bonus = w_bonus
        self.coverage_weight = coverage_weight

    def compute_sample_enhanced(
        self,
        orig_smiles: str,
        opt_smiles: str,
        reasoning: str,
        idx: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:

        ro_raw = _safe_search(orig_smiles, sample_idx=idx, role="ro")
        rp_raw = _safe_search(opt_smiles, sample_idx=idx, role="rp")


        ro = extract_numeric(normalize_props(ro_raw))
        rp = extract_numeric(normalize_props(rp_raw))

        if isinstance(ro_raw, dict):
            print("[DEBUG] ro_raw sample keys:", list(ro_raw.keys())[:10])
        if isinstance(rp_raw, dict):
            print("[DEBUG] rp_raw sample keys:", list(rp_raw.keys())[:10])

        print("ro keys:", sorted(ro.keys()))
        print("rp keys:", sorted(rp.keys()))
        print("common:", sorted(set(ro.keys()) & set(rp.keys())))

        main_reward = 0.0
        eval_feats: List[str] = []
        


        orig_bad_features = get_bad_features(ro)
        opt_bad_features = get_bad_features(rp)
        

        for feat in orig_bad_features:
            if feat in rp:

                o = ro.get(feat)
                p = rp.get(feat)
                if o is not None and p is not None:
                    if (feat in RANGE_TARGETS) or (is_more_is_better(feat) is not None):
                        eval_feats.append(feat)
                        main_reward += reward_for_improvement(o, p, feat)
            else:

                if rp:
                    main_reward += 0.5
                    eval_feats.append(f"{feat}_removed")
                else:
                    main_reward += 0.1
                    eval_feats.append(f"{feat}_unknown")
        

        new_bad_features = opt_bad_features - orig_bad_features
        for feat in new_bad_features:
            main_reward -= 0.3
            eval_feats.append(f"{feat}_new_bad")
        

        other_features = (set(ro.keys()) & set(rp.keys())) - orig_bad_features
        for feat in other_features:
            o = ro.get(feat)
            p = rp.get(feat)
            if o is not None and p is not None:
                if (feat in RANGE_TARGETS) or (is_more_is_better(feat) is not None):
                    eval_feats.append(feat)
                    main_reward += reward_for_improvement(o, p, feat) * 0.5


        denom_main = max(len(eval_feats), 1)
        


        max_possible_reward = sum(MAX_REWARD_PER_FEAT * get_feature_weight(f.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')) 
                                  for f in eval_feats)
        min_possible_penalty = sum(MIN_PENALTY_PER_FEAT * get_feature_weight(f.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')) 
                                   for f in eval_feats)
        

        if main_reward >= 0:

            if max_possible_reward > 0:
                base_score = main_reward / max_possible_reward
            else:
                base_score = 0.0
        else:

            if min_possible_penalty < 0:
                base_score = main_reward / abs(min_possible_penalty)
            else:
                base_score = -1.0
        


        core_eval_feats = [f for f in eval_feats if f in CORE_FEATURES]
        max_possible_feats = len(CORE_FEATURES)
        coverage_ratio = len(core_eval_feats) / max_possible_feats
        coverage_bonus = min(coverage_ratio, 1.0) * 0.1
        

        improvement_efficiency_bonus = 0.0
        

        has_reasoning = bool(reasoning and reasoning.strip())
        improved_count = 0
        degraded_count = 0
        
        if len(eval_feats) > 0:

            
            for feat in eval_feats:
                clean_feat = feat.replace('_removed', '').replace('_unknown', '').replace('_new_bad', '')
                if clean_feat in ro and clean_feat in rp:
                    o_val = ro[clean_feat]
                    p_val = rp[clean_feat]
                    if o_val is not None and p_val is not None:
                        direction = is_more_is_better(clean_feat)
                        if direction is not None:
                            delta = (p_val - o_val) if direction else (o_val - p_val)
                            if delta > 0:
                                improved_count += 1
                            elif delta < 0:
                                degraded_count += 1
                        elif clean_feat in RANGE_TARGETS:
                            lo, hi = RANGE_TARGETS[clean_feat]
                            dist_o = _interval_distance_to_band(o_val, lo, hi)
                            dist_p = _interval_distance_to_band(p_val, lo, hi)
                            if dist_o > dist_p:
                                improved_count += 1
                            elif dist_p > dist_o:
                                degraded_count += 1
            

            total_changed = improved_count + degraded_count
            if total_changed > 0:
                efficiency = improved_count / total_changed

                if efficiency > 0.5:
                    improvement_efficiency_bonus = (efficiency - 0.5) * 0.2
        


        if base_score >= 0:
            norm_main = min(base_score + coverage_bonus + improvement_efficiency_bonus, 1.0)
        else:
            norm_main = max(base_score, -1.0)


        if has_reasoning:
            feat_dirs = parse_reasoning(reasoning)
            pred_feats = {f for f, _ in feat_dirs}
        else:
            feat_dirs = []
            pred_feats = set()
        
        bad_feats = get_bad_features(ro)
        

        if not has_reasoning:
            bonus_hits = 0
            norm_bonus = 0.0
            bonus_f1 = 0.0

        elif len(bad_feats) == 0:

            bonus_hits = 0
            denom_bonus = max(len(pred_feats), 1)
            norm_bonus = bonus_hits / denom_bonus
            
            if len(pred_feats) == 0:
                bonus_f1 = 1.0
            else:

                bonus_f1 = 0.5
        else:

            bonus_hits = sum(1 for f in pred_feats if f in bad_feats)
            denom_bonus = max(len(pred_feats), 1)
            norm_bonus = bonus_hits / denom_bonus
            

            eval_domain = bad_feats | pred_feats
            if len(eval_domain) == 0:
                bonus_f1 = 0.0
            else:
                y_true = [1 if f in bad_feats else 0 for f in eval_domain]
                y_pred = [1 if f in pred_feats else 0 for f in eval_domain]
                

                if len(set(y_true)) < 2 and len(set(y_pred)) < 2:

                    bonus_f1 = 1.0 if y_true == y_pred else 0.0
                elif len(set(y_true)) < 2:

                    if 1 in y_true:

                        recall = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / len(y_true)
                        bonus_f1 = recall
                    else:

                        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
                        specificity = 1.0 - (fp / len(y_true))
                        bonus_f1 = max(specificity, 0.0)
                elif len(set(y_pred)) < 2:

                    if 1 in y_pred:

                        precision = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / len(y_pred)
                        bonus_f1 = precision
                    else:

                        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
                        specificity = 1.0 - (fn / len(y_true))
                        bonus_f1 = max(specificity, 0.0)
                else:

                    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
                    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
                    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
                    

                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    
                    if precision + recall == 0:
                        bonus_f1 = 0.0
                    else:

                        bonus_f1 = (1 + 4) * (precision * recall) / (4 * precision + recall)
                    

                    if precision < 0.3 and recall > 0.5:
                        bonus_f1 = max(bonus_f1, 0.3)
                    

                    if recall < 0.3 and precision > 0.5:
                        bonus_f1 = max(bonus_f1, 0.3)
                    

                    if precision < 0.3 and recall < 0.3 and tp > 0:
                        bonus_f1 = max(bonus_f1, 0.1)


        if not has_reasoning:

            norm_total = self.w_main * norm_main
        else:

            norm_total = self.w_main * norm_main + self.w_bonus * bonus_f1


        if not ro_raw:
            why_zero = "oracle(ro)查询为空：search_admet(orig_smiles)失败或返回空"
        elif not rp_raw:
            why_zero = "pred(rp)查询为空：search_admet(opt_smiles)失败或返回空"
        elif len(eval_feats) == 0:
            why_zero = "规范化后无可评估交集特征（请检查别名/规范化）"
        else:
            why_zero = ""


        detail = {
            "main_reward": main_reward,
            "overall_optimization_score": base_score,
            "norm_main": norm_main,
            "bonus_hits": bonus_hits,
            "norm_bonus": norm_bonus,
            "bonus_f1": bonus_f1,
            "bad_feats": sorted(list(bad_feats)),
            "num_eval_feats": len(eval_feats),
            "num_pred_feats": len(pred_feats),
            "eval_feats": sorted(list(eval_feats)),
            "pred_feats": sorted(list(pred_feats)),
            "feat_dirs": feat_dirs,
            "has_reasoning": has_reasoning,
            "reasoning_length": len(reasoning) if reasoning else 0,
            "why_zero": why_zero,

            "improved_count": improved_count,
            "degraded_count": degraded_count,
            "improvement_efficiency": improved_count / (improved_count + degraded_count) if (improved_count + degraded_count) > 0 else 0,
            "efficiency_bonus": improvement_efficiency_bonus,
        }
        

        print(f"[DEBUG] Sample {idx}: main_reward(raw)={main_reward:.3f}, eval_feats={len(eval_feats)}")
        print(f"[DEBUG] Sample {idx}: improved={improved_count}, degraded={degraded_count}, efficiency={improved_count/(improved_count+degraded_count) if (improved_count+degraded_count)>0 else 0:.2%}")
        print(f"[DEBUG] Sample {idx}: base_score={base_score:.3f}, coverage_bonus={coverage_bonus:.3f}, efficiency_bonus={improvement_efficiency_bonus:.3f}, norm_main={norm_main:.3f}")
        print(f"[DEBUG] Sample {idx}: has_reasoning={has_reasoning}, reasoning_length={len(reasoning) if reasoning else 0}")
        if has_reasoning:
            print(f"[DEBUG] Sample {idx}: bad_feats={sorted(list(bad_feats))}, pred_feats={sorted(list(pred_feats))}, bonus_f1={bonus_f1:.3f}")
            print(f"[DEBUG] Sample {idx}: feat_dirs={feat_dirs}")
        else:
            print(f"[DEBUG] Sample {idx}: No reasoning provided, bonus_f1=0.0")
        
        return norm_total, detail

    def compute_sample(self, orig_smiles: str, opt_smiles: str, reasoning: str, idx: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        return self.compute_sample_enhanced(orig_smiles, opt_smiles, reasoning, idx)

############################################################

############################################################

def compute_reward_with_bad_feats(orig_smiles, opt_smiles, reasoning):
    engine = RewardEngine(w_main=1.0, w_bonus=1.0)
    norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning)
    return norm_total, detail

############################################################

############################################################

def extract_orig_smiles(rec_o: Dict[str, Any]) -> str:
    instr = rec_o.get("input", "")
    if not instr:
        return ""
    first = instr.splitlines()[0]
    if ":" in first:
        return first.split(":", 1)[1].strip()
    return first.strip()

def parse_reasoning_and_smiles_from_output(output_text: str) -> Tuple[str, str]:
    if not output_text:
        return "", ""
    txt = output_text.strip()
    txt = re.sub(r"\s*<\s*end\s*>\s*$", "", txt, flags=re.IGNORECASE)

    reasoning = ""
    smiles = ""

    for line in txt.splitlines():
        line = line.strip()
        m_r = re.match(r"^reasoning\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m_r and not reasoning:
            reasoning = m_r.group(1).strip()
            continue
        m_s = re.match(r"^optimized[_\s-]*smiles\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m_s and not smiles:
            smiles = re.sub(r"\s*<\s*end\s*>\s*$", "", m_s.group(1).strip(), flags=re.IGNORECASE).strip()

    if not reasoning:

        m = re.search(r"reasoning\s*:\s*(.+?)(?=optimized[_\s-]*smiles|$)", txt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
    if not smiles:
        m = re.search(r"optimized[_\s-]*smiles\s*:\s*([^\s]+)", txt, flags=re.IGNORECASE)
        if m:
            smiles = m.group(1).strip()


    if smiles and '.' in smiles:
        smiles = handle_multiple_molecules(smiles)

    return reasoning, smiles

def handle_multiple_molecules(smiles: str) -> str:

    if not smiles or '.' not in smiles:
        return smiles
    

    molecules = smiles.split('.')
    

    molecules = [mol.strip() for mol in molecules if mol.strip()]
    
    if not molecules:
        return smiles
    



    longest_molecule = max(molecules, key=len)
    
    print(f"[SMILES-PROCESSING] 检测到{len(molecules)}个分子，选择最长的: {longest_molecule[:50]}...")
    
    return longest_molecule

def handle_multiple_molecules_advanced(smiles: str, strategy: str = "longest") -> str:

    if not smiles or '.' not in smiles:
        return smiles
    

    molecules = smiles.split('.')
    molecules = [mol.strip() for mol in molecules if mol.strip()]
    
    if not molecules:
        return smiles
    
    if len(molecules) == 1:
        return molecules[0]
    
    print(f"[SMILES-PROCESSING] 检测到{len(molecules)}个分子，使用策略: {strategy}")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        

        molecule_scores = []
        for i, mol_smiles in enumerate(molecules):
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol is not None:

                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    num_atoms = mol.GetNumAtoms()
                    num_rings = Descriptors.RingCount(mol)
                    

                    if strategy == "drug_like":

                        score = 0
                        if 150 <= mw <= 500: score += 1
                        if logp <= 5: score += 1
                        if tpsa <= 140: score += 1
                        if Descriptors.NumHDonors(mol) <= 5: score += 1
                        if Descriptors.NumHAcceptors(mol) <= 10: score += 1
                    elif strategy == "most_complex":

                        score = num_atoms + num_rings * 2 + len(mol_smiles)
                    elif strategy == "longest":

                        score = len(mol_smiles)
                    else:  # "first"
                        score = i
                    
                    molecule_scores.append((score, mol_smiles, mw, logp, tpsa))
                    print(f"  分子{i+1}: MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}, 分数={score}")
                else:
                    print(f"  分子{i+1}: 无效SMILES")
            except Exception as e:
                print(f"  分子{i+1}: 解析错误 - {e}")
        
        if molecule_scores:

            best_score, best_smiles, mw, logp, tpsa = max(molecule_scores, key=lambda x: x[0])
            print(f"  选择分子: MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}, 分数={best_score}")
            return best_smiles
        else:

            return max(molecules, key=len)
            
    except ImportError:
        print("  RDKit未安装，使用简单策略")
        return max(molecules, key=len)

def extract_pred_smiles(rec_p: Dict[str, Any]) -> str:
    for key, val in rec_p.items():
        if "smiles" in key.lower() and isinstance(val, str):
            s = val.strip()
            if s:
                return s
    out = rec_p.get("output", "")
    if isinstance(out, str) and out.strip():
        _r, s = parse_reasoning_and_smiles_from_output(out)
        return s or ""
    return ""

def extract_optimized_smiles(rec: Dict[str, Any]) -> str:


    out = rec.get("output", "")
    if isinstance(out, str) and out.strip():
        _r, s = parse_reasoning_and_smiles_from_output(out)
        if s:
            return s
    

    for key, val in rec.items():
        if "smiles" in key.lower() and isinstance(val, str):
            s = val.strip()
            if s:
                return s
    
    return ""

def extract_reasoning(rec: Dict[str, Any]) -> str:


    out = rec.get("output", "")
    if isinstance(out, str) and out.strip():
        r, _s = parse_reasoning_and_smiles_from_output(out)
        if r:
            return r
    

    for key, val in rec.items():
        if "reasoning" in key.lower() and isinstance(val, str):
            s = val.strip()
            if s:
                return s
    
    return ""

def _extract_from_output_text(prec: Dict[str, Any]) -> Tuple[str, str]:

    raw = prec.get("output", "")
    if not isinstance(raw, str):
        return "", ""

    txt = raw.strip()


    txt = re.sub(r"\s*<\s*end\s*>\s*$", "", txt, flags=re.IGNORECASE)


    m_r = re.search(
        r"(?is)\breasoning\s*:\s*(.+?)(?:\n+\s*(?:optimized[_\s-]*smiles|smiles)\s*:|$)",
        txt
    )
    reasoning = m_r.group(1).strip() if m_r else ""


    opt_smiles = ""
    for pat in (r"(?im)^\s*optimized[_\s-]*smiles\s*:\s*(.+)$",
                r"(?im)^\s*smiles\s*:\s*(.+)$"):
        m = re.search(pat, txt)
        if m:
            line_rest = m.group(1).strip()

            line_rest = re.sub(r"\s*<\s*end\s*>.*$", "", line_rest, flags=re.IGNORECASE).strip()

            first_token = line_rest.split()[0]

            first_token = first_token.strip(".,;，；。")
            opt_smiles = first_token
            break


    if opt_smiles and '.' in opt_smiles:
        opt_smiles = handle_multiple_molecules(opt_smiles)

    return reasoning, opt_smiles


############################################################

############################################################

def read_reasoning_file(reasoning_path: str) -> Dict[int, str]:

    reasoning_dict = {}
    try:
        with open(reasoning_path, "r", encoding="utf-8") as fr:
            content = fr.read()
            

            if "=== Line" in content:

                sections = re.split(r'=== Line (\d+) ===', content)
                
                for i in range(1, len(sections), 2):
                    if i + 1 < len(sections):
                        line_num = int(sections[i])
                        reasoning_text = sections[i + 1].strip()
                        if reasoning_text:
                            reasoning_dict[line_num] = reasoning_text
            else:

                lines = content.strip().split('\n')
                for line_num, line in enumerate(lines, start=1):
                    reasoning_text = line.strip()
                    if reasoning_text:
                        reasoning_dict[line_num] = reasoning_text
                        
    except Exception as e:
        print(f"[WARN] Failed to read reasoning file: {e}")
    
    return reasoning_dict

def main_from_extracted_enhanced(
    orig_jsonl: str,
    after_smi_path: str,
    out_path: str,
    reasoning_path: str = None,
    w_main: float = 1.0,
    w_bonus: float = 1.0,
    golden_path: str = None,
    calibrate_with_golden: bool = False
):

    engine = RewardEngine(
        w_main=w_main, 
        w_bonus=w_bonus
    )
    
    total_norm_main = 0.0
    total_f1 = 0.0
    total_norm_total = 0.0
    count = 0

    skipped_invalid_after = 0
    paired = 0


    bad_feat_stats = {}
    pred_feat_stats = {}
    reasoning_length_stats = []
    f1_score_distribution = []
    feature_coverage_stats = {}
    has_reasoning_count = 0


    with open(after_smi_path, "r", encoding="utf-8") as f_smiles:
        opt_smiles_list = [line.strip() for line in f_smiles if line.strip()]
    

    reasoning_dict: Dict[int, str] = {}
    if reasoning_path and os.path.exists(reasoning_path):
        reasoning_dict = read_reasoning_file(reasoning_path)
    
    with open(orig_jsonl, "r", encoding="utf-8") as fo, \
         open(out_path, "w", encoding="utf-8") as fw:

        for idx, lo in enumerate(fo):
            line_num = idx + 1
            paired += 1
            

            try:
                orec = json.loads(lo)
            except Exception:
                print(f"[WARN] sample {idx}: bad JSON in orig file, skip")
                continue


            orig_smiles = extract_orig_smiles(orec)


            if idx < len(opt_smiles_list):
                opt_smiles = opt_smiles_list[idx]
            else:
                print(f"[WARN] sample {idx}: no corresponding SMILES in after file, skip")
                continue


            if not _is_valid_smiles(opt_smiles):
                skipped_invalid_after += 1
                print(f"[SKIP] sample {idx}: invalid after SMILES -> {opt_smiles[:80]}")
                continue


            reasoning = reasoning_dict.get(line_num, "")


            norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning, idx=idx)


            reasoning_length_stats.append(len(reasoning))
            f1_score_distribution.append(detail["bonus_f1"])
            

            if detail["has_reasoning"]:
                has_reasoning_count += 1
            

            for feat in detail["bad_feats"]:
                bad_feat_stats[feat] = bad_feat_stats.get(feat, 0) + 1
            

            for feat in detail["pred_feats"]:
                pred_feat_stats[feat] = pred_feat_stats.get(feat, 0) + 1
            

            core_eval_feats = [f for f in detail["eval_feats"] if f in CORE_FEATURES]
            coverage = len(core_eval_feats) / len(CORE_FEATURES)
            feature_coverage_stats[idx] = coverage


            print(f"\n=== Sample {idx} ===")
            print("orig_smiles:", orig_smiles)
            print("opt_smiles :", opt_smiles)
            print("reasoning  :", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
            print("detail     :", detail)


            total_norm_main  += detail["overall_optimization_score"]
            total_f1         += detail["bonus_f1"]
            total_norm_total += norm_total
            count += 1


        if count > 0:
            averages = {
                "overall_optimization_score": round(total_norm_main / count, 4),
                "target_property_f1_score": round(total_f1 / count, 4),
                "avg_total_reward": round(total_norm_total / count, 4),
                "w_main": w_main,
                "w_bonus": w_bonus,
                "paired": paired,
                "used_pairs": count,
                "skipped_invalid_after": skipped_invalid_after,
                "reasoning_loaded": len(reasoning_dict),

                "avg_reasoning_length": round(sum(reasoning_length_stats) / len(reasoning_length_stats), 2),
                "f1_score_stats": {
                    "min": round(min(f1_score_distribution), 4),
                    "max": round(max(f1_score_distribution), 4),
                    "median": round(sorted(f1_score_distribution)[len(f1_score_distribution)//2], 4),
                    "std": round((sum((x - sum(f1_score_distribution)/len(f1_score_distribution))**2 for x in f1_score_distribution) / len(f1_score_distribution))**0.5, 4)
                },
                "top_bad_features": dict(sorted(bad_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_pred_features": dict(sorted(pred_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
                "avg_feature_coverage": round(sum(feature_coverage_stats.values()) / len(feature_coverage_stats), 4),
                "has_reasoning_count": has_reasoning_count,
                "has_reasoning_ratio": round(has_reasoning_count / count, 4) if count > 0 else 0.0
            }


            def _rankdata(vals: List[float]) -> List[float]:

                pairs = sorted((v, i) for i, v in enumerate(vals))
                ranks = [0.0] * len(vals)
                i = 0
                while i < len(pairs):
                    j = i
                    v = pairs[i][0]
                    while j < len(pairs) and pairs[j][0] == v:
                        j += 1

                    avg_rank = (i + 1 + j) / 2.0
                    for k in range(i, j):
                        ranks[pairs[k][1]] = avg_rank
                    i = j
                return ranks

            def _pearson(x: List[float], y: List[float]) -> float:
                n = min(len(x), len(y))
                if n == 0:
                    return 0.0
                x = x[:n]; y = y[:n]
                mx = sum(x)/n; my = sum(y)/n
                num = sum((a-mx)*(b-my) for a,b in zip(x,y))
                denx = sum((a-mx)**2 for a in x)
                deny = sum((b-my)**2 for b in y)
                if denx <= 1e-12 or deny <= 1e-12:
                    return 0.0
                return num / (denx**0.5 * deny**0.5)

            def _spearmanr(xs: List[float], ys: List[float]) -> float:
                rx = _rankdata(xs)
                ry = _rankdata(ys)
                return _pearson(rx, ry)

            def _tier(v: float) -> int:
                if v <= 0.33: return 0
                if v <= 0.66: return 1
                return 2

            def _tier_agree(pred: List[float], gold: List[float]) -> float:
                n = min(len(pred), len(gold))
                if n == 0:
                    return 0.0
                hit = 0
                for i in range(n):
                    if _tier(pred[i]) == _tier(gold[i]):
                        hit += 1
                return round(hit / n, 4)


            fw.write(json.dumps(averages, ensure_ascii=False) + "\n")

    print(f"✅ Averages saved to {out_path}")
    print(f"   paired={paired}, used={count}, skipped_invalid_after={skipped_invalid_after}")
    print(f"   reasoning_loaded={len(reasoning_dict)}")
    print(f"   avg_bonus_f1={averages['avg_bonus_f1']}, f1_stats={averages['f1_score_stats']}")


def main_from_extracted(orig_jsonl: str, after_smi_path: str, out_path: str, reasoning_path: str = None, 
                       w_main: float = 1.0, w_bonus: float = 1.0,
                       golden_path: str = None,
                       calibrate_with_golden: bool = False):
    return main_from_extracted_enhanced(orig_jsonl, after_smi_path, out_path, reasoning_path, 
                                       w_main, w_bonus,
                                       golden_path, calibrate_with_golden)

def main_from_training_data(
    training_jsonl: str,
    out_path: str,
    w_main: float = 1.0,
    w_bonus: float = 1.0
):

    engine = RewardEngine(w_main=w_main, w_bonus=w_bonus)
    
    total_norm_main = 0.0
    total_f1 = 0.0
    total_norm_total = 0.0
    count = 0
    
    skipped_invalid = 0
    skipped_parse_error = 0
    

    bad_feat_stats = {}
    pred_feat_stats = {}
    reasoning_length_stats = []
    f1_score_distribution = []
    feature_coverage_stats = {}
    has_reasoning_count = 0
    
    with open(training_jsonl, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as fw:
        
        for idx, line in enumerate(f):
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"[WARN] sample {idx}: JSON parse error - {e}")
                skipped_parse_error += 1
                continue
            

            input_text = record.get("input", "")
            orig_smiles = ""
            for line_part in input_text.split("\n"):
                if line_part.strip().startswith("Original SMILES:"):
                    orig_smiles = line_part.split(":", 1)[1].strip()
                    break
            
            if not orig_smiles:
                print(f"[WARN] sample {idx}: cannot extract original SMILES")
                skipped_parse_error += 1
                continue
            

            output_text = record.get("output", "")
            reasoning, opt_smiles = _extract_from_output_text({"output": output_text})
            
            if not opt_smiles:
                print(f"[WARN] sample {idx}: cannot extract optimized SMILES")
                skipped_parse_error += 1
                continue
            

            if not _is_valid_smiles(opt_smiles):
                skipped_invalid += 1
                print(f"[SKIP] sample {idx}: invalid optimized SMILES -> {opt_smiles[:80]}")
                continue
            

            norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning, idx=idx)
            

            reasoning_length_stats.append(len(reasoning))
            f1_score_distribution.append(detail["bonus_f1"])
            
            if detail["has_reasoning"]:
                has_reasoning_count += 1
            
            for feat in detail["bad_feats"]:
                bad_feat_stats[feat] = bad_feat_stats.get(feat, 0) + 1
            
            for feat in detail["pred_feats"]:
                pred_feat_stats[feat] = pred_feat_stats.get(feat, 0) + 1
            
            core_eval_feats = [f for f in detail["eval_feats"] if f in CORE_FEATURES]
            coverage = len(core_eval_feats) / len(CORE_FEATURES) if len(CORE_FEATURES) > 0 else 0
            feature_coverage_stats[idx] = coverage
            

            print(f"\n=== Sample {idx} ===")
            print("orig_smiles:", orig_smiles)
            print("opt_smiles :", opt_smiles)
            print("reasoning  :", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
            print("detail     :", detail)
            

            total_norm_main  += detail["overall_optimization_score"]
            total_f1         += detail["bonus_f1"]
            total_norm_total += norm_total
            count += 1
        

        if count > 0:
            averages = {
                "overall_optimization_score": round(total_norm_main / count, 4),
                "target_property_f1_score": round(total_f1 / count, 4),
                "avg_total_reward": round(total_norm_total / count, 4),
                "w_main": w_main,
                "w_bonus": w_bonus,
                "total_samples": idx + 1,
                "used_samples": count,
                "skipped_invalid": skipped_invalid,
                "skipped_parse_error": skipped_parse_error,

                "avg_reasoning_length": round(sum(reasoning_length_stats) / len(reasoning_length_stats), 2) if reasoning_length_stats else 0,
                "f1_score_stats": {
                    "min": round(min(f1_score_distribution), 4) if f1_score_distribution else 0,
                    "max": round(max(f1_score_distribution), 4) if f1_score_distribution else 0,
                    "median": round(sorted(f1_score_distribution)[len(f1_score_distribution)//2], 4) if f1_score_distribution else 0,
                } if f1_score_distribution else {},
                "top_bad_features": dict(sorted(bad_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_pred_features": dict(sorted(pred_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
                "avg_feature_coverage": round(sum(feature_coverage_stats.values()) / len(feature_coverage_stats), 4) if feature_coverage_stats else 0,
                "has_reasoning_count": has_reasoning_count,
                "has_reasoning_ratio": round(has_reasoning_count / count, 4) if count > 0 else 0.0
            }
            
            fw.write(json.dumps(averages, ensure_ascii=False) + "\n")
            
            print(f"\n✅ Results saved to {out_path}")
            print(f"   total_samples={idx+1}, used={count}, skipped_invalid={skipped_invalid}, skipped_parse_error={skipped_parse_error}")
            print(f"   overall_optimization_score={averages['overall_optimization_score']}")
            print(f"   avg_bonus_f1={averages['avg_bonus_f1']}")
            print(f"   avg_total_reward={averages['avg_total_reward']}")
        else:
            print(f"❌ No valid samples processed!")

############################################################

############################################################

if __name__ == "__main__":

    # main_from_training_data(
    #     training_jsonl="/data2/liuhaoran/reasoning_data/reasoning_progress_optimized_updated.jsonl",
    #     out_path="/data2/liuhaoran/result/training_data_rewards.jsonl",
    #     w_main=1.0,
    #     w_bonus=1.0
    # )
    

    main_from_extracted(
        #orig_jsonl="/data2/liuhaoran/val/sample_300_deduplicated_correct.jsonl", 
        #orig_jsonl="/data2/liuhaoran/ood_drug_admet_converted.jsonl",
        #orig_jsonl="/data2/liuhaoran/ood_pairs_reasoning_converted_rest168.jsonl",
        orig_jsonl="/data2/liuhaoran/val/converted_reasoning_test5_fixed.jsonl",

        #after_smi_path="/data2/liuhaoran/result/pretrain_to_sft.txt", 

        #after_smi_path="/data2/liuhaoran/result/base.txt",
        #after_smi_path="/data2/liuhaoran/result/base_clean_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/ether0_clean_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/ether0_ood_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/rl1000_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/rl1200_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/pretrain_to_sft_molnet_clean2_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/rl100_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/pretrain_to_sft_molnet_clean_formatted_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/rl200_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/pretrain_to_sft4_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/rl1600_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/base_smiles.txt",
        #after_smi_path="/data2/liuhaoran/ultrachat-clean/separated_output/smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/ether0_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/pretrain_to_sft3_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/sft_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/rl1200_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/pre_train_baseline_step600_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/rl1230_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/llama-3-8b_sft_ood_step400_rest168_smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/exllm_smiles.txt",
        after_smi_path="/data2/liuhaoran/generated_molecules_t2m_pred_smiles.clean.txt",
        #after_smi_path="/data2/liuhaoran/rl/mistral_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/qwen_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/base_ood_smiles.txt",              
        #after_smi_path="/data2/liuhaoran/rl/sft_ood_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/chemdfm_mini_baseline_smiles.txt",
        #after_smi_path="/data2/liuhaoran/rl/mobo_baseline_smiles.txt",
        #after_smi_path="/data2/liuhaoran/pred_smiles_aligned.txt",
        #after_smi_path="/data2/liuhaoran/rl/rl1220_change_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/noreasoning_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/sft_pretrain_llm8b2000_smiles.txt",
        #after_smi_path="/data2/liuhaoran/val/output/generated_results_smiles.txt",
        #after_smi_path="/data2/liuhaoran/change/sft_and_pretrain_smiles3.txt",
        #after_smi_path="/data2/liuhaoran/change/rl1400_smiles.txt",
        #after_smi_path="/data2/liuhaoran/val/output/generated_results_smiles.txt",
        #after_smi_path="/data2/liuhaoran/ultrachat-clean/separated_output_base/smiles.txt",
        #after_smi_path="/data2/liuhaoran/result/pretrain_to_sft_ood_smiles.txt",
        #reasoning_path="/data2/liuhaoran/result/pretrain_to_sft_reasoning.txt",

        #reasoning_path="/data2/liuhaoran/result/base_clean_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/ether0_clean_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/ether0_ood_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/rl1000_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/rl100_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/rl200_change_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/pretrain_to_sft_molnet_clean_formatted_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/sft_pretrain_llm8b2000_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/sft_and_pretrain_reasoning3.txt",
        #reasoning_path="/data2/liuhaoran/change/rl1400_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/rl200_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/pretrain_to_sft4_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/rl1600_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/base_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/ultrachat-clean/separated_output/reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/ether0_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/admet/sft_pretrain_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/change/sft_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/pre_train_baseline_step600_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/mistral_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/rl1230_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/llama-3-8b_sft_ood_codeblock_step200_rest168_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/exllm_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/qwen_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/base_ood_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/sft_ood_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/chemdfm_mini_baseline_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/mobo_baseline_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/mollm_baseline_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/rl/pretrain_to_sft3_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/val/output/generated_results_rationale.txt",
        #reasoning_path="/data2/liuhaoran/val/output/generated_results_rationale.txt",
        #reasoning_path="/data2/liuhaoran/ultrachat-clean/separated_output_base/reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/rl1200_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/pretrain_to_sft_molnet_clean2_reasoning.txt",
        #reasoning_path="/data2/liuhaoran/result/pretrain_to_sft_ood_reasoning.txt",
        #out_path="/data2/liuhaoran/admet/pretrain_to_sft_results.jsonl",
        #out_path="/data2/liuhaoran/admet/rl1200.jsonl",
        #out_path="/data2/liuhaoran/admet/rl100.jsonl",
        #out_path="/data2/liuhaoran/admet/pretrain_to_sft_molnet_clean_formatted.jsonl",
        #out_path="/data2/liuhaoran/admet/deepseek_results.jsonl",
        #out_path="/data2/liuhaoran/change/admet/pretrain_to_sft3.jsonl",
        #out_path="/data2/liuhaoran/change/admet/sft.jsonl",
        #out_path="/data2/liuhaoran/admet/mistral.jsonl",
        #out_path="/data2/liuhaoran/admet/qwen.jsonl",
        #out_path="/data2/liuhaoran/admet/base_ood.jsonl",
        #out_path="/data2/liuhaoran/change/admet/sft_ood.jsonl",
        #out_path="/data2/liuhaoran/change/admet/chemdfm_mini_baseline.jsonl",
        #out_path="/data2/liuhaoran/change/admet/mobo_baseline.jsonl",
        #out_path="/data2/liuhaoran/rl/pre_train_baseline_step600.jsonl",
        #out_path="/data2/liuhaoran/rl/rl1230.jsonl",
        out_path="/data2/liuhaoran/result/ood_train.jsonl",
        #out_path="/data2/liuhaoran/change/admet/pretrain_to_sft10.jsonl",
        #out_path="/data2/liuhaoran/change/admet/sft_pretrain_llm8b2000.jsonl",
        #out_path="/data2/liuhaoran/change/admet/sft_and_pretrain3.jsonl",
        #out_path="/data2/liuhaoran/change/admet/rl1400.jsonl",
        #out_path="/data2/liuhaoran/admet/base_test.jsonl",
        w_main=0.8,
        w_bonus=1.0
    )
