

import os
import re
import json
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class ReasoningRichnessEvaluator:

    
    def __init__(
        self,
        val_data_path: str = "val_data.jsonl",
        sent_limit: int = 5000,
        num_clusters: int = 64,
        pct_max_dist: float = 0.9,
        sentence_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = 'cpu',
        local_files_only: bool = False
    ):
        
        self.val_data_path = val_data_path
        self.sent_limit = sent_limit
        self.num_clusters = num_clusters
        self.pct_max_dist = max(0.75, min(0.99, pct_max_dist))
        self.device = device
        self.local_files_only = local_files_only
        

        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

            import transformers
            transformers.utils.hub.HF_HUB_OFFLINE = True
        

        self.sentence_model = None
        self.cluster_centers = None
        self.center_max_norm_dist = None
        self.richness_available = False
        

        self._build_cluster_centers(sentence_model_name)
    
    def _split_into_sentences(self, text: str) -> List[str]:

        parts = re.split(r"[\n\.!?]\s+", text)
        return [s.strip() for s in parts if len(s.strip()) >= 8]
    
    def _load_val_data(self) -> List[dict]:

        all_data = []
        path = self.val_data_path
        if not os.path.exists(path):
            print(f"[WARNING] 验证集数据文件不存在: {path}")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
 
            is_jsonl = False
            if lines:
                try:
                    json.loads(lines[0])
                    is_jsonl = True
                except (json.JSONDecodeError, ValueError):
                    is_jsonl = False
            

            if is_jsonl:

                for line in lines:
                    all_data.append(json.loads(line))
                print(f"[INFO] 从 {path} 加载了 {len(all_data)} 个JSONL样本")
            else:

                for line in lines:
                    all_data.append({'reasoning': line})
                print(f"[INFO] 从 {path} 加载了 {len(lines)} 个纯文本样本")
                
        except Exception as e:
            print(f"[ERROR] 加载验证集数据文件 {path} 失败: {e}")
        

        seen = set()
        unique_data = []
        for item in all_data:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        print(f"[INFO] 总共加载 {len(unique_data)} 个唯一样本")
        return unique_data
    
    def _extract_reasoning_sentences(self, data: List[dict]) -> List[str]:
        """从验证集数据中提取推理句子"""
        reasoning_sentences = []
        
        for item in data:

            reasoning = ""
            

            if 'output' in item:
                output_text = item['output']
                if "Reasoning:" in output_text:
                    reasoning = output_text.split("Reasoning:")[1]
                    if "Optimized_SMILES:" in reasoning:
                        reasoning = reasoning.split("Optimized_SMILES:")[0]
                    reasoning = reasoning.strip()
            

            if not reasoning and 'reasoning' in item:
                reasoning = item['reasoning']
            
   
            if reasoning:
                sentences = self._split_into_sentences(reasoning)
                reasoning_sentences.extend(sentences)
            

            if len(reasoning_sentences) >= self.sent_limit:
                break
        
        return reasoning_sentences
    
    def _check_model_cache(self, model_name: str) -> bool:
        """检查模型是否在缓存中"""
        import os
        from pathlib import Path
        

        model_cache_name = model_name.replace("/", "--")
        model_id = model_name.replace("sentence-transformers/", "")
        

        cache_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "sentence-transformers",
            Path.home() / ".cache" / "transformers",
        ]
        

        if os.environ.get("HF_HOME"):
            cache_paths.append(Path(os.environ["HF_HOME"]) / "hub")
        if os.environ.get("XDG_CACHE_HOME"):
            cache_paths.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
        

        for cache_path in cache_paths:
            if not cache_path.exists():
                continue
            try:

                for item in cache_path.iterdir():
                    if item.is_dir():

                        if model_cache_name in item.name or f"models--{model_cache_name}" in item.name:

                            if any(item.iterdir()):
                                return True

                        if model_id in item.name:
                            if any(item.iterdir()):
                                return True
            except (PermissionError, OSError):

                continue
        
        return False
    
    def _build_cluster_centers(self, sentence_model_name: str):

        try:
            print("[RICHNESS] 开始构建推理丰富度评估器...")
            
            # 如果设置了 local_files_only，提前设置离线模式环境变量
            if self.local_files_only:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                print("[RICHNESS] 已启用离线模式（local_files_only=True）")
            
            # 加载验证集数据
            print("[RICHNESS] 步骤1: 加载验证集数据...")
            val_data = self._load_val_data()
            
            # 提取推理句子
            print("[RICHNESS] 步骤2: 提取推理句子...")
            reasoning_sentences = self._extract_reasoning_sentences(val_data)
            
            if len(reasoning_sentences) < 8:
                print(f"[RICHNESS] 语料不足（{len(reasoning_sentences)}条），跳过丰富度评估器构建")
                return
            
            print(f"[RICHNESS] 收集到 {len(reasoning_sentences)} 条推理句子")
            
            # 初始化句子编码模型（使用官方示例的简洁方式）
            print(f"[RICHNESS] 步骤3: 初始化句子编码模型 ({sentence_model_name})...")
            
            # 检查是否有 modelscope 缓存路径
            modelscope_cache_path = os.path.expanduser(
                f"~/.cache/modelscope/hub/models/{sentence_model_name}"
            )
            if os.path.exists(modelscope_cache_path):
                print(f"[RICHNESS] 检测到 ModelScope 缓存路径: {modelscope_cache_path}")
                sentence_model_name = modelscope_cache_path
            
            # 如果设置了 local_files_only，先设置离线模式环境变量
            if self.local_files_only:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                print(f"[RICHNESS] 离线模式已启用（local_files_only=True）")
            
            # 直接使用官方示例的方式：SentenceTransformer(model_name)
            # 如果有本地路径，会自动识别；否则会尝试从 HuggingFace 下载或使用缓存
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name, device=self.device)
                print(f"[RICHNESS] 模型加载成功")
            except Exception as e:
                error_msg = str(e)
                if 'huggingface.co' in error_msg or 'connect' in error_msg.lower() or 'Network is unreachable' in error_msg:
                    if self.local_files_only:
                        print(f"[RICHNESS] 离线模式加载失败: 网络不可用且缓存中找不到模型")
                        print(f"[RICHNESS] 解决方案:")
                        print(f"[RICHNESS]   1. 在有网络的机器上下载模型:")
                        print(f"[RICHNESS]      python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{sentence_model_name}')\"")
                        print(f"[RICHNESS]   2. 将 ~/.cache/huggingface/ 目录复制到当前机器")
                        print(f"[RICHNESS]   3. 或使用本地模型路径（修改 sentence_model_name 参数）")
                        print(f"[RICHNESS]   4. 或设置 local_files_only=False 以允许在线下载（如果网络可用）")
                    else:
                        print(f"[RICHNESS] 网络连接失败，尝试从缓存加载...")
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        os.environ['HF_DATASETS_OFFLINE'] = '1'
                        try:
                            self.sentence_model = SentenceTransformer(sentence_model_name, device=self.device)
                            print(f"[RICHNESS] 成功从缓存加载模型")
                        except Exception as e2:
                            print(f"[RICHNESS] 缓存加载也失败: {e2}")
                            print(f"[RICHNESS] 请确保模型已下载到缓存或网络可用")
                            raise
                    raise
                else:
                    # 其他错误直接抛出
                    raise
            
            # 编码句子为向量
            print("[RICHNESS] 步骤4: 编码句子为向量...")
            embeddings = self.sentence_model.encode(
                reasoning_sentences,
                batch_size=64,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            print(f"[RICHNESS] 向量维度: {embeddings.shape}")
            
            # KMeans聚类
            k = min(self.num_clusters, max(8, len(reasoning_sentences) // 50))
            print(f"[RICHNESS] 步骤5: 进行KMeans聚类, k={k}...")
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            kmeans.fit(embeddings)
            self.cluster_centers = kmeans.cluster_centers_.astype(np.float32)
            print(f"[RICHNESS] 聚类中心形状: {self.cluster_centers.shape}")
            
            # 计算归一化基准（验证集句子到最近中心的距离分布）
            print("[RICHNESS] 步骤6: 计算归一化基准（基于验证集距离分布）...")
            dists = pairwise_distances(embeddings, self.cluster_centers, metric='cosine')
            nn_dists = dists.min(axis=1)  # 每个句子到最近中心的距离
            self.center_max_norm_dist = float(np.quantile(nn_dists, self.pct_max_dist))
            
            if self.center_max_norm_dist < 1e-6:
                self.center_max_norm_dist = 1.0
                print("[RICHNESS] 警告: 归一化基准过小，设置为1.0")
            
            self.richness_available = True
            print(f"[RICHNESS] 构建完成!")
            print(f"[RICHNESS] - 聚类中心数量: {self.cluster_centers.shape[0]}")
            print(f"[RICHNESS] - 归一化基准距离: {self.center_max_norm_dist:.4f}")
            print(f"[RICHNESS] - 距离分布: min={nn_dists.min():.4f}, "
                  f"median={np.median(nn_dists):.4f}, "
                  f"max={nn_dists.max():.4f}, "
                  f"{int(self.pct_max_dist*100)}%分位={self.center_max_norm_dist:.4f}")
            
        except Exception as e:
            print(f"[RICHNESS] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.richness_available = False
    
    def evaluate(self, reasoning_text: str) -> Tuple[float, dict]:

        if not self.richness_available:
            return 0.0, {
                'available': False,
                'error': 'Richness evaluator not initialized'
            }
        
        try:
            # 拆分句子
            sentences = self._split_into_sentences(reasoning_text)
            
            if len(sentences) == 0:
                return 0.0, {
                    'available': True,
                    'num_sentences': 0,
                    'error': 'No valid sentences found'
                }
            

            sentence_embeddings = self.sentence_model.encode(
                sentences,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            

            dists = pairwise_distances(sentence_embeddings, self.cluster_centers, metric='cosine')
            nn_dists = dists.min(axis=1)  # 每个句子到最近中心的距离
            mean_nn_dist = float(nn_dists.mean())

            normalized_distance = mean_nn_dist / self.center_max_norm_dist
            
            optimal_center = 0.85  # 最优中心点
            
            if normalized_distance < 0.5:

                richness_score = 0.4 + (normalized_distance / 0.5) * 0.3  # 0.4-0.7
                tier = "Tier 1: 过于接近验证集（可能缺乏创新）"
                
            elif 0.5 <= normalized_distance < 0.65:

                richness_score = 0.7 + ((normalized_distance - 0.5) / 0.15) * 0.2  # 0.7-0.9
                tier = "Tier 2: 良好（接近最优质量区间）"
                
            elif 0.65 <= normalized_distance < 1.05:

                deviation = abs(normalized_distance - optimal_center)
                sigma = 0.15  # 控制曲线宽度
                gaussian_factor = np.exp(-(deviation ** 2) / (2 * sigma ** 2))
                richness_score = 0.9 + 0.1 * gaussian_factor  # 0.9-1.0
                
                if 0.8 <= normalized_distance <= 0.9:
                    tier = "Tier 3: 优秀★★★（质量与创新完美平衡）"
                else:
                    tier = "Tier 3: 优秀（质量与创新良好平衡）"
                
            elif 1.05 <= normalized_distance < 1.3:

                richness_score = 0.9 - ((normalized_distance - 1.05) / 0.25) * 0.3  # 0.9-0.6
                tier = "Tier 4: 中等（稍偏离最优区间）"
                
            elif 1.3 <= normalized_distance < 1.8:

                richness_score = 0.6 - ((normalized_distance - 1.3) / 0.5) * 0.3  # 0.6-0.3
                tier = "Tier 5: 较差（偏离验证集，质量可疑）"
                
            else:

                richness_score = max(0.1, 0.3 - (normalized_distance - 1.8) * 0.05)  # 0.3->0.1
                tier = "Tier 6: 极差（与验证集完全不同，可能是模板或低质量）⚠️"
            

            dist_std = float(nn_dists.std())
            min_nn_dist = float(nn_dists.min())
            max_nn_dist = float(nn_dists.max())
            
            return richness_score, {
                'available': True,
                'num_sentences': len(sentences),
                'mean_distance': mean_nn_dist,
                'normalized_distance': normalized_distance,
                'normalization_ceiling': self.center_max_norm_dist,
                'distance_stats': {
                    'min': min_nn_dist,
                    'max': max_nn_dist,
                    'mean': mean_nn_dist,
                    'std': dist_std
                },
                'richness_score': richness_score,
                'tier': tier,
                'optimal_center': optimal_center,
                'scoring_philosophy': '钟形曲线：最优点0.85（质量与创新完美平衡），偏离越多分数越低，极度偏离(>1.8或<0.5)大幅降分'
            }
            
        except Exception as e:
            print(f"[RICHNESS] 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {
                'available': True,
                'error': str(e)
            }
    
    def batch_evaluate(self, reasoning_texts: List[str]) -> List[Tuple[float, dict]]:
        """批量评估多个推理文本"""
        results = []
        for reasoning_text in reasoning_texts:
            score, details = self.evaluate(reasoning_text)
            results.append((score, details))
        return results


def main():
    """评估生成数据相对于验证集数据的推理丰富度"""
    import json
    import numpy as np
    

    val_data_path = "val_data.jsonl"
    

    generated_data_path = "generated_data.jsonl"
    

    print("=" * 60)
    print("推理丰富度评估器 - 初始化（基于验证集数据）")
    print("=" * 60)
    evaluator = ReasoningRichnessEvaluator(
        val_data_path=val_data_path,
        sent_limit=5000,
        num_clusters=64,
        pct_max_dist=0.9,
        device='cpu',
        local_files_only=False 
    )
    
    if not evaluator.richness_available:
        print("[ERROR] 评估器初始化失败，退出")
        return
    
    def extract_reasoning_from_item(item):
        """从数据项中提取reasoning部分"""
        reasoning = ""
        if 'output' in item:
            output_text = item['output']
            if "Reasoning:" in output_text:
                reasoning = output_text.split("Reasoning:")[1]
                if "Optimized_SMILES:" in reasoning:
                    reasoning = reasoning.split("Optimized_SMILES:")[0]
                reasoning = reasoning.strip()
        if not reasoning and 'reasoning' in item:
            reasoning = item['reasoning']
        return reasoning
    
    def evaluate_data_file(file_path, data_name, max_samples=None):

        print(f"\n正在评估 {data_name}: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[ERROR] 数据文件不存在: {file_path}")
            return None
        
        all_scores = []
        all_details = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and line_num > max_samples:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    reasoning = extract_reasoning_from_item(item)
                    
                    if reasoning:
                        # 评估这个推理文本
                        score, details = evaluator.evaluate(reasoning)
                        all_scores.append(score)
                        all_details.append(details)
                        
                        if line_num <= 3:
                            print(f"  样本 {line_num}: 分数={score:.4f}, 距离={details.get('mean_distance', 0):.4f}")
                
                except Exception as e:
                    print(f"[WARNING] 处理第 {line_num} 行时出错: {e}")
                    continue
        
        return all_scores, all_details

    print("\n" + "=" * 60)
    print("步骤1: 评估验证集数据自身（用于对比）")
    print("=" * 60)
    
    val_scores = []
    val_details = []
    

    val_sample_size = 200  
    scores, details = evaluate_data_file(val_data_path, "验证集数据", max_samples=val_sample_size)
    if scores:
        val_scores.extend(scores)
        val_details.extend(details)
    

    print("\n" + "=" * 60)
    print("步骤2: 评估训练集数据")
    print("=" * 60)
    
    train_data_paths = [
        "train_data_1.jsonl",
        "train_data_2.jsonl"
    ]
    
    train_scores = []
    train_details = []
    train_sample_size = 200  
    
    for train_path in train_data_paths:
        scores, details = evaluate_data_file(train_path, f"训练集数据 ({os.path.basename(train_path)})", max_samples=train_sample_size)
        if scores:
            train_scores.extend(scores)
            train_details.extend(details)
    

    print("\n" + "=" * 60)
    print("步骤3: 评估生成数据")
    print("=" * 60)
    
    gen_scores, gen_details = evaluate_data_file(generated_data_path, "生成数据")
    

    print("\n" + "=" * 60)
    print("对比分析结果")
    print("=" * 60)
    
    def print_statistics(scores, name):
        """打印统计信息"""
        if not scores:
            print(f"{name}: 无数据")
            return
        
        print(f"\n{name}统计 ({len(scores)} 个样本):")
        print(f"  平均分: {np.mean(scores):.4f}")
        print(f"  中位数: {np.median(scores):.4f}")
        print(f"  标准差: {np.std(scores):.4f}")
        print(f"  最小值: {np.min(scores):.4f}")
        print(f"  最大值: {np.max(scores):.4f}")
        print(f"  25%分位: {np.percentile(scores, 25):.4f}")
        print(f"  75%分位: {np.percentile(scores, 75):.4f}")
        
 
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(scores, bins=bins)
        print(f"  分数分布:")
        for i in range(len(bins)-1):
            count = hist[i]
            pct = count / len(scores) * 100
            print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({pct:.1f}%)")
    
    print_statistics(val_scores, "验证集数据")
    if train_scores:
        print_statistics(train_scores, "训练集数据")
    print_statistics(gen_scores, "生成数据")
    

    if val_scores and train_scores:
        print("\n" + "=" * 60)
        print("对比分析：验证集 vs 训练集")
        print("=" * 60)
        
        val_mean = np.mean(val_scores)
        train_mean = np.mean(train_scores)
        diff = train_mean - val_mean
        diff_pct = (diff / val_mean) * 100 if val_mean > 0 else 0
        
        print(f"平均分差异: {diff:+.4f} ({diff_pct:+.2f}%)")
        print(f"  验证集数据: {val_mean:.4f} ± {np.std(val_scores):.4f}")
        print(f"  训练集数据: {train_mean:.4f} ± {np.std(train_scores):.4f}")
    

    if val_scores and gen_scores:
        print("\n" + "=" * 60)
        print("差异分析")
        print("=" * 60)
        
        val_mean = np.mean(val_scores)
        gen_mean = np.mean(gen_scores)
        diff = gen_mean - val_mean
        diff_pct = (diff / val_mean) * 100 if val_mean > 0 else 0
        

        val_std = np.std(val_scores)
        gen_std = np.std(gen_scores)
        pooled_std = np.sqrt((val_std**2 + gen_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        print(f"平均分差异: {diff:+.4f} ({diff_pct:+.2f}%)")
        print(f"  验证集数据: {val_mean:.4f} ± {val_std:.4f}")
        print(f"  生成数据: {gen_mean:.4f} ± {gen_std:.4f}")
        
 
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(gen_scores, val_scores)
            
            print(f"\n统计检验 (独立样本t-test):")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.6f}")
            print(f"  效应量 (Cohen's d): {cohens_d:.4f}")

            is_significant = p_value < 0.05
            

            if abs(cohens_d) < 0.2:
                effect_size = "很小"
            elif abs(cohens_d) < 0.5:
                effect_size = "小"
            elif abs(cohens_d) < 0.8:
                effect_size = "中等"
            else:
                effect_size = "大"
            
            print(f"\n结论:")
            if is_significant:
                if diff > 0:
                    if abs(cohens_d) >= 0.5:
                        print(f"  ✓ 生成数据的新颖度显著高于验证集数据")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 效应量{effect_size} (Cohen's d={cohens_d:.4f})")
                        print(f"     - 平均提升: {diff:+.4f} ({diff_pct:+.2f}%)")
                    else:
                        print(f"  ⚠ 生成数据的新颖度略高于验证集数据（统计显著但效应量小）")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 但效应量{effect_size} (Cohen's d={cohens_d:.4f})，实际提升有限")
                        print(f"     - 平均提升: {diff:+.4f} ({diff_pct:+.2f}%)")
                else:
                    if abs(cohens_d) >= 0.5:
                        print(f"  ✗ 生成数据的新颖度显著低于验证集数据")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 效应量{effect_size} (Cohen's d={cohens_d:.4f})")
                        print(f"     - 平均下降: {diff:+.4f} ({diff_pct:+.2f}%)")
                    else:
                        print(f"  ⚠ 生成数据的新颖度略低于验证集数据（统计显著但效应量小）")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 但效应量{effect_size} (Cohen's d={cohens_d:.4f})，实际下降有限")
                        print(f"     - 平均下降: {diff:+.4f} ({diff_pct:+.2f}%)")
            else:
                print(f"  ≈ 生成数据的新颖度与验证集数据无显著差异")
                print(f"     - 统计不显著 (p={p_value:.4f} >= 0.05)")
                print(f"     - 平均差异: {diff:+.4f} ({diff_pct:+.2f}%)")
            
        except ImportError:

            print(f"\n(需要scipy进行统计检验，使用简单阈值判断)")
            if abs(diff) > 0.05:
                if diff > 0:
                    print(f"  ✓ 生成数据的新颖度高于验证集数据 (+{diff:.4f})")
                else:
                    print(f"  ✗ 生成数据的新颖度低于验证集数据 ({diff:.4f})")
            else:
                print(f"  ≈ 生成数据的新颖度与验证集数据相近 (差异: {diff:+.4f})")
    

    if train_scores and gen_scores:
        print("\n" + "=" * 60)
        print("对比分析：训练集 vs 生成数据")
        print("=" * 60)
        
        train_mean = np.mean(train_scores)
        gen_mean = np.mean(gen_scores)
        diff = gen_mean - train_mean
        diff_pct = (diff / train_mean) * 100 if train_mean > 0 else 0
        

        train_std = np.std(train_scores)
        gen_std = np.std(gen_scores)
        pooled_std = np.sqrt((train_std**2 + gen_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        print(f"平均分差异: {diff:+.4f} ({diff_pct:+.2f}%)")
        print(f"  训练集数据: {train_mean:.4f} ± {train_std:.4f}")
        print(f"  生成数据: {gen_mean:.4f} ± {gen_std:.4f}")
        

        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_ind(gen_scores, train_scores)
            
            print(f"\n统计检验 (独立样本t-test):")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.6f}")
            print(f"  效应量 (Cohen's d): {cohens_d:.4f}")
            

            is_significant = p_value < 0.05
            
            # 判断效应量大小
            if abs(cohens_d) < 0.2:
                effect_size = "很小"
            elif abs(cohens_d) < 0.5:
                effect_size = "小"
            elif abs(cohens_d) < 0.8:
                effect_size = "中等"
            else:
                effect_size = "大"
            
            print(f"\n结论:")
            if is_significant:
                if diff > 0:
                    if abs(cohens_d) >= 0.5:
                        print(f"  ✓ 生成数据的新颖度显著高于训练集数据")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 效应量{effect_size} (Cohen's d={cohens_d:.4f})")
                        print(f"     - 平均提升: {diff:+.4f} ({diff_pct:+.2f}%)")
                    else:
                        # 对于小效应量，给出更详细的解释
                        if abs(cohens_d) >= 0.3:
                            print(f"  ✓ 生成数据的新颖度高于训练集数据（统计显著，效应量较小但可察觉）")
                            print(f"     - 统计显著 (p={p_value:.4f})")
                            print(f"     - 效应量{effect_size} (Cohen's d={cohens_d:.4f})")
                            print(f"     - 平均提升: {diff:+.4f} ({diff_pct:+.2f}%)")
                            print(f"     - 说明: 虽然效应量较小，但差异在统计上显著，且提升幅度为{diff_pct:.1f}%")
                        else:
                            print(f"  ⚠ 生成数据的新颖度略高于训练集数据（统计显著但效应量很小）")
                            print(f"     - 统计显著 (p={p_value:.4f})")
                            print(f"     - 但效应量{effect_size} (Cohen's d={cohens_d:.4f})，实际提升有限")
                            print(f"     - 平均提升: {diff:+.4f} ({diff_pct:+.2f}%)")
                            print(f"     - 说明: 差异在统计上显著，但实际差异很小（{diff_pct:.1f}%），可能不具有实际意义")
                else:
                    if abs(cohens_d) >= 0.5:
                        print(f"  ✗ 生成数据的新颖度显著低于训练集数据")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 效应量{effect_size} (Cohen's d={cohens_d:.4f})")
                        print(f"     - 平均下降: {diff:+.4f} ({diff_pct:+.2f}%)")
                    else:
                        print(f"  ⚠ 生成数据的新颖度略低于训练集数据（统计显著但效应量小）")
                        print(f"     - 统计显著 (p={p_value:.4f})")
                        print(f"     - 但效应量{effect_size} (Cohen's d={cohens_d:.4f})，实际下降有限")
                        print(f"     - 平均下降: {diff:+.4f} ({diff_pct:+.2f}%)")
            else:
                print(f"  ≈ 生成数据的新颖度与训练集数据无显著差异")
                print(f"     - 统计不显著 (p={p_value:.4f} >= 0.05)")
                print(f"     - 平均差异: {diff:+.4f} ({diff_pct:+.2f}%)")
            
        except ImportError:
            # 如果没有scipy，使用简单的阈值判断
            print(f"\n(需要scipy进行统计检验，使用简单阈值判断)")
            if abs(diff) > 0.05:
                if diff > 0:
                    print(f"  ✓ 生成数据的新颖度高于训练集数据 (+{diff:.4f})")
                else:
                    print(f"  ✗ 生成数据的新颖度低于训练集数据 ({diff:.4f})")
            else:
                print(f"  ≈ 生成数据的新颖度与训练集数据相近 (差异: {diff:+.4f})")


if __name__ == '__main__':
    main()
