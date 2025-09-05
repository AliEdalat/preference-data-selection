import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional

class RewardFusion:
    def __init__(self):
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
    
    def load_rewards(self, file1_path: str, file2_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load reward data from two JSONL files"""
        rewards1 = []
        with open(file1_path, 'r') as f:
            for line in f:
                rewards1.append(json.loads(line.strip()))
        
        rewards2 = []
        with open(file2_path, 'r') as f:
            for line in f:
                rewards2.append(json.loads(line.strip()))
        
        return rewards1, rewards2
    
    def extract_reward_values(self, rewards1: List[Dict], rewards2: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and normalize reward values from both files"""
        # Extract reward differences from first file
        reward_diffs1 = np.array([item['reward_diff'] for item in rewards1]).reshape(-1, 1)
        
        # Extract rewards from second file (handle different key names)
        reward_diffs2 = []
        for item in rewards2:
            if 'reward' in item:
                reward_diffs2.append(item['reward'][0] if isinstance(item['reward'], list) else item['reward'])
            elif 'reward_diff' in item:
                reward_diffs2.append(item['reward_diff'])
            else:
                # If no direct reward, calculate from ppl values
                ppl_chosen = np.mean(item['ppl_chosen']) if 'ppl_chosen' in item else 1.0
                ppl_rejected = np.mean(item['ppl_rejected']) if 'ppl_rejected' in item else 1.0
                reward_diffs2.append(ppl_rejected - ppl_chosen)  # Lower perplexity is better
        
        reward_diffs2 = np.array(reward_diffs2).reshape(-1, 1)
        
        # Handle different lengths by truncating to the shorter one
        min_length = min(len(reward_diffs1), len(reward_diffs2))
        reward_diffs1 = reward_diffs1[:min_length]
        reward_diffs2 = reward_diffs2[:min_length]
        
        # Normalize both reward sets
        norm_rewards1 = self.scaler1.fit_transform(reward_diffs1)
        norm_rewards2 = self.scaler2.fit_transform(reward_diffs2)
        
        return norm_rewards1.flatten(), norm_rewards2.flatten()
    
    def weighted_average_fusion(self, rewards1: np.ndarray, rewards2: np.ndarray, 
                              weight1: float = 0.5) -> np.ndarray:
        """Simple weighted average fusion"""
        weight2 = 1.0 - weight1
        return weight1 * rewards1 + weight2 * rewards2
    
    def bayesian_fusion(self, rewards1: np.ndarray, rewards2: np.ndarray, 
                       prior_strength: float = 1.0) -> np.ndarray:
        """
        Bayesian fusion as mentioned in the paper
        Uses uncertainty estimates to weight rewards
        """
        # Estimate uncertainty from variance (simplified)
        uncertainty1 = 1.0 / (np.std(rewards1) + 1e-8)
        uncertainty2 = 1.0 / (np.std(rewards2) + 1e-8)
        
        # Calculate weights based on uncertainty
        total_uncertainty = uncertainty1 + uncertainty2 + prior_strength
        weight1 = (uncertainty1 + prior_strength/2) / total_uncertainty
        weight2 = (uncertainty2 + prior_strength/2) / total_uncertainty
        
        return weight1 * rewards1 + weight2 * rewards2
    
    def attention_based_fusion(self, rewards1: np.ndarray, rewards2: np.ndarray,
                             temperature: float = 1.0) -> np.ndarray:
        """
        Attention-based fusion that dynamically weights rewards
        based on their confidence scores
        """
        # Calculate confidence scores (absolute values as simple proxy)
        confidence1 = np.abs(rewards1)
        confidence2 = np.abs(rewards2)
        
        # Apply softmax to get attention weights
        max_conf = np.maximum(confidence1, confidence2)
        exp_conf1 = np.exp((confidence1 - max_conf) / temperature)
        exp_conf2 = np.exp((confidence2 - max_conf) / temperature)
        
        sum_exp = exp_conf1 + exp_conf2
        weight1 = exp_conf1 / sum_exp
        weight2 = exp_conf2 / sum_exp
        
        return weight1 * rewards1 + weight2 * rewards2
    
    def orthogonal_fusion(self, rewards1: np.ndarray, rewards2: np.ndarray,
                        diversity_strength: float = 0.1) -> np.ndarray:
        """
        Fusion that encourages orthogonality between reward signals
        to reduce redundancy
        """
        # Calculate correlation
        correlation = np.corrcoef(rewards1, rewards2)[0, 1]
        
        # Adjust weights based on correlation
        if abs(correlation) > 0.8:  # Highly correlated
            # When highly correlated, emphasize diversity
            weight1 = 0.5 - diversity_strength * np.sign(correlation)
            weight2 = 0.5 + diversity_strength * np.sign(correlation)
        else:
            # When not highly correlated, use equal weighting
            weight1 = 0.5
            weight2 = 0.5
        
        return weight1 * rewards1 + weight2 * rewards2
    
    def save_fused_rewards(self, fused_rewards: np.ndarray, original_rewards: List[Dict], 
                          output_path: str, method_name: str):
        """Save fused rewards to a JSONL file"""
        with open(output_path, 'w') as f:
            for i, reward in enumerate(fused_rewards):
                # Create a new item with the fused reward
                new_item = original_rewards[i].copy()
                new_item['fused_reward'] = float(reward)
                new_item['fusion_method'] = method_name
                f.write(json.dumps(new_item) + '\n')

# Example usage
def main():
    fusion = RewardFusion()
    
    # # Load rewards from files
    # rewards1, rewards2 = fusion.load_rewards(
    #     '/home/jovyan/llm_project/Less-is-More/metric/hh_imp_rm_scores_2.jsonl', 
    #     '/home/jovyan/llm_project/Less-is-More/metric/dpo_2k_hh_ppl.jsonl'
    # )

    # Load rewards from files
    rewards1, rewards2 = fusion.load_rewards(
        '/home/jovyan/llm_project/Less-is-More/metric/uf_imp_rm_scores.jsonl', 
        '/home/jovyan/llm_project/Less-is-More/metric/dpo_2k_llama_uf_ppl.jsonl'
    )
    
    # Extract and normalize reward values
    norm_rewards1, norm_rewards2 = fusion.extract_reward_values(rewards1, rewards2)
    
    # Apply different fusion methods
    # weighted_fused = fusion.weighted_average_fusion(norm_rewards1, norm_rewards2, weight1=0.6)
    # bayesian_fused = fusion.bayesian_fusion(norm_rewards1, norm_rewards2)
    attention_fused = fusion.attention_based_fusion(norm_rewards1, norm_rewards2)
    orthogonal_fused = fusion.orthogonal_fusion(norm_rewards1, norm_rewards2)
    
    # Save results
    # fusion.save_fused_rewards(weighted_fused, rewards1, 'weighted_fused_rewards.jsonl', 'weighted_average')
    # fusion.save_fused_rewards(bayesian_fused, rewards1, 'bayesian_fused_rewards.jsonl', 'bayesian')
    # fusion.save_fused_rewards(attention_fused, rewards1, '/home/jovyan/llm_project/Less-is-More/metric/attention_fused_rewards.jsonl', 'attention_based')
    # fusion.save_fused_rewards(orthogonal_fused, rewards1, '/home/jovyan/llm_project/Less-is-More/metric/orthogonal_fused_rewards.jsonl', 'orthogonal')

    fusion.save_fused_rewards(attention_fused, rewards1, '/home/jovyan/llm_project/Less-is-More/metric/uf_attention_fused_rewards.jsonl', 'attention_based')
    fusion.save_fused_rewards(orthogonal_fused, rewards1, '/home/jovyan/llm_project/Less-is-More/metric/uf_orthogonal_fused_rewards.jsonl', 'orthogonal')
    
    # Print some statistics
    print(f"Original rewards 1: mean={np.mean(norm_rewards1):.3f}, std={np.std(norm_rewards1):.3f}")
    print(f"Original rewards 2: mean={np.mean(norm_rewards2):.3f}, std={np.std(norm_rewards2):.3f}")
    # print(f"Weighted fusion: mean={np.mean(weighted_fused):.3f}, std={np.std(weighted_fused):.3f}")
    # print(f"Bayesian fusion: mean={np.mean(bayesian_fused):.3f}, std={np.std(bayesian_fused):.3f}")
    print(f"Attention fusion: mean={np.mean(attention_fused):.3f}, std={np.std(attention_fused):.3f}")
    print(f"Orthogonal fusion: mean={np.mean(orthogonal_fused):.3f}, std={np.std(orthogonal_fused):.3f}")

if __name__ == "__main__":
    main()