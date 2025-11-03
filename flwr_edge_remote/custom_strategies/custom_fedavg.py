# custom_fedavg.py

import numpy as np
import torch
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, MetricRecord

# BinaryMaskAwareFedAvg
class MaskAwareFedAvg(FedAvg):
    """
    FedAvg that weights clients based on the fraction of samples with mask > 0
    
    Formula:
        w_i = n_i * (alpha + (1-alpha) * (1 - masked_fraction_i))
    
    Where:
        - masked_fraction = % of sample with mask > 0
        - alpha = minimum weight (default 0.5)
    
    Example:
        Client with 40% masked samples → higher weight ("clean" data)
        Client with 70% masked samples → lower weight ("confused" data)
    """
    
    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.weight_history = []
        
    def aggregate_fit(self, server_round, results):
        if not results:
            return None, MetricRecord()

        # Extracts metrics and calculates weights
        client_info = []
        for idx, (array_record, metric_record) in enumerate(results):
            num_examples = metric_record.metrics.get("num-examples", 1)
            masked_fraction = metric_record.metrics.get("masked-sample-fraction", 0.5)
            
            # Weight is determined by the number of "masked samples" (less masked samples --> more weight)
            weight = num_examples * (self.alpha + (1 - self.alpha) * (1 - masked_fraction))
            
            client_info.append({
                'idx': idx,
                'num_examples': num_examples,
                'masked_fraction': masked_fraction,
                'weight': weight,
                'params': array_record.to_ndarrays()
            })
        
        # Normalizes weights
        total_weight = sum(c['weight'] for c in client_info)
        for c in client_info:
            c['normalized_weight'] = c['weight'] / total_weight
        
        # Detailed logs
        print(f"\n{'='*70}")
        print(f"Round {server_round} - BinaryMaskAware Weighting (alpha={self.alpha})")
        print(f"{'='*70}")
        for c in client_info:
            print(f"  Client {c['idx']}:")
            print(f"    Samples:           {c['num_examples']}")
            print(f"    Masked Fraction:   {c['masked_fraction']:.2%}")
            print(f"    Raw Weight:        {c['weight']:.2f}")
            print(f"    Final Weight:      {c['normalized_weight']:.2%}")
        print(f"{'='*70}\n")
        
        # Saves stats for analysis
        self.weight_history.append({
            'round': server_round,
            'clients': client_info
        })
        
        # Aggregation
        aggregated_ndarrays = [
            np.sum([c['normalized_weight'] * c['params'][i] 
                    for c in client_info], axis=0)
            for i in range(len(client_info[0]['params']))
        ]
        
        aggregated_arrays = ArrayRecord.from_ndarrays(aggregated_ndarrays)
        
        # Aggregates metrics
        aggregated_metrics = {}
        for key in results[0][1].metrics.keys():
            if key not in ["masked-sample-fraction"]:
                aggregated_metrics[key] = sum(
                    c['normalized_weight'] * results[c['idx']][1].metrics[key]
                    for c in client_info
                )
        
        # Adds statistic "masked_fraction"
        masked_fractions = [c['masked_fraction'] for c in client_info]
        aggregated_metrics['mean-masked-fraction'] = np.mean(masked_fractions)
        aggregated_metrics['std-masked-fraction'] = np.std(masked_fractions)
        
        # Saves model
        state_dict = aggregated_arrays.to_torch_state_dict()
        from flwr_edge_remote.models.models import CNN
        model = CNN()
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), f"global_model_round_{server_round}.pt")
        
        return aggregated_arrays, MetricRecord(aggregated_metrics)