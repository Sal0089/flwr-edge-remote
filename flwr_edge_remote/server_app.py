"""
To start a local server, open a new console window (different from the server) and run:
flower-superlink --insecure 
"""


import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flwr_edge_remote.models.models import CNN, CNNWithAttention
from flwr_edge_remote.custom_strategies.custom_fedavg import MaskAwareFedAvg

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = CNN()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy_name = context.run_config.get("strategy-name", "FedAvg")
    if strategy_name == "FedAvg":
        strategy = FedAvg(
            fraction_train=fraction_train, 
            fraction_evaluate=fraction_evaluate,
        )
    elif strategy_name == "MaskAwareFedAvg":
        strategy = MaskAwareFedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            alpha=0.5  # Put clear comment about the alpha true meaning
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Start strategy for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
