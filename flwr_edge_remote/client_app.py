"""
To start a local client, open a new console window (different from the server) and run:

    flower-supernode \
        --insecure \
        --superlink="127.0.0.1:9092" \
        --clientappio-api-address="0.0.0.0:9094" \
        --node-config="client-id=0"

"""


import gc
import torch
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, RecordDict, MetricRecord, Message, Context
from flwr_edge_remote.task import (
    get_or_load_dataloaders, 
    train as train_fn, 
    test as test_fn,
)
from flwr_edge_remote.models.models import CNN, CNNWithAttention 


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model locally and return weights + metrics."""
    
    model = CNNWithAttention()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loads parameters from configuration
    epochs = context.run_config["local-epochs"]
    lr = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    partition_id = int(context.node_config.get("client-id"))

    # Loads local data
    trainloader, _ = get_or_load_dataloaders(client_id=partition_id, batch_size=batch_size)

    # Training function
    avg_trainloss = train_fn(
        net=model, 
        trainloader=trainloader, 
        epochs=epochs, 
        learning_rate=lr,
        device=device
    )


    # Builds response Message
    arrays = ArrayRecord(model.state_dict())
    metrics = {
        "train-loss": float(avg_trainloss),
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": arrays, "metrics": metric_record})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = CNNWithAttention()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = context.run_config["batch-size"]
    partition_id = int(context.node_config.get("client-id"))
    _, valloader = get_or_load_dataloaders(client_id=partition_id, batch_size=batch_size)

    eval_loss, eval_acc, mask_benefit, speed_stats = test_fn(
        net=model, 
        testloader=valloader, 
        device=device
    )

    metrics = {
        "eval-loss": float(eval_loss),
        "eval-acc": float(eval_acc),
        "mask-benefit": float(mask_benefit),
        "speed-range": float(speed_stats['range']),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)