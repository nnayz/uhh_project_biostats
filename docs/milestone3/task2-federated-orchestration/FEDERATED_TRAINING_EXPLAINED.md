# How Federated Training Works

## Quick Answer

**You only need ONE terminal!** The federated training script uses **Flower Simulation Mode**, which runs all clients sequentially in a single process.

## How It Works

### Flower Simulation Mode

The script uses `fl.simulation.start_simulation()`, which means:

1. **Single Process:** All clients run in the same Python process
2. **Sequential Execution:** Clients are processed one after another (not in parallel)
3. **No Network:** No actual network communication (simulated federated learning)
4. **Automatic:** Everything happens automatically - you just run one command

### Training Flow

```
Round 1:
├── Server sends global model to Client 1
├── Client 1 trains locally (2 epochs)
├── Client 1 returns updated weights
│
├── Server sends global model to Client 2
├── Client 2 trains locally (2 epochs)
├── Client 2 returns updated weights
│
├── Server sends global model to Client 3
├── Client 3 trains locally (2 epochs)
├── Client 3 returns updated weights
│
└── Server aggregates all updates (FedAvg)
    └── New global model created

Round 2:
├── (Same process repeats...)
└── ...

Round 5:
└── Final global model saved
```

### What You See in the Terminal

When you run the command, you'll see output like:

```
Starting Federated Training Simulation
============================================================

Round 1:
[client_01] Local epoch 1/2 - Loss: 2.3456, Acc: 0.2345
[client_01] Local epoch 2/2 - Loss: 2.1234, Acc: 0.3456
[client_02] Local epoch 1/2 - Loss: 2.4567, Acc: 0.2234
[client_02] Local epoch 2/2 - Loss: 2.2345, Acc: 0.3345
[client_03] Local epoch 1/2 - Loss: 2.5678, Acc: 0.2123
[client_03] Local epoch 2/2 - Loss: 2.3456, Acc: 0.3234

Round 2:
...
```

### Key Points

1. **One Command:** Just run the single command - no need for multiple terminals
2. **Sequential:** Clients train one after another (not simultaneously)
3. **Same GPU:** All clients share the same GPU (if using CUDA)
4. **Automatic:** Server aggregation happens automatically between rounds

### GPU Resource Allocation

When using `--device cuda`, the script allocates GPU resources:
- Each client gets `0.5` GPU fraction (as specified in `client_resources`)
- This means clients can share the GPU sequentially
- With 3 clients, each gets ~33% of GPU time per round

### Comparison: Simulation vs. Real Distributed

| Feature | Simulation Mode (What We Use) | Real Distributed Mode |
|---------|-------------------------------|----------------------|
| Terminals needed | **1** | Multiple (1 server + N clients) |
| Network | No | Yes (TCP/IP) |
| Parallelism | Sequential | Can be parallel |
| Complexity | Simple | More complex setup |
| Use case | Research/Development | Production deployment |

### Why Simulation Mode?

- ✅ **Simple:** One command, no network setup
- ✅ **Fast Development:** Easy to test and debug
- ✅ **Reproducible:** Same results every run
- ✅ **Sufficient:** Perfect for research and experiments

### If You Want Real Distributed Training

If you need actual distributed training (separate processes/machines), you would need to:

1. **Start Server:**
   ```bash
   python scripts/run_federated.py --mode server --port 8080
   ```

2. **Start Each Client (in separate terminals):**
   ```bash
   # Terminal 1
   python scripts/run_federated.py --mode client --client_id client_01 --server_address localhost:8080
   
   # Terminal 2
   python scripts/run_federated.py --mode client --client_id client_02 --server_address localhost:8080
   
   # Terminal 3
   python scripts/run_federated.py --mode client --client_id client_03 --server_address localhost:8080
   ```

**However, our current implementation uses simulation mode, so you don't need to do this!**

## Summary

**Just run the single command - everything happens automatically in one terminal!**

```bash
python scripts/run_federated.py \
    --data_dir data/processed \
    --output_dir results/federated \
    --num_rounds 5 \
    --clients_per_round 3 \
    --local_epochs 2 \
    --batch_size 1024 \
    --device cuda \
    --num_workers 4 \
    --use_amp
```

The script will:
1. Load all 3 clients automatically
2. Run 5 federated rounds
3. Train each client sequentially
4. Aggregate updates automatically
5. Save the final model

No need for multiple terminals! 🎉
