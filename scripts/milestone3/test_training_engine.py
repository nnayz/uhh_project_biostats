"""
Test Script for Model Integration & Training Engine

This script validates that:
1. Data loaders work correctly
2. Model wrapper implements the contract interface
3. Training engine functions work
4. Everything integrates properly

Run this after Milestone 2 data preparation is complete.
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data import (
    load_client_data,
    load_gene_list,
    load_label_map,
    create_dataloader,
    get_num_labels,
    get_num_genes
)
from src.model import create_model
from src.training import (
    train_one_epoch,
    evaluate,
    TrainingHistory,
    create_optimizer,
    create_scheduler,
    save_training_artifacts
)
from src.config import create_default_config


def test_data_loaders():
    """Test data loading functionality."""
    print("=" * 60)
    print("Testing Data Loaders")
    print("=" * 60)
    
    data_dir = "data/processed"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"[WARNING] {data_dir} not found.")
        print(f"   Skipping data loader tests. Run Milestone 2 data preparation to test with real data.")
        print(f"   Other tests will use dummy data and still validate the code.")
        return None  # Skip this test, don't fail
    
    try:
        # Load gene list
        print("\n1. Loading gene list...")
        genes = load_gene_list(data_dir)
        print(f"   [OK] Loaded {len(genes)} genes")
        print(f"   First 5 genes: {genes[:5]}")
        
        # Load label map
        print("\n2. Loading label map...")
        label_map = load_label_map(data_dir)
        num_labels = len(label_map)
        print(f"   [OK] Loaded {num_labels} labels")
        
        # Load client data
        print("\n3. Loading client data...")
        client_ids = ["client_01", "client_02", "client_03"]
        for client_id in client_ids:
            try:
                df = load_client_data(client_id, "train", data_dir, validate=True)
                print(f"   [OK] {client_id}: {len(df)} training samples")
            except FileNotFoundError:
                print(f"   [WARN] {client_id}: Data not found (this is OK if Milestone 2 not run yet)")
        
        # Create dataloader
        print("\n4. Creating dataloader...")
        try:
            df = load_client_data("client_01", "train", data_dir, validate=False)
            dataloader = create_dataloader(
                df, genes, batch_size=16, shuffle=True, include_spatial=True
            )
            print(f"   [OK] Created dataloader with {len(dataloader)} batches")
            
            # Test one batch
            batch = next(iter(dataloader))
            print(f"   [OK] Batch shape: features={batch['features'].shape}, labels={batch['label'].shape}")
        except (FileNotFoundError, KeyError) as e:
            print(f"   [WARN] Could not create dataloader: {e}")
            print("   (This is OK if Milestone 2 data not generated yet)")
        
        print("\n[PASSED] Data loaders test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Data loaders test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_wrapper():
    """Test model wrapper functionality."""
    print("\n" + "=" * 60)
    print("Testing Model Wrapper")
    print("=" * 60)
    
    try:
        # Create model
        print("\n1. Creating model...")
        num_genes = 248
        num_labels = 22
        
        model = create_model(
            num_genes=num_genes,
            num_labels=num_labels,
            fine_tune_mode="head_only",
            include_spatial=True
        )
        
        total_params, trainable_params = model.count_parameters()
        print(f"   [OK] Model created")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 4
        input_dim = num_genes + 2  # genes + spatial coords
        dummy_batch = {
            'features': torch.randn(batch_size, input_dim),
            'label': torch.randint(0, num_labels, (batch_size,))
        }
        
        model.eval()
        with torch.no_grad():
            logits = model(dummy_batch)
        
        print(f"   [OK] Forward pass successful")
        print(f"   Input shape: {dummy_batch['features'].shape}")
        print(f"   Output logits shape: {logits.shape}")
        assert logits.shape == (batch_size, num_labels), "Wrong output shape!"
        
        # Test loss computation
        print("\n3. Testing loss computation...")
        loss = model.compute_loss(logits, dummy_batch['label'])
        print(f"   [OK] Loss computed: {loss.item():.4f}")
        
        # Test weight get/set
        print("\n4. Testing weight get/set...")
        weights = model.get_weights()
        print(f"   [OK] Got weights: {len(weights)} parameters")
        
        # Create new model and set weights
        model2 = create_model(num_genes=num_genes, num_labels=num_labels)
        model2.set_weights(weights)
        print(f"   [OK] Set weights on new model")
        
        # Test fine-tuning modes
        print("\n5. Testing fine-tuning modes...")
        for mode in ["head_only", "partial", "full"]:
            model_mode = create_model(
                num_genes=num_genes,
                num_labels=num_labels,
                fine_tune_mode=mode
            )
            _, trainable = model_mode.count_parameters()
            print(f"   {mode}: {trainable:,} trainable parameters")
        
        print("\n[PASSED] Model wrapper test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_engine():
    """Test training engine functionality."""
    print("\n" + "=" * 60)
    print("Testing Training Engine")
    print("=" * 60)
    
    try:
        # Create model and dummy data
        print("\n1. Setting up model and data...")
        num_genes = 248
        num_labels = 22
        batch_size = 8
        
        model = create_model(
            num_genes=num_genes,
            num_labels=num_labels,
            fine_tune_mode="head_only"
        )
        
        # Create dummy dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dummy_features = torch.randn(32, num_genes + 2)  # 32 samples
        dummy_labels = torch.randint(0, num_labels, (32,))
        
        dataset = TensorDataset(dummy_features, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Convert to expected format
        def batch_to_dict(batch):
            features, labels = batch
            return {'features': features, 'label': labels}
        
        # Wrap dataloader
        class WrappedDataLoader:
            def __init__(self, dl):
                self.dl = dl
            def __iter__(self):
                for batch in self.dl:
                    yield batch_to_dict(batch)
            def __len__(self):
                return len(self.dl)
        
        wrapped_dl = WrappedDataLoader(dataloader)
        
        print(f"   [OK] Created model and dummy dataloader")
        
        # Create optimizer
        print("\n2. Creating optimizer...")
        optimizer = create_optimizer(model, learning_rate=1e-4, optimizer_type="adam")
        print(f"   [OK] Optimizer created")
        
        # Test training one epoch
        print("\n3. Testing train_one_epoch...")
        device = torch.device("cpu")
        metrics = train_one_epoch(model, wrapped_dl, optimizer, device, verbose=False)
        print(f"   [OK] Training epoch completed")
        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Test evaluation
        print("\n4. Testing evaluate...")
        eval_metrics = evaluate(model, wrapped_dl, device, verbose=False)
        print(f"   [OK] Evaluation completed")
        print(f"   Loss: {eval_metrics['loss']:.4f}")
        print(f"   Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"   F1-Macro: {eval_metrics['f1_macro']:.4f}")
        
        # Test history tracking
        print("\n5. Testing history tracking...")
        history = TrainingHistory()
        history.add_train_metrics(metrics)
        history.add_val_metrics(eval_metrics)
        print(f"   [OK] History tracking works")
        print(f"   Train epochs: {len(history.train_loss)}")
        print(f"   Val epochs: {len(history.val_loss)}")
        
        # Test scheduler
        print("\n6. Testing scheduler...")
        scheduler = create_scheduler(optimizer, scheduler_type="cosine", num_epochs=10)
        scheduler.step()
        print(f"   [OK] Scheduler created and stepped")
        
        print("\n[PASSED] Training engine test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Training engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration of all components."""
    print("\n" + "=" * 60)
    print("Testing Full Integration")
    print("=" * 60)
    
    try:
        print("\n1. Loading configuration...")
        config = create_default_config(
            num_genes=248,
            num_labels=22,
            fine_tune_mode="head_only",
            batch_size=16,
            num_epochs=2
        )
        print(f"   [OK] Configuration created")
        
        print("\n2. Creating model...")
        model = create_model(
            num_genes=config.num_genes,
            num_labels=config.num_labels,
            fine_tune_mode=config.fine_tune_mode,
            include_spatial=config.include_spatial
        )
        print(f"   [OK] Model created")
        
        print("\n3. Setting up training...")
        optimizer = create_optimizer(
            model,
            learning_rate=config.learning_rate,
            optimizer_type=config.optimizer_type
        )
        device = torch.device(config.device)
        history = TrainingHistory()
        
        # Create dummy data
        from torch.utils.data import TensorDataset, DataLoader
        dummy_features = torch.randn(64, config.num_genes + 2)
        dummy_labels = torch.randint(0, config.num_labels, (64,))
        train_dataset = TensorDataset(dummy_features, dummy_labels)
        val_dataset = TensorDataset(torch.randn(16, config.num_genes + 2), 
                                    torch.randint(0, config.num_labels, (16,)))
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        def batch_to_dict(batch):
            features, labels = batch
            return {'features': features, 'label': labels}
        
        class WrappedDataLoader:
            def __init__(self, dl):
                self.dl = dl
            def __iter__(self):
                for batch in self.dl:
                    yield batch_to_dict(batch)
            def __len__(self):
                return len(self.dl)
        
        train_dl = WrappedDataLoader(train_loader)
        val_dl = WrappedDataLoader(val_loader)
        
        print("\n4. Running training loop...")
        for epoch in range(config.num_epochs):
            print(f"\n   Epoch {epoch + 1}/{config.num_epochs}")
            
            # Train
            train_metrics = train_one_epoch(model, train_dl, optimizer, device, verbose=False)
            history.add_train_metrics(train_metrics)
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = evaluate(model, val_dl, device, verbose=False)
            history.add_val_metrics(val_metrics)
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        print("\n5. Saving artifacts...")
        output_dir = "outputs/test_integration"
        final_metrics = {
            'final_train_loss': history.train_loss[-1],
            'final_train_accuracy': history.train_accuracy[-1],
            'final_val_loss': history.val_loss[-1],
            'final_val_accuracy': history.val_accuracy[-1]
        }
        
        save_training_artifacts(
            output_dir,
            model,
            history,
            config.to_dict(),
            final_metrics
        )
        print(f"   [OK] Artifacts saved to {output_dir}")
        
        print("\n[PASSED] Full integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Model Integration & Training Engine - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test data loaders
    results.append(("Data Loaders", test_data_loaders()))
    
    # Test model wrapper
    results.append(("Model Wrapper", test_model_wrapper()))
    
    # Test training engine
    results.append(("Training Engine", test_training_engine()))
    
    # Test integration
    results.append(("Full Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        if passed is None:
            status = "[SKIPPED]"
        elif passed:
            status = "[PASSED]"
        else:
            status = "[FAILED]"
        print(f"{name:30s} {status}")
    
    # Only count non-skipped tests
    non_skipped = [r[1] for r in results if r[1] is not None]
    all_passed = all(non_skipped) if non_skipped else True
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed. Please check the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
