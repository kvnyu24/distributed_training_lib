from distributed_training_lib import ModelParallelConfig, ParallelTrainer

def main():
    # Initialize configuration
    config = ModelParallelConfig(
        mp_size=2,  # Model parallel size
        dp_size=1,  # Data parallel size
        is_megatron=False,  # Use naive model parallelism
        in_dim=256,  # Input dimension
        out_dim=512  # Output dimension
    )
    
    # Create trainer
    trainer = ParallelTrainer(config)
    print("Successfully initialized ParallelTrainer!")
    print(f"Model Parallel Size: {trainer.config.mp_size}")
    print(f"Data Parallel Size: {trainer.config.dp_size}")
    print(f"Using Megatron: {trainer.config.is_megatron}")
    print(f"Input Dimension: {trainer.config.in_dim}")
    print(f"Output Dimension: {trainer.config.out_dim}")

if __name__ == "__main__":
    main() 