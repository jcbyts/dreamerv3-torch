import pathlib
import numpy as np
import tools

# Test WandbLogger
logdir = pathlib.Path("./test_wandb_logs")
logdir.mkdir(exist_ok=True)

print("Testing WandbLogger...")
try:
    logger = tools.WandbLogger(
        logdir=logdir,
        step=0,
        entity="yateslab", 
        project="dreamer-test",
        config={"test": True, "model": "pDreamer"}
    )
    
    # Test scalar logging
    logger.scalar("test_metric", 42.0)
    logger.scalar("loss", 0.5)
    
    # Test video logging (create dummy video)
    dummy_video = np.random.randint(0, 255, (1, 10, 64, 64, 3), dtype=np.uint8)
    logger.video("test_video", dummy_video)
    
    # Write logs
    logger.write(fps=True, step=100)
    
    print("✓ WandbLogger test successful!")
    
except Exception as e:
    print(f"✗ WandbLogger test failed: {e}")
    import traceback
    traceback.print_exc()
