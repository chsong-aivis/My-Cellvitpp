#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import wandb
import logging
import argparse
from pathlib import Path

# Set environment variables
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

from cellvit.training.base_ml.base_cli import ExperimentBaseParser
from cellvit.training.experiments.experiment_cell_classifier import ExperimentCellVitClassifier
from cellvit.utils.logger import Logger

def setup_logging():
    """Setup logging configuration"""
    logger = Logger(
        level="INFO",
        log_dir="./logs",
        comment="train_single_gpu",
        use_timestamp=True
    ).create_logger()
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CellViT classifier on single GPU')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(config):
    """Setup Weights & Biases logging"""
    # Create logging directory if it doesn't exist
    log_dir = Path(config['logging']['wandb_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set WANDB directory
    os.environ["WANDB_DIR"] = str(log_dir.absolute())
    
    # Initialize wandb
    wandb.init(
        project=config['logging']['project'],
        notes=config['logging']['notes'],
        config=config,
        mode=config['logging']['mode']
    )

def main():
    """Main training function."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting single GPU training...")
    
    # Parse arguments
    parser = ExperimentBaseParser()
    config = parser.parse_arguments()
    
    # Set random seed for reproducibility
    if 'random_seed' in config:
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['random_seed'])
    
    # Setup experiment
    experiment = ExperimentCellVitClassifier(default_conf=config)
    
    # Setup W&B logging
    setup_wandb(config)
    
    try:
        # Run training
        output_dir = experiment.run_experiment()
        logger.info(f"Training completed. Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise e
    finally:
        # Cleanup
        wandb.finish()

if __name__ == "__main__":
    main() 