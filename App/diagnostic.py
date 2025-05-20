import argparse
import sys
import os
import yaml
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def validate_model_path(path):
    """Check if model file exists and has appropriate size."""
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return False
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 0.1:  # Model file should be at least 100KB
        logger.warning(f"Model file suspiciously small ({size_mb:.2f}MB): {path}")
    else:
        logger.info(f"Model file found: {path} ({size_mb:.2f}MB)")
    
    return True

def create_sample_config():
    """Create a sample config file."""
    config = {
        "model_path": "model/model.bin",
        "model_name": "bert-base-cased",
        "max_len": 300,
        "class_names": {
            0: "Not Fraud",
            1: "Fraud"
        }
    }
    
    # Ensure directory exists
    os.makedirs("App/config", exist_ok=True)
    
    # Write config file
    with open("config/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Sample config created at config/config.yaml")

def check_imports():
    """Check if all required packages are installed."""
    required_packages = ["torch", "transformers", "yaml"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"Package '{package}' is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"Package '{package}' is not installed")
    
    if missing_packages:
        install_cmd = "pip install " + " ".join(missing_packages)
        logger.error(f"Missing packages. Install them with: {install_cmd}")
        return False
    
    return True

def test_tokenizer(model_name):
    """Test if tokenizer can be loaded."""
    try:
        from transformers import BertTokenizer, AutoTokenizer
        
        logger.info(f"Attempting to load tokenizer: {model_name}")
        
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully with BertTokenizer")
        except Exception as e:
            logger.warning(f"Failed to load with BertTokenizer: {str(e)}")
            logger.info("Trying with AutoTokenizer...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("Tokenizer loaded successfully with AutoTokenizer")
            except Exception as e2:
                logger.error(f"Failed to load tokenizer with AutoTokenizer: {str(e2)}")
                return False
                
        # Test tokenizer with sample text
        sample = "Test tokenizer functionality"
        tokens = tokenizer(sample, return_tensors="pt")
        logger.info(f"Tokenization test successful. Input '{sample}' converted to tensor of shape {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing tokenizer: {str(e)}")
        return False

def main():
    """Diagnostics tool main function."""
    parser = argparse.ArgumentParser(description="Fraud Detection Diagnostics Tool")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--create-config", action="store_true", help="Create a sample config file")
    parser.add_argument("--check-imports", action="store_true", help="Check if required packages are installed")
    parser.add_argument("--test-model-path", action="store_true", help="Check if model file exists")
    parser.add_argument("--test-tokenizer", action="store_true", help="Test if tokenizer can be loaded")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    print("\n== Fraud Detection Model Diagnostics Tool ==\n")
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return 0
    
    # Check imports if requested
    if args.check_imports:
        if not check_imports():
            return 1
    
    # Load config file
    try:
        # Resolve the config path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, args.config)) if not os.path.isabs(args.config) else args.config
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            logger.info("You can create a sample config file with --create-config")
            return 1
        
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model information
        model_path = config.get("model_path")
        model_name = config.get("model_name")
        
        if not model_path:
            logger.error("model_path not found in config file")
            return 1
        
        if not model_name:
            logger.error("model_name not found in config file")
            return 1
        
        # Resolve model path relative to script location
        full_model_path = os.path.normpath(os.path.join(script_dir, model_path)) if not os.path.isabs(model_path) else model_path
        
        logger.info(f"Config loaded successfully")
        logger.info(f"Model path: {full_model_path}")
        logger.info(f"Model name: {model_name}")
        
        # Test model path if requested
        if args.test_model_path:
            if not validate_model_path(full_model_path):
                return 1
        
        # Test tokenizer if requested
        if args.test_tokenizer:
            if not test_tokenizer(model_name):
                return 1
        
        # If no specific tests were requested, run all tests
        if not any([args.create_config, args.check_imports, args.test_model_path, args.test_tokenizer]):
            logger.info("Running all diagnostics...")
            
            if not check_imports():
                return 1
            
            if not validate_model_path(full_model_path):
                return 1
            
            if not test_tokenizer(model_name):
                return 1
        
        print("\n== Diagnostics completed successfully ==")
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())