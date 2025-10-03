"""
Traffic GNN Classification - Main Launcher
==========================================

Easy launcher script for the Multi-Task Traffic GNN Classification system.
Provides options to train models, run evaluations, and launch the dashboard.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

def run_training(args):
    """Run model training"""
    print("🚀 Starting model training...")
    
    cmd = ["python", "train.py"]
    
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.hidden_dim:
        cmd.extend(["--hidden_dim", str(args.hidden_dim)])
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
            return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    
    return True

def run_dashboard(args):
    """Launch Streamlit dashboard"""
    print("🚀 Launching traffic dashboard...")
    
    dashboard_path = Path("app") / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    cmd = ["streamlit", "run", str(dashboard_path)]
    
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        result = subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print("❌ Failed to install requirements:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_setup():
    """Check if project setup is correct"""
    print("🔍 Checking project setup...")
    
    required_files = [
        "requirements.txt",
        "train.py",
        "src/data/data_processor.py",
        "src/models/multi_task_gnn.py",
        "src/utils/graph_constructor.py",
        "app/dashboard.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check if data directory exists
    data_path = Path("../Data")
    if not data_path.exists():
        print("⚠️  Data directory not found at ../Data")
        print("   Please ensure your data is properly placed.")
        return False
    
    print("✅ Project setup looks good!")
    return True

def show_status():
    """Show project status"""
    print("📊 Project Status:")
    print("=" * 50)
    
    # Check if model is trained
    model_path = Path("outputs/best_model.pth")
    if model_path.exists():
        print("✅ Model trained: YES")
        print(f"   Model location: {model_path}")
    else:
        print("❌ Model trained: NO")
        print("   Run: python run.py train")
    
    # Check processed data
    data_path = Path("outputs/processed_data.pkl")
    if data_path.exists():
        print("✅ Data processed: YES")
    else:
        print("❌ Data processed: NO")
    
    # Check training history
    history_path = Path("outputs/training_history.pkl")
    if history_path.exists():
        print("✅ Training history: YES")
    else:
        print("❌ Training history: NO")
    
    # Check results
    results_path = Path("outputs/evaluation_results.pkl")
    if results_path.exists():
        print("✅ Evaluation results: YES")
    else:
        print("❌ Evaluation results: NO")
    
    print()
    print("💡 Next steps:")
    if not model_path.exists():
        print("   1. Train model: python run.py train")
        print("   2. Launch dashboard: python run.py dashboard")
    else:
        print("   1. Launch dashboard: python run.py dashboard")
        print("   2. Re-train model: python run.py train")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Traffic GNN Classification Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup                    # Check project setup
  python run.py install                  # Install requirements
  python run.py train                    # Train model with default parameters
  python run.py train --epochs 100       # Train model with 100 epochs
  python run.py dashboard                # Launch dashboard
  python run.py dashboard --port 8502    # Launch dashboard on port 8502
  python run.py status                   # Show project status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Check project setup')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install requirements')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show project status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🚦 Multi-Task Traffic GNN Classification")
    print("=" * 50)
    
    if args.command == 'setup':
        check_setup()
    
    elif args.command == 'install':
        install_requirements()
    
    elif args.command == 'train':
        if check_setup():
            run_training(args)
    
    elif args.command == 'dashboard':
        if check_setup():
            run_dashboard(args)
    
    elif args.command == 'status':
        show_status()
    
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()