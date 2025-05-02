"""
Clean up model files and provide a fresh start.
This is useful if you encounter errors with corrupted model files.
"""

import os
import shutil
import sys

def clean_model_files():
    """Remove all model files and create fresh directories."""
    print("Cleaning model files...")
    
    # Model files to remove
    model_files = [
        "formation_recognizer.pt",
        "strategy_ai_model",
        "strategy_ai_model.zip"
    ]
    
    # Remove each file if it exists
    for file in model_files:
        if os.path.exists(file):
            try:
                if os.path.isdir(file):
                    shutil.rmtree(file)
                    print(f"Removed directory: {file}")
                else:
                    os.remove(file)
                    print(f"Removed file: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    # Remove any database file
    if os.path.exists("battle_data.db"):
        try:
            os.remove("battle_data.db")
            print("Removed battle database")
        except Exception as e:
            print(f"Error removing database: {e}")
    
    print("Cleanup complete! You can now start the simulator with a fresh state.")
    print("Run: python run_simulator.py")

if __name__ == "__main__":
    # Ask for confirmation
    print("This will remove all trained models and battle data.")
    print("You will need to train the AI from scratch.")
    confirm = input("Are you sure you want to proceed? (y/n): ")
    
    if confirm.lower() == 'y':
        clean_model_files()
    else:
        print("Aborted. No files were changed.") 