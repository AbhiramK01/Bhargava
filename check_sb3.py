"""
Utility to check if stable-baselines3 is configured correctly for our project.
"""

import sys
import os

print("Checking stable-baselines3 compatibility...")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check for stable-baselines3
try:
    import stable_baselines3
    print(f"\nFound stable_baselines3 version: {stable_baselines3.__version__}")
    
    # Check for specific features in this version
    print("\nChecking available policies in stable_baselines3...")
    from stable_baselines3.common import policies
    
    # Inspect available policies
    available_policies = [name for name in dir(policies) if not name.startswith('_')]
    print(f"Available policy classes: {available_policies}")
    
    # Check for PPO
    print("\nChecking PPO implementation...")
    from stable_baselines3 import PPO
    print(f"PPO version: {PPO.__module__}")
    
    # Create a simple environment for testing
    print("\nAttempting to create a simple test environment...")
    import gymnasium as gym
    import numpy as np
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    
    print("\nAttempting to create a PPO agent...")
    # Try to create a PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    print("Successfully created PPO agent with MlpPolicy")
    
    # Try with ActorCriticPolicy if available
    if 'ActorCriticPolicy' in available_policies:
        print("\nTrying with ActorCriticPolicy...")
        try:
            from stable_baselines3.common.policies import ActorCriticPolicy
            model = PPO(ActorCriticPolicy, env, verbose=1)
            print("Successfully created PPO agent with ActorCriticPolicy")
        except Exception as e:
            print(f"Error creating PPO with ActorCriticPolicy: {e}")
    
except ImportError as e:
    print(f"\nError importing stable-baselines3: {e}")
    print("Please install it with: pip install stable-baselines3[extra]")
    
    # Try to provide more helpful diagnostics
    try:
        import pkg_resources
        print("\nInstalled packages:")
        for pkg in pkg_resources.working_set:
            if any(name in pkg.key for name in ["stable", "gym", "torch", "numpy"]):
                print(f"  {pkg.key}: {pkg.version}")
    except Exception as ex:
        print(f"Error checking packages: {ex}")

print("\nCheck complete. Use this information to help diagnose any issues.") 