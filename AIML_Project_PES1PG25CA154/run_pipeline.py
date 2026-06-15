import os
import subprocess
import sys

def run_script(script_name):
    # Dynamically find the absolute path of the directory where run_pipeline.py is located
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path to the target script inside the scripts folder
    script_path = os.path.join(project_root, "scripts", script_name)
    
    print(f"\n🚀 Running {script_name}...")
    # Pass the absolute path to guarantee execution succeeds from any directory
    subprocess.run([sys.executable, script_path], check=True)

if __name__ == "__main__":
    try:
        run_script("1_data_cleaning.py")
        run_script("2_data_preparation.py")
        run_script("3_model_training.py")
        run_script("4_model_evaluation.py")
        print("\n✅ Lifecycle Complete! Run 'streamlit run app.py' to launch the website.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Execution stopped due to an error in the lifecycle pipeline: {e}")