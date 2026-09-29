"""
Deployment Check Script for BlastOpt Botswana
Verifies that all required files are present for Streamlit Cloud deployment.
"""

import os
import sys

def check_deployment_readiness():
    """Check if all required files are present for deployment."""

    print("=" * 60)
    print("BlastOpt Botswana - Deployment Readiness Check")
    print("=" * 60)
    print()

    required_files = {
        'app.py': 'Main Streamlit application',
        'requirements.txt': 'Python dependencies',
        'packages.txt': 'System packages (optional)',
        'README.md': 'Project documentation',
        '.gitignore': 'Git ignore rules',
        '.streamlit/config.toml': 'Streamlit configuration',
    }

    required_dirs = {
        'src': 'Source modules',
        'models': 'Model registry',
        'data': 'Data directory',
        'tests': 'Unit tests',
    }

    missing_files = []
    missing_dirs = []

    print("Checking required files...")
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  [OK] {file_path} - {description}")
        else:
            print(f"  [MISSING] {file_path} - {description}")
            missing_files.append(file_path)

    print()
    print("Checking required directories...")
    for dir_path, description in required_dirs.items():
        if os.path.exists(dir_path):
            print(f"  [OK] {dir_path}/ - {description}")
        else:
            print(f"  [MISSING] {dir_path}/ - {description}")
            missing_dirs.append(dir_path)

    print()
    print("=" * 60)

    if not missing_files and not missing_dirs:
        print("DEPLOYMENT READY: All required files and directories present")
        print()
        print("Next steps:")
        print("1. Install Git: https://git-scm.com/download/win")
        print("2. Run: connect_to_repo.bat")
        print("3. Deploy to Streamlit Cloud")
        print("Repository: https://github.com/mndolo2x/BlasterOPT")
        return True
    else:
        print("DEPLOYMENT NOT READY: Missing files or directories")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")
        return False

if __name__ == "__main__":
    check_deployment_readiness()
