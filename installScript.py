import subprocess
import sys
import pkg_resources

# List of packages with specific versions
required_packages = [
    "flask==2.3.3",
    "flask-socketio==5.3.6",
    "langchain==0.0.350",
    "langchain-community==0.0.10",
    "scikit-learn==1.3.2",
    "numpy==1.24.3",
    "python-socketio==5.9.0",
    "eventlet==0.33.3"
]

# Function to install a package
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check installed packages
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

for package in required_packages:
    name, version = package.split("==")
    if name.lower() in installed_packages and installed_packages[name.lower()] == version:
        print(f"{name}=={version} is already installed.")
    else:
        print(f"{name}=={version} not found. Installing...")
        install_package(package)
