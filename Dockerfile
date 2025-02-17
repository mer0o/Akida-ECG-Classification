# =====================================================================================================
# Dockerfile for Conda Environment Setup
# =====================================================================================================
#
# Description:
#   This Dockerfile creates a Conda environment based on a requirements.txt file.
#   The first line of requirements.txt must specify the Python version (e.g., python=3.9).
#   All subsequent lines list the required packages to be installed in the environment.
#
# Usage:
#   1. Ensure requirements.txt exists in the build context with proper format
#   2. Build: docker build --platform linux/amd64 -t <image_name> -f Dockerfile <build_context>
#   3. Run: docker run -it --platform linux/amd64 <image_name>
#
# Example:
#   Build: docker build --platform linux/amd64 -t miniconda_container1 -f Dockerfile ./container1
#   Run: docker run -it --platform linux/amd64 miniconda_container1
#
# Note:
#   - If requirements.txt is not present, defaults to Python 3.11.0
#   - Builds minimal conda environment for better performance
#   - Supports both conda and pip installations
#
# Example requirements.txt format:
#   python=3.9
#   numpy=1.21.0
#   pandas>=1.3.0
#   scikit-learn
# =====================================================================================================



FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Only install requirements if the file exists
RUN /bin/bash -c "if [ -f requirements.txt ]; then \
        PYTHON_VERSION=\$(head -n 1 requirements.txt | cut -d'=' -f2) && \
        echo \"Creating conda environment with Python \$PYTHON_VERSION\" && \
        conda create -y -n myenv python=\$PYTHON_VERSION && \
        echo \"source activate myenv\" > ~/.bashrc && \
        source activate myenv && \
        tail -n +2 requirements.txt | pip install --no-cache-dir -r /dev/stdin; \
    else \
        conda create -y -n myenv python=3.11.0 && \
        echo \"No requirements.txt found, current environment at python 3.11.0, skipping pip install.\" && \
        echo \"source activate myenv\" > ~/.bashrc; \
    fi"

# Default to bash shell inside the conda environment
CMD [ "/bin/bash", "-c", "source activate myenv && bash" ]
