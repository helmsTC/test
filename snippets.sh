sudo apt-get --purge remove "*cublas*" "cuda*" "nvidia-cuda*" "libcuda*"
sudo apt-get autoremove
sudo rm -rf /usr/local/cuda*

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda-repo-ubuntu1804-9-1-local_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-9-1-local/7fa2af80.pub
sudo apt-get update

sudo apt-get install -y cuda-9-1

ls -l /usr/local/cuda/bin/nvcc
nvcc --version

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
