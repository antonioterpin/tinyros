# ROS 2 Latency Benchmark

This benchmark measures end-to-end message latency using **ROS 2**
(publisher → subscriber) for different payload sizes.

It is intended as a baseline for comparison with the TinyROS and Portal
benchmarks in this repository.

---

## Install ROS2
   Install the necessary dependencies **in the order below**:

   ### Create a dedicated conda environment

   ```bash
   conda create -n ros2-bench python=3.10
   conda activate ros2-bench
   ```

   ### Install Python requirements for the benchmark

   ```bash
   pip install -r benchmark_ros2/requirements.txt
   ```

   ### Install ROS 2

   Add the following channels to the environment:

   ```bash
   conda config --env --add channels conda-forge
   conda config --env --add channels robostack-staging
   conda config --env --remove channels defaults
   ```

   Then, install ROS2 as follows:

   ```bash
   conda install ros-humble-desktop
   conda deactivate
   conda activate ros2-bench
   ```

   ### Run benchmark

   To run the benchmark use this line:
   ```bash
   python -m benchmark_ros2.runner
   ```
