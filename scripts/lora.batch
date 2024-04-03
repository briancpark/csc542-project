#!/bin/bash
#SBATCH -N 6
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH -p max
#SBATCH -w c[21,22,26,27,28,29]
#SBATCH -J lora-tune
#SBATCH -o lora-tune.o%j
#SBATCH -e lora-tune.e%j


# Export the CUDA_VISIBLE_DEVICES variable
####### #SBATCH -p rtx2060super
####### #SBATCH -p max
####### #SBATCH -w c[21,26,27,28,29]
# [20-22,26-29], queue: -p rtx4060ti8g
########### SBATCH -w c[20,21,26,27,28,29]
#25
source ~/.bashrc
conda activate csc542
cd /home/bcpark/csc542-project

set_cuda_visible_devices() {
    GPU_IDS=$(nvidia-smi --query-gpu=gpu_name,index --format=csv,noheader | grep 'RTX 4060' | awk -F ', ' '{print $2}')
    CUDA_VISIBLE_DEVICES=$(echo $GPU_IDS | tr ' ' ',')
    export CUDA_VISIBLE_DEVICES
}


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "$(declare -f set_cuda_visible_devices); set_cuda_visible_devices"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 32 --num-gpus 1 --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "$(declare -f set_cuda_visible_devices); set_cuda_visible_devices"
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 32 --num-gpus 1 --block &
    sleep 5
done
# "${SLURM_CPUS_PER_TASK}"


echo "Workers successfully initialized!"

python3 main.py --hyperparameter-tune