#!/bin/bash
#SBATCH --constraint=MI250
#SBATCH --account=@account@
#SBATCH --job-name=@name@     # nom du job
#SBATCH --nodes=@nodes@                   # nombre de noeuds
#SBATCH --ntasks-per-node=@core@         # nombre de tache MPI par noeud  (=nb de GPU)
#SBATCH --gres=gpu:@core@                # nombre de GPU par noeud
#SBATCH --cpus-per-task=8          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread        # hyperthreading desactive
#SBATCH --time=0:10:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=idefix%j.out     # nom du fichier de sortie
#SBATCH --error=idefix-error%j.out      # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=geoffroy.lesur@univ-grenoble-alpes.fr

# nettoyage des modules charges en interactif et herites par defaut
module purge
module load cpe/24.07
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-cray

#module load amd-mixed/6.3.3
## Rocm 6.4 from CINES support mail
export ROCM_PATH="/opt/software/rocm/6.4.0"
export PATH="${ROCM_PATH}/bin:${PATH}"
export PATH="${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib/llvm/lib:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${CMAKE_PREFIX_PATH:-}"
export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} --no-default-config "

export MPICH_GPU_SUPPORT_ENABLED=1

#module load nvidia-nsight-systems/2021.1.1

# echo des commandes lancees
set -x



# execution du code
srun ./astra -i @input_file@
#/idefix
