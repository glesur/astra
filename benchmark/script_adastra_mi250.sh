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

module load cpe/25.09
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-amd
module load cray-python/3.11.7
module load cmake

export MPICH_GPU_SUPPORT_ENABLED=1
export ASTRA_FLAGS="-DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DAstra_MPI=ON -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON -DKokkos_ARCH_AMD_GFX90A=ON"

export HIPCC_COMPILE_FLAGS_APPEND="-isystem ${CRAY_MPICH_PREFIX}/include"
export HIPCC_LINK_FLAGS_APPEND="-L${CRAY_MPICH_PREFIX}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lstdc++fs"

# echo des commandes lancees
set -x



# execution du code
srun ./astra -i @input_file@
#/idefix
