#!/bin/bash
#SBATCH -A @accoun@
#SBATCH -C h100
#SBATCH --job-name=@name@     # nom du job
#SBATCH --nodes=@nodes@                   # nombre de noeuds
#SBATCH --ntasks-per-node=@core@         # nombre de tache MPI par noeud  (=nb de GPU)
#SBATCH --gres=gpu:@core@                # nombre de GPU par noeud
#SBATCH --cpus-per-task=24          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread        # hyperthreading desactive
#SBATCH --time=0:10:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=idefix%j.out     # nom du fichier de sortie
#SBATCH --error=idefix-error%j.out      # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=geoffroy.lesur@univ-grenoble-alpes.fr

# nettoyage des modules charges en interactif et herites par defaut
module purge

module load arch/h100
module load cmake/3.30.1
module load cuda/12.8.0
module load openmpi/4.1.5-cuda

#module load nvidia-nsight-systems/2021.1.1

# echo des commandes lancees
set -x



# execution du code
srun ./astra -i @input_file@
#/idefix