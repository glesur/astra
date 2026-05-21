import numpy as np
from shutil import copy2
from shutil import rmtree
from datetime import datetime
import os
import re
import stat
import argparse

parser = argparse.ArgumentParser(
    prog="launch.py",
    description="Le programme de lancement des benchmarks Idefix"
)
parser.add_argument('--max-cores', type=int)
parser.add_argument('--cores-per-node', type=int)
parser.add_argument('--input-file', type=str, default='kelvin_helmholtz.ini')
parser.add_argument('--problem-size', type=int)
parser.add_argument('--account', type=str)
parser.add_argument('--strong', type=bool, default=False)
parser.add_argument('--build-directory', type=str, default='./build')
parser.add_argument('--run-directory', type=str, default='./run')
parser.add_argument('--script-file', type=str, default='script.slurm')

args = parser.parse_args()

# Number of cores which we want to explore
minCores=1
maxCores=args.max_cores

#elementary 1D dimension
problemSize=args.problem_size

# coreperNode on the cluster we're running
coresPerNode=args.cores_per_node

strongScaling=args.strong

#set number of cores
coreList=(2**(np.arange(np.log2(minCores),np.log2(maxCores)+1))).astype(int)

for ncores in coreList:
    date = datetime.today().strftime('%Y-%m-%d')
    targetDir=args.run_directory+"/"+date+"/%d/%d"%(problemSize,ncores)
    print("Doing %d cores setup"%ncores)
    if os.path.exists(targetDir):
        rmtree(targetDir)
    os.makedirs(targetDir)
    copy2(args.build_directory+"/astra",targetDir)
    copy2(args.input_file,targetDir)

    # compute number of cores and node
    nodes=ncores//coresPerNode
    if nodes<1:
        nodes=1
        cores=ncores
    else:
        cores=coresPerNode

    print("%d nodes and %d cores"%(nodes,cores))

    # compute problem total resolution
    nx1=problemSize
    nx2=problemSize
    nx3=problemSize

    nproc1=ncores
    nproc2=1
    nproc3=1

    if strongScaling:
        nproc1=1

    inputOptions={}
    inputOptions['resx1']="%d"%(nproc1*problemSize)
    inputOptions['resx2']="%d"%(nproc2*problemSize)
    inputOptions['resx3']="%d"%(nproc3*problemSize)
    inputOptions['lx1']="%f"%(nproc1*1.0)
    inputOptions['lx2']="%f"%(nproc2*1.0)
    inputOptions['lx3']="%f"%(nproc3*1.0)

    # inifile substitution
    with open(args.input_file, 'r') as file:
        inifile = file.read()

    for key, val in inputOptions.items():
        inifile = re.sub(r'@{0}@'.format(key), val, inifile)

    with open(targetDir+"/"+args.input_file ,'w') as file:
        file.write(inifile)


    scriptOptions={}
    scriptOptions['nodes']="%d"%(nodes)
    scriptOptions['core']="%d"%(cores)
    scriptOptions['name']="benchmark-%d"%(ncores)
    scriptOptions['account']=args.account
    scriptOptions['input_file']=args.input_file

    # inifile substitution
    with open(args.script_file, 'r') as file:
        scriptfile = file.read()

    for key, val in scriptOptions.items():
        scriptfile = re.sub(r'@{0}@'.format(key), val, scriptfile)

    with open(targetDir+"/script.slurm",'w') as file:
        file.write(scriptfile)

    os.chmod(targetDir+"/script.slurm",stat.S_IRWXU)
    cwd = os.getcwd()
    os.chdir(targetDir)
    os.system('sbatch ./script.slurm')
    os.chdir(cwd)
