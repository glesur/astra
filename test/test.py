import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pytools/"))
import astraTest as tst
import argparse
import shutil


def runTest(test, testDir, currentDir, testName):
  runDir = os.path.join(currentDir, testName)
  os.makedirs(runDir, exist_ok=True)
  # List all input files from test
  inputFiles = []

  for f in os.listdir(os.path.join(testDir, testName)):
    print("Got file: ", f)
    if f.endswith(".ini"):
      fullPath = os.path.join(testDir, testName, f)
      inputFiles.append(fullPath)
    # copy python files to runDir
    if f.endswith(".py"):
      fullPath = os.path.join(testDir, testName, f)
      shutil.copy(fullPath, runDir)
  if not inputFiles:
    print("No input file found for test ", testName)
    return


  for inputFile in inputFiles:
    print("Running test ", testName, " with input file ", inputFile)
    test.run(inputFile=inputFile, problemDir=runDir)
    test.standardTest(problemDir=runDir,inputFile=inputFile)


##############################################
## The Meat
#############################################
parser = argparse.ArgumentParser()

parser.add_argument("action",
                    choices=["build", "run", "checkOnly"],
                    help="Action to perform")

parser.add_argument("testName",
                    help="Name of the test to perform",
                    nargs='?',
                    default="all")

parser.add_argument("buildDir",
                    help="Directory for the build",
                    nargs='?',
                    default="build")

args, unknown=parser.parse_known_args()
thisDir = os.getcwd()
testDir = os.path.dirname(os.path.abspath(__file__))
buildDir = os.path.join(thisDir, args.buildDir)
test = tst.astraTest(buildDirInput=buildDir)

if args.action == "build":
  test.configure()
  test.build()

if args.action == "run":
  # Assume the code is already built
  if args.testName == "all":
    # Walk all the tests
    for testName in os.listdir(testDir):
      if os.path.isdir(os.path.join(testDir, testName)) and testName != "env":
        print("Running test ", testName)
        runTest(test, testDir, thisDir, testName)
  else:
    runTest(test, testDir, thisDir, args.testName)
