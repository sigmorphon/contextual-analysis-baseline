import os
import sys

with open("scripts/run_cpu.sh") as f:
  data = f.readlines()

lang = str(sys.argv[1]).split("UD_")[1][:-1]

model_directory = "task2_jackknife_models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

pyscript = "python -u baselineTagger.py --treebank_path " + str(sys.argv[1]) + " --langs " + lang + " --batch_size 32 --model_type mono --model_path " + model_directory + " --jackknife"

for k in range(10):
    wdata = data + [pyscript + " --fold " + str(k)]
    script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_baseline_"+ str(k) + ".sh"
    with open(script_name, 'w') as f:
        f.writelines(wdata)
    os.system("sbatch -J " + str(k) + lang + "_baseline " + script_name)
