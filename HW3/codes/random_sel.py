models = ["Tfmr-scratch", "Tfmr-finetune"]
strats = ["random", "top-p"]
p = 0.9
temps = [1.0, 0.7]

import random
random.seed(42)
random_list = random.sample(range(5000), 10)

with open("random_sel.txt", "w") as fout:
    for t in temps:
        for model in models:
            name_random = f"output_{model}_random_{t}_1.0.txt"
            name_top_p = f"output_{model}_top-p_{t}_0.9.txt"
            with open(name_random, "r") as fin:
                lines = fin.readlines()
                fout.write(f"random t={t} {model}\n")
                for i in random_list:
                    fout.write(lines[i])
                fout.write("\n")

    for t in temps:
        for model in models:
            with open(name_top_p, "r") as fin:
                lines = fin.readlines()
                fout.write(f"top-p=0.9 t={t} {model}\n")
                for i in random_list:
                    fout.write(lines[i])
                fout.write("\n")

    with open("output_None_random_1_1.0.txt", "r") as fin:
        fout.write("random t=1.0 Tfmr-finetune-1-6-12\n")
        for i in random_list:
            fout.write(lines[i])
        fout.write("\n")
