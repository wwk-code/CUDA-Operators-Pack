import random

datas_size = 12800
datas_upper_bound = 10000
datas_lower_bound = -10000

file_path = "../input_datas/inputs_" + str(datas_size) + ".txt"

datas = [random.uniform(datas_lower_bound,datas_upper_bound) for _ in range(datas_size)]

with open(file_path,"w") as file:
    for num in datas:
        file.write(str(num) + "\n")
        
print("Datas Generating finished!")