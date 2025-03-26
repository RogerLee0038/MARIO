import os

def get_counts(path, name):
    count_dict = {}
    for root, dirs, files in os.walk(path):
        count_name = os.path.abspath(os.path.join(root,".."))
        if name in files:
            if count_name in count_dict.keys():
                count_dict[count_name] += 1
            else:
                count_dict[count_name] = 1
    return count_dict

dir_name = "alldatas"
for count_name, cnt in  get_counts(dir_name, "summary.pkl").items():
    print_name = "/".join(count_name.split("/")[-2:])
    print(print_name,":",cnt)
