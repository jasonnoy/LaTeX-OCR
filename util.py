import os
import math
from tqdm import tqdm
from multiprocessing import Process


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def write_to_file(texes, part):
    output_path = f"/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_parts/part_{part}.txt"
    with open(output_path, "w") as f:
        for tex in texes:
            f.write(tex + "\n")
    f.close()


def filter_and_save():
    grand_texes = {'0'}
    # 获取当前目录
    base_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all"
    for dir in os.listdir(base_path):
        print("processing ", dir)
        dir_path = os.path.join(base_path, dir)
        for filename in tqdm(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    grand_texes.add(line)
    print("total texes:", len(grand_texes))
    grand_texes = list(grand_texes)
    splited_texes = split_list_by_n(grand_texes, 100)

    print("start writing to file...")
    process_list = []
    for i, texes in enumerate(splited_texes):
        p = Process(target=write_to_file, args=(texes, i))
        p.start()
        process_list.append(p)
        if len(process_list) >= 32:
            for p in process_list:
                p.join()
            process_list = []


if __name__ == "__main__":
    filter_and_save()
