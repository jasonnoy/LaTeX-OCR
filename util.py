import json
import os
import math
from tqdm import tqdm
from multiprocessing import Process
import shutil
from pix2tex.dataset.extract_latex import extract_formula_from_tex


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def write_to_file(texes, part):
    output_path = f"/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_parts_new_new/part_{part}.txt"
    with open(output_path, "w") as f:
        for tex in texes:
            f.write(tex + "\n")
    f.close()


def filter_and_save():
    grand_texes = set({})
    # 获取当前目录
    base_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_new"
    for dir in os.listdir(base_path):
        print("processing ", dir)
        dir_path = os.path.join(base_path, dir)
        for filename in tqdm(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(line) < 3:
                        continue
                    line = line.strip()
                    grand_texes.add(line)
    print("total texes:", len(grand_texes))
    json_path = "/nxchinamobile2/shared/img_datasets/math_ocr/img2latex/latex_32G.jsonl"
    count = 0
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tex = data['data']
            grand_texes.add(tex)
            count += 1
            if count % 1000 == 0:
                print(f'{count}/677168335', end='\r')
    grand_texes = list(grand_texes)
    splited_texes = split_list_by_n(grand_texes, 672)

    print("start writing to file...")
    process_list = []
    for i, texes in enumerate(splited_texes):
        p = Process(target=write_to_file, args=(texes, i))
        p.start()
        process_list.append(p)
        if len(process_list) >= 128:
            for p in process_list:
                p.join()
            process_list = []


def count_all_parts():
    parts = range(672)
    dataset_path = "/nxchinamobile2/shared/img_datasets/math_ocr/aminer_math_short"
    count = 0
    for part in parts:
        part_path = os.path.join(dataset_path, "part-%03d" % part)
        stat_path = os.path.join(part_path, "stat.txt")
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                text = f.read()
            text = text.split(sep=',')[0]
            count += int(text[7:])
        except Exception as e:
            print(f'{part_path} error: {e}')
    print(count)
    return count


def make_dataset(dataset_path, output_path, part_num=32):
    last_to_dir = 0
    tar_id = 0
    for part in tqdm(range(672)):
        to_dir = part // part_num
        if to_dir != last_to_dir:
            last_to_dir = to_dir
            tar_id = 0
        to_dir_path = os.path.join(output_path, "part-%03d" % to_dir)
        os.makedirs(to_dir_path, exist_ok=True)
        part_path = os.path.join(dataset_path, "part-%03d" % part)
        for file in os.listdir(part_path):
            if file.endswith(".tar"):
                to_file = "%07d.tar" % tar_id
                tar_id += 1
                file_path = os.path.join(part_path, file)
                to_file_path = os.path.join(to_dir_path, to_file)
                shutil.move(file_path, to_file_path)
                # print(file_path, to_file_path)


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def make_dataset_temp(dataset_path, output_path, part_num=32):
    tar_files = []
    for part in tqdm(range(11)):
        part_path = os.path.join(dataset_path, "part-%03d" % part)
        for file in os.listdir(part_path):
            tar_path = os.path.join(part_path, file)
            tar_files.append(tar_path)
    divided_tar_files = split_list_by_n(tar_files, part_num)
    for part, tar_files in enumerate(divided_tar_files):
        tar_id = 0
        to_dir_path = os.path.join(output_path, "part-%03d" % part)
        os.makedirs(to_dir_path, exist_ok=True)
        for file_path in tar_files:
            to_file = "%07d.tar" % tar_id
            tar_id += 1
            to_file_path = os.path.join(to_dir_path, to_file)
            shutil.move(file_path, to_file_path)
            # print(file_path, to_file_path)


def get_and_save_colors(file='colornames.txt'):
    with open(file, 'r') as f1, open('color_names.txt', 'w') as f2:
        for line in f1:
            color = line.strip().split(':')[0].replace('\'', '')
            f2.write(color + '\n')
    f2.close()


if __name__ == "__main__":
    make_dataset("/nxchinamobile2/shared/img_datasets/math_ocr/aminer_math_short", "/nxchinamobile2/shared/img_datasets/math_ocr/Aminer_Math_Short")
    # filter_and_save()
    # count_all_parts()
    # output_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/latex_32G.jsonl"
    # input_path = "/nxchinamobile2/shared/img_datasets/math_ocr/img2latex/latex_32G.jsonl"
    # with open(input_path, 'r') as f1, open(output_path, 'w') as f2:
    #     for line in f1:
    #         data = json.loads(line)
    #         tex = data['data']
    #         formulas = extract_formula_from_tex(tex, str_input=True)
    #         for f in formulas:
    #             if len(f) < 5:
    #                 continue
    #             f_json = json.dumps({'data': f})
    #             f2.write(f_json + '\n')
    # f2.close()
