import argparse
import math
import os
import re
import subprocess
import shlex
from tqdm import tqdm
from multiprocessing import Pool


def convert_tex_to_png(output_paths, tex_paths, pool_size=30):
    subprocess_pool = []
    for tex_path, output_path in tqdm(zip(tex_paths, output_paths)):
        cmd = 'xelatex -interaction=batchmode -output-directory %s %s' % (output_path, tex_path)
        p = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        subprocess_pool.append(p)
        if len(subprocess_pool) >= pool_size:
            for p in subprocess_pool:
                sout, serr = p.communicate()
            subprocess_pool = []
    for p in subprocess_pool:
        sout, serr = p.communicate()
    subprocess_pool = []
    # xelatex needs second processing.
    for tex_path, output_path in tqdm(zip(tex_paths, output_paths)):
        cmd = 'xelatex -interaction=batchmode -output-directory %s %s' % (output_path, tex_path)
        p = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        subprocess_pool.append(p)
        if len(subprocess_pool) >= pool_size:
            for p in subprocess_pool:
                sout, serr = p.communicate()
            subprocess_pool = []
    for p in subprocess_pool:
        sout, serr = p.communicate()
    subprocess_pool = []


def rec_tex_files(subpath, tex_files: list) -> list:
    subpaths = os.listdir(subpath)
    for path in subpaths:
        cur_filepath = os.path.join(subpath, path)
        if os.path.isdir(cur_filepath):
            tex_files.extend(rec_tex_files(cur_filepath, []))
        elif cur_filepath.endswith('.tex'):
            tex_files.append(cur_filepath)
    return tex_files


def dig_tex_files(subpath, tex_files: list, pool_size=50) -> list:
    p_pool = Pool(processes=pool_size)
    results = []
    subpaths = os.listdir(subpath)
    for i, path in enumerate(tqdm(subpaths)):
        cur_filepath = os.path.join(subpath, path)
        if os.path.isdir(cur_filepath):
            res = p_pool.apply_async(rec_tex_files, (cur_filepath, []))
            results.append(res)
            # tex_files.extend(rec_tex_files(cur_filepath, []))
        elif cur_filepath.endswith('.tex'):
            tex_files.append(cur_filepath)
    p_pool.close()
    p_pool.join()
    for res in results:
        tex_files.extend(res.get())
    return tex_files


def get_tex_children(tex_path):
    matches = []
    with open(tex_path, 'r', errors='ignore') as f:
        tex = f.read()
        match = re.findall(r'\\input{(.*?)}', tex)
        matches.extend(match)
        match = re.findall(r'\\include{(.*?)}', tex)
        matches.extend(match)
    return matches


def preprocess_tex_project(project_path, pool_size=50):
    tex_paths = dig_tex_files(project_path, [])
    tex_names = [os.path.basename(tex_path)[:-4] for tex_path in tex_paths]
    tex_dict = dict(zip(tex_names, tex_paths))

    p_pool = Pool(processes=pool_size)
    results = []

    for tex_path in tqdm(tex_paths):
        res = p_pool.apply_async(get_tex_children, (tex_path,))
        results.append(res)
        # children = get_tex_children(tex_path)
    p_pool.close()
    p_pool.join()
    for children in results:
        for child in children.get():
            if child in tex_dict:
                tex_dict.pop(child)
    return list(tex_dict.values())


def split_list_by_n(origin_list, n):
    origin_list.sort()
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--world_size', type=int, default=1)
#     parser.add_argument('--rank', type=int, default=0)
#
#     args = parser.parse_args()
#
#     p_paths = ["/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Computer_Science",
#                "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Mathematics",
#                "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Quantitative_Biology",
#                "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/other",
#                "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Statistics"]
#     p_path = p_paths[0]
#     meta_path = os.path.join(p_path, "meta.txt")
#     pdf_out_paths = []
#     tex_paths = []
#     with open(meta_path, "r") as f:
#         for tex_path in f:
#             tex_paths.append(tex_path)
#             pdf_out_path = os.path.dirname(tex_path)
#             pdf_out_paths.append(pdf_out_path)
#     convert_tex_to_png(pdf_out_paths, tex_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    args = parser.parse_args()

    dataset_path = "/share/home/jijunhui/Aminer_Texfiles_Clean"
    parts = os.listdir(dataset_path)
    divided_parts = split_list_by_n(parts, args.world_size)
    select_parts = divided_parts[args.rank]

    for part in select_parts:
        p_path = os.path.join(dataset_path, part)
        meta_path = os.path.join(p_path, "meta.txt")
        parent_texes = preprocess_tex_project(p_path, pool_size=128)
        with open(meta_path, "w") as f:
            for tex in parent_texes:
                f.write(tex + "\n")
        print(f"{len(parent_texes)} root texes written to {meta_path}")



#     matches = []
#     tex = r"""
#     \input{begin}
# \includeonly{qianyan,end}
# \begin{document}
# 	\include{qianyan}
# 	\include{end}
# \end{document}"""
#     match = re.findall(r'\\input{(.*?)}', tex)
#     print(match)
#     matches.extend(match)
#     match = re.findall(r'\\include{(.*?)}', tex)
#     print(match)
#     matches.extend(match)
#     print(matches)

    # matches.extend(match.groups())
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--world_size', type=int, default=1)
#     parser.add_argument('--rank', type=int, default=0)
#     args = parser.parse_args()
#
#     dataset_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Physics"
#     all_texes = []
#     for dir in os.listdir(dataset_path):
#         dir_path = os.path.join(dataset_path, dir)
#         for tex in os.listdir(dir_path):
#             tex_path = os.path.join(dir_path, tex)
#             all_texes.append(tex_path)
#     divided_texes = split_list_by_n(all_texes, args.world_size)
#     selected_texes = divided_texes[args.rank]
#     for tex_path in tqdm(selected_texes):
#         convert_tex_to_png('./Physics_output', tex_path)
