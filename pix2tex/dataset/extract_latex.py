import argparse
from tqdm import tqdm
import math
import os
import re
import numpy as np
from typing import List
import sys
import signal
import eventlet

eventlet.monkey_patch()

sys.path.append("/nxchinamobile2/shared/jjh/projects/LaTeX-OCR")


MIN_CHARS = 1
MAX_CHARS = 3000
dollar = re.compile(r'((?<!\$)\${1,2}(?!\$))(.{%i,%i}?)(?<!\\)(?<!\$)\1(?!\$)' % (1, MAX_CHARS))
inline = re.compile(r'(\\\((.*?)(?<!\\)\\\))|(\\\[(.{%i,%i}?)(?<!\\)\\\])' % (1, MAX_CHARS))
equation = re.compile(r'\\begin\{(equation|math|displaymath)\*?\}(.{%i,%i}?)\\end\{\1\*?\}' % (1, MAX_CHARS), re.S)
align = re.compile(r'(\\begin\{(align|alignedat|alignat|flalign|eqnarray|aligned|split|gather)\*?\}(.{%i,%i}?)\\end\{\2\*?\})' % (1, MAX_CHARS), re.S)
displaymath = re.compile(r'(?:\\displaystyle)(.{%i,%i}?)((?<!\\)\}?(?:\"|<))' % (1, MAX_CHARS), re.S)
outer_whitespace = re.compile(
    r'^\\,|\\,$|^~|~$|^\\ |\\ $|^\\thinspace|\\thinspace$|^\\!|\\!$|^\\:|\\:$|^\\;|\\;$|^\\enspace|\\enspace$|^\\quad|\\quad$|^\\qquad|\\qquad$|^\\hspace{[a-zA-Z0-9]+}|\\hspace{[a-zA-Z0-9]+}$|^\\hfill|\\hfill$')
label_names = [re.compile(r'\\%s\s?\{(.*?)\}' % s) for s in ['ref', 'cite', 'label', 'eqref']]



import functools
def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator


def check_brackets(s):
    a = []
    surrounding = False
    for i, c in enumerate(s):
        if c == '{':
            if i > 0 and s[i-1] == '\\':  # not perfect
                continue
            else:
                a.append(1)
            if i == 0:
                surrounding = True
        elif c == '}':
            if i > 0 and s[i-1] == '\\':
                continue
            else:
                a.append(-1)
    b = np.cumsum(a)
    if len(b) > 1 and b[-1] != 0:
        raise ValueError(s)
    surrounding = s[-1] == '}' and surrounding
    if not surrounding:
        return s
    elif (b == 0).sum() == 1:
        return s[1:-1]
    else:
        return s


def remove_labels(string):
    for s in label_names:
        string = re.sub(s, '', string)
    return string


def clean_matches(matches, min_chars=MIN_CHARS):
    faulty = []
    for i in range(len(matches)):
        if 'tikz' in matches[i]:  # do not support tikz at the moment
            faulty.append(i)
            continue
        matches[i] = remove_labels(matches[i])
        matches[i] = matches[i].replace('\n', '').replace(r'\notag', '').replace(r'\nonumber', '')
        matches[i] = re.sub(outer_whitespace, '', matches[i])
        if len(matches[i]) < min_chars:
            faulty.append(i)
            continue
        # try:
        #     matches[i] = check_brackets(matches[i])
        # except ValueError:
        #     faulty.append(i)
        if matches[i][-1] == '\\' or 'newcommand' in matches[i][-1]:
            faulty.append(i)

    matches = [m.strip() for i, m in enumerate(matches) if i not in faulty]
    return list(set(matches))


def find_math(s: str, wiki=False) -> List[str]:
    r"""Find all occurences of math in a Latex-like document. 

    Args:
        s (str): String to search
        wiki (bool, optional): Search for `\displaystyle` as it can be found in the wikipedia page source code. Defaults to False.

    Returns:
        List[str]: List of all found mathematical expressions
    """
    matches = []
    x = re.findall(inline, s)
    matches.extend([(g[1] if g[1] != '' else g[-1]) for g in x])
    if not wiki:
        patterns = [dollar, equation, align]
        groups = [1, 1, 0]
    else:
        patterns = [displaymath]
        groups = [0]
    for i, pattern in zip(groups, patterns):
        x = re.findall(pattern, s)
        matches.extend([g[i] for g in x])

    return clean_matches(matches)


@timeout(1)
def extract_formula_from_tex(filepath, wiki=False) -> List[str]:
    r"""Extract all equations from a Latex-like document.

    Args:
        filepath (str): Path to the file to extract equations from
        wiki (bool, optional): Search for `\displaystyle` as it can be found in the wikipedia page source code. Defaults to False.

    Returns:
        List[str]: List of all found mathematical expressions
    """
    from pix2tex.dataset.demacro import pydemacro
    try:
        s = pydemacro(open(filepath, 'r', encoding='utf-8', errors='ignore').read())
        with open("mid_out.tex", 'w') as f:
            f.write(s)
        f.close()
        equations = find_math(s, wiki)
    except Exception as e:
        equations = []
    return equations


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--world_size', type=int, default=1)
#     parser.add_argument('--rank', type=int, default=0)
#     parser.add_argument('--subject', type=str, default="Quantitative_Biology")
#     parser.add_argument('--wiki', type=bool, default=False)
#
#     args = parser.parse_args()
#
#     rank = args.rank
#     subject = args.subject
#
#     input_dir = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/"+subject
#     output_dir = os.path.join("/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all", subject)
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"rank_{rank}.txt")
#
#     divided_dirs = split_list_by_n(os.listdir(input_dir), args.world_size)
#     select_dirs = divided_dirs[rank]
#
#     print(f"rank{rank} starting task, file num: {len(select_dirs)}")
#     # count = 0
#     with open(output_path, 'w', 1, encoding='utf-8') as f:
#         for select_dir in tqdm(select_dirs):
#             # if count < 23220:
#             #     count += 1
#             #     continue
#             file_dir_path = os.path.join(input_dir, select_dir)
#             # print("file_dir_path: ", file_dir_path)
#             tex_files = []
#             for filename in os.listdir(file_dir_path):
#                 file_path = os.path.join(file_dir_path, filename)
#                 if filename.endswith(".tex"):
#                     tex_files.append(file_path)
#                 elif os.path.isdir(file_path):
#                     for inner_file in os.listdir(file_path):
#                         if inner_file.endswith(".tex"):
#                             tex_files.append(os.path.join(file_path, inner_file))
#             if len(tex_files) > 100:
#                 print("path:", file_path)
#             for tex_file in tex_files:
#                 if os.path.getsize(tex_file) > 1024*1024*1024:
#                     print("large file path:", tex_file)
#                 # with eventlet.Timeout(1, False):
#                 try:
#                     formulas = extract_formula_from_tex(tex_file)
#                     formulas = [f for f in formulas if len(f) >= 5]
#                     f.write("\n".join(formulas))
#                     f.write("\n")
#                 except TimeoutError:
#                     print("timeout:", tex_file)
#                     pass
#             # count += 1
#     f.close()


if __name__ == "__main__":
    extract_formula_from_tex("/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/Physics/123f5df5f54d9f24d3646f03c8db1fa2/brane_grav_taxonomy.tex")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(dest='file', type=str, help='file to find equations in')
#     parser.add_argument('--out','-o', type=str, default=None, help='file to save equations to. If none provided, print all equations.')
#     parser.add_argument('--wiki', action='store_true', help='only look for math starting with \\displaystyle')
#     parser.add_argument('--unescape', action='store_true', help='call `html.unescape` on input')
#     args = parser.parse_args()
#
#     if not os.path.exists(args.file):
#         raise ValueError('File can not be found. %s' % args.file)
#
#     from pix2tex.dataset.demacro import pydemacro
#     s = pydemacro(open(args.file, 'r', encoding='utf-8').read())
#     if args.unescape:
#         s = html.unescape(s)
#     math = '\n'.join(sorted(find_math(s, args.wiki)))
#     if args.out is None:
#         print(math)
#     else:
#         with open(args.out, 'w') as f:
#             f.write(math)
    