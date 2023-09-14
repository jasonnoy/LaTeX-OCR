import math
from retry import retry
import argparse
import glob
import os
import re
import sys
import io
import tempfile
import shlex
import subprocess
import traceback
import random
from io import BytesIO
from PIL import Image
from textimg_gen import augment_image
import matplotlib.pyplot as plt
import webdataset as wds
from tqdm import tqdm
import matplotlib.font_manager as mfm
from matplotlib import mathtext
import numpy as np


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\usepackage{amsmath}
%s
\begin{document}
\thispagestyle{empty}
\begin{tiny}
%s
\end{tiny}
\end{document}
'''

    def __init__(self, math, dpi=550, fonts=None, debug=False, **kwargs):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        if fonts is None:
            fonts = ['lmodern']
        self.math = math
        self.dpi = dpi
        self.fonts = fonts
        self.debug = debug
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        try:
            # workdir = tempfile.gettempdir()
            workdir = "/nxchinamobile2/shared/jjh/projects/LaTeX-OCR/temp"
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            font = random.choice(self.fonts)
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (font, '\n'.join(self.math))
                f.write(document)

            pngs, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes, debug=self.debug)
            return pngs, error_index
        finally:
            if not self.debug:
                if os.path.exists(texfile):
                    try:
                        os.remove(texfile)
                    except PermissionError:
                        pass

    def convert_file(self, infile, workdir, return_bytes=False, debug=False):
        infile = infile.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = 'xelatex -interaction=batchmode -output-directory %s %s' % (
                workdir.replace('\\', '/'), infile)
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            sout, serr = p.communicate()
            # extract error line from sout
            error_index, _ = extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
            # extract success rendered equation
            if error_index != []:
                # offset index start from 0, same as self.math
                error_index = [int(_) - self.prefix_line - 1 for _ in error_index]
            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            # print(sout)
            # result, _ = extract(
            #     text=sout, expression="Output written on %s \((\d+)? page" % pdffile)
            # print(result)
            # if not result:
            #     raise Exception(
            #         'xelatex rendering error, file: %s' % pdffile)
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))

            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (
                self.dpi,
                pdffile,
                pngfile,
            )  # -bg Transparent -z 9
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(
                    pdffile), os.path.exists(infile))
            if return_bytes:
                if len(self.math) > 1:
                    png = [open(pngfile.replace('.png', '') + '-%i.png' %
                                i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace(
                        '.png', '') + '.png', 'rb').read()]
            else:
                # return path
                if len(self.math) > 1:
                    png = [(pngfile.replace('.png', '') + '-%i.png' % i)
                           for i in range(len(self.math))]
                else:
                    png = [(pngfile.replace('.png', '') + '.png')]
            return png, error_index
        finally:
            # Cleanup temporaries
            if not debug:
                basefile = infile.replace('.tex', '')
                tempext = ['.aux', '.pdf', '.log']
                if return_bytes:
                    ims = glob.glob(basefile + '*.png')
                    for im in ims:
                        os.remove(im)
                for te in tempext:
                    tempfile = basefile + te
                    if os.path.exists(tempfile):
                        os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def matplot_lex2pil(text, size=64, out=None, **kwds):
    """LaTex数学公式转图片

        text        - 文本字符串，其中数学公式须包含在两个$符号之间
        size        - 字号，整型，默认64
        out         - 文件名，仅支持后缀名为.png的文件名。若维None，则返回PIL图像对象
        color_path  - 颜色文件 默认从中随机
        kwds        - 关键字参数
                        dpi         - 输出分辨率（每英寸像素数），默认72
                        family      - 系统支持的字体，None表示当前默认的字体
                        weight      - 笔画轻重，可选项包括：normal（默认）、light和bold
        """

    assert out is None or os.path.splitext(out)[1].lower() == '.png', '仅支持后缀名为.png的文件名'

    dpi = kwds.get('dpi', 72)

    families = ['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace']
    family = random.choice(families)
    math_families = ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
    math_family = random.choice(math_families)
    weights = ['ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
    weight = random.choice(weights)
    # 50%几率黑色，50%几率其他随机颜色
    if random.random() < 0.5:
        color = 'black'
    else:
        colors = ['tab:blue', 'tab:grey', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive']
        # with open(color_path, 'r') as f:
        #     colors = f.read().splitlines()
        color = random.choice(colors)
    bfo = BytesIO()  # 创建二进制的类文件对象
    prop = mfm.FontProperties(family=family, math_fontfamily=math_family, size=size, weight=weight)
    mathtext.math_to_image(text, bfo, prop=prop, dpi=dpi, color=color)
    im = Image.open(bfo).convert('RGB')
    w, h = im.size
    if w * h == 0 or w / h >= 50:
        # print(f"invalid image text: {text}")
        if kwds.get('debug'):
            raise Exception(f"invalid image text: {text}")
        raise Exception(f"invalid image text")

    if out:
        im.save(out)
    return im


def tex2pil(tex_text, use_xelatex=False, **kwargs):
    if use_xelatex:
        pngs, error_index = Latex([tex_text], **kwargs).write(return_bytes=True)
        image = Image.open(io.BytesIO(pngs[0])).convert('RGB')
    else:
        image = matplot_lex2pil(tex_text, **kwargs)
        error_index = None

    return image, error_index


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


def formula2img(str_latex, out_file, img_size=(5, 3), font_size=16):
    fig = plt.figure(figsize=img_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, str_latex, fontsize=font_size, verticalalignment='center', horizontalalignment='center')
    plt.savefig(out_file)


def formula2pil(str_latex, img_size=(10, 1), font_size=16):
    fig = plt.figure(figsize=img_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, str_latex, fontsize=font_size, verticalalignment='center', horizontalalignment='center')
    # Image.open(io.BytesIO(d))
    # plt.savefig(out_file)
    plt.savefig("test.png")
    byte_io = BytesIO()
    plt.savefig(byte_io)
    return Image.open(byte_io)


def is_str_close(a):
    """
    判断括号是否闭合
    """
    b = []
    flag = True
    for i in a:
        if i == "{" or i == "[" or i == "(":
            # 左边的括号加进去
            b.append(i)
        elif i == "}":
            # 遇到右边括号}弹出最后面的一个{
            if len(b) == 0 or b.pop() != "{":
                return False
        elif i == "]":
            # 遇到右边括号]弹出最后面的一个[
            if len(b) == 0 or b.pop() != "[":
                return False
        elif i == ")":
            # 遇到右边括号)弹出最后面的一个(
            if len(b) == 0 or b.pop() != "(":
                return False
    # 判断最后列表b里面的左边括号是否全部被弹出
    if len(b) != 0:
        flag = False
    return flag


def tex2pil_with_augment(tex, use_xelatex=True, **kwargs):
    if use_xelatex:
        pil_imgs = tex2pil(tex, **kwargs)
    else:
        pil_imgs = [formula2pil(tex)]
    avail_background_colors = [
        'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray', 'grey', 'maroon', 'navy',
        'olive', 'purple', 'silver', 'teal'
    ]
    rotation_angle = (-20, 20)
    tilt_angle = (-15, 15)
    alphas = (0.95, 1.05)
    betas = (0, 0)
    background_color = 'white'
    aug_imgs = []
    for image in pil_imgs:
        if random.random() < 0.25:
            background_color = random.choice(avail_background_colors)
        if random.random() < 0.5:
            rotation_angle, tilt_angle = random.uniform(*rotation_angle), random.uniform(*tilt_angle)
        else:
            rotation_angle, tilt_angle = 0, 0
        if random.random() < 0.5:
            alpha, beta = random.uniform(*alphas), random.randint(*betas)
        else:
            alpha, beta = 1, 0
        aug_image = augment_image(image, rotation_angle, tilt_angle, alpha, beta, background_color=background_color)
        aug_imgs.append(aug_image)
    return aug_imgs


def process_single_tex(tex):
    tex = tex.replace("&", "")
    if "{align" in tex:
        tex = tex.replace(r"\begin{align}", "").replace(r"\end{align}", "").replace(r"\begin{aligned}",
                                                                                    "").replace(
            r"\end{aligned}", "").replace(r"\begin{align*}", "").replace(r"\end{align*}", "")
    if "{gather" in tex:
        tex = tex.replace(r"\begin{gather}", "").replace(r"\end{gather}", "")
    return tex


def process_batch_tex(batch_texes, format_base, base_format):
    res_tex_strs = []
    tex_strs = []
    for texes in batch_texes:
        texes = [process_single_tex(tex) for tex in texes]
        res_tex_strs.extend(texes)
        tex_str = r" \\ ".join(texes)

        base_str = format_base[base_format]
        if base_format == "equation":
            if random.random() < 0.5:
                nonumber = r"\nonumber"
            else:
                nonumber = ""
            tex_str = base_str % (tex_str, nonumber)
        else:
            tex_str = base_str % tex_str
        tex_strs.append(tex_str)
    return tex_strs, res_tex_strs


def process_tex(tex, base_str, base_format):
    tex = process_single_tex(tex)
    if base_format == "equation":
        if random.random() < 0.5:
            nonumber = r"\nonumber"
        else:
            nonumber = ""
        tex_str = base_str % (tex, nonumber)
    else:
        tex_str = base_str % tex
    return tex_str, tex


def process_texes(tex, base_format="random", use_xelatex=False, **kwargs):
    # format_base = {"equation": r"\begin{equation} \begin{aligned} %s \end{aligned} %s \end{equation}",
    #                "displaymath": r"\begin{displaymath} \begin{aligned} %s \end{aligned} \end{displaymath}",
    #                "normal": r"$%s$"}
    # if base_format == "random":
    #     base_format = random.choice(list(format_base.keys()))

    format_base = r"$%s$"

    # tex_lines, res_tex_strs = process_batch_tex(batch_texes, format_base, base_format)
    tex_line, res_tex_str = process_tex(tex, format_base, base_format)

    math_fonts = [r'\usepackage{concmath}', r'\usepackage{cmbright}',
                  r'\usepackage{unicode-math} \setmathfont{Erewhon Math} \setmainfont{Erewhon Regular}',
                  r'\usepackage[sfdefault,lining]{FiraSans} \usepackage[fakebold]{firamath-otf} \renewcommand*\oldstylenums[1]{{\firaoldstyle #1}}',
                  r'\usepackage[sfdefault,scaled=.85]{FiraSans} \usepackage{newtxsf}',
                  r'\usepackage[sfmath]{kpfonts} \renewcommand*\familydefault{\sfdefault}',
                  r'\usepackage{kpfonts}',
                  r'\usepackage{lmodern}',
                  r'\usepackage{libertinus}',
                  r'\usepackage{libertine} \usepackage{libertinust1math}',
                  r'\usepackage{lxfonts}',
                  r'\usepackage{newpxtext,newpxmath}',
                  r'\usepackage{newtxtext,newtxmath}',
                  r'\usepackage{mathptmx}',
                  r'\usepackage{notomath}',
                  r'\usepackage[sc]{mathpazo} \linespread{1.05}',
                  r'\usepackage{pxfonts}',
                  r'\usepackage[notext]{stix} \usepackage{step}',
                  r'\usepackage{txfonts}',
                  ]

    pil_img, error_no = tex2pil(tex_line, fonts=math_fonts, use_xelatex=use_xelatex, **kwargs)

    return pil_img, res_tex_str


def check_validity(tex_str):
    if tex_str == "":
        return False
    if not is_str_close(tex_str):
        return False
    if re.match(r"[A-Za-z\s]+$", tex_str):
        return False
    return True


def preprocess_line(string):
    string = re.sub(" +", " ", string)
    return string


@retry(delay=0.1, tries=10)
def convert_pil_to_bytes(aug_img):
    byte_io = BytesIO()
    aug_img.save(byte_io, format='JPEG')
    return byte_io.getvalue()


def split_list_by_n(origin_list, n):
    origin_list.sort()
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1, help="num of pages per xelatex compilation")
    parser.add_argument('--math_type', type=str, default="short", help='choose from short, long')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()

    if args.debug:
        print("=================debug mode=================")
    if args.save_image:
        print("=================save image mode=================")

    input_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_parts_new_new"
    all_parts = range(672)
    divided_parts = split_list_by_n(all_parts, args.world_size)
    selected_parts = divided_parts[args.rank]

    for part in selected_parts:
        print(f"rank {args.rank} processing part {part}...")

        total = 0
        total_error = 0
        tar_index = 0
        input_file_path = os.path.join(input_path, f"part_{part}.txt")
        save_dir = os.path.join(f"/nxchinamobile2/shared/img_datasets/math_ocr/aminer_math_{args.math_type}", 'part-%03d' % part)
        os.makedirs(save_dir, exist_ok=True)
        tar_path = os.path.join(save_dir, "%06d.tar" % tar_index)
        sink = wds.TarWriter(tar_path)

        count = 0
        error_count = 0
        invalid_count = 0

        with open(input_file_path, "r") as f:
            if args.debug or args.math_type == 'long':
                lines = tqdm(f.readlines())
            else:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not check_validity(line):
                    invalid_count += 1
                    if args.debug or args.math_type == 'long':
                        lines.set_description(f'rank {args.rank}')
                        lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                    continue
                line = preprocess_line(line)

                if r"\\" in line:
                    use_xelatex = True
                    if args.math_type == "short":
                        continue
                else:
                    use_xelatex = False
                    if args.math_type == "long":
                        continue

                if args.debug:
                    pil_img, tex_str = process_texes(line, use_xelatex=use_xelatex, debug=args.debug,
                                                     batch_size=args.batch_size)
                else:
                    try:
                        pil_img, tex_str = process_texes(line, use_xelatex=use_xelatex, debug=args.debug,
                                                         batch_size=args.batch_size)
                    except Exception as e:
                        error_count += 1
                        if args.debug or args.math_type == 'long':
                            lines.set_description(f'rank {args.rank}')
                            lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                        continue

                if args.debug or args.save_image:
                    pil_img.save(f"./test/{count}.jpg")
                # 将PIL图像转换为字节
                try:
                    img_bytes = convert_pil_to_bytes(pil_img)
                    # 将图像和文本保存到webdataset中
                    key = "%03d%07d" % (args.rank, count)
                    tar_content = {
                        "__key__": key,
                        "png": img_bytes,
                        "txt": tex_str,
                    }
                    if args.debug:
                        print(f"count {count}, content:{tar_content}")

                    sink.write(tar_content)
                except Exception as e:
                    error_count += 1
                    print(f'image convert error, part{part}, line:{line}')
                    pil_img.save(f"./test/{count}.jpg")
                    if args.debug or args.math_type == 'long':
                        lines.set_description(f'rank {args.rank}')
                        lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                    continue
                count += 1
                if args.debug or args.math_type == 'long':
                    lines.set_description(f'rank {args.rank}')
                    lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                if count % 10000 == 0:
                    sink.close()
                    print(f'rank {args.rank} finish {tar_path}.', flush=True)
                    tar_index += 1
                    tar_path = os.path.join(save_dir, "%06d.tar" % tar_index)
                    sink = wds.TarWriter(tar_path)
        total_error += error_count
        error_count = 0
        if count % 10000 != 0:
            sink.close()
            print(f'rank {args.rank} finish {tar_path}.', flush=True)
            tar_index += 1
            tar_path = os.path.join(save_dir, "%06d.tar" % tar_index)

        total += count
        with open(os.path.join(save_dir,  'stat.txt'), 'w') as f:
            f.write(f'total: {total}, error: {total_error}')
        f.close()
