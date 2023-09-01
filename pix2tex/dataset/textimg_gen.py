from PIL import Image, ImageDraw, ImageFont, ImageColor
import textwrap
from io import BytesIO
import logging
import os, sys
import webdataset as wds
from tqdm import tqdm
import random
# set logging level to error
logging.basicConfig(level=logging.ERROR)

def generate_image(text, font_path, img_width, font_size, line_spacing=10, align='center', text_color='black', background_color='white'):
    # 创建字体对象
    font = ImageFont.truetype(font_path, font_size)

    if len(text[:20]) == 0:
        raise ValueError('Text must contain at least one character')

    avg_char_width = sum(font.getsize(text[i])[0] for i in range(len(text[:20]))) // len(text[:20])  # 使用大写字母A作为参考
    if avg_char_width == 0:
        raise ValueError("avg_char_width is 0")
    estimated_chars_per_line = img_width // avg_char_width

    # 使用textwrap库将文本包装到多行
    lines = textwrap.wrap(text, width=estimated_chars_per_line)  # 根据图片宽度和字体大小调整宽度
    lines = [line for line in lines]

    # 计算文本高度和宽度
    img_height = sum(font.getsize(line)[1] for line in lines) + (len(lines) - 1) * line_spacing

    # 创建一个背景的图像
    img = Image.new('RGB', (img_width, img_height), color=background_color)

    # 创建一个绘图对象
    d = ImageDraw.Draw(img)

    # 在图像上绘制文本
    y = 0
    for line in lines:
        width, height = d.textsize(line, font=font)
        if align == 'center':
            text_x = (img_width - width) / 2
        else:
            text_x = 0
        text_y = y
        d.text((text_x, text_y), line, fill=text_color, font=font)
        y += height + line_spacing

    return img
    # # 保存图像到文件
    # with BytesIO() as f:
    #     img.save(f, format="jpg")

    # return f.getvalue()

from PIL import Image
import cv2
import numpy as np

def augment_image(pil_image, rotation_angle, tilt_angle, alpha, beta, background_color=(255, 255, 255)):
    """
    Augment the given image.
    
    Parameters:
    - pil_image: Input PIL Image to be augmented
    - rotation_angle: Angle in degrees for simple rotation
    - tilt_angle: Angle in degrees for 3D tilt (tilt around the y-axis)
    - alpha: Contrast control
    - beta: Brightness control
    - background_color: RGB tuple for the background color (default is white)

    Returns:
    - Augmented PIL Image
    """
    if isinstance(background_color, str): # pil name
        background_color = ImageColor.getrgb(background_color)
        # to bgr
        background_color = (background_color[2], background_color[1], background_color[0])
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Compute the scale to keep all the image content after rotation
    rows, cols, _ = img.shape
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
    abs_cos = abs(M_rot[0,0])
    abs_sin = abs(M_rot[0,1])
    new_cols = int(rows * abs_sin + cols * abs_cos)
    new_rows = int(rows * abs_cos + cols * abs_sin)
    M_rot[0, 2] += (new_cols - cols) / 2
    M_rot[1, 2] += (new_rows - rows) / 2
    
    img_rot = cv2.warpAffine(img, M_rot, (new_cols, new_rows), borderValue=background_color)
    
    # 3D tilt (around y-axis)
    focal_length = new_cols
    pts_src = np.float32([[0,0], [new_cols-1,0], [0,new_rows-1], [new_cols-1,new_rows-1]])
    pts_dst = np.float32([
        [-focal_length*np.tan(np.radians(tilt_angle)),0],
        [new_cols-1+focal_length*np.tan(np.radians(tilt_angle)),0],
        [0,new_rows-1],
        [new_cols-1,new_rows-1]
    ])
    M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
    corners = np.array([[0, 0], [new_cols-1, 0], [0, new_rows-1], [new_cols-1, new_rows-1]], dtype=np.float32)
    new_corners = cv2.perspectiveTransform(corners[None, :, :], M_persp).squeeze()
    x_min, y_min = new_corners.min(axis=0)
    x_max, y_max = new_corners.max(axis=0)

    M_translate = np.eye(3)
    M_translate[0, 2] = -x_min
    M_translate[1, 2] = -y_min
    M_persp = M_translate @ M_persp
    img_tilt = cv2.warpPerspective(img_rot, M_persp, (int(x_max - x_min), int(y_max - y_min)), borderValue=background_color)
    
    # Adjust contrast and brightness
    img_contrast = cv2.convertScaleAbs(img_tilt, alpha=alpha, beta=beta)
    
    # Convert the augmented image back to PIL format
    augmented_pil_image = Image.fromarray(cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB))
    
    return augmented_pil_image


avail_font_colors = [
    'aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgrey','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkslategrey','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dimgrey','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','grey','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightgrey','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightslategrey','lightsteelblue','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','rebeccapurple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','slategrey','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen'
]
avail_background_colors = [
    'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray', 'grey', 'maroon', 'navy', 'olive', 'purple', 'silver', 'teal'
]

en_font_pool = [
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/arial.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/times.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Supplemental/Times New Roman.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Supplemental/Georgia.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Supplemental/Courier New.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Supplemental/Arial Italic.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Helvetica.ttf'
]
zh_font_pool = [
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/STXIHEI.TTF',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/simkai.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/STSONG.TTF',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/simhei.ttf',
    '/nxchinamobile2/shared/img_datasets/txt_data/Fonts/STKAITI.TTF',
    '/nxchinamobile2/shared/img_datasets/txt_data/System/Library/Fonts/SimSun.ttf'
]
# # read file each line a ttf
# with open('/nxchinamobile2/shared/dm/vlbackbone/tests/fonts.txt', 'r') as fin:
#     en_font_pool2 = fin.readlines()
# with open('/nxchinamobile2/shared/dm/vlbackbone/tests/fonts_zh.txt', 'r') as fin:
#     zh_font_pool2 = fin.readlines()

def txt2img(
    txt: str,
    en: bool,
    add_aug: bool,
    width_sizes=(480, 2000),
    rotation_angle=(-20, 20), tilt_angle=(-15, 15), alphas=(0.95, 1.05), betas=(0, 0),
):
    import random

    # select font from pool
    if en:
        font = random.choice(en_font_pool)
    else:
        font = random.choice(zh_font_pool)
    
    width = random.randint(*width_sizes)
    # print("width", width)
    # if len(txt) < 80 and width > 1000:
    #     width = random.randint(280, 600)
    font_size = random.randint(20, 40)
    # if len(txt) < 80:
    #     font_size *= 3
    background_color = 'white'
    font_color = 'black'
    if add_aug:
        # color, bg etc
        if random.random() < 0.25:
            background_color = random.choice(avail_background_colors)
            # print('background_color', background_color)
        if random.random() < 0.25:
            font_color = random.choice(avail_font_colors)
            # print('font_color', font_color)
        a0, b0, c0 = ImageColor.getrgb(background_color) 
        a1, b1, c1 = ImageColor.getrgb(font_color)
        while abs(a0 - a1) + abs(b0 - b1) + abs(c0 - c1) < 100:
            font_color = random.choice(avail_font_colors)
            a1, b1, c1 = ImageColor.getrgb(font_color)
            # print('font_color', font_color)
            # print('background_color', background_color)
    spacing = random.randint(font_size // 3, int(font_size))
    # generate image
    image = generate_image(txt, font, width, font_size, spacing, align=random.choice(['left', 'center']), text_color=font_color, background_color=background_color)
    if add_aug:
        if random.random() < 0.5:
            # random float between rotation_angle[0] and rotation_angle[1]
            rotation_angle, tilt_angle = random.uniform(*rotation_angle), random.uniform(*tilt_angle)
            # print('rotation_angle', rotation_angle)
            # print('tilt_angle', tilt_angle)
        else:
            rotation_angle, tilt_angle = 0, 0
        if random.random() < 0.5:
            alpha, beta = random.uniform(*alphas), random.randint(*betas)
        else:
            alpha, beta = 1, 0
        image = augment_image(image, rotation_angle, tilt_angle, alpha, beta, background_color=background_color)
    return image

# # 生成图像
# text_to_display = """
# We present symbol tuning—finetuning language models on in-context input–label pairs where natural language labels (e.g., “positive/negative sentiment”) are re- placed with arbitrary symbols (e.g., “foo/bar”). Symbol tuning leverages the intuition that when a model cannot use instructions or natural language labels to figure out a task, it must instead do so by learning the input–label mappings. $a^2+bx +c_{fick} = 0 \mathbf{2+67}$
# We experiment with symbol tuning across Flan-PaLM models up to 540B param- eters and observe benefits across various settings. First, symbol tuning boosts performance on unseen in-context learning tasks and is much more robust to under- specified prompts, such as those without instructions or without natural language.
# """
# # text_to_display = '来说，该实际预报距离就不能满足施工进度要求。因为预报的准备时间较长（含等待TBM检修或不掘进的时间）。当然，如果能尽量减小或避免爆破震动对TBM设备的影响，改变爆破震动方式，TSP也还是可以改进或变通使用的，只要能达到地质超前预报效果即可。 [1] 物探技术在地质勘察中的作用 （1）提供基础资料 工作人员在充分掌握地质灾害易发地区的区域地质资料后．初步预测并圈定地质灾害调查的目标，选择合理的物探方法和技术对目标地质体进行勘查．通过分析、解释，可以掌握灾害体的范围、分布、性质及现状，并对地质灾害是否继续扩大的可能性做出迅速的判断，为后期预防和控制地质灾害提供科学的基础资料。 （2）'


# for i in range(10):
#     print(i)
#     image = txt2img(text_to_display, True, True)
#     image.save(f"./samples/testocr/test_{i}.jpg")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, required=True, help='txt path')
    parser.add_argument('--en', action='store_true', help='whether to use english font')
    parser.add_argument('--add_aug', action='store_true', help='whether to add augmentation')
    parser.add_argument('--save_dir', type=str, required=True, help='save dir')
    # parser.add_argument('--split_line', type=int, default=512, help='split line')
    # parser.add_argument('--rank', type=int, default=0, help='rank')
    # parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args()

    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    
    # if args.en:
    #     args.split_line *= 2
    if args.en:
        split_line = (64, 128)
    else:
        split_line = (16, 32)
        
    args.save_dir = os.path.join(args.save_dir, 'part-%3d'%args.rank)

    # 确保保存目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    count = 0
    tar_index = 0
    tar_path = os.path.join(args.save_dir, "%06d.tar"% tar_index)
    sink = wds.TarWriter(tar_path)

    # 读取文本文件
    with open(args.txt_path, 'r') as f:
        filenames = f.readlines()
    
    for filename in filenames[args.rank::args.world_size]:
        with open(filename.strip(), 'r') as fin:
            lines = fin.readlines()

        for line in lines:
            line = line.strip()
            while len(line) > 0:
                # 如果必要的话，分割行
                current_split_line = random.randint(*split_line)
                text_segment = line[:current_split_line]
                line = line[current_split_line:]
                
                if args.en:
                    text_segment_list = text_segment.split(" ")[1:-1]
                    if len(text_segment_list) != 0:
                        text_segment = " ".join(text_segment_list)
                
                try: 
                    # 将文本转换为图像
                    img = txt2img(text_segment, args.en, args.add_aug, width_sizes=(100, 600))
                except ValueError as e:
                    continue
                
                # 将PIL图像转换为字节
                byte_io = BytesIO()
                if args.add_aug:
                    save_quality = int(60 + random.random()*40)
                else:
                    save_quality = 75
                # print("save quality=", save_quality)
                img.save(byte_io, format='JPEG', quality=save_quality)
                img_bytes = byte_io.getvalue()

                # 将图像和文本保存到webdataset中
                key = "%06d" % count
                sink.write({
                    "__key__": key,
                    "jpg": img_bytes,
                    "txt": text_segment,
                })
                
                count += 1
                # 如果达到10,000个样本，分割.tar文件
                if count % 10000 == 0:
                    sink.close()
                    tar_index += 1
                    tar_path = os.path.join(args.save_dir, "%06d.tar"% tar_index)
                    sink = wds.TarWriter(tar_path)
                    print(f'rank {args.rank} finish {tar_path}.', flush=True)

    sink.close()

# srun -N 1 --ntasks-per-node=50 --cpus-per-task=2 --export=ALL --job-name=synocr-en --partition=dev --time=24:00:00 --output=synocr-en-aug.out --error=synocr-en-aug.err python -W ignore  tests/textimg_gen.py --en --add_aug --txt_path tests/enlist_aug.txt --save_dir /nxchinamobile2/shared/img_datasets/synocr/en_aug

# srun -N 1 --ntasks-per-node=50 --cpus-per-task=2 --export=ALL --job-name=synocr-cn-aug --partition=dev --time=24:00:00 --output=synocr-cn.out --error=synocr-cn.err python -W ignore  tests/textimg_gen.py --txt_path tests/cnlist.txt --save_dir /nxchinamobile2/shared/img_datasets/synocr/cn2


# srun -N 1 --ntasks-per-node=50 --cpus-per-task=2 --export=ALL --job-name=synocr-cn-10 --partition=dev --time=24:00:00 --output=tests/synocr-cn-10.out --error=tests/synocr-cn-10.err python -W ignore  tests/textimg_gen.py --txt_path tests/cnlist.txt --save_dir /nxchinamobile2/shared/hwy/tmp/synocr_cn_10 --split_line 10

# srun -N 1 --ntasks-per-node=50 --cpus-per-task=2 --export=ALL --job-name=synocr-en-32 --partition=dev --time=24:00:00 --output=tests/synocr-en-32.out --error=tests/synocr-en-32.err python -W ignore  tests/textimg_gen.py --en --txt_path tests/enlist.txt --save_dir /nxchinamobile2/shared/hwy/tmp/synocr_en_32 --split_line 32

# 08.21
# srun -N 10 --ntasks-per-node=50 --cpus-per-task=1 --export=ALL --job-name=synocr-cn-32-part2 --partition=dev --time=24:00:00 --output=tests/synocr-cn-32aug-part2.out --error=tests/synocr-cn-32aug-part2.err python -W ignore  tests/textimg_gen.py --add_aug --txt_path tests/cnlist.txt --save_dir /nxchinamobile2/shared/hwy/tmp/synocr_cn_32aug_part2

# srun -N 20 --ntasks-per-node=50 --cpus-per-task=1 --export=ALL --job-name=synocr-en-128 --partition=dev --time=24:00:00 --output=tests/synocr-en-128aug-v2.out --error=tests/synocr-en-128aug-v2.err  --exclude=g0018 python -W ignore  tests/textimg_gen.py --add_aug --en --txt_path tests/enlist_aug.txt --save_dir /nxchinamobile2/shared/hwy/tmp/synocr_en_128aug_v2 