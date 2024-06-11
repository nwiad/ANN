import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image

def export_tensorboard_curves(logdir, output_dir):
    # 加载TensorBoard日志文件
    ea = event_accumulator.EventAccumulator(logdir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tag in ea.Tags()['scalars']:
        # print(*ea.Scalars(tag))
        times = [scalar.wall_time for scalar in ea.Scalars(tag)]
        steps = [scalar.step for scalar in ea.Scalars(tag)]
        values = [scalar.value for scalar in ea.Scalars(tag)]
        # times, steps, values = zip(*ea.Scalars(tag))
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label=tag)
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, f"{tag.replace('/', '_')}.png"))
        plt.close()
        
def concat_images(image_folder, subdir, output_image_path):
    images = [Image.open(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg')) and not "sum" in f and not "fid" in f]

    images.append(Image.open(f"./results/{subdir}/4999/samples.png"))
    print(image_folder)
    print(images)
    widths, heights = zip(*(i.size for i in images))
    max_width, max_height = max(widths), max(heights)

    # 创建新图片
    new_im = Image.new('RGB', (max_width * 2, max_height * 3))

    # 拼接图片
    y_offset = 0
    for i, im in enumerate(images):
        if i % 2 == 0 and i != 0:
            y_offset += im.size[1]
        x_offset = (i % 2) * max_width
        new_im.paste(im.resize((max_width, max_height)), (x_offset, y_offset))

    # 保存图片
    new_im.save(output_image_path)
    
logdir = './runs'  # TensorBoard日志文件目录
for subdir in os.listdir(logdir):
    subdir_path = os.path.join(logdir, subdir)
        
    if os.path.isdir(subdir_path):
        output_dir = os.path.join(logdir, subdir)
        export_tensorboard_curves(subdir_path, output_dir)
        concat_images(subdir_path, subdir, f"./pics/{subdir}.png")

