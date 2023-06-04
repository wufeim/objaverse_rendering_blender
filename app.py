import os
import shutil

import gradio as gr
import objaverse
from PIL import Image
import wget

gr.close_all()

title = """## Render Objaverse Models

Find Objaverse models from [here](https://objaverse.allenai.org/explore).

Make sure your objects have pose aligned with the examples images below.

Source code available [here](https://github.com/wufeim/objaverse_rendering_blender).
"""
OBJAVERSE_MODEL_URL = 'https://huggingface.co/datasets/allenai/objaverse/resolve/main'
OBJAVERSE_OBJECT_PATHS = objaverse._load_object_paths()

example_imgs = [Image.open(f'eg/pose_{i}.png') for i in range(4)]


def get_objaverse_url_from_uid(uid):
    return os.path.join(OBJAVERSE_MODEL_URL, OBJAVERSE_OBJECT_PATHS[uid])


def render_objects(object_ids, distance, azimuth, elevation):
    imgs = []
    for model_id in object_ids.split(','):
        model_id = model_id.strip()
        try:
            temp_dir = f'temp_{model_id}'
            os.makedirs(temp_dir, exist_ok=True)

            model_path = os.path.join(temp_dir, f"{model_id}.glb")
            if not os.path.isfile(model_path):
                wget.download(get_objaverse_url_from_uid(model_id), out=temp_dir)

            cmd = (
                f'export DISPLAY=:0.{0} && '
                f'blender-3.2.2-linux-x64/blender -b -P render_script.py -- '
                f'--object_path {model_path} --engine CYCLES '
                f'--output_dir {temp_dir} --distance {distance} --azimuth {azimuth} '
                f'--elevation {elevation}'
            )
            print(cmd)
            os.system(cmd)

            for i in range(1):
                imgs.append(Image.open(os.path.join(temp_dir, f"{i:03d}.png")))
            shutil.rmtree(temp_dir)
        except:
            raise gr.Error(f'Error when processing model {model_id}')
            imgs.append(Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)))
    return imgs


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(title)
    with gr.Row():
        with gr.Column(scale=1):
            object_ids = gr.Textbox(label='List of Objaverse object IDs, separated by commas (e.g., tpvzmLUXAURQ7ZxccJIBZvcIDlr)')
            distance = gr.Slider(label='Distance', minimum=0.5, maximum=3.0, value=1.5, step=0.1)
            azimuth = gr.Slider(label='Azimuth (degrees)', minimum=-180, maximum=180, value=0, step=5)
            elevation = gr.Slider(label='Elevation (degrees)', minimum=-90, maximum=90, value=0, step=5)
            render_btn = gr.Button(value='Render Objects')
        with gr.Column(scale=4):
            display_box = gr.Gallery(label='Rendered Images', show_label=False, elem_id='gallery').style(grid=4, height='auto')
    render_btn.click(fn=render_objects, inputs=[object_ids, distance, azimuth, elevation], outputs=[display_box])
    with gr.Row():
        gr.Gallery(label='Rendered Images', show_label=False, elem_id='gallery', value=example_imgs).style(grid=4, height='auto')


if __name__ == '__main__':
    block.launch(server_name="0.0.0.0", server_port=7880)
