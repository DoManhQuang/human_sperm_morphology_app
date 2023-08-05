import os
import PIL
import gradio as gr


def demo_UI(img):
    # print('img', img)
    input_length = len(os.listdir("/content/inputs"))
    PIL.Image.fromarray(img).save(f'/content/inputs/imgs_input_{input_length + 1}.jpg')
    os.system(f'bash run_yolov6.sh /content/inputs/imgs_input_{input_length + 1}.jpg /content/ouputs')
    return PIL.Image.open(f'/content/ouputs/imgs_input_{input_length + 1}.jpg')

demo = gr.Interface(
    demo_UI, 
    inputs=gr.inputs.Image(), 
    outputs="image",
    examples=[
      ["/content/tokyo.jpg"],
      ["/content/conan.jpg"]
  ]
)

demo.launch(
    debug=True,
    share = True
)