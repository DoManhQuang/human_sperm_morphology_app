import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from detect_track import DeepSortTracker,node_to_dict,node_to_json
import my_yolov6
yolov6_model = my_yolov6.my_yolov6("./weights/1.0/last_ckpt.pt", "cpu", 
                                   "./weights/dataset.yaml", 640, False)
import json
# Load your video processing model (replace this with your actual model)

def process_video(video):
    cap = cv2.VideoCapture(video.name)
    tracker =  DeepSortTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        tracker.detect_per_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return json.dumps(tracker.memo,default=node_to_dict)

# Define the Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs="video",
    outputs="text",
    capture_session=True,  # To release GPU memory after processing
    title="Video to Text",
    description="Convert a video into text using a model.",
    article="Some article describing the project or model."
)

# Define a function to save the text output to a file
def save_output_text(output_text, output_file):
    with open(output_file, "w") as f:
        f.write(output_text)

# Launch the Gradio interface
if __name__ == "__main__":
    input_file = "input_video.mp4"  # Path to the input video file
    output_file = "output_text.txt"  # Path to save the output text file
    
    interface_input = gr.inputs.File()  # Input is a video file
    interface_output = gr.outputs.Textbox()  # Output is a textbox
    
    interface = gr.Interface(
        fn=process_video,
        inputs=interface_input,
        outputs=interface_output,
        capture_session=True,
        title="Video to Text",
        description="Convert a video into text using a model.",
        article="Some article describing the project or model."
    )

    interface.launch()

    # After processing, save the output text to a file
    processed_text = process_video(input_file)  # Process the video
    save_output_text(processed_text, output_file)  # Save the output text to a file
