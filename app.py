import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess
import ffmpeg

#loading the saved model
loaded_model = load_model("")
#loaded_model2 = load_model("")

IMAGE_HEIGHT, IMAGE_WIDTH = 180,180
CLASSES_LIST = ["cover","defense","flick","hook","late_cut","lofted","pull","square_cut","straight","sweep"]
SEQUENCE_LENGTH = 15

def predict_on_video(video_file_path,output_file_path,SEQUENCE_LENGTH,model_name):
  video_reader = cv2.VideoCapture(video_file_path)

  #get height and width of the video
  original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)) 
  original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

  video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M','P','4','V'),video_reader.get(cv2.CAP_PROP_FPS),(original_video_width),(original_video_height))
  
  frames_queue = deque(maxlen = SEQUENCE_LENGTH)

  predicted_class_name = ''

  while video_reader.isOpened():
    ok,frame = video_reader.read()
    if not ok:
      break
    resized_frame = cv2.resize(frame,(IMAGE_HEIGHT,IMAGE_WIDTH))
    normalized_frame = resized_frame/255
    frames_queue.append(normalized_frame)

    if len(frames_queue) == SEQUENCE_LENGTH:
      predicted_label_probabilities = model_name.predict(np.expand_dims(frames_queue,axis = 0))[0]
      predicted_label = np.argmax(predicted_label_probabilities)
      predicted_class_name = CLASSES_LIST[predicted_label]

    cv2.putText(frame,predicted_class_name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    video_writer.write(frame)

  video_reader.release()
  video_writer.release()

def predict_single_action(video_file_path,SEQUENCE_LENGTH):

  video_reader = cv2.VideoCapture(video_file_path)

  original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)) 
  original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

  frames_list = []
  predicted_class_name = ''
  video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
  skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

  for frame_counter in range(SEQUENCE_LENGTH):
    video_reader.set(cv2.CAP_POS_FRAMES, frame_counter * skip_frames_window)
    success, frame = video_reader.read()
    if not success:
      break
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = cv2.resized_frame/255
    frame_list.append(normalized_frame)
  predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_list,axis=0))[0]
  predicted_label = np.argmax(predicted_labels_probabilities)
  predicted_class_name = CLASSES_LIST[predicted_label]

  #displaying the predicted action along with the predicted confidence
  print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

  video_reader.release()

# def home():
#     try:
#         st.title("Home")
#         st.write("Choose another option")

def main(): 
    
    # giving a title
    st.title('Video Classification Web App')

    # menu = ["Home","Predict on video","Predict on Single Action"]
    # choice = st.sidebar.selectbox("Menu",menu)
    # if choice == 'Home':


    #Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        #store the uploaded video locally
        with open(os.path.join("C:/Users/HP/Desktop/cricket_shots/",uploaded_file.name.split("/")[-1]),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")
                       
        if st.button('Classify The Video'):
            # Construct the output video path.
            output_video_file_path = "C:/Users/HP/Desktop/cricket_shots/"+uploaded_file.name.split("/")[-1].split(".")[0]+"_output1.mp4"
            with st.spinner('Wait for it...'):
                # Perform Action Recognition on the Test Video.
                predict_on_video("C:/Users/HP/Desktop/cricket_shots/"+uploaded_file.name.split("/")[-1], output_video_file_path, SEQUENCE_LENGTH)
                #OpenCV’s mp4v codec is not supported by HTML5 Video Player at the moment, one just need to use another encoding option which is x264 in this case 
                os.chdir('C:/Users/HP/Desktop/cricket_shots/')
                subprocess.call(['ffmpeg','-y', '-i', uploaded_file.name.split("/")[-1].split(".")[0]+"_output1.mp4",'-vcodec','libx264','-f','mp4','output4.mp4'],shell=True)
                st.success('Done!')
            
            #displaying a local video file
            video_file = open("C:/Users/HP/Desktop/cricket_shots/" + 'output4.mp4', 'rb') #enter the filename with filepath
            video_bytes = video_file.read() #reading the file
            st.video(video_bytes) #displaying the video
    
    else:
        st.text("Please upload a video file")
    
    
    
if __name__ == '__main__':
    main()