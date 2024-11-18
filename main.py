import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
import face_recognition as frg
import yaml
from utils import recognize, build_dataset,submitNew, get_info_from_id, deleteOne
from camera_input_live import camera_input_live

"# Streamlit camera input live Demo"
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

st.sidebar.title("Settings")
#Create a menu bar
menu = ["Webcam","Picture"]
choice = st.sidebar.selectbox("Input type",menu)
#Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.6,0.01)
st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

#Infomation section
st.sidebar.title("Visitor Information")
name_container = st.sidebar.empty()
id_container = st.sidebar.empty()
name_container.info('Name: Unnknown')
id_container.success('ID: Unknown')


if "found_qr" not in st.session_state:
    st.session_state.found_qr = False

if "qr_code_image" not in st.session_state:
    st.session_state.qr_code_image = None

add_btn = st.button("Add", key="Add_btn")
FRAME_WINDOW = st.image([])

if not st.session_state["found_qr"]:
    image = camera_input_live(show_controls=False)
else:
    image = st.session_state.qr_code_image

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    image, name, id = recognize(frame, TOLERANCE)
    name_container.info(f"Name: {name}")
    id_container.success(f"ID: {id}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(image)

