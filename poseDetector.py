#Title: POSE DETECTOR
#Developer: Vishwas Puri
#Purpose: A program that detects the 33 unique points on your body  on a live stream camera!

#It uses media pipe (by Google) and its pre-trained models with a data set of thousands of body photos to determine the unique 33 points in our body.

#This program is made using python supported by streamlit.
import streamlit as st
import mediapipe as mp
import cv2
st.set_page_config(layout="wide")
col = st.empty()

#defining mediapipe's inbuilt pose recogignition models
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
st.write("Press start to turn on camera, move back and pose your body!")

def poseDetector():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            # converting image to rgb
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # draw points over the 33 recognized pose points
            results = pose.process(imgRGB)
            mpDraw.draw_landmarks(
                img,
                results.pose_landmarks,
                mpPose.POSE_CONNECTIONS)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # setting up streamlit camera configuration
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

    # Info Block
    st.write("If camera doesn't turn on, please ensure that your camera permissions are on!")
    with st.expander("Steps to enable permission"):
        st.write("1. Click the lock button at the top left of the page")
        st.write("2. Slide the camera slider to on")
        st.write("3. Reload your page!")

    st.subheader("Testimonials")
    st.image("testimonials.gif")
    st.caption("MediaPipe ML Solutions, https://google.github.io/mediapipe/solutions/pose.html")

if __name__ == "__main__":
    poseDetector()
