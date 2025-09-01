import streamlit as st

# Simulated video analysis function
def analyze_video(video_path: str, description: str):
    mistakes = []

    # Simulated findings as bullet points
    mistakes.append("Aspect ratio is not TikTok vertical (9:16).")
    mistakes.append("Detected potential short clip segments (<0.25 seconds).")
    mistakes.append("Black frames / black-screen gaps detected.")
    mistakes.append("Flashes or flickers detected in video.")
    mistakes.append("Frozen frames or stuttering detected.")
    mistakes.append("Potential typos in on-screen text.")
    mistakes.append("Audio issues or missing segments detected.")
    mistakes.append("Cuts or transitions might be abrupt.")

    return mistakes

# Streamlit UI
st.title("TikTok Video QA Web App")
st.markdown("This app allows you to upload a TikTok video from your device and generates a bullet-pointed list of potential editing mistakes.")

# File uploader
uploaded_file = st.file_uploader("Upload your TikTok video", type=["mp4"])

# Description input
description = st.text_area("Describe what the video should be:", "A short dancing clip with text captions.")

if uploaded_file is not None:
    st.write(f"Uploaded video: {uploaded_file.name}")
    st.write(f"Description: {description}")

    mistakes = analyze_video(uploaded_file.name, description)

    st.subheader("Video Mistakes")
    for mistake in mistakes:
        st.write(f"- {mistake}")

    st.markdown("---")
    st.info("To make this app accessible via a public URL, deploy it on Streamlit Cloud by connecting a Git repository containing this code. Alternatively, use ngrok to temporarily expose your local server with `ngrok http 8501`.")
else:
    st.info("Please upload a video to see analysis results.")