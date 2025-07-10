import streamlit as st
import numpy as np
import tensorflow as tf
from decord import VideoReader
import math
import os
import logging
import datetime
from tensorflow import keras
import plotly.graph_objects as go
from tensorflow.keras import layers

# Constants
input_size = 224
num_frame = 16

# Label mappings
uc_label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4}
uc_id2label = {v: k for k, v in uc_label2id.items()}

# Logging Setup
log_filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Application started.")

# Streamlit Page Config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Custom Layer for TF SavedModel
class TFSMLayer(tf.keras.layers.Layer):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = tf.saved_model.load(model_path)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        try:
            return self.model.signatures['serving_default'](input_1=inputs)['output_1']
        except KeyError:
            outputs = self.model.signatures['serving_default'](input_1=inputs)
            return outputs[list(outputs.keys())[0]]

# Cached model loader for Streamlit
@st.cache_resource
def load_model():
    model_path = r'D:\prrrr\Demo-for-master-thesis-real-time-emotion-detection\model_multi_task'
    
    # Create input layer
    inputs = keras.Input(shape=(num_frame, input_size, input_size, 3))
    
    # Load the base model
    base_model = tf.saved_model.load(model_path)
    
    # Get the base model outputs
    base_outputs = TFSMLayer(model_path)(inputs)
    
    # Debug the shape
    debug_model = keras.Model(inputs, base_outputs)
    debug_output = debug_model.predict(np.zeros((1, num_frame, input_size, input_size, 3)))
    print(f"Base model output shape: {debug_output.shape}")
    
    # If output is (1,1), we need to modify our approach
    if debug_output.shape == (1, 1):
        print("Detected single-value output from base model")
        
        # Alternative approach - use the base model directly without reshape
        x = layers.Dense(256, activation='relu')(base_outputs)
    else:
        # Original approach with reshape
        total_features = debug_output.shape[-1]
        features_per_frame = total_features // num_frame
        if total_features % num_frame != 0:
            features_per_frame = total_features
        
        reshaped = layers.Reshape((num_frame, features_per_frame))(base_outputs)
        x = layers.LSTM(256, return_sequences=False)(reshaped)
    
    # Valence branch
    valence_branch = layers.Dense(128, activation='relu')(x)
    valence_branch = layers.Dropout(0.3)(valence_branch)
    valence_out = layers.Dense(1, activation='tanh', name='valence')(valence_branch)
    
    # Arousal branch
    arousal_branch = layers.Dense(128, activation='relu')(x)
    arousal_branch = layers.Dropout(0.3)(arousal_branch)
    arousal_out = layers.Dense(1, activation='tanh', name='arousal')(arousal_branch)
    
    # Emotion branch
    emotion_branch = layers.Dense(256, activation='relu')(x)
    emotion_branch = layers.Dropout(0.5)(emotion_branch)
    emotion_out = layers.Dense(len(uc_id2label), activation='softmax', name='emotion')(emotion_branch)
    
    # Create final model
    model = keras.Model(inputs, [valence_out, arousal_out, emotion_out])
    
    return model

def read_video_decord(file_path):
    try:
        vr = VideoReader(file_path)
        total_frames = len(vr)
        frame_indices = np.linspace(0, total_frames - 1, num_frame, dtype=np.int32)
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = tf.image.convert_image_dtype(frames, tf.float32)
        frames = tf.image.resize(frames, [input_size, input_size])
        return frames
    except Exception as e:
        st.error(f"Error reading video: {e}")
        return None

def create_semi_circle_mood_meter(emotion=None):
    emotions = ['anger', 'disgust', 'fear', 'sadness', 'happiness']
    colors = ['red', 'orange', 'darkorange', 'yellow', 'skyblue']
    emojis = ['😠', '🤢', '😨', '😢', '😄']
    angles_deg = [18, 54, 90, 126, 162]
    angles_rad = [math.radians(a) for a in angles_deg]

    fig = go.Figure()

    for i, angle in enumerate(angles_rad):
        start_angle = angle - math.radians(18)
        end_angle = angle + math.radians(18)
        theta = [start_angle + (end_angle - start_angle) * t / 50 for t in range(51)]
        x = [0.5] + [0.5 + 0.45 * math.cos(t) for t in theta] + [0.5]
        y = [0.0] + [0.45 * math.sin(t) for t in theta] + [0.0]

        fig.add_trace(go.Scatter(
            x=x, y=y, fill="toself", fillcolor=colors[i],
            line=dict(color='white', width=2), mode='lines',
            hoverinfo="skip", showlegend=False
        ))

        emoji_x = 0.5 + 0.6 * math.cos(angle)
        emoji_y = 0.6 * math.sin(angle)
        fig.add_annotation(x=emoji_x, y=emoji_y, text=emojis[i],
                          showarrow=False, font=dict(size=22))

        label_x = 0.5 + 0.6 * math.cos(angle)
        label_y = 0.48 * math.sin(angle)
        fig.add_annotation(x=label_x, y=label_y, 
                          text=emotions[i].capitalize(),
                          showarrow=False, font=dict(size=13))

    if emotion and emotion.lower() in emotions:
        idx = emotions.index(emotion.lower())
        pointer_angle = angles_rad[idx]
    else:
        pointer_angle = math.radians(180)

    center_x, center_y = 0.5, 0.0
    black_circle_radius = 0.03
    theta_circle = [t * 2 * math.pi / 50 for t in range(51)]
    x_circle = [center_x + black_circle_radius * math.cos(t) for t in theta_circle]
    y_circle = [center_y + black_circle_radius * math.sin(t) for t in theta_circle]

    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle, fill='toself', fillcolor='black',
        line=dict(color='black'), mode='lines', hoverinfo='skip', showlegend=False
    ))

    white_circle_radius = 0.15
    theta_white = [t * math.pi / 50 for t in range(51)]
    x_white = [center_x + white_circle_radius * math.cos(t) for t in theta_white]
    y_white = [center_y + white_circle_radius * math.sin(t) for t in theta_white]

    fig.add_trace(go.Scatter(
        x=[center_x] + x_white + [center_x], y=[center_y] + y_white + [center_y],
        fill='toself', fillcolor='white', line=dict(color='white'),
        mode='lines', hoverinfo='skip', showlegend=False
    ))

    pointer_length = 0.3
    x_end = center_x + pointer_length * math.cos(pointer_angle)
    y_end = center_y + pointer_length * math.sin(pointer_angle)

    fig.add_shape(
        type='line',
        x0=center_x, y0=center_y,
        x1=x_end, y1=y_end,
        line=dict(color='black', width=6)
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        width=600, height=350, margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor='white', plot_bgcolor='white', title=""
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

def video_analysis_page():
    st.title("🎥 Video Emotion Analysis")
    
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mpeg4'])
    if uploaded_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_path)
        
        if st.button("Analyze Emotions", key="analyze_button"):
            with st.spinner('Processing video...'):
                model = load_model()
                frames = read_video_decord(temp_path)
                if frames is None:
                    return

                video_tensor = tf.expand_dims(frames, axis=0)
                predictions = model.predict(video_tensor)
                
                # Get all three outputs - now in correct order
                valence_val = predictions[0][0][0]  # First output is valence
                arousal_val = predictions[1][0][0]  # Second output is arousal
                emotion_pred = np.argmax(predictions[2][0])  # Third output is emotion probabilities
                
                detected_emotion = uc_id2label.get(emotion_pred, "Unknown")
                emotion_emoji = {
                    "anger": "😠", "disgust": "🤢", "fear": "😨",
                    "happiness": "😊", "sadness": "😢"
                }.get(detected_emotion, "")

                # Store all results in session state
                st.session_state.detected_emotion = detected_emotion
                st.session_state.arousal = float(arousal_val)
                st.session_state.valence = float(valence_val)

                # Display results in a single div
                st.markdown("""
                <div class="metric-card" style="padding:20px; margin-bottom:20px;">
                    <h2 style="border-bottom:2px solid #3498db; padding-bottom:10px;">Detection Results</h2>
                    <div style="display:flex; justify-content:space-between;">
                        <div style="text-align:center; flex:1;">
                            <h3>Detected Emotion</h3>
                            <h1 style="font-size:2.5em;">{emotion} {emoji}</h1>
                        </div>
                        <div style="text-align:center; flex:1;">
                            <h3>Arousal Level</h3>
                            <h1 style="font-size:2.5em;">{arousal:.2f}</h1>
                        </div>
                        <div style="text-align:center; flex:1;">
                            <h3>Valence Level</h3>
                            <h1 style="font-size:2.5em;">{valence:.2f}</h1>
                        </div>
                    </div>
                </div>
                """.format(
                    emotion=detected_emotion.capitalize(),
                    emoji=emotion_emoji,
                    arousal=arousal_val,
                    valence=valence_val
                ), unsafe_allow_html=True)

                # Enhanced risk analysis with bigger text and emojis
                negative_emotions = ['fear', 'disgust', 'sadness', 'anger']
                if detected_emotion in negative_emotions:
                    st.markdown("""
                    <div style="background-color:#ffebee; border-left:6px solid #f44336; 
                                padding:15px; margin:20px 0; border-radius:5px;">
                        <h1 style="color:#d32f2f; font-size:28px;">🚨🚨🚨 ALERT: DANGEROUS EMOTIONAL STATE DETECTED! 🚨🚨🚨</h1>
                        <p style="font-size:22px; margin-top:15px;">
                        ⚠️⚠️⚠️ <strong>DRIVER, PLEASE BE EXTREMELY CAREFUL TO PREVENT ACCIDENTS!</strong> ⚠️⚠️⚠️
                        </p>
                        <p style="font-size:20px; margin-top:10px;">
                        🛑 Slow down immediately <br>
                        🧘 Take deep breaths <br>
                        🚗 Pull over if necessary
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

def mood_meter_page():
    st.title("📊 Mood Meter Dashboard")
    
    if 'detected_emotion' not in st.session_state:
        st.session_state.detected_emotion = None
        st.info("No emotion detected yet. Please analyze a video first.")
        return

    # Standard emotion coordinates (valence, arousal)
    emotion_coords = {
        'anger':     [-0.8571,  0.4615],
        'disgust':   [-0.2571, -0.8889],
        'fear':      [-0.2857,  1.0000],
        'happiness': [ 1.0000,  0.3846],
        'sadness':   [-1.0000, -1.0000]
    }

    # Create layout with larger mood meter
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Make mood meter larger by adjusting width and height
        st.markdown("<div style='margin-bottom:30px;'>", unsafe_allow_html=True)
        create_semi_circle_mood_meter(st.session_state.detected_emotion)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create valence-arousal plot
        fig = go.Figure()
        
        # Add quadrant rectangles
        fig.add_shape(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="red", opacity=0.1, line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="green", opacity=0.1, line=dict(width=0))
        fig.add_shape(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="blue", opacity=0.1, line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="yellow", opacity=0.1, line=dict(width=0))
        
        # Add quadrant labels
        fig.add_annotation(x=-0.5, y=0.5, text="High Arousal<br>Negative Valence", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=0.5, y=0.5, text="High Arousal<br>Positive Valence", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=-0.5, y=-0.5, text="Low Arousal<br>Negative Valence", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=0.5, y=-0.5, text="Low Arousal<br>Positive Valence", showarrow=False, font=dict(size=12))
        
        # Add standard emotion points
        for emotion, coords in emotion_coords.items():
            fig.add_trace(go.Scatter(
                x=[coords[0]],
                y=[coords[1]],
                mode='markers+text',
                marker=dict(size=15, line=dict(width=2)),
                name=emotion.capitalize(),
                text=emotion.capitalize(),
                textposition="top center"
            ))
        
        # Add predicted point (larger and more prominent)
        fig.add_trace(go.Scatter(
            x=[st.session_state.valence],
            y=[st.session_state.arousal],
            mode='markers+text',
            marker=dict(
                size=25,  # Increased size
                color='#FF00FF',  # Magenta color
                symbol='star',
                line=dict(width=3, color='black')  # Thicker border
            ),
            name='Predicted Emotion',
            text='Predicted Emotion',
            textposition="bottom center",
            textfont=dict(size=14, color='black')
        ))
        
        # Add center lines
        fig.add_shape(type="line", x0=-1.1, y0=0, x1=1.1, y1=0, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=0, y0=-1.1, x1=0, y1=1.1, line=dict(color="black", width=1))
        
        # Style the plot
        fig.update_layout(
            title='Valence-Arousal Emotion Map with Quadrants',
            xaxis=dict(title='Valence (Pleasure)', range=[-1.1, 1.1]),
            yaxis=dict(title='Arousal (Intensity)', range=[-1.1, 1.1]),
            width=800,
            height=550,  # Slightly taller to accommodate quadrant labels
            showlegend=False,
            plot_bgcolor='rgba(240,240,240,0.8)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a container for the metrics
        with st.container():
            st.markdown("""
            <style>
                .metric-box {
                    background-color: white;
                    border-radius: 10px;
                    padding: 25px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
                }
                .metric-title {
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 12px;
                    margin-bottom: 25px;
                    color: #2c3e50;
                    font-size: 1.3em;
                }
                .metric-row {
                    display: flex;
                    margin-bottom: 25px;
                    align-items: center;
                }
                .metric-label {
                    color: #7f8c8d;
                    width: 45%;
                    font-size: 1em;
                    font-weight: 500;
                }
                .metric-value {
                    color: #2c3e50;
                    font-size: 1.4em;
                    font-weight: 600;
                    width: 55%;
                    text-align: right;
                }
                .arousal-value {
                    color: #e74c3c;
                }
                .valence-value {
                    color: #2ecc71;
                }
                .emotion-value {
                    font-size: 1.6em;
                    font-weight: 700;
                }
                .status-box {
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    border-left: 5px solid;
                    font-weight: 600;
                    font-size: 1.1em;
                }
                .high-risk {
                    background-color: #ffebee;
                    border-color: #d32f2f;
                    color: #d32f2f;
                }
                .moderate-risk {
                    background-color: #fff8e1;
                    border-color: #ff8f00;
                    color: #ff8f00;
                }
                .positive {
                    background-color: #e8f5e9;
                    border-color: #388e3c;
                    color: #388e3c;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Metric box
            # st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            
            # Title
            st.markdown('<div class="metric-box">Emotion Statistics</div>', unsafe_allow_html=True)
            
            # Detected Emotion
            st.markdown("""
            <div class="metric-row">
                <div class="metric-label">Detected Emotion</div>
                <div class="metric-value emotion-value">{emotion}</div>
            </div>
            """.format(emotion=st.session_state.detected_emotion.capitalize()), unsafe_allow_html=True)
            
            # Arousal Level
            st.markdown("""
            <div class="metric-row">
                <div class="metric-label">Arousal Level</div>
                <div class="metric-value arousal-value">{arousal:.2f}</div>
            </div>
            """.format(arousal=st.session_state.arousal), unsafe_allow_html=True)
            
            # Valence Level
            st.markdown("""
            <div class="metric-row">
                <div class="metric-label">Valence Level</div>
                <div class="metric-value valence-value">{valence:.2f}</div>
            </div>
            """.format(valence=st.session_state.valence), unsafe_allow_html=True)
            
            # Emotional State
            st.markdown('<div class="metric-label" style="margin-top:25px;">Emotional State</div>', unsafe_allow_html=True)
            
            # Emotional state analysis
            if st.session_state.detected_emotion in ['anger', 'fear', 'disgust']:
                st.markdown("""
                <div class="status-box high-risk">⚠️ High risk emotional state</div>
                """, unsafe_allow_html=True)
            elif st.session_state.detected_emotion == 'sadness':
                st.markdown("""
                <div class="status-box moderate-risk">⚠️ Moderate risk emotional state</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-box positive">✅ Positive emotional state</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close metric-box
            
def home_page():
    st.markdown("""
<h1 style='text-align: center;'>🎥🔊 Audio-Visual Emotion Recognition System 📊🎯</h1>
""", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50, #3498db); color:white; padding:25px; border-radius:15px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h2 style="color:white; text-align:center;">Emotion Detection using Audio-visual Modalities for Accident Prevention</h2>
        <p style="text-align:center; font-size:1.1em;">Enhancing Emotion Recognition in Safety-Critical Systems Through Psychological Dimensions and Multimodal Fusion</p>
        <p style="text-align:center; font-size:1.1em;">A collaboration between Technical University of Munich and IAV GmbH</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Key Features:
    
    <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; border-left:5px solid #3498db;">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background-color:white; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h4>🎭 Multi-Task Analysis</h4>
            <p>Simultaneous emotion classification and valence/arousal prediction</p>
        </div>
        <div style="background-color:white; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h4>🧠 TF-VideoMAE + LSTM</h4>
            <p>State-of-the-art architecture combining vision transformer and recurrent networks</p>
        </div>
        <div style="background-color:white; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h4>📊 Comprehensive Metrics</h4>
            <p>Valence MAE | Arousal MAE | Emotion Accuracy </p>
        </div>
        <div style="background-color:white; padding:15px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h4>🚨 Safety Focus</h4>
            <p>Specialized for driver monitoring and accident prevention</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🛠️ How It Works
    
    <div style="background-color:#e8f4f8; padding:20px; border-radius:10px; margin-bottom:20px;">
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px; margin:10px;">
            <h4>1️⃣ Video Input</h4>
            <p>16 frames @224×224 resolution</p>
        </div>
        <div style="flex: 1; min-width: 200px; margin:10px;">
            <h4>2️⃣ Feature Extraction</h4>
            <p>TF-VideoMAE processes spatial-temporal features</p>
        </div>
        <div style="flex: 1; min-width: 200px; margin:10px;">
            <h4>3️⃣ Temporal Analysis</h4>
            <p>LSTM captures emotion dynamics</p>
        </div>
        <div style="flex: 1; min-width: 200px; margin:10px;">
            <h4>4️⃣ Multi-Task Output</h4>
            <p>Emotion, valence, and arousal predictions</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📚 Getting Started
    
    1. Navigate to the **Video Analysis** page
    2. Upload a short video clip (MP4/AVI)
    3. Click "Analyze Emotions"
    4. View comprehensive results including:
       - Predicted emotion category
       - Valence and arousal levels
       - Mood meter visualization
       - Safety recommendations
    
    <div style="background-color:#e8f5e9; padding:15px; border-radius:8px; margin-top:20px;">
        <h4>💡 Technical Requirements:</h4>
        <ul style="margin-top:10px; margin-bottom:0; padding-left:20px;">
            <li>Video length: 3-10 seconds recommended</li>
            <li>Resolution: Minimum 480p</li>
            <li>Clear frontal face visibility</li>
            <li>Good lighting conditions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def about_page():
    st.title("📘 Research Project Details")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3498db, #2c3e50); color:white; padding:25px; border-radius:15px;">
        <h2 style="color:white;">Emotion Detection for Safety-Critical Systems</h2>
        <p style="font-size:1.1em;">Collaboration between Technical University of Munich and IAV GmbH</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎓 Academic Context
        - **Institution:** Technical University of Munich (TUM)
        - **Chair:** Hybrid Electronic Systems
        - **Department:** Electrical and Computer Engineering
        - **Developer:** Mohammed Ghanemi
        - **IAV Supervisor:** Pant Diva (IAV GmbH)
        
        ### 🔬 Research Focus
        This project investigates:
        - Audio-Visual emotion recognition
        - Multi-task learning approaches
        - Safety applications in automotive
        - Psychological dimensional modeling
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Model Architecture
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px;">
        - **Base Model:** TF-VideoMAE (Video Masked Autoencoder)
        - **Temporal Processing:** LSTM with 256 units
        - **Multi-Task Heads:**
          - Emotion: 5-class classification
          - Valence: Regression (tanh activation)
          - Arousal: Regression (tanh activation)
        - **Parameters:** ~290K trainable
        </div>
        
        ### 🏆 Performance Metrics
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-top:10px;">
        - **Valence MAE:** 0.3631
        - **Arousal MAE:** 0.2686  
        - **Emotion Accuracy:** 45.54%
        - **Training Samples:** 928
        - **Validation Samples:** 199
        - **Test Samples:** 202
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Training Details
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top:20px;">
        <div style="background-color:#e3f2fd; padding:15px; border-radius:8px; text-align:center;">
            <h4>Optimizer</h4>
            <p style="font-size:1.2em; font-weight:bold;">AdamW</p>
            <p>Weight decay 1e-4</p>
        </div>
        <div style="background-color:#e8f5e9; padding:15px; border-radius:8px; text-align:center;">
            <h4>Learning Rate</h4>
            <p style="font-size:1.2em; font-weight:bold;">1.5e-4</p>
            <p>Exponential decay</p>
        </div>
        <div style="background-color:#fff3e0; padding:15px; border-radius:8px; text-align:center;">
            <h4>Batch Size</h4>
            <p style="font-size:1.2em; font-weight:bold;">10</p>
            <p>Mixed precision</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📄 Technical Highlights
    <div style="background-color:#f5f5f5; padding:20px; border-radius:10px; border-left:4px solid #3498db;">
    <ul>
        <li><strong>TF-VideoMAE Backbone:</strong> Pretrained masked autoencoder for efficient video representation learning</li>
        <li><strong>Class Balancing:</strong> Weighted loss function for imbalanced emotion classes</li>
        <li><strong>Multi-Task Learning:</strong> Joint optimization of classification and regression tasks</li>
        <li><strong>Data Augmentation:</strong> Random temporal sampling, flipping, and brightness adjustment</li>
        <li><strong>Early Stopping:</strong> Patience of 40 epochs to prevent overfitting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def documentation_page():
    st.title("📖 Technical Documentation")
    
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px;">
        <h3>System Specifications and Implementation Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🧠 Model Architecture Deep Dive"):
        st.markdown("""
        ### TF-VideoMAE + LSTM Architecture
        ```python
        Input: (16, 224, 224, 3) video frames
        ↓
        TF-VideoMAE Feature Extraction (400-dim features)
        ↓
        Reshape to (16, 25) temporal sequence
        ↓
        LSTM(256 units) for temporal modeling
        ↓
        Multi-Task Heads:
          - Valence: Dense(128)→Dropout→Dense(1, tanh)
          - Arousal: Dense(128)→Dropout→Dense(1, tanh)  
          - Emotion: Dense(256)→Dropout→Dense(5, softmax)
        ```
        
        ### Key Hyperparameters
        - **Input Size:** 224×224 pixels
        - **Frame Count:** 16 frames
        - **Batch Size:** 10
        - **Learning Rate:** 1.5e-4 with exponential decay
        - **Loss Weights:** Emotion:3.0, Valence:1.0, Arousal:1.0
        - **Class Weights:** Computed for imbalanced data
        """)
    
    with st.expander("📊 Performance Analysis"):
        st.markdown("""
        ### Test Set Results
        | Metric          | Value   |
        |-----------------|---------|
        | Valence MAE     | 0.3631  |
        | Arousal MAE     | 0.2686  |
        | Emotion Accuracy| 45.54%  |
        
        ### Training Dynamics
        - **Best Epoch:** Early stopping at epoch ~40
        - **Training Loss:** 2.1244 (final epoch)
        - **Validation Loss:** 6.4802 (final epoch)
        - **Emotion Accuracy:** 76.51% (train), 40.20% (val)
        
        ### Sample Predictions
        ```text
        Video: happiness_01621.mp4
        True: happiness (V:0.70, A:0.13)
        Pred: sadness (V:-0.15, A:0.15)
        
        Video: anger_00556.mp4  
        True: anger (V:-0.34, A:0.67)
        Pred: anger (V:-0.13, A:0.35)
        ```
        """)
    
    with st.expander("⚙️ System Requirements"):
        st.markdown("""
        ### Hardware Requirements
        - **Training:**
          - GPU: NVIDIA with ≥8GB VRAM
          - RAM: ≥16GB
          - Storage: SSD recommended
        
        - **Inference:**
          - CPU: Modern Intel/AMD
          - RAM: ≥8GB
        
        ### Software Dependencies
        - Python 3.8+
        - TensorFlow 2.6+
        - DECORD for video processing
        - Streamlit for interface
        
        ### Dataset Specifications
        - **Source:** MAFW dataset + custom collection
        - **Classes:** anger, happiness, disgust, fear, sadness
        - **Split:**
          - Train: 928 samples
          - Val: 199 samples  
          - Test: 202 samples
        """)

def main():
    st.sidebar.title("Navigation")
    menu_options = {
        "Home": "🏠",
        "Video Analysis": "🎥", 
        "Mood Meter": "📊",
        "Documentation": "📖",
        "About": "📘"
    }
    
    selected = st.sidebar.radio("Go to:", list(menu_options.keys()),
                             format_func=lambda x: f"{menu_options[x]} {x}")
    
    if selected == "Home":
        home_page()
    elif selected == "Video Analysis":
        video_analysis_page()
    elif selected == "Mood Meter":
        mood_meter_page()
    elif selected == "Documentation":
        documentation_page()
    elif selected == "About":
        about_page()
    
    st.sidebar.markdown("---")

    
    st.sidebar.markdown("""
    <div style="text-align:center;">
        <p>Developed by<strong> Mohammed Ghanemi </strong></p>
        <p>Supervised by<strong> Pant Diva </strong></p>    
        <p>© 2025 Master's Thesis Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()