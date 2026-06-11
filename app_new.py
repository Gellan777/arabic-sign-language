"""
=============================================================
Arabic Sign Language Recognition - Streamlit App (Enhanced)
تطبيق التعرف على لغة الإشارة العربية - نسخة محسّنة
=============================================================
تشغيل:
    streamlit run app_new.py

التحسينات:
    - CLAHE لتحسين الإضاءة
    - Test Time Augmentation (TTA) لدقة أفضل
    - دعم أشخاص مختلفين
=============================================================
"""

import os
import json
import pickle
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from tensorflow.keras.models import load_model

# ============================================================
# CONFIG
# ============================================================

ASSETS_DIR  = os.path.join(os.path.dirname(__file__), 'streamlit_assets')
IMG_SIZE    = 128
MAX_FRAMES  = 20
NUM_CLASSES = 50

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title = "لغة الإشارة العربية",
    page_icon  = "🤟",
    layout     = "wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    .prediction-word {
        font-size: 3.5rem;
        font-weight: 900;
        color: #00d4ff;
        direction: rtl;
        unicode-bidi: plaintext;
    }
    .prediction-conf { font-size: 1.2rem; color: #a0aec0; margin-top: 0.5rem; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .section-title {
        color: #00d4ff; font-size: 1.4rem; font-weight: 700;
        border-bottom: 2px solid #00d4ff33;
        padding-bottom: 0.5rem; margin-bottom: 1rem; direction: rtl;
    }
    .stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7b2ff7); }
    .top-k-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.4rem 0.8rem; border-radius: 8px;
        background: rgba(0, 212, 255, 0.08); margin-bottom: 0.4rem; direction: rtl;
    }
    .word-card {
        text-align: center; padding: 0.5rem;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        border: 1px solid rgba(0,212,255,0.2);
        margin-bottom: 0.5rem;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_resources():
    model = load_model(os.path.join(ASSETS_DIR, 'model.h5'))
    with open(os.path.join(ASSETS_DIR, 'reverse_map.pkl'), 'rb') as f:
        reverse_map = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, 'label_map.pkl'), 'rb') as f:
        label_map = pickle.load(f)
    return model, reverse_map, label_map

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_frame(frame, brightness=1.0, flip=False):
    """معالجة frame واحد مع CLAHE وتعديل الإضاءة"""
    # CLAHE لتحسين التباين
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.merge((l, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    # تعديل الإضاءة
    if brightness != 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)

    # عكس أفقي
    if flip:
        frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    return frame


def load_video_frames(video_path: str, brightness=1.0, flip=False) -> np.ndarray:
    """تحميل الفيديو مع preprocessing"""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, brightness=brightness, flip=flip))
    cap.release()

    if len(frames) == 0:
        raise ValueError("الفيديو فاضي أو تالف")
    if len(frames) > MAX_FRAMES:
        idx    = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in idx]
    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    return np.expand_dims(np.array(frames, dtype=np.float32), axis=0)


def load_video_for_prediction(video_path: str) -> np.ndarray:
    """للتوافق مع الكود القديم"""
    return load_video_frames(video_path)


def text_to_speech(text: str) -> bytes:
    tts = gTTS(text=str(text), lang='ar', slow=False)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        tts.save(f.name)
        with open(f.name, 'rb') as audio_file:
            return audio_file.read()


def predict_with_tta(model, video_path: str, reverse_map, top_k=5):
    """
    Test Time Augmentation:
    بيبعت الفيديو للموديل 3 مرات بتعديلات مختلفة
    وياخد متوسط النتايج عشان يكون أدق
    """
    augmentations = [
        {"brightness": 1.0, "flip": False},   # فيديو عادي
        {"brightness": 1.3, "flip": False},   # إضاءة أكتر
        {"brightness": 0.8, "flip": False},   # إضاءة أقل
    ]

    all_probs = []
    for aug in augmentations:
        try:
            video_array = load_video_frames(video_path, **aug)
            probs       = model.predict(video_array, verbose=0)[0]
            all_probs.append(probs)
        except Exception:
            continue

    if not all_probs:
        raise ValueError("فشل التحليل")

    # متوسط النتايج
    avg_probs   = np.mean(all_probs, axis=0)
    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    top_results = [
        {
            'label': int(idx),
            'word':  str(reverse_map.get(int(idx), f'كلاس {idx}')),
            'prob':  float(avg_probs[idx])
        }
        for idx in top_indices
    ]
    return top_results[0]['word'], top_results[0]['prob'], top_results


def predict_video(model, video_array, reverse_map, top_k=5):
    probs       = model.predict(video_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [
        {'label': int(idx), 'word': str(reverse_map.get(int(idx), f'كلاس {idx}')), 'prob': float(probs[idx])}
        for idx in top_indices
    ]
    return top_results[0]['word'], top_results[0]['prob'], top_results


def _draw_landmarks_new_api(input_path: str, output_path: str) -> bool:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request, tempfile

    def _dl(url, name):
        dst = os.path.join(tempfile.gettempdir(), name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(url, dst)
        return dst

    pose_model = _dl(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "pose_landmarker.task"
    )
    hand_model = _dl(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "hand_landmarker.task"
    )

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    POSE_CONNECTIONS = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
    ]
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def _draw_pose(frame, lms, h, w):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for s, e in POSE_CONNECTIONS:
            if lms[s].visibility > 0.5 and lms[e].visibility > 0.5:
                cv2.line(frame, pts[s], pts[e], (0, 255, 255), 2)
        for s, e in POSE_CONNECTIONS:
            if lms[s].visibility > 0.5:
                cv2.circle(frame, pts[s], 6, (255, 255, 0), -1)
            if lms[e].visibility > 0.5:
                cv2.circle(frame, pts[e], 6, (255, 255, 0), -1)

    def _draw_hand(frame, lms, h, w, color):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for s, e in HAND_CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)

    cap = cv2.VideoCapture(input_path)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    with mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_det, \
         mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms  = int(frame_idx * 1000 / fps)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_res = pose_det.detect_for_video(mp_img, ts_ms)
            hand_res = hand_det.detect_for_video(mp_img, ts_ms)

            if pose_res.pose_landmarks:
                _draw_pose(frame, pose_res.pose_landmarks[0], fh, fw)
            for hand_lms in hand_res.hand_landmarks:
                _draw_hand(frame, hand_lms, fh, fw, (0, 200, 255))

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _draw_landmarks_legacy_api(input_path: str, output_path: str) -> bool:
    import mediapipe as mp
    mp_holistic   = mp.solutions.holistic
    mp_drawing    = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles

    POSE_CONNECTIONS = [
        (mp_holistic.PoseLandmark.LEFT_SHOULDER,  mp_holistic.PoseLandmark.LEFT_ELBOW),
        (mp_holistic.PoseLandmark.LEFT_ELBOW,     mp_holistic.PoseLandmark.LEFT_WRIST),
        (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW),
        (mp_holistic.PoseLandmark.RIGHT_ELBOW,    mp_holistic.PoseLandmark.RIGHT_WRIST),
        (mp_holistic.PoseLandmark.LEFT_SHOULDER,  mp_holistic.PoseLandmark.RIGHT_SHOULDER),
    ]

    cap = cv2.VideoCapture(input_path)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    with mp_holistic.Holistic(min_detection_confidence=0.3,
                               min_tracking_confidence=0.3) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            if result.pose_landmarks:
                for s_lm, e_lm in POSE_CONNECTIONS:
                    s = result.pose_landmarks.landmark[s_lm]
                    e = result.pose_landmarks.landmark[e_lm]
                    if s.visibility > 0.5 and e.visibility > 0.5:
                        sx, sy = int(s.x * fw), int(s.y * fh)
                        ex, ey = int(e.x * fw), int(e.y * fh)
                        cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 255), 2)
                        cv2.circle(frame, (sx, sy), 6, (255, 255, 0), -1)
                        cv2.circle(frame, (ex, ey), 6, (255, 255, 0), -1)

            if result.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style())
            if result.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_style.get_default_hand_landmarks_style())

            out.write(frame)

    cap.release()
    out.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def add_landmarks_direct(input_path: str, output_path: str) -> bool:
    import mediapipe as mp
    version     = tuple(int(x) for x in getattr(mp, "__version__", "0.0.0").split(".")[:2])
    use_new     = version >= (0, 10) or not hasattr(mp, "solutions")

    try:
        if use_new:
            return _draw_landmarks_new_api(input_path, output_path)
        else:
            return _draw_landmarks_legacy_api(input_path, output_path)
    except Exception as e1:
        try:
            alt_out = output_path.replace(".mp4", "_alt.mp4")
            if use_new:
                ok = _draw_landmarks_legacy_api(input_path, alt_out)
            else:
                ok = _draw_landmarks_new_api(input_path, alt_out)
            if ok:
                import shutil
                shutil.move(alt_out, output_path)
            return ok
        except Exception as e2:
            st.warning(f"MediaPipe error: {e1} | {e2}")
            return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🤟 إعدادات التطبيق")
    st.markdown("---")
    play_audio = st.toggle("قراءة الكلمة بصوت", value=True)
    top_k      = st.slider("عدد أفضل التوقعات", min_value=3, max_value=10, value=5)
    use_tta    = st.toggle("تحسين التعرف (TTA)", value=True,
                           help="بيحلل الفيديو 3 مرات بإضاءات مختلفة لدقة أفضل")
    st.markdown("---")
    st.markdown("### 📊 معلومات الموديل")
    st.markdown("""
    - **Architecture:** MobileNetV2 + BiLSTM
    - **Classes:** 50 كلمة عربية
    - **Input:** 20 frame × 128×128
    - **Dataset:** KArSL-502
    - **Enhancement:** CLAHE + TTA
    """)
    st.markdown("---")
    page = st.radio(
        "الصفحة",
        ["🎯 التنبؤ", "📚 مكتبة الكلمات", "📈 نتائج التدريب"],
        index=0
    )

# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("جاري تحميل الموديل..."):
    try:
        model, reverse_map, label_map = load_resources()
        model_loaded = True
    except Exception as e:
        st.error(f"خطأ في تحميل الموديل: {e}")
        model_loaded = False
        reverse_map  = {}
        label_map    = {}

# ============================================================
# PAGE 1: PREDICTION
# ============================================================

if page == "🎯 التنبؤ":

    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0;'>
        <h1 style='color:#00d4ff; font-size:2.5rem; font-weight:900;'>🤟 التعرف على لغة الإشارة العربية</h1>
        <p style='color:#a0aec0; font-size:1.1rem;'>ارفع فيديو إشارة والموديل هيتعرف على الكلمة</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📁 ارفع فيديو الإشارة (mp4, avi, mov)",
        type=['mp4', 'avi', 'mov', 'mkv']
    )

    if uploaded_file and model_loaded:

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.markdown('<div class="section-title">📹 الفيديو الأصلي</div>', unsafe_allow_html=True)
            st.video(tmp_path)

        with col2:
            st.markdown('<div class="section-title">🧠 التنبؤ</div>', unsafe_allow_html=True)

            with st.spinner("جاري التحليل..."):
                progress = st.progress(0, text="تحميل الفيديو...")
                time.sleep(0.3)

                try:
                    if use_tta:
                        progress.progress(20, text="TTA — تحليل الفيديو 3 مرات...")
                        predicted_word, confidence, top_results = predict_with_tta(
                            model, tmp_path, reverse_map, top_k=top_k
                        )
                    else:
                        video_array = load_video_for_prediction(tmp_path)
                        progress.progress(40, text="التنبؤ...")
                        predicted_word, confidence, top_results = predict_video(
                            model, video_array, reverse_map, top_k=top_k
                        )

                    progress.progress(100, text="✅ تم!")

                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style='font-size:1rem; color:#a0aec0; margin-bottom:0.5rem;'>الكلمة المتوقعة</div>
                        <div class="prediction-word">{predicted_word}</div>
                        <div class="prediction-conf">نسبة الثقة: {confidence*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if play_audio:
                        st.markdown("---")
                        st.markdown("**🔊 استمع للكلمة:**")
                        try:
                            audio_bytes = text_to_speech(predicted_word)
                            st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                        except Exception as e:
                            st.warning(f"تعذر تشغيل الصوت: {e}")

                    st.markdown("---")
                    st.markdown("**📊 أفضل التوقعات:**")
                    for i, res in enumerate(top_results):
                        emoji     = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                        bar_color = "#00d4ff" if i == 0 else "#7b2ff7"
                        st.markdown(f"""
                        <div class="top-k-item">
                            <span>{emoji} {res['word']}</span>
                            <span style='color:{bar_color}; font-weight:700;'>{res['prob']*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(res['prob']))

                except Exception as e:
                    st.error(f"خطأ في التحليل: {e}")
                    progress.empty()

        st.markdown("---")
        st.markdown('<div class="section-title">🦴 فيديو مع MediaPipe Landmarks</div>',
                    unsafe_allow_html=True)

        landmark_out = tmp_path.replace('.mp4', '_landmarks.mp4')

        with st.spinner("جاري رسم الـ Landmarks..."):
            success = add_landmarks_direct(tmp_path, landmark_out)

        if success:
            landmark_web = tmp_path.replace('.mp4', '_landmarks_web.mp4')
            os.system(f'ffmpeg -i "{landmark_out}" -vcodec libx264 -pix_fmt yuv420p -y "{landmark_web}" -loglevel quiet')
            final_video = landmark_web if os.path.exists(landmark_web) else landmark_out
            st.video(final_video)
            st.caption(f"فيديو Landmarks — كلمة: {predicted_word}")
        else:
            st.warning("تعذّر رسم الـ Landmarks.")


# ============================================================
# PAGE 2: WORD LIBRARY
# ============================================================

elif page == "📚 مكتبة الكلمات":

    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0;'>
        <h1 style='color:#00d4ff;'>📚 مكتبة الكلمات</h1>
        <p style='color:#a0aec0;'>استعرض الـ 50 كلمة مع فيديوهات الـ Landmarks</p>
    </div>
    """, unsafe_allow_html=True)

    landmark_dir = os.path.join(ASSETS_DIR, 'landmark_videos')

    if not os.path.exists(landmark_dir):
        st.error(f"فولدر landmark_videos مش موجود في: {landmark_dir}")
        st.stop()

    search_term = st.text_input("🔍 ابحث عن كلمة", placeholder="اكتب جزء من الكلمة...")
    all_labels  = sorted(reverse_map.keys())
    if search_term:
        all_labels = [lbl for lbl in all_labels if search_term in str(reverse_map.get(lbl, ''))]

    st.markdown(f"**{len(all_labels)} كلمة**")
    st.markdown("---")

    cols = st.columns(4)
    for i, label_id in enumerate(all_labels):
        word     = str(reverse_map.get(label_id, f'Label {label_id}'))
        vid_path = os.path.join(landmark_dir, f'label_{label_id:02d}.mp4')

        with cols[i % 4]:
            st.markdown(f"""
            <div class="word-card">
                <div style='color:#00d4ff; font-size:1.1rem; font-weight:700;'>{word}</div>
                <div style='color:#718096; font-size:0.8rem;'>Label {label_id}</div>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists(vid_path):
                st.video(vid_path)
            else:
                st.caption(f"⚠️ {os.path.basename(vid_path)} غير موجود")

            if st.button(f"🔊", key=f"audio_{label_id}", help=f"استمع لـ {word}"):
                try:
                    audio = text_to_speech(word)
                    st.audio(audio, format='audio/mp3', autoplay=True)
                except Exception as e:
                    st.warning(f"تعذر: {e}")

            st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PAGE 3: TRAINING RESULTS
# ============================================================

elif page == "📈 نتائج التدريب":

    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0;'>
        <h1 style='color:#00d4ff;'>📈 نتائج التدريب</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🎯 Test Accuracy", "92.75%", "#00d4ff"),
        ("📉 Test Loss",     "0.193",  "#7b2ff7"),
        ("📦 Classes",       "50",     "#48bb78"),
        ("🎬 Train Videos",  "~6,291", "#ed8936"),
    ]
    for col, (title, value, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='color:#a0aec0; font-size:0.9rem;'>{title}</div>
                <div style='color:{color}; font-size:2rem; font-weight:900;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    curves_path = os.path.join(ASSETS_DIR, 'training_curves.png')
    if os.path.exists(curves_path):
        st.markdown("### منحنيات التدريب")
        st.image(curves_path, use_container_width=True)
    else:
        st.info("صور التدريب مش متاحة")

    st.markdown("---")

    cm_path = os.path.join(ASSETS_DIR, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.markdown("### Confusion Matrix")
        st.image(cm_path, use_container_width=True)
    else:
        st.info("Confusion Matrix مش متاح")

    st.markdown("---")

    st.markdown("### تاريخ التدريب")
    import pandas as pd
    history_data = {
        "Phase":        ["Phase 1"] * 5 + ["Phase 2"] * 6,
        "Epoch":        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6],
        "Val Accuracy": ["67.25%", "83.08%", "86.42%", "90.00%", "90.67%",
                         "90.92%", "92.33%", "92.08%", "92.75%", "92.42%", "93.25%"],
    }
    st.dataframe(pd.DataFrame(history_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ معمارية الموديل")
    st.code("""
Input: (batch, 20 frames, 128, 128, 3)
    ↓
TimeDistributed(MobileNetV2)   → CNN features per frame
    ↓
TimeDistributed(GlobalAvgPool) → (batch, 20, 1280)
    ↓
TimeDistributed(BatchNorm)
    ↓
Bidirectional(LSTM, 128)       → temporal context
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.5)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(50, softmax)             → 50 كلمة
    """, language='text')
