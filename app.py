# app.py
import streamlit as st
import cv2, os, time
import numpy as np
from PIL import Image
from gtts import gTTS
import io

st.set_page_config(page_title="Face Recognition + Voice", layout="centered")
st.title("👤 Face Recognition with Voice Output (Streamlit)")

# folders & model names
DATA_DIR = "dataset"
MODEL_FILE = "classifier.xml"
LABELS_FILE = "labels.txt"
os.makedirs(DATA_DIR, exist_ok=True)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

st.markdown("**How it works:** Upload images (ID + name) → Train → Upload test image → Hear result.")

# ----------------- Upload dataset image -----------------
st.header("1) Upload dataset image")
with st.form("upload_form"):
    uid = st.text_input("User ID (a number)", help="Type a number like 1, 2, 3")
    name = st.text_input("User name (e.g. Alice)")
    uploaded = st.file_uploader("Choose face image (jpg/png) — on mobile you can choose Camera", type=["jpg","jpeg","png"])
    save_btn = st.form_submit_button("Save image")
if save_btn:
    if not uid or not name or not uploaded:
        st.warning("Please provide ID, name and an image.")
    else:
        safe_name = name.strip().replace(" ", "_")
        fname = f"{uid}__{safe_name}__{int(time.time())}.jpg"
        path = os.path.join(DATA_DIR, fname)
        img = Image.open(uploaded).convert("RGB")
        img.save(path)
        st.success(f"Saved dataset image: {fname}")
        st.image(img, width=240)

# ----------------- Train model -----------------
st.header("2) Train recognizer")
if st.button("Train model"):
    image_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    if not image_paths:
        st.warning("No dataset images found. Upload images first.")
    else:
        with st.spinner("Training... this may take a few seconds"):
            # create LBPH recognizer
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            except Exception as e:
                st.error("OpenCV 'face' module not available. Make sure opencv-contrib-python is in requirements.")
                st.stop()

            faces = []
            ids = []
            labels = {}

            face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
            for p in image_paths:
                try:
                    gray = Image.open(p).convert("L")
                except:
                    continue
                img_np = np.array(gray, "uint8")
                parts = os.path.basename(p).split("__")
                if len(parts) < 2:
                    continue
                uid_parsed = int(parts[0])
                name_parsed = parts[1]
                labels[uid_parsed] = name_parsed

                rects = face_cascade.detectMultiScale(img_np, 1.3, 5)
                if len(rects) == 0:
                    continue
                x,y,w,h = rects[0]
                face = img_np[y:y+h, x:x+w]
                face = cv2.resize(face, (200,200))
                faces.append(face)
                ids.append(uid_parsed)

            if len(faces) == 0:
                st.warning("No faces detected in dataset images. Use clear frontal photos.")
            else:
                recognizer.train(faces, np.array(ids))
                recognizer.save(MODEL_FILE)
                # save labels
                with open(LABELS_FILE, "w") as f:
                    for k,v in labels.items():
                        f.write(f"{k}__{v}\n")
                st.success(f"Training complete. Trained on {len(faces)} faces.")

# ----------------- Recognize -----------------
st.header("3) Recognize (upload test image)")
test_file = st.file_uploader("Upload test image (jpg/png)", type=["jpg","jpeg","png"], key="test")
if st.button("Recognize"):
    if not os.path.exists(MODEL_FILE):
        st.warning("No trained model found. Train first.")
    elif not test_file:
        st.warning("Upload a test image to recognize.")
    else:
        # read image
        img = Image.open(test_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.info("No face detected in the test image.")
            result_text = "No face detected"
        else:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_FILE)
            # load labels
            labels_map = {}
            if os.path.exists(LABELS_FILE):
                with open(LABELS_FILE) as f:
                    for line in f:
                        k,v = line.strip().split("__")
                        labels_map[int(k)] = v

            x,y,w,h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200,200))
            id_pred, conf = recognizer.predict(face)
            if conf < 70:
                name = labels_map.get(id_pred, f"ID_{id_pred}")
                result_text = f"Recognized: {name} (conf {conf:.1f})"
            else:
                result_text = f"Unknown (conf {conf:.1f})"
            # draw rectangle
            cv2.rectangle(img_cv, (x,y),(x+w,y+h),(0,255,0),2)

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption=result_text, use_column_width=True)
        st.write(result_text)

        # TTS using gTTS (writes to memory, plays on the page)
        try:
            tts = gTTS(result_text, lang="en")
            mp3_bytes = io.BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
            st.audio(mp3_bytes.read(), format="audio/mp3")
        except Exception as e:
            st.error("TTS failed: " + str(e))
