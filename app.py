# app.py
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

MODEL_PATH = "best.pt"  # ใส่ชื่อไฟล์โมเดลของคุณ (22MB)

# โหลดโมเดล (โหลดครั้งเดียว)
model = YOLO(MODEL_PATH)

VALUE_MAP = {"1baht": 1, "5baht": 5, "10baht": 10}

def predict(img):
    """
    รับภาพ (PIL / ndarray) -> คืนค่า (ผลลัพธ์ภาพที่มี bbox เป็น numpy array, ข้อความสรุป)
    """
    if img is None:
        return None, "❌ ไม่มีภาพส่งมา"
    # ถ้าเป็น PIL -> แปลงเป็น numpy RGB
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = img

    # ปรับขนาดเพื่อความเร็ว (YOLO จะปรับให้อยู่แล้ว แต่ใช้ขนาดพอสมควร)
    # ไม่เปลี่ยน aspect ratio เพื่อให้ bbox ตรง
    h, w = img_np.shape[:2]
    scale = 640 / max(h, w) if max(h, w) > 640 else 1.0
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_np, (new_w, new_h))
    else:
        img_resized = img_np

    # ทำการทำนาย
    results = model(img_resized, conf=0.45, save=False)
    res = results[0]

    # วาด bbox (ultralytics result.plot() ให้ BGR)
    plotted = res.plot()  # BGR numpy
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # นับเหรียญ
    coin_count = {"1baht": 0, "5baht": 0, "10baht": 0}
    total_value = 0
    try:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in VALUE_MAP:
                coin_count[class_name] += 1
                total_value += VALUE_MAP[class_name]
    except Exception:
        # ถ้าไม่มี boxes ก็ข้าม
        pass

    # สร้างข้อความสรุป
    lines = []
    for k, v in coin_count.items():
        lines.append(f"{k}: {v} เหรียญ → {v * VALUE_MAP[k]} บาท")
    if sum(coin_count.values()) == 0:
        summary = "⚠️ ไม่พบเหรียญในภาพ"
    else:
        summary = "\n".join(lines) + f"\n\n💰 ยอดรวม: {total_value} บาท"

    return plotted_rgb, summary


# สร้าง UI ด้วย Gradio
title = "🪙 Thai Coin Detector (YOLOv8)"
description = "อัปโหลดรูปหรือใช้กล้องเพื่อให้ระบบตรวจจับและนับเหรียญไทย (1, 5, 10 บาท)"

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="อัปโหลดภาพ / ถ่ายจากเว็บแคม")
            btn = gr.Button("วิเคราะห์")
        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="ผลลัพธ์ (bounding boxes)")
            output_text = gr.Textbox(label="สรุปผล", interactive=False)

    btn.click(predict, inputs=input_img, outputs=[output_img, output_text])

# ถ้า run บนเครื่อง: เปิดพอร์ทให้ทดสอบด้วย share=True (แต่บน Spaces จะรันอัตโนมัติ)
if __name__ == "__main__":
    demo.launch()
