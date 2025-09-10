import qrcode
from io import BytesIO
import streamlit as st
from PIL import Image

# ✅ 正確的應用網址
app_url = "https://research-msgfuuqy6pzr3hqgghv5v8.streamlit.app/"

# 生成二维码
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(app_url)
qr.make(fit=True)

# 轉成圖片
img = qr.make_image(fill_color="black", back_color="white")
buffer = BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)

# 在 Streamlit 页面中展示二维码
st.title("Welcome to the AKD/AKI Prediction App!")
st.write("Scan the QR code below to access the app:")
st.image(Image.open(buffer), caption="Scan to access")
