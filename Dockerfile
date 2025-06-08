# Gunakan Python 3.10 untuk kompatibilitas dengan TensorFlow 2.19.0
FROM python:3.10-slim

# Disable bytecode dan enable output langsung
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Buat folder kerja
WORKDIR /app

# Salin dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi
COPY . .

# Expose port Flask
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]
