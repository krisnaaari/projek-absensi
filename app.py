from flask import Flask, flash, render_template, request, redirect, url_for, session, send_from_directory
#import mysql.connector
import pymysql # type: ignore
import os
from datetime import date, datetime
import base64
import cv2
import numpy as np
import pymysql.cursors # type: ignore
import math
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = 'your_secret_key'

def load_image(image_path):
    """
    Memuat gambar dari path yang diberikan.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gambar tidak dapat dimuat dari path: {image_path}")
    return image

def preprocess_image(image):
    """
    Praproses gambar untuk deteksi wajah.
    """
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram untuk meningkatkan kontras
    gray = cv2.equalizeHist(gray)
    
    return gray

def detect_faces(image, face_cascade):
    """
    Mendeteksi wajah dalam gambar menggunakan Haar Cascade.
    """
    faces = face_cascade.detectMultiScale(
        image, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def detect_faces_with_lbp(image):
    """
    Mendeteksi wajah menggunakan LBP cascade yang lebih akurat daripada Haar.
    """
    lbp_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface_improved.xml')
    
    # Jika file XML tidak ditemukan, gunakan yang default
    if lbp_face_cascade.empty():
        lbp_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface.xml')
    
    # Jika masih empty, gunakan Haar
    if lbp_face_cascade.empty():
        lbp_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = lbp_face_cascade.detectMultiScale(
        image, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def extract_face(image, face_coords, margin=0.2):
    """
    Mengekstrak wilayah wajah dari gambar dengan margin tambahan.
    """
    x, y, w, h = face_coords
    
    # Tambahkan margin ke wajah
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    # Pastikan koordinat tidak keluar dari batas gambar
    start_x = max(0, x - margin_x)
    start_y = max(0, y - margin_y)
    end_x = min(image.shape[1], x + w + margin_x)
    end_y = min(image.shape[0], y + h + margin_y)
    
    face = image[start_y:end_y, start_x:end_x]
    return face

def extract_facial_landmarks(face_gray):
    """
    Ekstrak facial landmarks menggunakan SIFT.
    SIFT adalah algoritma yang dapat mendeteksi dan deskripsi fitur lokal dalam gambar.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(face_gray, None)
    return keypoints, descriptors

def get_face_features(face_img):
    """
    Ekstrak berbagai fitur dari gambar wajah.
    """
    # Resize wajah ke ukuran standar
    face_resized = cv2.resize(face_img, (100, 100))
    
    # 1. SIFT features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(face_resized, None)
    
    # Jika tidak ada keypoints yang terdeteksi
    if descriptors is None:
        descriptors = np.zeros((1, 128), dtype=np.float32)
    
    # 2. HOG features
    win_size = (100, 100)
    block_size = (20, 20)
    block_stride = (10, 10)
    cell_size = (10, 10)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(face_resized)
    
    # 3. LBP features (Local Binary Patterns)
    lbp_features = get_lbp_features(face_resized)
    
    return {
        'sift': descriptors,
        'hog': hog_features,
        'lbp': lbp_features
    }

def get_lbp_features(image):
    """
    Compute Local Binary Pattern features.
    """
    radius = 1
    n_points = 8 * radius
    
    lbp = np.zeros_like(image)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            center = image[i, j]
            pattern = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = i + radius * np.cos(angle)
                y = j + radius * np.sin(angle)
                x = int(round(x))
                y = int(round(y))
                if image[x, y] >= center:
                    pattern |= (1 << k)
            lbp[i, j] = pattern
    
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**n_points + 1), density=True)
    return hist

def match_descriptors(desc1, desc2):
    """
    Match SIFT descriptors between two faces.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return 0
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        match_ratio = len(good_matches) / max(len(desc1), len(desc2))
        return match_ratio
    except:
        # Jika FLANN gagal, gunakan brute force matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        match_ratio = len(good_matches) / max(len(desc1), len(desc2))
        return match_ratio

def cosine_similarity(v1, v2):
    """
    Menghitung cosine similarity antara dua vektor.
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot / (norm1 * norm2)

def compare_faces(face1, face2, threshold=0.55):
    """
    Membandingkan dua wajah menggunakan beberapa metode.
    """
    # Ekstrak fitur dari kedua wajah
    features1 = get_face_features(face1)
    features2 = get_face_features(face2)
    
    # 1. SIFT similarity
    #sift_similarity = match_descriptors(features1['sift'], features2['sift'])
    
    # 2. HOG similarity
    hog_similarity = cosine_similarity(features1['hog'].flatten(), features2['hog'].flatten())
    
    # 3. LBP similarity
    lbp_similarity = cosine_similarity(features1['lbp'], features2['lbp'])
    
    # 4. Histogram similarity (metode awal)
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_similarity = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    # Gabungkan semua skor dengan bobot
    final_similarity = (0 * 0.4 + 
                       hog_similarity * 0.25 + 
                       lbp_similarity * 0.25 + 
                       hist_similarity * 0.50)
    
    #print(f"SIFT Similarity: {sift_similarity:.4f}")
    print(f"HOG Similarity: {hog_similarity:.4f}")
    print(f"LBP Similarity: {lbp_similarity:.4f}")
    print(f"Histogram Similarity: {hist_similarity:.4f}")
    print(f"Final Similarity: {final_similarity:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Is Same Person: {final_similarity >= threshold}")
    
    return final_similarity >= threshold, final_similarity

def face_comparison(image1_path, image2_path, threshold=0.51):
    """
    Fungsi utama untuk membandingkan wajah dalam dua gambar.
    """
    # Muat gambar
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Praproses gambar
    gray1 = preprocess_image(image1)
    gray2 = preprocess_image(image2)

    # Deteksi wajah dengan cascade yang lebih baik
    faces1 = detect_faces_with_lbp(gray1)
    faces2 = detect_faces_with_lbp(gray2)

    if len(faces1) == 0:
        print("Tidak ada wajah yang terdeteksi di gambar pertama.")
        return False, 0
    elif len(faces2) == 0:
        print("Tidak ada wajah yang terdeteksi di gambar kedua.")
        return False, 0

    # Ekstrak wajah pertama dari gambar pertama dengan margin tambahan
    face1 = extract_face(gray1, faces1[0], margin=0.2)

    # Ekstrak wajah pertama dari gambar kedua dengan margin tambahan
    face2 = extract_face(gray2, faces2[0], margin=0.2)
    
    # Resize kedua wajah ke ukuran yang sama
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))

    # Bandingkan wajah
    is_same, similarity = compare_faces(face1, face2, threshold)
    
    return is_same, similarity

def cek_radius(id_proyek, latitude, longitude, toleransi_radius_km=2.0):
    
    def haversine(lat1, lon1, lat2, lon2):
        # Radius bumi dalam kilometer
        R = 6371.0

        # Konversi koordinat dari derajat ke radian
        lat1 = math.radians(float(lat1))
        lon1 = math.radians(float(lon1))
        lat2 = math.radians(float(lat2))
        lon2 = math.radians(float(lon2))

        # Perbedaan koordinat
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Rumus Haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Jarak dalam kilometer
        distance = R * c
        return distance

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    hasil = True
    cursor.execute('''
                   SELECT latitude_proyek, longitude_proyek 
                   FROM lokasi_proyek 
                   WHERE id_proyek = %s 
                   ''', (id_proyek,))
    hasil_query = cursor.fetchone()
    if hasil_query:
        latitude_proyek = hasil_query.get('latitude_proyek')
        longitude_proyek = hasil_query.get('longitude_proyek')

        # Pastikan nilai latitude dan longitude adalah float
        latitude_proyek = float(latitude_proyek)
        longitude_proyek = float(longitude_proyek)
        latitude = float(latitude)
        longitude = float(longitude)
        
        # Hitung jarak menggunakan rumus Haversine
        jarak = haversine(latitude_proyek, longitude_proyek, latitude, longitude)
        print(f"latitude, longitude: {latitude}, {longitude}")
        print(f"jarak dari lokasi: {jarak}")
        # Cek apakah jarak dalam toleransi radius
        if jarak <= toleransi_radius_km:
            return hasil
        else:
            hasil = False
            return hasil
    else:
        hasil = False
        return hasil
    
def handle_attendance(jenis_absensi, id_karyawan, image_data, longitude, latitude, id_proyek):
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    try:
        waktu_sekarang = datetime.now()

        if jenis_absensi == 'pulang':
            # Cek apakah ada absensi datang yang belum memiliki absensi pulang
            cursor.execute('''
                SELECT id
                FROM absensi_datang1
                WHERE id_karyawan = %s
                AND id NOT IN (
                    SELECT absensi_datang_id
                    FROM absensi_pulang1
                )
                ORDER BY waktu DESC
                LIMIT 1
            ''', (id_karyawan,))
            absensi_datang = cursor.fetchone()

            if not absensi_datang:
                flash("Anda belum melakukan absensi datang atau sudah melakukan absensi pulang.", "warning")
                return redirect(url_for('attendance'))

        # Validasi foto dan lokasi
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        temp_image_path = os.path.join('static', 'images', 'data_wajah', f"temp_{id_karyawan}.png")
        with open(temp_image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        reference_image_data = os.path.join('static', 'images', 'data_wajah', f"{id_karyawan}.png")
        is_same, similarity = face_comparison(temp_image_path, reference_image_data, threshold=0.4)
        os.remove(temp_image_path)

        if not is_same:
            flash("Foto selfie tidak cocok dengan foto referensi. Silakan ambil foto ulang.", "danger")
            return redirect(url_for("absensi", jenis_absensi=jenis_absensi))
        
        if not cek_radius(id_proyek, latitude, longitude):
            flash("Anda melakukan absensi di luar dari toleransi radius proyek (2km)", "warning")
            return redirect(url_for("absensi", jenis_absensi=jenis_absensi))
        
        # Simpan absensi
        if jenis_absensi == 'datang':
            image_filename = f"{id_karyawan}_{waktu_sekarang.strftime('%Y%m%d_%H%M%S')}.jpeg"
            image_path = os.path.join('static', 'images', image_filename)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            cursor.execute('''
                INSERT INTO absensi_datang1 (id_karyawan, waktu, latitude, longitude, photo, id_proyek)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (id_karyawan, waktu_sekarang, latitude, longitude, image_filename, id_proyek))
        else:
            cursor.execute('''
                INSERT INTO absensi_pulang1 (absensi_datang_id, id_karyawan, waktu, latitude, longitude, id_proyek)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (absensi_datang['id'], id_karyawan, waktu_sekarang, latitude, longitude, id_proyek))
            
        conn.commit()
        flash("Absensi berhasil disimpan.", "success")
        return redirect(url_for('attendance'))
    
    except Exception as e:
        print(f"Error saving time: {e}")
        flash("Terjadi kesalahan saat menyimpan absensi.", "danger")
        return redirect(url_for("absensi", jenis_absensi=jenis_absensi))
    
    finally:
        cursor.close()
        conn.close()
    
def get_log_absensi(id_karyawan):
    # Fetch user details
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    # Ambil data karyawan
    cursor.execute('SELECT ID, nama FROM karyawan WHERE id = %s', (id_karyawan,))
    user = cursor.fetchone()

    # Ambil log absensi
    cursor.execute('''
        SELECT 
            k.nama AS nama_karyawan,
            lp.nama_proyek,
            d.waktu AS waktu_datang,
            d.latitude AS latitude_datang,
            d.longitude AS longitude_datang,
            d.photo AS foto,
            p.waktu AS waktu_pulang,
            p.latitude AS latitude_pulang,
            p.longitude AS longitude_pulang
        FROM absensi_datang1 d
        LEFT JOIN absensi_pulang1 p ON d.id = p.absensi_datang_id
        INNER JOIN karyawan k ON d.id_karyawan = k.id
        INNER JOIN lokasi_proyek lp ON d.id_proyek = lp.id_proyek
        WHERE d.id_karyawan = %s
        ORDER BY d.waktu DESC
    ''', (id_karyawan,))
    
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    return user, logs

def get_db_connection():
    conn = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'), 
        database=os.getenv('DB_NAME')
    )
    return conn

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        query = 'SELECT * FROM user WHERE username = %s AND password = %s'
        conn = get_db_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(query, (username, password))
                user = cursor.fetchone()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

        if user:
            session['id_karyawan'] = user['id_karyawan']
            if username == 'admin':  # Assuming 'admin' is the admin username
                return redirect(url_for('admin_dashboard'))
            else:
                flash("Login Berhasil", "success")
                return redirect(url_for('attendance'))
        else:
            if username == '' and password == '':
                flash('Username dan Password harus di isi', 'warning')
            elif username == '':
                flash("Username harus diisi", "warning")
            elif password == '':
                flash("Password harus diisi!", "warning")
            else:
                flash("Username atau Password salah!", "danger")  # Flash message
            
    return render_template('login.html')

@app.route('/attendance')
def attendance():
    if 'id_karyawan' not in session or session['id_karyawan'] == 0:
        return redirect(url_for('login'))

    id_karyawan = session['id_karyawan']

    # Fetch user details
    user, logs = get_log_absensi(id_karyawan)

    return render_template('attendance2.html', user=user, logs=logs)

@app.route('/attendance/absensi-<jenis_absensi>', methods=["GET", "POST"])
def absensi(jenis_absensi):
    if 'id_karyawan' not in session or session['id_karyawan'] == 0:
        return redirect(url_for('login'))
    
    id_karyawan = session["id_karyawan"]
    print(f"jenis absensi: {jenis_absensi}")

    if(jenis_absensi == 'pulang'):
        # Check if user has already done the attendance
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor) 
        cursor.execute('''
                SELECT id
                FROM absensi_datang1
                WHERE id_karyawan = %s
                AND id NOT IN (
                    SELECT absensi_datang_id
                    FROM absensi_pulang1
                )
                ORDER BY waktu DESC
                LIMIT 1
            ''', (id_karyawan,))
        absensi_datang = cursor.fetchone()

        if not absensi_datang:
            flash("Anda belum melakukan absensi datang atau sudah melakukan absensi pulang.", "warning")
            return redirect(url_for('attendance'))
    
    if request.method == 'POST':
        jenis_absensi = jenis_absensi
        image_data = request.form['image']
        longitude = request.form['longitude']
        latitude = request.form['latitude']
        id_proyek = request.form['project']
        return handle_attendance(jenis_absensi, id_karyawan, image_data, longitude, latitude, id_proyek)

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute('SELECT * FROM lokasi_proyek ORDER BY id_proyek DESC')    
    lokasi_proyek = cursor.fetchall()

    user, logs = get_log_absensi(id_karyawan)

    return render_template('attendance.html', user=user, logs=logs, projects=lokasi_proyek)


@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    if session['id_karyawan'] != 0:
        return redirect(url_for('login'))
    
    # Ambil parameter tanggal dari query string
    tanggal1 = request.args.get('tanggal1')
    tanggal2 = request.args.get('tanggal2')
    
    logs = []

    if tanggal1 and tanggal2:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute('''
            SELECT 
                k.nama AS nama_karyawan,
                lp.nama_proyek,
                d.waktu AS waktu_datang,
                d.latitude AS latitude_datang,
                d.longitude AS longitude_datang,
                d.photo AS foto,
                p.waktu AS waktu_pulang,
                p.latitude AS latitude_pulang,
                p.longitude AS longitude_pulang
            FROM absensi_datang1 d
            LEFT JOIN absensi_pulang1 p ON d.id = p.absensi_datang_id
            INNER JOIN karyawan k ON d.id_karyawan = k.id
            INNER JOIN lokasi_proyek lp ON d.id_proyek = lp.id_proyek
            WHERE DATE(d.waktu) BETWEEN %s AND %s
            ORDER BY d.waktu DESC
        ''', (tanggal1, tanggal2))
        logs = cursor.fetchall()
        cursor.close()
        conn.close()

    return render_template('admin_dashboard.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True, port=5500)