def handle_attendance(jenis_absensi, id_karyawan, image_data, longitude, latitude, id_proyek):
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    existing_attendance = False
    print(f"existing attendance: {existing_attendance}")
    try:
        if(jenis_absensi == 'pulang'):
            # Check if user has already done the attendance
            cursor.execute('''
                        SELECT MAX(d.id) AS max_d, MAX(p.id) AS max_p
                        FROM absensi_datang d
                        LEFT JOIN absensi_pulang p
                        ON d.id = p.id
                        WHERE d.id_karyawan = %s
                        ''', (id_karyawan,))
            hasil = cursor.fetchone()

            if hasil:
                max_d = hasil.get('max_d')
                max_p =  hasil.get('max_p')
                
                if max_d is not None and max_p is not None and max_d == max_p:
                    existing_attendance = True
                
        if existing_attendance:
            cursor.close()
            conn.close()
            flash(f"Lakukan Absensi Datang terlebih dahulu", "warning")
            return redirect(url_for('attendance'))
    
        # Remove prefix "data:image/png;base64;" from data URL
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Save the captured image to a temporary file
        temp_image_path = os.path.join( 'static', 'images', 'data_wajah', f"temp_{id_karyawan}.png")
        with open(temp_image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        print(f"temp image: {temp_image_path}")
        # Get the reference image path from storage
        reference_image_data = os.path.join( 'static', 'images', 'data_wajah', f"{id_karyawan}.png")
        print(f"reference image: {reference_image_data}")
        # Compare images
        is_same, similarity = face_comparison(temp_image_path, reference_image_data, threshold=0.4)
        print(f"Hasil Perbandingan: {'SAMA': if is_same else 'BERBEDA'}")
        print(f"Similarity score: {similarity:.4f}")

        print(f"id karyawan: {id_karyawan} {type(id_karyawan)}")
        print(f"waktu sekarang: {waktu_sekarang} {type(waktu_sekarang)}")
        print(f"latitude, longitude: {latitude}, {longitude} {type(latitude)} {type(longitude)}")
        print(f"id proyek: {id_proyek} {type(id_proyek)}")

        # Remove temporary image
        os.remove(temp_image_path)
        print(f"test1")

        if not is_same:
            flash(f"Foto selfie tidak cocok dengan foto referensi. Silakan ambil foto ulang.", "danger")
            return redirect(url_for(f"absensi", jenis_absensi=jenis_absensi))
        

        # Save attendance log to database
        waktu_sekarang = datetime.now()



        if jenis_absensi == 'datang':
            photo_path = f"{id_karyawan}_{waktu_sekarang.strftime('%Y%m%d_%H%M%S')}.jpeg"
            cursor.execute('''
                           INSERT INTO absensi_datang (ID_karyawan, waktu, latitude, longitude, photo, id_proyek)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ''', (id_karyawan, waktu_sekarang, latitude, longitude, photo_path, id_proyek))
        else:
            cursor.execute('''
                           INSERT INTO absensi_pulang (ID_karyawan, waktu, latitude, longitude, id_proyek)
                           VALUES (%s, %s, %s, %s, %s)
                           ''', (id_karyawan, waktu_sekarang, latitude, longitude, id_proyek))
            
        conn.commit()
        flash("Absensi berhasil disimpan.", "success")
        return redirect(url_for(f'absensi', jenis_absensi=jenis_absensi))
    
    except Exception as e:
        print(f"Error saving time: {e}")
        flash("Terjadi kesalahan saat menyimpan absensi.", "danger")
        return redirect(url_for(f"absensi", jenis_absensi=jenis_absensi))
    
    finally:
        # Pastikan koneksi ke database selalu ditutup
        cursor.close()
        conn.close()