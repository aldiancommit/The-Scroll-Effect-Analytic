import joblib
import pandas as pd
import os

class ScrollEffectPredictor:
    def __init__(self):
        print("[SISTEM] Memuat Model Random Forest dan Encoders...")
        self.model = joblib.load(os.path.join('models', 'random_forest_addiction_model.pkl'))
        
        encoder_dir = os.path.join('models', 'encoders')
        self.encoders = {
            'Gender': joblib.load(os.path.join(encoder_dir, 'Gender_encoder.pkl')),
            'Academic_Level': joblib.load(os.path.join(encoder_dir, 'Academic_Level_encoder.pkl')),
            'Country': joblib.load(os.path.join(encoder_dir, 'Country_encoder.pkl')),
            'Most_Used_Platform': joblib.load(os.path.join(encoder_dir, 'Most_Used_Platform_encoder.pkl')),
            'Affects_Academic_Performance': joblib.load(os.path.join(encoder_dir, 'Affects_Academic_Performance_encoder.pkl')),
            'Relationship_Status': joblib.load(os.path.join(encoder_dir, 'Relationship_Status_encoder.pkl'))
        }

    def predict(self, form_data):
        """Memproses data mentah dari form HTML ke bentuk prediksi akhir"""
        df_input = pd.DataFrame([form_data])
        
        # 1. URUTAN WAJIB (Sesuai dengan file dibersihkan saat training)
        expected_columns = [
            'Age', 'Gender', 'Academic_Level', 'Country', 
            'Avg_Daily_Usage_Hours', 'Most_Used_Platform', 
            'Affects_Academic_Performance', 'Sleep_Hours_Per_Night', 
            'Relationship_Status', 'Conflicts_Over_Social_Media'
        ]
        
        # Susun ulang urutan kolom dataframe agar tidak ditolak model
        df_input = df_input[expected_columns]
        
        # 2. Transformasi teks ke angka (Encoding)
        for col, le in self.encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = le.transform(df_input[col])
                except ValueError as e:
                    # Lempar error spesifik jika ada teks yang tidak terdaftar di encoder
                    raise ValueError(f"Terdapat nilai form yang tidak valid pada kategori {col}: {df_input[col].iloc[0]}")
                
        # 3. Jalankan prediksi
        prediction = self.model.predict(df_input)[0]
        return round(prediction, 2)