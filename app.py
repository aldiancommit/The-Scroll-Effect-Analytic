from flask import Flask, render_template, request
from core.predictor import ScrollEffectPredictor

app = Flask(__name__, template_folder='src/templates')

ml_service = ScrollEffectPredictor()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction_result=None, error_msg=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil semua data form
        raw_data = request.form.to_dict()
        
        # Konversi tipe data
        raw_data['Age'] = float(raw_data['Age'])
        raw_data['Avg_Daily_Usage_Hours'] = float(raw_data['Avg_Daily_Usage_Hours'])
        raw_data['Sleep_Hours_Per_Night'] = float(raw_data['Sleep_Hours_Per_Night'])
        raw_data['Conflicts_Over_Social_Media'] = float(raw_data['Conflicts_Over_Social_Media'])

        # Lakukan kalkulasi
        final_score = ml_service.predict(raw_data)
        
        # TAMBAHKAN user_data=raw_data DI SINI
        return render_template('index.html', prediction_result=final_score, error_msg=None, user_data=raw_data)

    except Exception as e:
        print(f"Error Sistem: {e}")
        # Jika gagal, tampilkan pesan error di halaman web!
        return render_template('index.html', prediction_result=None, error_msg=str(e))

if __name__ == '__main__':
    app.run(debug=True)