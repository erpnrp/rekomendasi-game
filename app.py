from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Global Variables ---
df_games = None
cosine_sim = None
indices = None
DATA_LOADED_SUCCESSFULLY = False
APP_ERROR_MESSAGE = ""

# --- Data Loading and Processing ---
def muat_dan_proses_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(dir_path, 'games.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise Exception(f"Error: File 'games.csv' tidak ditemukan di lokasi: {csv_path}")

    if len(df) > 10000:
        df = df.head(10000)

    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower()

    kolom_mapping = {
        'title': ['name', 'title', 'judul'],
        'positive_ratio': ['positive_ratio', 'positive ratio', 'rating', 'rasio positif'],
        'price_original': ['price', 'price_original', 'harga']
    }

    rename_dict = {}
    for standard, possibles in kolom_mapping.items():
        for p in possibles:
            if p in df.columns:
                rename_dict[p] = standard
                break
    df.rename(columns=rename_dict, inplace=True)

    required_cols = ['title', 'positive_ratio']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_message = (
            f"<b>MASALAH:</b> Kolom hilang.<br><b>Kolom yang hilang:</b> <font color='red'>{', '.join(missing_cols)}</font>.<br>"
            f"<b>Kolom tersedia:</b><br><code>{original_columns}</code>"
        )
        raise Exception(error_message)

    df.dropna(subset=['title'], inplace=True)
    df['title'] = df['title'].astype(str)
    df.drop_duplicates(subset=['title'], inplace=True)

    df['positive_ratio'] = pd.to_numeric(df['positive_ratio'], errors='coerce').fillna(0)
    df.loc[df['positive_ratio'] <= 1, 'positive_ratio'] *= 100

    if 'price_original' not in df.columns:
        df['price_original'] = 0.0

    return df

# --- Recommendation System ---
def dapatkan_rekomendasi(title):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    game_indices = [i[0] for i in sim_scores]
    return df_games.iloc[game_indices].copy()

# --- Initialization ---
def inisialisasi_sistem():
    global df_games, cosine_sim, indices, DATA_LOADED_SUCCESSFULLY, APP_ERROR_MESSAGE
    try:
        df_games = muat_dan_proses_data()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_games['title'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        df_games.reset_index(drop=True, inplace=True)
        indices = pd.Series(df_games.index, index=df_games['title'])
        DATA_LOADED_SUCCESSFULLY = True
    except Exception as e:
        DATA_LOADED_SUCCESSFULLY = False
        APP_ERROR_MESSAGE = str(e)

inisialisasi_sistem()

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def halaman_utama():
    if not DATA_LOADED_SUCCESSFULLY:
        return f"""
        <div style='font-family: Arial; padding: 20px; margin: 40px; background-color: #f8d7da;'>
            <h1 style='color: #721c24;'>Error Saat Inisialisasi Aplikasi</h1>
            <p style='color: #721c24;'>{APP_ERROR_MESSAGE}</p>
        </div>
        """

    recommendations = None
    selected_game = None
    search_error = None

    if request.method == 'POST':
        search_query = request.form.get('search_query')
        if search_query:
            matching_games = df_games[df_games['title'].str.contains(search_query, case=False, na=False)]
            if not matching_games.empty:
                selected_game = matching_games.iloc[0]['title']
                recommendations_df = dapatkan_rekomendasi(selected_game)
                if recommendations_df is not None:
                    recommendations_df['Rating'] = recommendations_df['positive_ratio'].map('{:.2f}%'.format)
                    recommendations_df['Price'] = recommendations_df['price_original'].apply(lambda x: "Free to Play" if x == 0 else f"$ {x:.2f}")
                    recommendations_df['Review'] = recommendations_df['rating'] if 'rating' in recommendations_df else "N/A"
                    recommendations = recommendations_df.to_dict('records')
            else:
                search_error = f"Game dengan judul '{search_query}' tidak ditemukan."

    return render_template('index.html', 
                           recommendations=recommendations, 
                           selected_game=selected_game,
                           search_error=search_error)

if __name__ == '__main__':
    app.run(debug=True)
