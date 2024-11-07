import sys
sys.setrecursionlimit(10000)
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from flask import Flask, request, render_template, jsonify, send_file
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configure o backend para não-GUI
import re
import os
import tempfile
import wave
import pickle
import random
import pandas as pd
from gtts import gTTS
from WordMetrics import edit_distance_python2
from WordMatching import get_best_mapped_words
from unidecode import unidecode
import pronouncing
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Carregar o Modelo de Reconhecimento de Fala em Inglês
model_name_asr = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name_asr)
model = Wav2Vec2ForCTC.from_pretrained(model_name_asr)

# Carregar o Modelo de Tradução
model_name_translator = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
tokenizer = MarianTokenizer.from_pretrained(model_name_translator, use_auth_token=False)
translation_model = MarianMTModel.from_pretrained(model_name_translator, use_auth_token=False)

# Função de Tradução
def translate_to_portuguese(text):
    try:
        # Tokenizar o texto de entrada
        tokens = tokenizer(text, return_tensors='pt', padding=True)
        # Gerar a tradução
        translated = translation_model.generate(**tokens)
        # Decodificar o texto traduzido
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return "Tradução indisponível."



# Carregar ou inicializar dados de desempenho
performance_file = 'performance_data.pkl'

def load_performance_data():
    if os.path.exists(performance_file):
        with open(performance_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def save_performance_data(data):
    with open(performance_file, 'wb') as f:
        pickle.dump(data, f)

performance_data = load_performance_data()

# Arquivo para armazenar o progresso do usuário
user_progress_file = 'user_progress.pkl'

def load_user_progress():
    if os.path.exists(user_progress_file):
        with open(user_progress_file, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_user_progress(progress):
    with open(user_progress_file, 'wb') as f:
        pickle.dump(progress, f)

user_progress = load_user_progress()

def get_pronunciation(word):
    phones = pronouncing.phones_for_word(word.lower())
    if phones:
        return phones[0]
    else:
        return word  # Retorna a palavra original se a pronúncia não for encontrada

# Mapeamento de fonemas ARPABET para fonemas aproximados em português
arpabet_to_portuguese_phonemes = {
    'AA': 'á',
    'AE': 'é',
    'AH': 'ã',
    'AO': 'ó',
    'AW': 'au',
    'AY': 'ai',
    'B': 'b',
    'CH': 'tch',
    'D': 'd',
    'DH': 'd',  # Aproximação
    'EH': 'é',
    'ER': 'âr',
    'EY': 'ei',
    'F': 'f',
    'G': 'g',
    'HH': 'h',
    'IH': 'i',
    'IY': 'i',
    'JH': 'dj',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'n',
    'OW': 'ou',
    'OY': 'ói',
    'P': 'p',
    'R': 'r',
    'S': 's',
    'SH': 'ch',
    'T': 't',
    'TH': 'f',  # Aproximação
    'UH': 'u',
    'UW': 'u',
    'V': 'v',
    'W': 'u',
    'Y': 'i',
    'Z': 'z',
    'ZH': 'j',
}

def convert_pronunciation_to_portuguese(pronunciation):
    phonemes = pronunciation.split()
    mapped_phonemes = []
    for phoneme in phonemes:
        # Remove números de estresse (ex: 'AA1' -> 'AA')
        phoneme = re.sub(r'\d', '', phoneme)
        if phoneme in arpabet_to_portuguese_phonemes:
            mapped_phonemes.append(arpabet_to_portuguese_phonemes[phoneme])
        else:
            mapped_phonemes.append(phoneme)
    return ''.join(mapped_phonemes)

# Carregar frases para seleção aleatória de 'data_de_en_2.pickle'
try:
    with open('data_de_en_2.pickle', 'rb') as f:
        random_sentences_df = pickle.load(f)
    # Verificar se é um DataFrame e converter para lista de dicionários
    if isinstance(random_sentences_df, pd.DataFrame):
        random_sentences = random_sentences_df.to_dict(orient='records')
    else:
        random_sentences = random_sentences_df
except Exception as e:
    print(f"Erro ao carregar data_de_en_2.pickle: {e}")
    random_sentences = []

# Carregar frases categorizadas de 'frases_categorias_en.pickle'
try:
    with open('frases_categorias_en.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar frases_categorias_en.pickle: {e}")
    categorized_sentences = {}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def transliterate_and_convert(word):
    pronunciation = get_pronunciation(word)
    pronunciation_pt = convert_pronunciation_to_portuguese(pronunciation)
    return pronunciation_pt

def compare_pronunciations(correct_pronunciation, user_pronunciation, similarity_threshold=0.9):
    distance = edit_distance_python2(correct_pronunciation, user_pronunciation)
    max_length = max(len(correct_pronunciation), len(user_pronunciation))
    if max_length == 0:
        return False  # Evita divisão por zero
    similarity = 1 - (distance / max_length)
    return similarity >= similarity_threshold

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    text = request.form['text']
    category = request.form.get('category', 'random')

    # Salvar o arquivo enviado em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Ler o arquivo de áudio usando o módulo wave
        with wave.open(tmp_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            waveform = wav_file.readframes(num_frames)
            waveform = np.frombuffer(waveform, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # Reamostrar se necessário
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        waveform = waveform.squeeze(0)

        # Normalizar o volume
        waveform = waveform / waveform.abs().max()

        # Ajustar os parâmetros do modelo
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    finally:
        os.remove(tmp_file_path)

    # Normalizar textos
    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)

    # Palavras estimadas e reais
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()

    # Utilizar get_best_mapped_words para obter o mapeamento
    mapped_words, mapped_words_indices = get_best_mapped_words(words_estimated, words_real)

    # Gerar HTML com palavras coloridas e feedback
    diff_html = []
    pronunciations = {}
    feedback = {}
    correct_count = 0
    incorrect_count = 0

    for idx, real_word in enumerate(words_real):
        if idx < len(mapped_words):
            mapped_word = mapped_words[idx]
            correct_pronunciation = transliterate_and_convert(real_word)
            user_pronunciation = transliterate_and_convert(mapped_word)
            if compare_pronunciations(correct_pronunciation, user_pronunciation):
                diff_html.append(f'<span class="word correct" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
                correct_count += 1
            else:
                diff_html.append(f'<span class="word incorrect" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
                incorrect_count += 1
                feedback[real_word] = {
                    'correct': correct_pronunciation,
                    'user': user_pronunciation,
                    'suggestion': f"Tente pronunciar '{real_word}' como '{correct_pronunciation}'"
                }
            pronunciations[real_word] = {
                'correct': correct_pronunciation,
                'user': user_pronunciation
            }
        else:
            # Palavra não reconhecida
            diff_html.append(f'<span class="word missing" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            incorrect_count += 1
            feedback[real_word] = {
                'correct': transliterate_and_convert(real_word),
                'user': '',
                'suggestion': f"Tente pronunciar '{real_word}' como '{transliterate_and_convert(real_word)}'"
            }
            pronunciations[real_word] = {
                'correct': transliterate_and_convert(real_word),
                'user': ''
            }

    diff_html = ' '.join(diff_html)

    # Calcular taxa de acerto e completude
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0
    completeness_score = (len(words_estimated) / len(words_real)) * 100 if len(words_real) > 0 else 0

    # Armazenar resultados diários
    performance_data.append({
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'ratio': ratio,
        'completeness_score': completeness_score,
        'sentence': text
    })
    save_performance_data(performance_data)

    # Atualizar o progresso do usuário
    if category != 'random':
        user_category_progress = user_progress.get(category, {'sentences_done': [], 'performance': []})

        if text not in user_category_progress['sentences_done']:
            user_category_progress['sentences_done'].append(text)

        user_category_progress['performance'].append({
            'sentence': text,
            'ratio': ratio,
            'completeness_score': completeness_score
        })

        user_progress[category] = user_category_progress
        save_user_progress(user_progress)

    # Logging para depuração
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {total_words}, Ratio: {ratio}")
    formatted_ratio = "{:.2f}".format(ratio)
    formatted_completeness = "{:.2f}".format(completeness_score)

    return jsonify({
        'ratio': formatted_ratio,
        'diff_html': diff_html,
        'pronunciations': pronunciations,
        'feedback': feedback,
        'completeness_score': formatted_completeness
    })

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    # Use a função de tradução existente
    translated_text = translate_to_portuguese(text)
    return jsonify({'translation': translated_text})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    try:
        category = request.form.get('category', 'random')

        if category == 'random':
            if random_sentences:
                sentence = random.choice(random_sentences)
                sentence_text = sentence.get('en_sentence', "Sentence not found.")
            else:
                return jsonify({"error": "No sentences available for random selection."}), 500
        else:
            if category in categorized_sentences:
                sentences_in_category = categorized_sentences[category]
                user_category_progress = user_progress.get(category, {'sentences_done': [], 'performance': []})
                sentences_done = user_category_progress.get('sentences_done', [])
                sentences_remaining = list(set(sentences_in_category) - set(sentences_done))

                if not sentences_remaining:
                    # Todas as frases foram praticadas
                    performance_list = user_category_progress.get('performance', [])
                    if performance_list:
                        avg_ratio = sum(p['ratio'] for p in performance_list) / len(performance_list)
                        avg_completeness = sum(p['completeness_score'] for p in performance_list) / len(performance_list)
                    else:
                        avg_ratio = 0
                        avg_completeness = 0

                    return jsonify({
                        "message": "Você completou todas as frases desta categoria.",
                        "average_ratio": f"{avg_ratio:.2f}",
                        "average_completeness": f"{avg_completeness:.2f}"
                    })
                else:
                    sentence_text = random.choice(sentences_remaining)
                    sentence_text = remove_punctuation_end(sentence_text)
            else:
                return jsonify({"error": "Categoria não encontrada."}), 400

        return jsonify({
            'en_sentence': sentence_text,
            'category': category
        })

    except Exception as e:
        print(f"Erro no endpoint /get_sentence: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500



@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form['text']
    words = text.split()
    pronunciations = [transliterate_and_convert(word) for word in words]
    return jsonify({'pronunciations': ' '.join(pronunciations)})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    tts = gTTS(text=text, lang='en')
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    return send_file(file_path, as_attachment=True, mimetype='audio/mp3')

@app.route('/performance', methods=['GET'])
def performance():
    if not performance_data:
        return "No performance data available.", 204
    df = pd.DataFrame(performance_data)
    if 'date' not in df.columns:
        return "Invalid performance data.", 500
    grouped = df.groupby('date').agg({
        'ratio': 'mean',
        'completeness_score': 'mean'
    }).reset_index()

    dates = grouped['date']
    ratios = grouped['ratio']
    completeness_scores = grouped['completeness_score']

    x = np.arange(len(dates))  # as posições dos rótulos

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ratios, marker='o', label='Pronunciation Accuracy Rate (%)')
    ax.plot(dates, completeness_scores, marker='x', label='Completeness Rate (%)')

    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage')
    ax.set_title('Daily Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()

    graph_path = 'static/performance_graph.png'
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()

    return send_file(graph_path, mimetype='image/png')

@app.route('/get_progress', methods=['GET'])
def get_progress():
    progress_data = {}
    for category in categorized_sentences.keys():
        total_sentences = len(categorized_sentences[category])
        user_category_progress = user_progress.get(category, {'sentences_done': []})
        sentences_done = len(user_category_progress.get('sentences_done', []))
        progress_data[category] = {
            'total_sentences': total_sentences,
            'sentences_done': sentences_done
        }
    return jsonify(progress_data)
    
# Inicialização e execução do aplicativo
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=5000))
