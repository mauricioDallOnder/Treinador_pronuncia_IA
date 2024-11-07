import asyncio
import sys
sys.setrecursionlimit(10000)
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from flask import Flask, request, render_template, jsonify, send_file
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Configura o backend para não-GUI
import re
import os
import tempfile
import wave
import pickle
import random
import pandas as pd
from gtts import gTTS
from WordMatching import get_best_mapped_words
import epitran
import noisereduce as nr
import jellyfish
from concurrent.futures import ThreadPoolExecutor
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__, template_folder="templates", static_folder="static")

# Variáveis globais para modelos
model, processor, translation_model, tokenizer = None, None, None, None

# Executor para processamento assíncrono
executor = ThreadPoolExecutor(max_workers=1)

# Carregar frases categorizadas e arquivos --------------------------------------------------------------------------------------------------

# Caminhos dos arquivos de desempenho e progresso do usuário
performance_file = 'performance_data.pkl'
user_progress_file = 'user_progress.pkl'

# Carregar frases aleatórias
try:
    with open('data_de_en_2.pickle', 'rb') as f:
        random_sentences_df = pickle.load(f)
    # Verificar se é um DataFrame e converter para lista de dicionários
    if isinstance(random_sentences_df, pd.DataFrame):
        random_sentences = random_sentences_df.to_dict(orient='records')
    else:
        random_sentences = random_sentences_df
except Exception as e:
    print(f"Erro ao carregar data_en_pt.pickle: {e}")
    random_sentences = []

try:
    with open('frases_categorias_en.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar frases_categorias_en.pickle: {e}")
    categorized_sentences = {}

# Carregar o Modelo de SST Inglês atualizado
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Modelo para tradução
translation_model_name = 'facebook/m2m100_418M'
tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

# Carregar progresso do usuário
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

##-----------------------------------------------------------------------------------------------------------
# Iniciar o epitran e funções de tradução

# Inicializar Epitran para Inglês
epi = epitran.Epitran('eng-Latn')

source_language = 'en'  # Inglês
target_language = 'pt'  # Português

# Mapeamento de fonemas ingleses para português
english_to_portuguese_phonemes = {
    # Vogais
    'i': 'i',
    'ɪ': 'i',
    'eɪ': 'ei',
    'ɛ': 'é',
    'æ': 'é',
    'ɑ': 'á',
    'ʌ': 'â',
    'ɔ': 'ó',
    'oʊ': 'ô',
    'u': 'u',
    'ʊ': 'u',
    'aɪ': 'ai',
    'aʊ': 'au',
    'ɔɪ': 'ói',
    'ər': 'â',
    'ɜr': 'âr',

    # Consoantes
    'p': 'p',
    'b': 'b',
    't': 't',
    'd': 'd',
    'k': 'k',
    'g': 'g',
    'f': 'f',
    'v': 'v',
    'θ': 'th',
    'ð': 'dh',
    's': 's',
    'z': 'z',
    'ʃ': 'ch',
    'ʒ': 'j',
    'h': 'h',
    'm': 'm',
    'n': 'n',
    'ŋ': 'ng',
    'l': 'l',
    'r': 'r',
    'w': 'w',
    'j': 'i',
    'tʃ': 'tch',
    'dʒ': 'dj',
}

# Função de Tradução
def translate_to_portuguese(text):
    try:
        tokenizer.src_lang = source_language
        encoded = tokenizer(text, return_tensors='pt')
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_language)
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return "Tradução indisponível."

# Função para obter pronúncia fonética
def get_english_phonetic(word):
    return epi.transliterate(word)

def get_pronunciation(word):
    pronunciation = epi.transliterate(word)
    return pronunciation

def normalize_vowels(pronunciation):
    # Normaliza vogais para consistência, se necessário
    return pronunciation

def handle_special_cases(pronunciation):
    # Regras especiais para contextos específicos
    return pronunciation

def convert_pronunciation_to_portuguese(pronunciation):
    pronunciation = normalize_vowels(pronunciation)
    pronunciation = handle_special_cases(pronunciation)

    # Substituir símbolos fonéticos usando o mapeamento
    # Ordenar os fonemas por tamanho decrescente para evitar conflitos
    sorted_phonemes = sorted(english_to_portuguese_phonemes.keys(), key=len, reverse=True)
    for phoneme in sorted_phonemes:
        pronunciation = pronunciation.replace(phoneme, english_to_portuguese_phonemes[phoneme])

    return pronunciation

def transliterate_and_convert(word):
    pronunciation = get_pronunciation(word)
    pronunciation_pt = convert_pronunciation_to_portuguese(pronunciation)
    return pronunciation_pt

def compare_phonetics(phonetic1, phonetic2, threshold=0.85):
    # Calcular distância Damerau-Levenshtein normalizada
    damerau_score = 1 - jellyfish.damerau_levenshtein_distance(phonetic1, phonetic2) / max(len(phonetic1), len(phonetic2))

    # Calcular similaridade Jaro-Winkler
    jaro_winkler_score = jellyfish.jaro_winkler_similarity(phonetic1, phonetic2)

    # Combinar ambos os resultados com uma ponderação
    combined_score = 0.7 * damerau_score + 0.3 * jaro_winkler_score

    # Suavização para pontuações próximas ao limite
    smooth_threshold = threshold - 0.05 if combined_score >= threshold - 0.05 else threshold

    # Verificar se a pontuação combinada atinge o limite ajustado
    return combined_score >= smooth_threshold

# Normalização de texto para comparação
def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'")
    # Manter apóstrofos dentro das palavras e remover outros caracteres especiais
    text = re.sub(r"[^\w\s']", '', text)
    # Garantir que não haja espaços desnecessários ao redor dos apóstrofos
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text.strip()

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')

##--------------------------------------------------------------------------------------------------------------------------------
# Processamento de áudio:

# Função para melhorar a qualidade do áudio com redução de ruído
def reduce_noise(waveform, sample_rate):
    return nr.reduce_noise(y=waveform, sr=sample_rate)

# Função de processamento de áudio
def process_audio(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            waveform = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16) / 32768.0
        waveform = reduce_noise(waveform, sample_rate)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if sample_rate != 16000:
            waveform_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_tensor)
        inputs = processor(waveform_tensor.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    finally:
        os.remove(file_path)

##--------------------------------------------------------------------------------------------------------------------------------
# Rotas de API
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form['text']
    words = text.split()
    pronunciations = [transliterate_and_convert(word) for word in words]
    return jsonify({'pronunciations': ' '.join(pronunciations)})

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    translated_text = translate_to_portuguese(text)
    return jsonify({'translation': translated_text})

@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    try:
        category = request.form.get('category', 'random')

        if category == 'random':
            if random_sentences:
                sentence = random.choice(random_sentences)
                sentence_text = remove_punctuation_end(sentence.get('en_sentence', "Frase não encontrada."))
            else:
                return jsonify({"error": "Nenhuma frase disponível para seleção aleatória."}), 500
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

        return jsonify({'en_sentence': sentence_text, 'category': category})

    except Exception as e:
        print(f"Erro no endpoint /get_sentence: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    text = request.form['text']
    category = request.form.get('category', 'random')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    # Executar processamento assíncrono de áudio
    transcription = executor.submit(process_audio, tmp_file_path).result()

    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()
    mapped_words, mapped_words_indices = get_best_mapped_words(words_estimated, words_real)

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
            if compare_phonetics(correct_pronunciation, user_pronunciation):
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
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0
    completeness_score = (len(words_estimated) / len(words_real)) * 100
    performance_data.append({
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'ratio': ratio,
        'completeness_score': completeness_score,
        'sentence': text
    })
    save_performance_data(performance_data)

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

    return jsonify({
        'ratio': f"{ratio:.2f}",
        'diff_html': diff_html,
        'pronunciations': pronunciations,
        'feedback': feedback,
        'completeness_score': f"{completeness_score:.2f}"
    })

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
        return "Nenhum dado de desempenho disponível.", 204
    df = pd.DataFrame(performance_data)
    if 'date' not in df.columns:
        return "Dados de desempenho inválidos.", 500
    grouped = df.groupby('date').agg({
        'ratio': 'mean',
        'completeness_score': 'mean'
    }).reset_index()

    dates = grouped['date']
    ratios = grouped['ratio']
    completeness_scores = grouped['completeness_score']

    x = np.arange(len(dates))  # the label locations

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ratios, marker='o', label='Taxa de Acerto na Pronúncia (%)')
    ax.plot(dates, completeness_scores, marker='x', label='Taxa de Completude (%)')

    ax.set_xlabel('Data')
    ax.set_ylabel('Percentagem')
    ax.set_title('Desempenho Diário')
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
