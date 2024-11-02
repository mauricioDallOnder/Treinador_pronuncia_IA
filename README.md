# Pronunciation Trainer AI

Pronunciation Trainer AI is a web application designed to help users improve their pronunciation in French. The application allows users to input text, record their pronunciation, and receive detailed feedback on their accuracy. It utilizes state-of-the-art speech-to-text models to analyze the pronunciation and provides visual and textual feedback to help users improve.

## Features

- **Text Input**: Users can write or generate text using predefined categories to practice their pronunciation.
- **Audio Recording**: The application allows users to record their pronunciation directly through the browser.
- **Pronunciation Feedback**: Detailed feedback is provided, highlighting correctly and incorrectly pronounced words, along with suggestions for improvement.
- **Performance Tracking**: Users can track their performance over time with visual graphs showing accuracy and errors.
- **Responsive Design**: The user interface is designed to be responsive, ensuring a seamless experience across different devices, including mobile phones and tablets.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **g2pk**: A Korean grapheme-to-phoneme conversion tool.
- **unidecode**: A text transliteration tool.
- **epitran**: A tool for transliterating text to IPA.
- **torch**: A deep learning framework.
- **torchaudio**: An audio library for PyTorch.
- **transformers**: Hugging Face's library for state-of-the-art natural language processing.
- **matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **gtts**: Google Text-to-Speech, a Python library and CLI tool to interface with Google Translate’s text-to-speech API.
- **Bootstrap**: A front-end framework for developing responsive and mobile-first websites.
- **JavaScript**: A programming language that enables interactive web pages.

## Setup and Installation

### Prerequisites

- Python 3.6 or higher

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pronunciation-trainer-ai.git
    cd pronunciation-trainer-ai
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask application**:
    ```bash
    python main.py
    ```

4. **Access the application**:
    - Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

Steps on how to use the application:

1. **Writing or Generating Text**: 
    - Navigate to the main page of the application.
    - You can write your own text in the textarea provided or select a category from the dropdown menu and click the "Generate Sentence" button to generate a random sentence from the selected category.

2. **Recording Audio**:
    - Click on the microphone button to start recording your pronunciation.
    - Speak clearly into your device’s microphone.
    - Click the microphone button again to stop the recording.

3. **Receiving Feedback**:
    - Once the recording is stopped, the application will process the audio and provide feedback on your pronunciation.
    - The feedback section will display the accuracy rate and completeness score.
    - Words will be highlighted in green if pronounced correctly and in red if pronounced incorrectly.
    - Click on any highlighted word to see detailed pronunciation feedback.

4. **Visualizing Performance**:
    - Scroll down to the "Performance Graph" section.
    - The graph will show your daily performance in terms of correct and incorrect pronunciations.
    - The performance graph will be updated each time you use the application, allowing you to track your progress over time.

## Project Structure

```plaintext
pronunciation-trainer-ai/
│
├── static/
│   ├── performance_graph.png
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── data/
│   ├── data_de_en_fr.pickle
│   └── frases_categorias.pickle
│
├── main.py
├── requirements.txt
└── README.md
```

4. **Access the application**:
    - Open your browser and navigate to `http://127.0.0.1:5000`.

## Contributing
Open an issue or submit a pull request for improvements or bug fixes.

## License:
This project is licensed under the MIT License.

## Acknowledgements

1. The Wav2Vec2 model from Facebook AI.
2. The g2pk library for Korean pronunciation conversion.
3. The Epitran library for phoneme transliteration.
4. The GTTS library for text-to-speech conversion.
