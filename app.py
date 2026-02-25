
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
import io
import csv
from urllib.parse import urlparse

app = Flask(__name__)

model = joblib.load('best_model.pkl')
le    = joblib.load('label_encoder.pkl')

def extract_features(url):
    url = str(url)
    try:
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
    except:
        hostname = ''
        path = url

    features = {
        'url_length':          len(url),
        'hostname_length':     len(hostname),
        'path_length':         len(path),
        'num_dots':            url.count('.'),
        'num_hyphens':         url.count('-'),
        'num_underscores':     url.count('_'),
        'num_slashes':         url.count('/'),
        'num_at':              url.count('@'),
        'num_question':        url.count('?'),
        'num_equals':          url.count('='),
        'num_ampersand':       url.count('&'),
        'num_percent':         url.count('%'),
        'num_digits':          sum(c.isdigit() for c in url),
        'has_ip':              1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0,
        'has_https':           1 if url.startswith('https') else 0,
        'has_www':             1 if 'www.' in url else 0,
        'has_at_sign':         1 if '@' in url else 0,
        'has_double_slash':    1 if '//' in url[7:] else 0,
        'has_hex_encoding':    1 if '%' in url else 0,
        'num_subdomains':      len(hostname.split('.')) - 2 if hostname else 0,
        'has_suspicious_word': 1 if any(w in url.lower() for w in [
            'login', 'secure', 'account', 'update', 'bank', 'verify',
            'confirm', 'paypal', 'signin', 'ebay', 'admin', 'password'
        ]) else 0,
        'digit_ratio':         sum(c.isdigit() for c in url) / max(len(url), 1),
        'letter_ratio':        sum(c.isalpha() for c in url) / max(len(url), 1),
        'special_ratio':       sum(not c.isalnum() for c in url) / max(len(url), 1),
    }
    return features


def get_dna_scores(url, features):
    """Return 8 normalized threat signal scores for radar chart"""
    return {
        'Length Risk':      min(len(url) / 150, 1.0),
        'Special Chars':    min(features['special_ratio'] * 3, 1.0),
        'Suspicious KW':    float(features['has_suspicious_word']),
        'IP Address':       float(features['has_ip']),
        'Subdomain Depth':  min(max(features['num_subdomains'], 0) / 4, 1.0),
        'Digit Density':    min(features['digit_ratio'] * 4, 1.0),
        'Hyphen Abuse':     min(features['num_hyphens'] / 6, 1.0),
        'Encoding Tricks':  float(features['has_hex_encoding']),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    features = extract_features(url)
    df_input = pd.DataFrame([features])

    prediction      = model.predict(df_input)[0]
    probabilities   = model.predict_proba(df_input)[0]
    predicted_class = le.inverse_transform([prediction])[0]
    prob_dict       = {cls: round(float(p), 4) for cls, p in zip(le.classes_, probabilities)}
    dna             = get_dna_scores(url, features)

    return jsonify({
        'url':           url,
        'prediction':    predicted_class,
        'probabilities': prob_dict,
        'dna':           dna,
        'confidence':    round(float(max(probabilities)) * 100, 1),
    })


@app.route('/bulk', methods=['POST'])
def bulk():
    data = request.get_json()
    urls = data.get('urls', [])
    if not urls:
        return jsonify({'error': 'No URLs provided'}), 400
    if len(urls) > 500:
        return jsonify({'error': 'Max 500 URLs per batch'}), 400

    results = []
    for url in urls:
        url = str(url).strip()
        if not url:
            continue
        features      = extract_features(url)
        df_input      = pd.DataFrame([features])
        prediction    = model.predict(df_input)[0]
        probs         = model.predict_proba(df_input)[0]
        pred_class    = le.inverse_transform([prediction])[0]
        confidence    = round(float(max(probs)) * 100, 1)
        results.append({
            'url':        url,
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': {cls: round(float(p), 4) for cls, p in zip(le.classes_, probs)},
        })

    summary = {c: sum(1 for r in results if r['prediction'] == c) for c in le.classes_}
    return jsonify({'results': results, 'summary': summary, 'total': len(results)})


if __name__ == '__main__':
    app.run(debug=True)
