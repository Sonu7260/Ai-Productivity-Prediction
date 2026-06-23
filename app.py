import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Essential for Vercel Deployment: Map absolute path to look for the pickle file correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'adaboost1_pkl.pkl')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Model loading warning: {e}. Make sure 'adaboost1_pkl.pkl' is in the same directory.")

# Combined HTML & CSS UI Layout
UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Engineering Productivity Suite</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-main: #f4f6fa;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --brand-color: #4f46e5;
            --brand-hover: #4338ca;
            --border-color: #e2e8f0;
            --accent-success: #10b981;
            --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }
        .dashboard-container {
            width: 100%;
            max-width: 1150px;
            background: var(--card-bg);
            border-radius: 20px;
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }
        .app-header {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            color: #ffffff;
            padding: 1.5rem 2.5rem;
            border-bottom: 4px solid var(--brand-color);
        }
        .logo-area { display: flex; align-items: center; gap: 1rem; }
        .icon-logo { font-size: 2.2rem; color: #818cf8; }
        .app-header h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
        .app-header p { font-size: 0.85rem; color: #c7d2fe; }
        .workspace { display: grid; grid-template-columns: 1.4fr 1fr; min-height: 600px; }
        @media (max-width: 900px) { .workspace { grid-template-columns: 1fr; } }
        .prediction-form { padding: 2.5rem; border-right: 1px solid var(--border-color); }
        .results-panel { padding: 2.5rem; background-color: #fafbfd; display: flex; align-items: center; justify-content: center; }
        .form-section { margin-bottom: 2rem; }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .grid-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
        .wide-column { grid-column: span 2; }
        .input-group { display: flex; flex-direction: column; gap: 0.4rem; }
        .input-group label { font-size: 0.85rem; font-weight: 500; color: #475569; }
        .input-wrapper { position: relative; display: flex; align-items: center; }
        .field-icon { position: absolute; left: 1rem; color: var(--text-secondary); font-size: 0.95rem; }
        .input-wrapper input, .input-wrapper select {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border: 1px solid var(--border-color);
            background-color: #f8fafc;
            border-radius: 10px;
            font-size: 0.9rem;
            font-family: inherit;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }
        .input-wrapper input:focus, .input-wrapper select:focus {
            outline: none;
            border-color: var(--brand-color);
            background-color: #ffffff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }
        .submit-btn {
            width: 100%;
            background-color: var(--brand-color);
            color: #ffffff;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            transition: all 0.2s ease;
        }
        .submit-btn:hover { background-color: var(--brand-hover); transform: translateY(-1px); }
        .output-card {
            width: 100%;
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 16px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .output-icon {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem auto;
            font-size: 1.8rem;
        }
        .empty .output-icon { background-color: #eff6ff; color: var(--brand-color); }
        .success .output-icon { background-color: #ecfdf5; color: var(--accent-success); }
        .error .output-icon { background-color: #fef2f2; color: #ef4444; }
        .output-card h3 { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }
        .output-card p { font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5; }
        .prediction-value { font-size: 3rem; font-weight: 800; color: var(--accent-success); margin: 1rem 0; letter-spacing: -1px; }
        .prediction-meta { font-size: 0.8rem !important; border-top: 1px dashed var(--border-color); padding-top: 1rem; margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="app-header">
            <div class="logo-area">
                <i class="fa-solid fa-brain-circuit icon-logo"></i>
                <div>
                    <h1>ProdPredict AI</h1>
                    <p>AdaBoost-Powered Engineering Analytics</p>
                </div>
            </div>
        </header>

        <main class="workspace">
            <form method="POST" action="/" class="prediction-form">
                <div class="form-section">
                    <h3 class="section-title"><i class="fa-solid fa-code"></i> Engineering Metrics</h3>
                    <div class="grid-layout">
                        <div class="input-group">
                            <label>Coding Duration (Hours)</label>
                            <div class="input-wrapper">
                                <i class="fa-regular fa-clock field-icon"></i>
                                <input type="number" step="0.1" name="Hours_Coding" min="0" max="24" required placeholder="e.g. 6.5" value="{{ form.get('Hours_Coding', '') }}">
                            </div>
                        </div>
                        <div class="input-group">
                            <label>AI Copilot Usage (Hours)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-microchip field-icon"></i>
                                <input type="number" step="0.1" name="AI_Usage_Hours" min="0" max="24" required placeholder="e.g. 3.0" value="{{ form.get('AI_Usage_Hours', '') }}">
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Lines of Code Written</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-terminal field-icon"></i>
                                <input type="number" name="Lines_of_Code" min="0" required placeholder="e.g. 250" value="{{ form.get('Lines_of_Code', '') }}">
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Git Commits Pushed</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-code-commit field-icon"></i>
                                <input type="number" name="Commits" min="0" required placeholder="e.g. 4" value="{{ form.get('Commits', '') }}">
                            </div>
                        </div>
                        <div class="input-group wide-column">
                            <label>Bugs Linked / Logged</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-bug field-icon"></i>
                                <input type="number" name="Bugs_Reported" min="0" required placeholder="e.g. 2" value="{{ form.get('Bugs_Reported', '') }}">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3 class="section-title"><i class="fa-solid fa-bolt"></i> Focus & Environment</h3>
                    <div class="grid-layout">
                        <div class="input-group">
                            <label>Sleep Quality (Hours)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-bed field-icon"></i>
                                <input type="number" step="0.1" name="Sleep_Hours" min="0" max="24" required placeholder="e.g. 7.5" value="{{ form.get('Sleep_Hours', '') }}">
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Distraction Level (0-10)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-bell-slash field-icon"></i>
                                <input type="number" name="Distractions" min="0" max="10" required placeholder="0-10" value="{{ form.get('Distractions', '') }}">
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Cognitive Load</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-layer-group field-icon"></i>
                                <select name="Cognitive_Load" required>
                                    <option value="Low" {% if form.get('Cognitive_Load') == 'Low' %}selected{% endif %}>Low Complexity Tasks</option>
                                    <option value="Medium" {% if form.get('Cognitive_Load') == 'Medium' or not form.get('Cognitive_Load') %}selected{% endif %}>Medium Complexity Tasks</option>
                                    <option value="High" {% if form.get('Cognitive_Load') == 'High' %}selected{% endif %}>High Architecture Tasks</option>
                                </select>
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Perceived Stress Level</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-gauge-high field-icon"></i>
                                <select name="Stress_Level" required>
                                    <option value="Low" {% if form.get('Stress_Level') == 'Low' %}selected{% endif %}>Low / Calm Environment</option>
                                    <option value="Medium" {% if form.get('Stress_Level') == 'Medium' or not form.get('Stress_Level') %}selected{% endif %}>Moderate Pace</option>
                                    <option value="High" {% if form.get('Stress_Level') == 'High' %}selected{% endif %}>High Stress / Tight Deadline</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <button type="submit" class="submit-btn">
                    <span>Analyze Productivity</span>
                    <i class="fa-solid fa-arrow-right"></i>
                </button>
            </form>

            <div class="results-panel">
                {% if prediction %}
                    <div class="output-card success">
                        <div class="output-icon"><i class="fa-solid fa-chart-line"></i></div>
                        <h3>Engineered Productivity Result</h3>
                        <div class="prediction-value">{{ prediction }}</div>
                        <p class="prediction-meta">Computed via AdaBoost decision ensemble mapping metrics against task density.</p>
                    </div>
                {% elif error_message %}
                    <div class="output-card error">
                        <div class="output-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
                        <h3>Processing Error</h3>
                        <p>{{ error_message }}</p>
                    </div>
                {% else %}
                    <div class="output-card empty">
                        <div class="output-icon"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
                        <h3>Awaiting Input</h3>
                        <p>Fill out telemetry logs on the left panel to execute your engineering productivity calculation pipeline.</p>
                    </div>
                {% endif %}
            </div>
        </main>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        if model is None:
            error_message = "Model file 'adaboost1_pkl.pkl' could not be found or loaded on the server."
        else:
            try:
                # Mapping user interface categoricals safely into model numerical labels
                categorical_mapping = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
                
                features = [
                    float(request.form.get('Hours_Coding', 0)),
                    float(request.form.get('AI_Usage_Hours', 0)),
                    float(request.form.get('Lines_of_Code', 0)),
                    float(request.form.get('Commits', 0)),
                    float(request.form.get('Bugs_Reported', 0)),
                    float(request.form.get('Sleep_Hours', 0)),
                    float(request.form.get('Distractions', 0)),
                    categorical_mapping.get(request.form.get('Cognitive_Load', 'Medium')),
                    categorical_mapping.get(request.form.get('Stress_Level', 'Medium'))
                ]
                
                raw_prediction = model.predict(np.array([features]))[0]
                prediction = str(raw_prediction)
            except Exception as e:
                error_message = f"Prediction pipeline breakdown: {str(e)}"

    return render_template_string(UI_TEMPLATE, prediction=prediction, error_message=error_message, form=form_data)

# WSGI setup for Vercel
app = app
