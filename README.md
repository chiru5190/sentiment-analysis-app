# Sentiment-Analysis-App

📊 Sentiment Analysis App

A machine-learning powered web application that analyzes text sentiment (Positive, Negative, Neutral) using a TF-IDF-based model integrated with an interactive Streamlit interface.

🚀 Features

🔍 Single Text Analysis – Enter any text to get instant sentiment prediction.

📂 Bulk Sentiment Analysis – Upload a CSV file and classify multiple texts at once.

📈 Interactive UI – Built with Streamlit for a smooth and intuitive user experience.

⚡ Fast & Accurate Predictions – Uses a trained ML model with TF-IDF vectorization.

🧹 Automated Preprocessing – Cleans and prepares text before prediction.

🧠 Technologies Used

Python

Streamlit

scikit-learn

NumPy

Pandas

TF-IDF Vectorizer

📁 Project Structure
Sentiment-Analysis-App/
│
├── app.py                 # Main Streamlit application  
├── pages/                 # Multi-page UI
│   ├── 1_About.py  
│   └── 2_Bulk_Analysis.py  
│
├── utils.py               # Preprocessing & helper functions
├── view_utils.py          # UI helper components
├── sentiment_models.pkl   # Trained ML model
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── requirements.txt       # Dependencies
└── Readme.txt

🛠️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/<yourusername>/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

🎯 How It Works

User enters text (or uploads a dataset).

Text is cleaned and processed using custom preprocessing steps.

TF-IDF vectorizer converts text into numerical features.

A trained ML model predicts the sentiment.

Streamlit displays results clearly for the user.



🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

📜 License

This project is open-source and available under the MIT License.

🙌 Acknowledgements

Streamlit for the easy UI framework

scikit-learn for ML algorithms

Dataset used for training (customized/curated)
