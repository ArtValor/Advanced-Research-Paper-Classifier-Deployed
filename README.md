# Advanced Research Paper Classifier

This project aims to help label research papers to fit arXiv's extensive labelling system. We used SciBERT to create a robust classification pipeline that uses the abstracts of research papers to sort them into 50+ labels. This leverages the now popular transformers mechanism to help the model understand the abstracts, and then use this information to accurately classify the papers.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-evaluation-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```
2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
