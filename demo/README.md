Welcome to Our Streamlit Web Application
This repository hosts a powerful web application built with Streamlit, designed to provide an interactive and intuitive user experience.

Requirements
To ensure optimal performance and access to the latest features, this application requires Streamlit version 1.46.0 or newer.

Getting Started
Follow these simple steps to get your web application up and running:

Navigate to the Application Directory:
Open your terminal or command prompt and change your current directory to the location containing the webapp.py file.

Install Dependencies and Run:
Execute the following commands in your terminal:

Bash

pip install -r requirements.txt
streamlit run webapp.py
Important Note on First Launch
Please be aware that the initial launch of the web application may experience a slight delay compared to subsequent runs. This is due to the necessary download of the 315 MB FasterRCNN model from Hugging Face, which occurs only once. Subsequent launches will be significantly faster.
