---
title: CI/CD Risk and Failure Classification
emoji: 🚦
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
---

Classifies a GitHub Actions run as likely to fail or succeed before it starts, and assigns a Low, Medium or High risk level.
The scikit-learn pipeline is loaded from the model repository on the Hub (`zopi12/ci-failure-risk`).