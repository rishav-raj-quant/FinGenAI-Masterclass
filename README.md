# FinGenAI Masterclass: LLMs for Quantitative Finance 📈🤖

Welcome to the **FinGenAI Masterclass**. This repository contains a complete, step-by-step guide to applying Generative AI to quantitative finance, structured as a series of Jupyter Notebooks.

## 📁 Repository Structure

1. **`01_Financial_Data_Acquisition.ipynb`**: Learn how to fetch live market data, news, and financial text using Python.
2. **`02_Financial_RAG_System.ipynb`**: Build a Retrieval-Augmented Generation (RAG) system to query SEC filings and financial reports.
3. **`03_Sentiment_Signal_Extraction.ipynb`**: Prompt engineering to extract structured trading signals (BUY/SELL/HOLD) from raw news text.
4. **`04_Autonomous_Trading_Agent.ipynb`**: Build an autonomous LangGraph agent that can reason over market data and execute simulated trades.

## 🚀 Getting Started

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Google Gemini API key (or OpenAI key if you modify the code):
   ```
   GOOGLE_API_KEY=your_key_here
   ```
4. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

## 🛠️ Tech Stack
* Python, Pandas, yfinance
* LangChain, LangGraph
* Google Gemini API (gemini-2.0-flash)
* ChromaDB (Vector Search)
