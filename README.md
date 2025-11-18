# 📈 Stock Market Protocol

A framework for analyzing and evaluating stock return predictors on global market data. A detailed documentation of this project can be found here: [Global Stock Market Protocol Documentation](docs/Documentation.pdf)

---

## 🚀 Usage

Follow these steps to set up and run the framework locally:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Proenchen/stock-market-protocol.git
```

### 2️⃣ Navigate into the project directory
```bash
cd stock-market-protocol
```

### 3️⃣ Create a virtual environment
```bash
python -m venv venv
```

### 4️⃣ Activate the virtual environment
- **Windows**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS / Linux**
  ```bash
  source venv/bin/activate
  ```

### 5️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 6️⃣ Prepare the data
Create a folder named `data` inside the project directory and insert all necessary data files.

If required, update the data file paths in `logic/analysis.py` (lines 53–56):

```python
crsp_full = pd.read_csv("./data/dsws_crsp.csv")
factors_full = pd.read_csv("./data/Factors.csv")
fm_full = pd.read_csv("./data/Fama_Macbeth.csv")
corr_full = pd.read_parquet("./data/Predictor_different.parquet")
```

### 7️⃣ Configure mailing service (optional)
To enable the mailing service, create a `.env` file in the project root (`stock-market-protocol`) and add your SMTP and email credentials:

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=globalstockmarketprotocol@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop
```

> **Note:** If you use Gmail, `EMAIL_PASSWORD` must be an [App Password](https://support.google.com/accounts/answer/185833).

### 8️⃣ Run the application
```bash
python -u app.py
```

### 9️⃣ Open in browser
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to start using the application.

---

## 📸​ Screenshots

Below is a screenshot of the webpage for test configuration and execution:

<img width="1628" height="877" alt="grafik" src="https://github.com/user-attachments/assets/38009e7e-5419-4f8f-96e9-266f3ad90b50" />


---

## 🧩 Troubleshooting

If you encounter any issues or bugs, please create an issue on the [GitHub Issues page](https://github.com/Proenchen/stock-market-protocol/issues).
