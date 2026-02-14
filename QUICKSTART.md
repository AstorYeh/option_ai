# 台指期選擇權預測系統 - 快速開始指南

## 📋 前置需求

- Python 3.10 或以上
- Git
- Ollama (Local LLM)

## 🚀 5 分鐘快速啟動

### 步驟 1: 安裝依賴套件
```bash
cd D:\option
pip install -r requirements.txt
```

### 步驟 2: 設定環境變數
已自動建立 `.env` 檔案,請編輯以下設定:

```env
# FinMind API (必填)
FINMIND_API_TOKEN=your_token_here

# Discord Webhook (選填)
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Ollama (預設值通常不需修改)
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

**取得 FinMind API Token**:
1. 前往 https://finmindtrade.com/
2. 註冊帳號
3. 登入後在個人設定中取得 API Token

### 步驟 3: 啟動 Ollama
```bash
# 啟動 Ollama 服務
ollama serve

# 下載模型(另開一個終端)
ollama pull qwen2.5:3b
```

### 步驟 4: 初始化資料庫
```bash
python scripts\init_database.py
```

### 步驟 5: 下載歷史資料
```bash
# 下載最近 1 年資料
python scripts\daily_update.py --initial --days 365
```

### 步驟 6: 測試連線
```bash
python scripts\test_connection.py
```

### 步驟 7: 啟動 Web 介面
```bash
streamlit run streamlit_app\Home.py
```

瀏覽器會自動開啟 http://localhost:8501

## 📊 使用流程

### 每日操作
1. **收盤後更新資料** (約 15:30 執行)
   ```bash
   python scripts\daily_update.py
   ```

2. **查看預測結果**
   - 開啟 Streamlit 介面
   - 前往「📊 即時預測」頁面

3. **發送通知** (選填)
   - 在即時預測頁面啟用「發送 Discord 通知」

## ⚙️ 系統設定

在 Streamlit 介面的「⚙️ 系統設定」頁面可以:
- 測試 API 連線
- 設定 Discord Webhook
- 調整風險管理參數

## ❓ 常見問題

### Q: 安裝套件時出現錯誤?
A: 某些套件(如 ta-lib)需要編譯,可能需要安裝 Visual Studio Build Tools

### Q: FinMind API 連線失敗?
A: 請確認:
1. API Token 是否正確填入 `.env`
2. 網路連線是否正常
3. 是否超過免費版 API 限制(600次/小時)

### Q: Ollama 連線失敗?
A: 請確認:
1. Ollama 服務是否運行中 (`ollama serve`)
2. 模型是否已下載 (`ollama pull qwen2.5:3b`)
3. API URL 是否正確(預設 http://localhost:11434)

### Q: Discord 通知未收到?
A: 請確認:
1. Webhook URL 是否正確
2. 在系統設定頁面測試連線
3. 檢查 Discord 伺服器權限

## 📝 下一步

系統啟動後,建議:
1. 先查看主頁的市場數據
2. 前往即時預測頁面體驗 AI 預測
3. 在系統設定頁面測試所有連線
4. 設定 Discord 通知(選填)

## ⚠️ 重要提醒

- 本系統僅供學習研究,不構成投資建議
- 選擇權交易具有高風險
- 請做好風險管理

---

**祝您使用愉快! 🚀**
