# 台指期選擇權買方策略預測系統

## 🎯 專案簡介

本系統專注於 **Buy Call** 和 **Buy Put** 策略的選擇權預測,整合 AI 預測、回測驗證與即時通知功能。

**核心特色**:
- 🤖 AI 預測引擎 (XGBoost + LSTM + LLM)
- 📊 完整回測系統
- 🌐 Streamlit Web 介面
- 📈 22 個技術指標
- 🔔 Discord 通知

**開發進度**: 約 90% 完成

---

## 🚀 快速開始

### 1. 環境設定

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入 FinMind API Token
```

### 2. 初始化資料庫

```bash
# 建立資料庫
python scripts/init_database.py

# 下載歷史資料
python scripts/daily_update.py
```

### 3. 訓練 AI 模型

```bash
# 訓練所有模型
python scripts/train_models.py --all

# 或分別訓練
python scripts/train_models.py --direction   # 方向性預測
python scripts/train_models.py --volatility  # 波動率預測
python scripts/train_models.py --ensemble    # 測試集成系統
```

### 4. 執行回測

```bash
# 執行回測測試
python scripts/run_backtest.py
```

### 5. 啟動 Web 介面

```bash
# 啟動 Streamlit
streamlit run streamlit_app/app.py

# 訪問 http://localhost:8501
```

---

## 📊 系統功能

### AI 預測引擎

- **方向性預測** (XGBoost)
  - 測試集準確率: 54.81%
  - 三分類: 漲/跌/盤整
  - 13 個技術指標特徵

- **波動率預測** (移動平均)
  - 預測未來波動率
  - 輔助選擇權定價

- **集成預測系統**
  - 整合 AI + LLM 建議
  - 綜合信心度評分
  - 風險等級評估

### 回測系統

- 歷史資料回測
- 績效指標:
  - 勝率
  - 總報酬率
  - 最大回撤
  - Sharpe Ratio
  - Profit Factor
- 自動評級系統 (⭐~⭐⭐⭐⭐⭐)

### Streamlit Web 介面

1. **📊 首頁儀表板** - 系統狀態總覽
2. **🔮 即時預測** - AI 預測與 LLM 建議
3. **📈 回測分析** - 歷史績效驗證
4. **📉 績效追蹤** - 交易表現監控
5. **⚙️ 系統設定** - 參數調整

---

## 📁 專案結構

```
option/
├── config/              # 配置檔案
│   ├── api_config.py   # API 設定
│   ├── model_config.py # 模型參數
│   └── settings.py     # 系統設定
├── data/               # 資料目錄
│   └── database/       # SQLite 資料庫
├── models/             # 訓練好的模型
├── scripts/            # 執行腳本
│   ├── init_database.py
│   ├── daily_update.py
│   ├── train_models.py
│   └── run_backtest.py
├── src/                # 原始碼
│   ├── backtest/       # 回測引擎
│   ├── data/           # 資料處理
│   ├── features/       # 特徵工程
│   ├── models/         # AI 模型
│   ├── notifications/  # 通知系統
│   └── utils/          # 工具函數
├── streamlit_app/      # Web 介面
│   ├── app.py          # 主程式
│   └── pages/          # 各頁面
└── logs/               # 日誌檔案
```

---

## 🔧 技術棧

- **後端**: Python 3.10+
- **資料來源**: FinMind API
- **機器學習**: scikit-learn, XGBoost
- **LLM**: Ollama (Qwen2.5:3B)
- **Web 框架**: Streamlit
- **資料庫**: SQLite
- **視覺化**: Plotly
- **通知**: Discord Webhook

---

## 📝 使用說明

### 每日更新資料

```bash
# 手動更新
python scripts/daily_update.py

# 或設定排程 (Windows)
# 使用工作排程器執行 daily_update.py
```

### 獲取即時預測

1. 訪問 http://localhost:8501
2. 點選「🔮 即時預測」
3. 查看 AI 預測結果與 LLM 建議

### 執行回測驗證

1. 訪問 http://localhost:8501
2. 點選「📈 回測分析」
3. 設定回測參數
4. 點擊「🚀 執行回測」
5. 查看績效報告

---

## ⚠️ 注意事項

1. **免責聲明**
   - 本系統僅供學習與研究使用
   - 預測結果不構成投資建議
   - 實際交易請自行評估風險

2. **資料限制**
   - 使用 FinMind 免費版 API (600 次/小時)
   - 歷史資料約 90 天
   - 建議累積更多資料以提升準確度

3. **模型限制**
   - 測試集準確率 54.81% (三分類)
   - 存在過擬合現象
   - 需持續優化與驗證

---

## 📈 績效指標

### AI 模型表現

- **方向性預測**
  - 訓練集: 97.36%
  - 測試集: 54.81%
  - 特徵數: 13 個

- **重要特徵**
  1. high (16.51%)
  2. low (14.37%)
  3. volume (13.00%)

### 回測結果

執行 `python scripts/run_backtest.py` 查看完整回測報告

---

## 🔜 未來規劃

1. ⏭️ 每日市場分析報告
2. ⏭️ 策略參數自動優化
3. ⏭️ 更多技術指標
4. ⏭️ 實時推播通知
5. ⏭️ 模型持續優化

---

## 📚 相關文件

- [FinMind 免費版使用指南](FINMIND_FREE_TIER.md)
- [開發任務清單](C:\Users\jayye\.gemini\antigravity\brain\be637411-b071-4997-be2b-26b671ca34a3\task.md)
- [完整開發記錄](C:\Users\jayye\.gemini\antigravity\brain\be637411-b071-4997-be2b-26b671ca34a3\walkthrough.md)

---

## 📧 聯絡資訊

如有問題或建議,歡迎提出 Issue 或 Pull Request!

**最後更新**: 2026-02-14
**版本**: 1.0.0
**狀態**: ✅ 可用 (90% 完成)
