# 大學專題 — 機器學習在量化交易的應用

## 研究流程

```mermaid
flowchart LR
    A[資料收集]-->B[計算技術指標]-->C[訓練模型]-->D[產生買賣訊號]-->E[回測]-->F[評估效益]
```

---

### 1. 資料收集

- 來源：Yahoo Finance (Python)
- 資料間隔：15 分鐘
- 標的
  - 台積電 (2330.TW)
  - 元大台灣50 (0050.TW)，為了比較穩定情況
  - 其他波動較大的個股 (未定)

### 2. 計算技術指標

- 移動平均 (MA)
  - 簡單移動平均 (SMA)

    ```math
    \begin{align*}
      &SMA_n=\frac{P_t+P_{t-1}+P_{t-2}+\cdots+P_{t-(n-1)}}{n}\\
      &n\text{: 天數}
    \end{align*}
    ```

  - 指數移動平均 (EMA)

    ```math
    \begin{align*}
      \alpha&=\frac{2}{n+1}\\
      EMA_n&=\alpha\times(P_t+(1-\alpha)P_{t-1}+(1-\alpha)^2P_{t-2}+\cdots+(1-\alpha)^{n-1}P_{t-(n-1)})\\
      n\text{: 天數}&
    \end{align*}
    ```

  - 加權移動平均 (WMA)

    ```math
    \begin{align*}
      &WMA_n=\frac{nP_t+(n-1)P_{t-1}+(n-2)P_{t-2}+\cdots+P_{t-(n-1)}}{n+(n-1)+(n-2)+\cdots+1}\\
      &n\text{: 天數}
    \end{align*}
    ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/移動平均>

- 指數平滑異同移動平均線 (MACD)

  ```math
  \begin{align*}
    &DIF=EMA_{(close,12)}-EMA_{(close,26)}\\
    &DIF\text{: 差離值 (快線)}\\
    &EMA_{(close,n)}\text{: n 日收盤價 EMA}\\
    \\
    &DEM=EMA_{(DIF,9)}\\
    &DEM\text{: 訊號線 (MACD、慢線)}\\
    \\
    &OSC=DIF-DEM\\
    &OSC\text{: MACD bar}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/指数平滑移动平均线>

- 隨機指標 (KD)

  ```math
  \begin{align*}
    &RSV=\frac{C_9-L_9}{H_9-L_9}\times100\%\\
    &RSV\text{: 未成熟隨機值，股價與過去 9 天比的強弱勢}\\
    &C_9\text{: 第 9 日收盤價}\\
    &L_9\text{: 9 日內最低價}\\
    &H_9\text{: 9 日內最高價}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    K_n&=\frac{1}{3}\times RSV_n+\frac{2}{3}\times K_{n-1}\\
    D_n&=\frac{1}{3}\times K_n+\frac{2}{3}\times D_{n-1}\\
    K_n&\text{: 快速平均值}\\
    D_n&\text{: 慢速平均值}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/随机指标>
  - <https://quantpass.org/kd/>

- 相對強弱指數 (RSI)

  ```math
  \begin{align*}
    &RSI=\frac{SMA_{(U,n)}}{SMA_{(U,n)}+SMA_{(D,n)}}\times100\%\\
    &SMA_{(U,n)}\text{: n 日內上漲日的SMA}\\
    &SMA_{(D,n)}\text{: n 日內下跌日的SMA}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/相對強弱指數>

- 布林通道 (Bollinger Band)

  ```math
  \begin{align*}
    中軌&=SMA_{20}\\
    上軌&=中軌+2\times\sigma_{20}\\
    下軌&=中軌-2\times\sigma_{20}\\
    \sigma_{20}&\text{: 20 日收盤價標準差}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    &PB=\frac{收盤價-下軌}{上軌-下軌}\\
    &PB\text{: \%b指標}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    &BW=\frac{上軌-下軌}{中軌}\\
    &BW\text{: 帶寬指標}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/布林带>
  - <https://www.oanda.com/bvi-ft/lab-education/technical_analysis/bollinger_bands/>

- 乖離率 (BIAS)

  ```math
  \begin{align*}
    &BIAS_n=\frac{Close-MA_n}{MA_n}\times100\%\\
    &n\text{: 天數}\\
    &Close\text{: 今日收盤價}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/乖離率>
  - <https://quantpass.org/bias-2/>

- 動向指數 (DMI)

  ```math
  \begin{cases}
    +DM=H_t-H_{t-1}&,\;\text{if}\;H_t-H_{t-1}>0\;\text{and}\;H_t-H_{t-1}>L_{t-1}-L_t\\
    -DM=L_{t-1}-L_t&,\;\text{if}\;L_{t-1}-L_t>0\;\text{and}\;H_t-H_{t-1}\lt L_{t-1}-L_t\\
    DM=0&,\;\text{otherwise}
  \end{cases}
  ```

  ```math
  \begin{align*}
    &DM\text{: 股價趨勢}\\
    &H_n\text{: 第 n 日最高價}\\
    &L_n\text{: 第 n 日最低價}
  \end{align*}
  ```

  ```math
  TR=\begin{cases}
    H_t-L_t&,\;\text{if}\;H_t>C_{t-1}>L_t\\
    H_t-C_{t-1}&,\;\text{if}\;H_t>L_t>C_{t-1}\\
    C_{t-1}-L_t&,\;\text{if}\;C_{t-1}>H_t>L_t
  \end{cases}
  ```

  ```math
  TR\text{: 真實波幅}\\
  \begin{align*}\\
    +DM14_t&=\frac{+DM14_{t-1}\times13}{14}+\frac{+DM14_{new}\times1}{14}\\
    TR14_t&=\frac{TR14_{t-1}\times13}{14}+\frac{TR14_{new}\times1}{14}\\
    -DM14_t&=\frac{-DM14_{t-1}\times13}{14}+\frac{-DM14_{new}\times1}{14}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    +DI14&=\frac{+DM14}{TR14}\times100\%\\
    -DI14&=\frac{-DM14}{TR14}\times100\%\\
    DI&\text{: 方向指標}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    &DX=\frac{\lvert(+DI14)-(-DI14)\rvert}{\lvert(+DI14)+(-DI14)\rvert}\\
    &DX\text{: 趨向指數}
  \end{align*}
  ```

  ```math
  \begin{align*}\\
    &ADX=SMA_{(DX,14)}\\
    &ADX\text{: 平均趨向指數}
  \end{align*}
  ```

  來源：
  - <https://zh.wikipedia.org/zh-tw/動向指數>
  - <https://www.thinkmarkets.com/tw/trading-academy/tech-indicators/dmi/>

### 3. 訓練模型

- 期間：1 年 (2024 年)
- 模型：
  - 預計使用 DQN 模型用於決策
  - LSTM 用於預測股價 (若有)

### 4. 產生買賣訊號

- 使用 DQN 模型作出買賣訊號決策

### 5. 回測

- 期間：1 年 (2025 年)

### 6. 評估效益

- 比較標的：
  1. 買進並持續持倉
  2. 傳統交易訊號 (MA 交叉)
- 指標：
  1. 回報率
  2. 最大回撤
  3. 夏普比率

---

## References

1. A. Keesari, "AI-Driven Algorithmic Trading: Integrating Machine Learning, Hybrid Technical Indicators, and Risk Management for Momentum Strategies," _International Journal of Engineering and Computer Science_, vol. 14, no. 11, pp. 27958-27970, 2025. <https://doi.org/10.18535/ijecs.v14i11.5335> (<https://www.ijecs.in/index.php/ijecs/article/view/5335>)
2. J. Guo, "Applications of machine learning in quantitative trading," _ACE_, vol. 82, 2024. <https://doi.org/10.54254/2755-2721/82/20240984>
3. Y. Duan, X.M. Gu, T. Lei, "Application of machine learning in quantitative timing model based on factor stock selection," _Electronic Research Archive_, vol. 14, no. 1, pp. 174-192, 2024. <https://doi.org/10.3934/era.2024009>
