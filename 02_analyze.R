# ============================================================
# 02_analyze.R（完整版 v3）
# 三种分析方法 + 描述统计 + TOPSIS
# 输出文件：
#   data/summary_stats.csv
#   data/method_comparison.csv
#   data/method_comparison_plot.png
#   data/main_effects.png
#   data/topsis_result.csv
#   data/topsis_ranking.png
# ============================================================

library(tidyverse)
library(car)
library(gridExtra)
library(DiceKriging)
library(lmtest)
library(Metrics)

# ════════════════════════════════════════
# 0. 读取数据
# ════════════════════════════════════════
df <- read.csv("data/results.csv", fileEncoding = "UTF-8")

df <- df %>%
  mutate(
    A_code = case_when(
      A_quant == "Q6K"  ~ 1,
      A_quant == "Q4KM" ~ 2,
      A_quant == "Q2K"  ~ 3
    ),
    B_code = B_batch,
    C_code = C_cpu_pct,
    D_code = D_mem_gb,
    A_f = factor(A_quant,   levels = c("Q6K", "Q4KM", "Q2K")),
    B_f = factor(B_batch,   levels = c(1, 2, 4)),
    C_f = factor(C_cpu_pct, levels = c(10, 40, 70)),
    D_f = factor(D_mem_gb,  levels = c(0, 2, 4))
  )

df9 <- df %>%
  group_by(combo, A_code, B_code, C_code, D_code,
           A_quant, B_batch, C_cpu_pct, D_mem_gb,
           A_f, B_f, C_f, D_f) %>%
  summarise(
    y1_mean = mean(y1_latency_ms),
    y1_sd   = sd(y1_latency_ms),
    y2_cv   = first(y2_cv_pct),
    .groups = "drop"
  )

cat("数据加载完成：", nrow(df), "条原始记录，",
    nrow(df9), "个组合均值\n")

# ════════════════════════════════════════
# 1. 描述统计 summary_stats.csv
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  描述统计\n")
cat(strrep("=", 50), "\n")

summary_stats <- df %>%
  group_by(combo, A_quant, B_batch, C_cpu_pct, D_mem_gb) %>%
  summarise(
    n         = n(),
    mean_ms   = round(mean(y1_latency_ms),              2),
    sd_ms     = round(sd(y1_latency_ms),                2),
    cv_pct    = round(sd(y1_latency_ms) /
                        mean(y1_latency_ms) * 100,         4),
    min_ms    = round(min(y1_latency_ms),                2),
    max_ms    = round(max(y1_latency_ms),                2),
    median_ms = round(median(y1_latency_ms),             2),
    q25_ms    = round(quantile(y1_latency_ms, 0.25),     2),
    q75_ms    = round(quantile(y1_latency_ms, 0.75),     2),
    .groups   = "drop"
  )

write.csv(summary_stats,
          "data/summary_stats.csv",
          row.names    = FALSE,
          fileEncoding = "UTF-8")

cat("✓ summary_stats.csv 已生成\n")
print(summary_stats %>%
        select(combo, A_quant, B_batch,
               C_cpu_pct, D_mem_gb,
               mean_ms, sd_ms, cv_pct))

# ════════════════════════════════════════
# 2. 多因素ANOVA
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  方法1：多因素ANOVA\n")
cat(strrep("=", 50), "\n")

model_anova   <- aov(y1_latency_ms ~ A_f + B_f + C_f + D_f,
                     data = df)
anova_summary <- summary(model_anova)
print(anova_summary)

ss_table  <- anova_summary[[1]]
ss_vals   <- ss_table$`Sum Sq`
eta2_vals <- ss_vals / sum(ss_vals)
eta2_df   <- data.frame(
  Factor = c("A_量化精度", "B_Batch",
             "C_CPU负载",  "D_内存压力", "残差"),
  SS     = round(ss_vals,   2),
  eta2   = round(eta2_vals, 4)
)
cat("\nη²效应量：\n")
print(eta2_df)

cat("\nTukey HSD事后检验（因子A）：\n")
print(TukeyHSD(model_anova, "A_f"))

# ANOVA LOO
anova_pred <- numeric(9)
for (i in 1:9) {
  train          <- df[df$combo != i, ]
  test           <- df[df$combo == i, ]
  m_loo          <- aov(y1_latency_ms ~ A_f + B_f + C_f + D_f,
                        data = train)
  anova_pred[i]  <- mean(suppressWarnings(
    predict(m_loo, newdata = test)))
}
anova_rmse <- rmse(df9$y1_mean, anova_pred)
anova_mae  <- mae(df9$y1_mean,  anova_pred)
cat(sprintf("\nANOVA LOO-RMSE: %.2f ms | LOO-MAE: %.2f ms\n",
            anova_rmse, anova_mae))

# ════════════════════════════════════════
# 3. 多元线性回归
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  方法2：多元线性回归\n")
cat(strrep("=", 50), "\n")

df9_scaled <- df9 %>%
  mutate(
    A_s = scale(A_code)[, 1],
    B_s = scale(B_code)[, 1],
    C_s = scale(C_code)[, 1],
    D_s = scale(D_code)[, 1]
  )

model_lm   <- lm(y1_mean ~ A_s + B_s + C_s + D_s,
                 data = df9_scaled)
lm_summary <- summary(model_lm)
print(lm_summary)

cat(sprintf("R² = %.4f | 调整R² = %.4f\n",
            lm_summary$r.squared,
            lm_summary$adj.r.squared))

coef_df <- data.frame(
  Factor    = c("截距", "A_量化精度", "B_Batch",
                "C_CPU负载", "D_内存压力"),
  Coeff     = round(coef(model_lm),                        3),
  Std_Error = round(lm_summary$coefficients[, 2],          3),
  p_value   = round(lm_summary$coefficients[, 4],          4)
)
cat("\n标准化回归系数：\n")
print(coef_df)

cat("\nDurbin-Watson检验：\n")
print(dwtest(model_lm))
cat("\nBreusch-Pagan检验：\n")
print(bptest(model_lm))

# 线性回归 LOO（数值编码，避免秩亏）
lm_pred <- numeric(9)
for (i in 1:9) {
  train_num <- df9[-i, ] %>%
    mutate(
      A_s = scale(A_code)[, 1],
      B_s = scale(B_code)[, 1],
      C_s = scale(C_code)[, 1],
      D_s = scale(D_code)[, 1]
    )
  # 保存scale中心和尺度
  a_center <- mean(df9[-i, ]$A_code)
  b_center <- mean(df9[-i, ]$B_code)
  c_center <- mean(df9[-i, ]$C_code)
  d_center <- mean(df9[-i, ]$D_code)
  a_scale  <- sd(df9[-i, ]$A_code)
  b_scale  <- sd(df9[-i, ]$B_code)
  c_scale  <- sd(df9[-i, ]$C_code)
  d_scale  <- sd(df9[-i, ]$D_code)
  
  test_num <- df9[i, ] %>%
    mutate(
      A_s = (A_code - a_center) / a_scale,
      B_s = (B_code - b_center) / b_scale,
      C_s = (C_code - c_center) / c_scale,
      D_s = (D_code - d_center) / d_scale
    )
  
  m_loo       <- lm(y1_mean ~ A_s + B_s + C_s + D_s,
                    data = train_num)
  lm_pred[i]  <- predict(m_loo, newdata = test_num)
}
lm_rmse <- rmse(df9$y1_mean, lm_pred)
lm_mae  <- mae(df9$y1_mean,  lm_pred)
cat(sprintf("\n线性回归 LOO-RMSE: %.2f ms | LOO-MAE: %.2f ms\n",
            lm_rmse, lm_mae))

b <- coef(model_lm)
cat(sprintf(
  "\n预测方程：延迟(ms) = %.1f + %.1f×A' + %.1f×B' + %.1f×C' + %.1f×D'\n",
  b[1], b[2], b[3], b[4], b[5]))

# ════════════════════════════════════════
# 4. 克里金代理模型
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  方法3：克里金代理模型\n")
cat(strrep("=", 50), "\n")

X_raw  <- df9[, c("A_code", "B_code", "C_code", "D_code")]
X_norm <- as.data.frame(lapply(X_raw, function(x) {
  (x - min(x)) / (max(x) - min(x))
}))
colnames(X_norm) <- c("A", "B", "C", "D")
y_kriging <- df9$y1_mean

km_model <- km(
  formula      = ~1,
  design       = X_norm,
  response     = y_kriging,
  covtype      = "matern5_2",
  nugget.estim = TRUE,
  optim.method = "BFGS"
)
cat("\n克里金模型摘要：\n")
print(km_model)

kriging_pred <- numeric(9)
kriging_sd   <- numeric(9)
for (i in 1:9) {
  X_train <- X_norm[-i, ]
  y_train <- y_kriging[-i]
  X_test  <- X_norm[i, , drop = FALSE]
  
  km_loo <- km(
    formula      = ~1,
    design       = X_train,
    response     = y_train,
    covtype      = "matern5_2",
    nugget.estim = TRUE,
    optim.method = "BFGS",
    control      = list(trace = FALSE)
  )
  pred            <- predict(km_loo, newdata = X_test,
                             type = "UK")
  kriging_pred[i] <- pred$mean
  kriging_sd[i]   <- pred$sd
}
kriging_rmse <- rmse(y_kriging, kriging_pred)
kriging_mae  <- mae(y_kriging,  kriging_pred)
cat(sprintf("\n克里金 LOO-RMSE: %.2f ms | LOO-MAE: %.2f ms\n",
            kriging_rmse, kriging_mae))

cat("\n克里金插值示例：\n")
new_points <- data.frame(
  A = c(0.5, 0.0, 1.0),
  B = c(0.5, 1.0, 0.0),
  C = c(0.5, 0.0, 1.0),
  D = c(0.5, 0.5, 0.5)
)
new_pred <- predict(km_model, newdata = new_points, type = "UK")
print(data.frame(
  new_points,
  预测延迟_ms = round(new_pred$mean,                        1),
  预测标准差  = round(new_pred$sd,                          1),
  CI_下       = round(new_pred$mean - 1.96 * new_pred$sd,   1),
  CI_上       = round(new_pred$mean + 1.96 * new_pred$sd,   1)
))

# ════════════════════════════════════════
# 5. 三方法比较汇总
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  三种方法比较汇总\n")
cat(strrep("=", 50), "\n")

comparison <- data.frame(
  方法             = c("ANOVA", "线性回归", "克里金"),
  LOO_RMSE_ms      = round(c(anova_rmse, lm_rmse, kriging_rmse), 2),
  LOO_MAE_ms       = round(c(anova_mae,  lm_mae,  kriging_mae),  2),
  能否预测新点     = c("否", "是", "是"),
  能否量化不确定性 = c("否", "部分", "是"),
  适合样本量       = c("大样本", "中等", "小样本"),
  主要优势         = c("显著性检验严格",
                   "系数直观可解释",
                   "小样本插值精度高")
)
print(comparison)

write.csv(comparison,
          "data/method_comparison.csv",
          row.names    = FALSE,
          fileEncoding = "UTF-8")

# ════════════════════════════════════════
# 6. TOPSIS 综合评分
# ════════════════════════════════════════
cat("\n", strrep("=", 50), "\n")
cat("  TOPSIS 综合评分\n")
cat(strrep("=", 50), "\n")

topsis_input <- summary_stats %>%
  select(combo, A_quant, B_batch,
         C_cpu_pct, D_mem_gb,
         mean_ms, cv_pct)

X_t <- as.matrix(topsis_input[, c("mean_ms", "cv_pct")])

# 向量归一化
norm_t <- X_t
for (j in 1:ncol(X_t)) {
  norm_t[, j] <- X_t[, j] / sqrt(sum(X_t[, j]^2))
}

# 加权（延迟0.6，稳定性0.4）
W   <- c(0.6, 0.4)
V_t <- norm_t
for (j in 1:ncol(V_t)) {
  V_t[, j] <- norm_t[, j] * W[j]
}

# 正/负理想解（成本型）
PIS <- apply(V_t, 2, min)
NIS <- apply(V_t, 2, max)

# 距离
d_pos <- apply(V_t, 1, function(r) sqrt(sum((r - PIS)^2)))
d_neg <- apply(V_t, 1, function(r) sqrt(sum((r - NIS)^2)))

# 贴近度
score <- d_neg / (d_pos + d_neg)

topsis_result <- topsis_input %>%
  mutate(
    d_positive   = round(d_pos,  6),
    d_negative   = round(d_neg,  6),
    topsis_score = round(score,  4),
    rank         = rank(-score, ties.method = "first")
  ) %>%
  arrange(rank)

write.csv(topsis_result,
          "data/topsis_result.csv",
          row.names    = FALSE,
          fileEncoding = "UTF-8")

cat("✓ topsis_result.csv 已生成\n\n")
print(topsis_result %>%
        select(rank, combo, A_quant, B_batch,
               C_cpu_pct, D_mem_gb,
               mean_ms, cv_pct, topsis_score))

best <- topsis_result %>% filter(rank == 1)
cat("\n============================================\n")
cat("  最优参数组合（TOPSIS第1名）\n")
cat("============================================\n")
cat(sprintf("  量化精度 : %s\n",      best$A_quant))
cat(sprintf("  Batch大小: %d\n",      best$B_batch))
cat(sprintf("  CPU负载  : %d%%\n",    best$C_cpu_pct))
cat(sprintf("  内存压力 : %d GB\n",   best$D_mem_gb))
cat(sprintf("  均值延迟 : %.1f ms\n", best$mean_ms))
cat(sprintf("  CV       : %.2f%%\n",  best$cv_pct))
cat(sprintf("  TOPSIS   : %.4f\n",    best$topsis_score))
cat("============================================\n")

# ════════════════════════════════════════
# 7. 可视化
# ════════════════════════════════════════

# ── 7.1 三方法预测 vs 实测 ──
plot_data <- data.frame(
  实测值   = df9$y1_mean,
  ANOVA    = anova_pred,
  线性回归 = lm_pred,
  克里金   = kriging_pred
) %>%
  pivot_longer(cols      = c(ANOVA, 线性回归, 克里金),
               names_to  = "方法",
               values_to = "预测值")

p_compare <- ggplot(plot_data,
                    aes(x = 实测值, y = 预测值,
                        color = 方法, shape = 方法)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray60") +
  geom_point(size = 3.5, alpha = 0.85) +
  labs(title    = "三种方法 LOO 预测值 vs 实测值",
       subtitle = sprintf(
         "RMSE — ANOVA: %.0fms | 线性回归: %.0fms | 克里金: %.0fms",
         anova_rmse, lm_rmse, kriging_rmse),
       x = "实测延迟均值 (ms)",
       y = "LOO 预测延迟 (ms)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

png("data/method_comparison_plot.png",
    width = 700, height = 600, res = 120)
print(p_compare)
dev.off()

# ── 7.2 主效应图 ──
me_A <- df %>% group_by(A_f) %>%
  summarise(m = mean(y1_latency_ms), .groups = "drop")
me_B <- df %>% group_by(B_f) %>%
  summarise(m = mean(y1_latency_ms), .groups = "drop")
me_C <- df %>% group_by(C_f) %>%
  summarise(m = mean(y1_latency_ms), .groups = "drop")
me_D <- df %>% group_by(D_f) %>%
  summarise(m = mean(y1_latency_ms), .groups = "drop")

p_A <- ggplot(me_A, aes(x = A_f, y = m, group = 1)) +
  geom_line(color = "#2196F3", linewidth = 1) +
  geom_point(size = 3, color = "#2196F3") +
  labs(title = "主效应：量化精度",
       x = "量化精度", y = "平均延迟 (ms)") +
  theme_minimal()

p_B <- ggplot(me_B, aes(x = B_f, y = m, group = 1)) +
  geom_line(color = "#4CAF50", linewidth = 1) +
  geom_point(size = 3, color = "#4CAF50") +
  labs(title = "主效应：Batch",
       x = "Batch大小", y = "平均延迟 (ms)") +
  theme_minimal()

p_C <- ggplot(me_C, aes(x = C_f, y = m, group = 1)) +
  geom_line(color = "#FF9800", linewidth = 1) +
  geom_point(size = 3, color = "#FF9800") +
  labs(title = "主效应：CPU负载",
       x = "CPU负载 (%)", y = "平均延迟 (ms)") +
  theme_minimal()

p_D <- ggplot(me_D, aes(x = D_f, y = m, group = 1)) +
  geom_line(color = "#9C27B0", linewidth = 1) +
  geom_point(size = 3, color = "#9C27B0") +
  labs(title = "主效应：内存压力",
       x = "内存压力 (GB)", y = "平均延迟 (ms)") +
  theme_minimal()

png("data/main_effects.png",
    width = 1200, height = 800, res = 120)
grid.arrange(p_A, p_B, p_C, p_D, ncol = 2)
dev.off()

# ── 7.3 TOPSIS 排名图 ──
topsis_plot <- topsis_result %>%
  mutate(label = paste0(A_quant,
                        "  B=", B_batch,
                        "  C=", C_cpu_pct, "%",
                        "  D=", D_mem_gb, "G"))

p_topsis <- ggplot(
  topsis_plot,
  aes(x    = reorder(label, topsis_score),
      y    = topsis_score,
      fill = topsis_score)
) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = sprintf("%.3f  #%d", topsis_score, rank)),
    hjust = -0.05, size = 3.5, color = "gray20"
  ) +
  scale_fill_gradient(low = "#FFB74D", high = "#1565C0") +
  scale_y_continuous(limits = c(0, 1.08)) +
  coord_flip() +
  labs(
    title    = "TOPSIS 综合评分排名",
    subtitle = "权重：延迟 × 0.6 + CV × 0.4，分数越高越优",
    x        = NULL,
    y        = "TOPSIS 贴近度得分"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position    = "none",
    plot.title         = element_text(face = "bold", size = 14),
    plot.subtitle      = element_text(color = "gray40", size = 10),
    panel.grid.major.y = element_blank()
  )

png("data/topsis_ranking.png",
    width = 950, height = 550, res = 130)
print(p_topsis)
dev.off()

# ════════════════════════════════════════
# 8. 文件清单确认
# ════════════════════════════════════════
files_all <- c(
  "data/results.csv",
  "data/summary_stats.csv",
  "data/method_comparison.csv",
  "data/topsis_result.csv",
  "data/method_comparison_plot.png",
  "data/main_effects.png",
  "data/topsis_ranking.png"
)

cat("\n", strrep("=", 50), "\n")
cat("  输出文件确认\n")
cat(strrep("=", 50), "\n")
for (f in files_all) {
  ok   <- file.exists(f)
  size <- if (ok) paste0(round(file.size(f) / 1024, 1), " KB")
  else "缺失！"
  cat(sprintf("  %s  %-45s  %s\n",
              if (ok) "✓" else "✗", f, size))
}
cat(strrep("=", 50), "\n")
cat("全部完成\n")