library(dplyr)
library(simhelpers)
library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
library(tidyr)

dados_preditor0 <- data.frame()
dados_preditor1 <- data.frame()
dados_aux2_0 <- data.frame()
dados_aux3_0 <- data.frame()
dados_aux2_1 <- data.frame()
dados_aux3_1 <- data.frame()

for (p_cens in c(0.25, 0.4, 0.5, 0.6, 0.7)) {
  for (tempo in c(1, 3, 5)) {
    for (n in c(500, 1000, 2500, 5000)) {
      dados_aux <- read.csv(glue::glue("resultados2/df_predicoes_{p_cens}_{tempo}_{n}.csv")) |>
        pivot_longer(
          cols = starts_with("predicao"),
          names_to = "modelo",
          values_to = "predicao"
        ) |>
        mutate(
          p_cens = p_cens,
          tempo = tempo,
          n = n,
          modelo = ifelse(modelo == "predicao_xgboost", "xgboost", "rsf"),
          .before = "preditor"
        )
      
      dados_aux1_0 <- dados_aux |>
        filter(preditor == 0) 
      
      dados_aux1_1 <- dados_aux |>
        filter(preditor == 1)
      
      dados_aux2_0 <- bind_rows(dados_aux2_0, dados_aux1_0)
      dados_aux2_1 <- bind_rows(dados_aux2_1, dados_aux1_1)
    }
    dados_aux3_0 <- bind_rows(dados_aux3_0, dados_aux2_0)
    dados_aux3_1 <- bind_rows(dados_aux3_1, dados_aux2_1)
  }
  dados_preditor0 <- bind_rows(dados_preditor1, dados_aux3_0)
  dados_preditor1 <- bind_rows(dados_preditor1, dados_aux3_1)
}

rm(list = ls()[grepl("dados_aux", ls())])

# Plotando e salvando os gráficos
iteracao <- 1
for (preditor in c(0, 1)) {
  for (tempo_interesse in c(1, 3, 5)) {
    dados_plot <- get(paste0("dados_preditor", preditor)) |>
      select(p_cens, tempo, n, modelo, predicao, prob_teorica) |>
      filter(tempo == tempo_interesse) |>
      mutate(
        p_cens = factor(paste("Cens. =", p_cens), levels = c("Cens. = 0.25", "Cens. = 0.4", "Cens. = 0.5", "Cens. = 0.6", "Cens. = 0.7")),
        n = factor(paste("n =", n), levels = c("n = 500", "n = 1000", "n = 2500", "n = 5000"))
      )
    
    plot <- dados_plot |>
      ggplot(aes(x = modelo, y = predicao, fill = modelo)) + 
      geom_hline(yintercept = dados_plot$prob_teorica[1], linetype = "dashed") + 
      geom_boxplot(alpha = .7) + 
      facet_grid(p_cens ~ n, scales = "fixed") + 
      labs(x = "Modelo", y = "Predições", fill = "Modelo", caption = glue::glue("FIGURA {iteracao}. Boxplots das predições obtidas quando o tempo de interesse era de {tempo_interesse} ano{ifelse(tempo_interesse == 1, '', 's')} \ne o valor do preditor era {preditor}. As linhas pontilhadas representam o valor teórico da \nprobabilidade de sobrevivência.")) + 
      theme_bw() +
      theme(legend.position = "top",
            text = element_text(size = 17)) +
      theme(plot.caption=element_text(hjust = 0, size = 16)) 
    
    ggsave(
      glue::glue("figuras/plot_preditor{preditor}_tempo{tempo_interesse}.png"), plot, width = 10, height = 8, units = "in", 
      dpi = 300
    )
    
    iteracao <- iteracao + 1
    
  }
}
