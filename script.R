options(scipen = 999)
library(dplyr)
library(caret)
library(survival)
library(randomForestSRC)
library(glue)

# Importando os arquivos com funções auxiliares
source("funcao_gera_amostras.R")
source("funcao_tune_xgboost.R")

simulacao_predicoes_xgboost_rsf <- function(t1, t2, p1_1, p2_1, p1_2, p2_2, seed, vetor_p_cens, vetor_n, vetor_tempos_interesse, M, prop_n0, prop_n1) {
  # Criando um dataframe que irá guardar os vícios e EQMs médios
  df_estatisticas_medias <- data.frame()
  
  for (p_cens in vetor_p_cens) {
    
    for (tempo in vetor_tempos_interesse) {
      
      # Gerando uma amostra inicial para treinar os hiperparâmetros dos modelos
      amostras_aux <- gera_valores_censurados(
        t1 = t1, t2 = t2,
        p1_1 = p1_1, p2_1 = p2_1,
        p1_2 = p1_2, p2_2 = p2_2,
        p_cens = p_cens,
        n = 10000, prop_n0 = prop_n0, prop_n1 = prop_n1,
        seed = seed
      )
      
      # Salvando o dataframe de amostras e os parâmetros das variáveis Weibull utilizadas
      df_amostras <- amostras_aux$df_amostras
      params0 <- amostras_aux$params0
      params1 <- amostras_aux$params1
      
      # Criando um conjunto de treinamento 
      set.seed(seed)
      linhas_treino <- createDataPartition(
        df_amostras$falha,
        p = 0.7, 
        list = FALSE,
        times = 1
      )
      
      df_treino <- df_amostras[linhas_treino, ]
      
      # Para o treino/teste dos modelos de classificação usuais, retirando as observações
      # cujo tempo de falha seja censurado e menor que o dado tempo e criando a variável resposta
      df_treino_classificacao <- df_treino |>
        filter(!(t_obs < tempo & falha == 0)) |>
        mutate(
          !!paste0("sobrev_", tempo, "ano") := ifelse(t_obs > tempo, 1, 0)
        )
      
      # Garantindo que os dados estão no formato correto para as bibliotecas do Python
      X_train_classificacao <- df_treino_classificacao |> dplyr::select(preditor) |> as.matrix()
      y_train_classificacao <- df_treino_classificacao |> pull(starts_with("sobrev")) |> as.integer()
      
      # Tunando os hiperparâmetros do XGBoost
      print(glue("Tunando os hiperparâmetros do XGBoost para o cenário de p_cens = {p_cens} e tempo de interesse = {tempo}"))
      params_xgboost <- funcao_tune_xgboost(X_train = X_train_classificacao, y_train = y_train_classificacao, seed = as.integer(seed))
      ## Para a RSF, não faz sentido tunar os hiperparâmetros, pois só temos um preditor binário
      
      # Iniciando as simulações
      for (n in vetor_n) {
        
        #Criando um dataframe que irá receber predições
        df_predicoes <- data.frame()
        
        for (i in 1:M) {
          # Mudando o valor da semente
          seed <- seed + 1
          
          # Gerando uma amostra de tamanho n de (tempo de falha, indicador de falha, preditor)
          df_amostras <- gera_valores_censurados(
            t1 = t1, t2 = t2,
            p1_1 = p1_1, p2_1 = p2_1,
            p1_2 = p1_2, p2_2 = p2_2,
            p_cens = p_cens,
            n = n, prop_n0 = prop_n0, prop_n1 = prop_n1,
            seed = seed
          )$df_amostras
          
          # Criando conjunto de treinamento e teste
          set.seed(seed)
          linhas_treino <- createDataPartition(
            df_amostras$falha,
            p = 0.7, 
            list = FALSE,
            times = 1
          )
          
          df_treino <- df_amostras[linhas_treino, ]
          df_teste <- df_amostras[-linhas_treino, ]
          
          # Para o treino/teste dos modelos de classificação usuais, retirando as observações
          # cujo tempo de falha seja censurado e menor que o dado tempo e criando a variável resposta
          df_treino_classificacao <- df_treino |>
            filter(!(t_obs < tempo & falha == 0)) |>
            mutate(
              !!paste0("sobrev_", tempo, "ano") := ifelse(t_obs > tempo, 1, 0)
            )
          
          df_teste_classificacao <- df_teste |>
            filter(!(t_obs < tempo & falha == 0)) |>
            mutate(
              !!paste0("sobrev_", tempo, "ano") := ifelse(t_obs > tempo, 1, 0)
            )
          
          observacoes_excluidas_teste <- df_teste |>
            mutate(
              index = 1:nrow(df_teste),
              flag_exclusao = ifelse(t_obs < tempo & falha == 0, 1, 0)
            ) |>
            filter(flag_exclusao == 1) |>
            pull(index)
          
          # Garantindo que os dados estão no formato correto para as bibliotecas do Python
          X_train_classificacao <- df_treino_classificacao |> dplyr::select(preditor) |> as.matrix()
          y_train_classificacao <- df_treino_classificacao |> pull(starts_with("sobrev")) |> as.integer()
          
          X_test_classificacao <- df_teste_classificacao |> dplyr::select(preditor) |> as.matrix()
          y_test_classificacao <- df_teste_classificacao |> pull(starts_with("sobrev")) |> as.integer()
          
          # Treinando os modelos 
          ## Para o XGBoost
          params_xgboost['random_state'] <- as.integer(seed)
          params_xgboost['scale_pos_weight'] <- 0.3
          
          fit_xgboost <- xgb$XGBClassifier()
          do.call(fit_xgboost$set_params, params_xgboost)
          
          fit_xgboost$fit(X_train_classificacao, y_train_classificacao) 
          
          ## Para a RSF
          fit_rsf <- rfsrc(
            Surv(t_obs, falha) ~ preditor, data = df_treino, ntime = NULL, save.memory = TRUE
          )
          
          # Fazendo as predições
          ## Para o XGBoost (salvando apenas as predições para a classe 1 da resposta)
          predicoes_xgboost <- fit_xgboost$predict_proba(X_test_classificacao)[, 1]
          
          ## Para a RSF
          predicoes_rsf <- predict.rfsrc(fit_rsf, df_teste)
          probs_rsf_tempo_interesse <- predicoes_rsf$survival[, which.min(abs(predicoes_rsf$time.interest[which(predicoes_rsf$time.interest <= tempo)] - tempo))]
          
          ## Calculando as probabilidades de sobrevivência teóricas para cada grupo
          prob_teorica0 <- pweibull(tempo, shape = params0["shape"], scale = params0["scale"], lower.tail = FALSE)
          prob_teorica1 <- pweibull(tempo, shape = params1["shape"], scale = params1["scale"], lower.tail = FALSE)
          
          # Criando um dataframe com as probabilidades estimadas e teóricas de cada observação no conjunto de teste
          df_predicoes_aux <- data.frame(
            index = 1:nrow(df_teste),
            preditor = df_teste$preditor
          ) |>
            mutate(
              predicao_xgboost = ifelse(index %in% observacoes_excluidas_teste, NA, predicoes_xgboost),
              predicao_rsf = probs_rsf_tempo_interesse,
              prob_teorica = ifelse(preditor == 0, prob_teorica0, prob_teorica1)
            ) |>
            select(!index)
          
          df_predicoes <- bind_rows(df_predicoes, df_predicoes_aux)
          
          print(paste(p_cens, tempo, n, i))
          
          write.csv(df_predicoes, glue("resultados2/df_predicoes_{p_cens}_{tempo}_{n}.csv"), row.names = FALSE)
        }
        
        # Calculando os vícios e EQMs ao final das M simulações
        df_estatisticas_medias_aux <- data.frame(
          p_cens = p_cens,
          tempo_anos = tempo,
          n = n,
          vicio_medio_xgboost = as.numeric(round(mean(df_predicoes$predicao_xgboost - df_predicoes$prob_teorica, na.rm = TRUE), 3)),
          vicio_medio_rsf = as.numeric(round(mean(df_predicoes$predicao_rsf - df_predicoes$prob_teorica, na.rm = TRUE), 3)),
          eqm_xgboost = as.numeric(round(sum((df_predicoes$predicao_xgboost - df_predicoes$prob_teorica)^2, na.rm = TRUE)/M, 3)),
          eqm_rsf = as.numeric(round(sum((df_predicoes$predicao_rsf - df_predicoes$prob_teorica)^2, na.rm = TRUE)/M, 3))
        )
        
        # Juntando com o restante dos resultados
        df_estatisticas_medias <- bind_rows(df_estatisticas_medias, df_estatisticas_medias_aux)
        
        write.csv(df_estatisticas_medias, "resultados2/resultados_finais.csv", row.names = FALSE)
      }
    }
  }
  return(df_estatisticas_medias)
}

resultados <- simulacao_predicoes_xgboost_rsf(
  t1 = 5,
  t2 =  10,
  p1_1 = 0.6011338,
  p2_1 = 0.5440756,
  p1_2 = 0.5138367,
  p2_2 = 0.4321000,
  seed = 803,
  vetor_p_cens = c(0.25, 0.4, 0.5, 0.6, 0.7),
  vetor_n = c(500, 1000, 2500, 5000),
  vetor_tempos_interesse = c(1, 3, 5),
  M = 100,
  prop_n0 = 0.3,
  prop_n1 = 0.7
)

# Testes
t1 <- 5
t2 <-  10
p1_1 <- 0.6011338
p2_1 <- 0.5440756
p1_2 <- 0.5138367
p2_2 <- 0.4321000
seed <- 803
p_cens <- c(0.7)
n <- c(500)
tempo <- c(3)
M <- 50
prop_n0 <- 0.3
prop_n1 <- 0.7


