library(dplyr)
library(caret)

# Criando uma função para gerar valores da distribuição Weibull com censura
gera_valores_censurados <- function(t1, t2, p1_1, p2_1, p1_2, p2_2, p_cens, n, prop_n0, prop_n1, seed) {
  # Criando uma função que encontra os parâmetros da distribuição Weibull dados dois tempos de sobrevivência e probs
  find_weibull_params <- function(t1, t2, p1, p2) {
    # Calculando o valor do parâmetro de forma 
    shape <- (log(-log(p1)) - log(-log(p2))) / (log(t1) - log(t2))
    
    # Calculando o valor do parâmetro de escala
    scale <- t1 / (-log(p1))^(1/shape)
    
    return(list(shape = shape, scale = scale))
  }
  
  # Criando uma função para encontrar o valor de theta que satisfaça P(T > C) = p
  find_theta <- function(scale, shape, p) {
    # Função para calcular P(T > C) dado um valor de theta
    P_T_greater_than_C <- function(theta, scale, shape, p) {
      # Verificar se theta é muito pequeno ou muito grande, o que pode causar problemas numéricos
      if (theta <= 0) return(Inf)  # Evitar que theta seja zero ou negativo
      if (theta > 10 * scale) return(0)  # Evitar intervalos muito grandes
      
      # Integrando P(T > c) para c de 0 até theta
      integral_result <- tryCatch({
        integrate(function(c) exp(-(c/scale)^shape), lower = 0, upper = theta)
      }, error = function(e) NULL)
      
      if (is.null(integral_result)) return(NA)  # Se a integração falhar, retornar NA
      
      integral_value <- integral_result$value
      # Dividido por theta, conforme a fórmula
      P <- (1/theta) * integral_value
      return(P - p)  # Queremos que isso seja 0
    }
    
    # Usar uma busca numérica para encontrar theta com um intervalo ajustado
    result <- tryCatch({
      uniroot(P_T_greater_than_C, c(0, 10*scale), scale = scale, shape = shape, p = p)
    }, error = function(e) NULL)
    
    if (is.null(result)) {
      cat("Erro ao encontrar a raiz para theta.\n")
      return(NULL)
    }
    
    return(result$root)
  }
  
  set.seed(seed)
  # Definindo os parâmetros das v.a.'s com distribuição Weibull (T0 e T1), das quais geraremos valores
  # Obs.: esses parâmetros são calculados dados dois tempos de sobrevivência e duas probabilidades 
  # de sobrevivência associadas a esses tempos
  
  ## Calculando os parâmetros da distribuição de T0, assumindo que T0 ~ Weibull(shape0, scale0)
  params0 <- find_weibull_params(t1 = t1, t2 = t2, p1 = p1_1, p2 = p2_1)
  shape0 <- params0$shape
  scale0 <- params0$scale
  
  ## Calculando os parâmetros da distribuição de T1, assumindo que T1 ~ Weibull(shape1, scale1)
  params1 <- find_weibull_params(t1 = t1, t2 = t2, p1 = p1_2, p2 = p2_2)
  shape1 <- params1$shape
  scale1 <- params1$scale
  
  # Calculando os valores de theta0 e theta1, parâmetros das variáveis com distribuição
  # uniforme que utilizaremos para simular a censura
  theta0 <- find_theta(scale = scale0, shape = shape0, p = p_cens)
  theta1 <- find_theta(scale = scale1, shape = shape1, p = p_cens)
  
  # Gerando n0 observações de T0 e C0
  amostra_t0 <- rweibull(prop_n0*n, shape = shape0, scale = scale0)
  amostra_c0 <- runif(prop_n0*n, 0, theta0)
  
  # Gerando n1 observações de T1 e C1
  amostra_t1 <- rweibull(prop_n1*n, shape = shape1, scale = scale1)
  amostra_c1 <- runif(prop_n1*n, 0, theta1)
 
  # Organizando as amostras em um dataframe e criando variáveis com o tempo observado e a indicação de falha
  df_amostras_aux1 <- data.frame(
    t = c(amostra_t0, amostra_t1),
    c = c(amostra_c0, amostra_c1),
    preditor = c(rep(0, times = prop_n0*n), rep(1, times = prop_n1*n))
  ) |>
    mutate(
      t_obs = ifelse(t < c, t, c),
      falha = ifelse(t < c, 1, 0),
      .after = "c"
    ) 
  
  # Embaralhando as linhas do dataframe gerado
  df_amostras_aux2 <- df_amostras_aux1[sample(1:nrow(df_amostras_aux1)), ]
  row.names(df_amostras_aux2) <- 1:nrow(df_amostras_aux2)
  
  # Retirando as colunas com o tempo teórico e o tempo de censura do dataframe
  df_amostras <- df_amostras_aux2 |> select(!c(t, c))
  
  # Retornando o dataframe de amostras
  return(
    list(
      df_amostras = df_amostras,
      params0 = c("scale" = scale0, "shape" = shape0),
      params1 = c("scale" = scale1, "shape" = shape1) 
    )
  )
}

