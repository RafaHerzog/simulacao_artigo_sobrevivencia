library(reticulate)
library(caret)

py_install("optuna")
py_install("scikit-learn")
py_install("xgboost")

# Carregando as bibliotecas necessárias do Python
optuna <- import("optuna")
sklearn_model_selection <- import("sklearn.model_selection")
xgb <- import("xgboost")


funcao_tune_xgboost <- function(X_train, y_train, seed) {
  # Definindo o formato da validação cruzada 
  skf <- sklearn_model_selection$StratifiedKFold(n_splits = 10L, shuffle = TRUE, random_state = seed)
  
  # Definindo a função objetivo 
  objective <- function(trial) {
    
    # Definindo o mesmo grid de parâmetros do artigo original
    objective <- ""
    n_estimators <- as.integer(trial$suggest_int("n_estimators", 50, 250))
    max_depth <- as.integer(trial$suggest_int("max_depth", 3, 18))
    learning_rate <- trial$suggest_float("learning_rate", 0.05, 0.2, step = 0.05)
    gamma <- trial$suggest_float("gamma", 0.0, 0.3, step = 0.1)
    min_child_weight <- as.integer(trial$suggest_int("min_child_weight", 1, 7))
    colsample_bytree <- trial$suggest_float("colsample_bytree", 0.3, 0.7, step = 0.1)
    
    # Criando o modelo do XGBoost
    cls <- xgb$XGBClassifier(
      n_estimators = n_estimators,
      max_depth = max_depth,
      learning_rate = learning_rate,
      gamma = gamma,
      min_child_weight = min_child_weight,
      colsample_bytree = colsample_bytree,
      random_state = seed
    )
    
    # Realizando a validação cruzada 
    scores <- sklearn_model_selection$cross_val_score(
      cls, X_train, y_train, cv = skf, scoring = "balanced_accuracy"
    )
    
    return(mean(scores))
  }
  
  # Reduzindo a verbosidade para suprimir mensagens no console
  optuna$logging$set_verbosity(optuna$logging$ERROR)
  
  # Criando o estudo e fazendo a otimização dos hiperparâmetros
  studyXGB <- optuna$create_study(direction = "maximize", sampler = optuna$samplers$RandomSampler(seed = seed))
  studyXGB$optimize(objective, n_trials = 100)
  
  # Retornando os melhores parâmetros encontrados
  return(studyXGB$best_params)
}


