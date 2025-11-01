# Projeto-Transformador-2
Projeto transformador 2 - Detecção de anomalia em painéis solares

Este repositório contém os principais arquivos de código e resultados referentes ao projeto da disciplina.

Database: https://github.com/RaptorMaps/InfraredSolarModules/tree/master.

Imagens geradas: https://drive.google.com/file/d/1IqB25GE_N7521AtTvcUIOrGL4Ex4qgwp/view?usp=sharing

Modelos Treinados: https://drive.google.com/file/d/1SdMK5GreZimHrzA3QtyXRwQ1tSUY61VB/view?usp=sharing

Arquivos deste repositório: 

  1. Melhores modelos - Cada pasta contém a matriz de confusão e log do treinamento 

  2. main_training.py - Como o nome indica, este arquivo é onde ocorre o treinamento dos modelos. Todos os parametros são configurados em constantes dentro do próprio arquivo.

  Variáveis configuráveis:
  
    BASE_DIR → Diretório com o dataset principal
    OUTPUT_DIR → Diretório com as informações do dataset aumentado e outras saídas de programa
    AUG_IMAGES_DIR → Diretório com as imagens do dataset aumentado
    DF_TRAIN_PATH → Json com os caminhos das imagens de treino
    DF_VAL_PATH → Json com os caminhos das imagens de validação
    DF_TEST_PATH → Json com os caminhos das imagens de teste
    
    TRAINING_OUTPUT_DIR → Diretório com as saídas do programa
    FEATURE_CACHE_DIR → Diretório com os vetores de features salvos
    LOG_FILENAME → Arquivo de log
    
    - Variáveis do treino autoexplicativas
    BATCH_SIZE = 64 
    IMAGE_SIZE = (224, 224) 
    NUM_WORKERS = 0 
    EPOCHS = 400 
    LEARNING_RATE = 1e-4 
    EARLY_STOPPING_PATIENCE = 20 
    LR_SCHEDULER_PATIENCE = 7
    
    CLASS_SUBSET_MODE → De que forma lerá os arquivos de entrada para treinoalos. Opções: 'ALL', 'ANOMALY_ONLY', 'BINARY'.
    
    - Número de imagens que serão usadas para cada classe
    TARGET_SAMPLES_PER_CLASS_MAP = {
        'Cell': 1000,
        'Cell-Multi': 1000,
        'Cracking': 753,
        'Diode': 1000,
        'Diode-Multi': 141,
        'Hot-Spot': 199,
        'Hot-Spot-Multi': 198,
        'No-Anomaly': 8010, 
        'Offline-Module': 662,
        'Shadowing': 845,
        'Soiling': 144,
        'Vegetation': 1000,
    }
    
    - Peso de cada classe na validação do treino
    CLASS_WEIGHTS_MAP = {
        'Cell': 1.0,        
        'Cell-Multi': 1.0,    
        'Cracking': 1.0,
        'Diode': 1.0,        
        'Diode-Multi': 1.0,
        'Hot-Spot': 1.0,    
        'Hot-Spot-Multi': 1.0,
        'No-Anomaly': 1.0,   
        'Offline-Module': 1.0,
        'Shadowing': 1.0,
        'Soiling': 1.0,     
        'Vegetation': 1.0,   
    }
    
    BINARY_WEIGHTS_MAP = {
        'Anomaly': 1.0,     
        'No-Anomaly': 1.0,  
    }

  3. main_data_augumentation - Arquivo com o código que gera as imagens artificiais para posteriormente treinar o modelo.

  Variáveis configuráveis:

    - Similar aos caminhos citados anteriormente. Autoexplicativos
    BASE_DIR = "./InfraredSolarModules" 
    OUTPUT_DIR = "./augmented_data_output" 
    AUG_IMAGES_DIR = os.path.join(OUTPUT_DIR, "augmented_images") 
    DF_TRAIN_PATH = os.path.join(OUTPUT_DIR, "df_train_augmented.json")
    DF_VAL_PATH = os.path.join(OUTPUT_DIR, "df_validation.json")
    DF_TEST_PATH = os.path.join(OUTPUT_DIR, "df_test.json")
    NEW_METADATA_PATH = os.path.join(OUTPUT_DIR, "generated_metadata.json")
    
    - Parametros de treino do VAE
    
    PARAMETROS_AUG = {
        "INTERPOLACAO": True,
        "VARIACAO_SUTIL": True,
        "PONTOS_ALEATORIOS": True,
        "desvio": 0.25,
        "dimensao_latente": 64,
        "fator_baixo": 2,
        "fator_alto": 4,
        "proporcao_aleatoria": 0.3,
        "proporcao_interp": 0.3,
        "usar_threshold": True,
        "threshold": 0.5, 
    }
    
    FINAL_SAVE_H = 40
    FINAL_SAVE_W = 24
    VAE_IMAGE_H = 160
    VAE_IMAGE_W = 96 
    VAE_LATENT_DIM = PARAMETROS_AUG["dimensao_latente"]
    VAE_BATCH_SIZE = 128
    VAE_EPOCHS = 70 
    
    - Parâmetros da última camada do Encoder para cálculo do Flatten/Reshape
    VAE_FINAL_H = 10 
    VAE_FINAL_W = 6 
    VAE_FLATTENED_SIZE = 256 * VAE_FINAL_H * VAE_FINAL_W 
    
    MAX_COUNT → Número alvo de imagens para cada classe.
