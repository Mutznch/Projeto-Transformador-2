import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import sys # Para logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import ResNetModel, ViTModel, MobileNetV2Model 
import torchvision.models as models 
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt # Para a matriz de confusão
import itertools # Para plotagem da matriz

# =========================================================================
# 0. CONFIGURAÇÃO GLOBAL E PARÂMETROS
# =========================================================================

BASE_DIR = "./InfraredSolarModules" 
OUTPUT_DIR = "./augmented_data_output" 
AUG_IMAGES_DIR = os.path.join(OUTPUT_DIR, "augmented_images") 
DF_TRAIN_PATH = os.path.join(OUTPUT_DIR, "df_train_augmented.json")
DF_VAL_PATH = os.path.join(OUTPUT_DIR, "df_validation.json")
DF_TEST_PATH = os.path.join(OUTPUT_DIR, "df_test.json")

TRAINING_OUTPUT_DIR = "./training_output"
FEATURE_CACHE_DIR = os.path.join(TRAINING_OUTPUT_DIR, "feature_cache")
LOG_FILENAME = os.path.join(TRAINING_OUTPUT_DIR, "training_log.txt") 

# Parâmetros de Treinamento
BATCH_SIZE = 64 
IMAGE_SIZE = (224, 224) 
NUM_WORKERS = 0 
EPOCHS = 400 
LEARNING_RATE = 1e-4 
EARLY_STOPPING_PATIENCE = 20 
LR_SCHEDULER_PATIENCE = 7

# Opções: 'ALL', 'ANOMALY_ONLY', 'BINARY'
CLASS_SUBSET_MODE = 'BINARY' 

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

BINARY_TARGET_COUNT = max(sum(TARGET_SAMPLES_PER_CLASS_MAP.values()) - TARGET_SAMPLES_PER_CLASS_MAP['No-Anomaly'], TARGET_SAMPLES_PER_CLASS_MAP['No-Anomaly'])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================================================================
# 1. FUNÇÃO DE LOGGING
# =========================================================================

def setup_logging(filename):
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush() # Garante que o buffer é descarregado
        def flush(self):
            for f in self.files:
                f.flush()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    log_file = open(filename, 'a')
    sys.stdout = Tee(sys.stdout, log_file)
    print(f"--- LOG INICIADO: {filename} ---")


# =========================================================================
# 2. FUNÇÃO DE MATRIZ DE CONFUSÃO
# =========================================================================

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão Sem Normalização')

    plt.figure(figsize=(len(classes) * 0.8, len(classes) * 0.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')
    plt.tight_layout()
    
    # Salva o plot
    filename = f"confusion_matrix_{'normalized' if normalize else 'raw'}_{CLASS_SUBSET_MODE}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f"\nMatriz salva em: {save_path}")
    plt.close()


def save_confusion_matrix(all_labels, all_preds, class_names): 
    # 1. Matriz de Confusão Bruta
    cm_raw = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm_raw, class_names, normalize=False, title=f'Matriz de Confusão - {CLASS_SUBSET_MODE}')
    
    # 2. Matriz de Confusão Normalizada
    plot_confusion_matrix(cm_raw, class_names, normalize=True, title=f'Matriz de Confusão Normalizada - {CLASS_SUBSET_MODE}')


# =========================================================================
# 3. FUNÇÕES DE PREPARAÇÃO E FILTRAGEM DE DADOS (EXISTENTES)
# =========================================================================

def map_to_binary_label(df):
    df_mapped = df.copy()
    # Mapeia todas as classes que NÃO SÃO 'No-Anomaly' para 'Anomaly'
    df_mapped['anomaly_class'] = df_mapped['anomaly_class'].apply(
        lambda x: 'No-Anomaly' if x == 'No-Anomaly' else 'Anomaly'
    )
    return df_mapped

def remake_balanced_train_df(df_train_raw, target_count_map, mode):
    #Filtra o DataFrame com base no modo e aplica Undersampling/Oversampling.
    
    # 1. FILTRAGEM INICIAL (apenas para ANOMALY_ONLY)
    if mode == 'ANOMALY_ONLY':
        print("\n--- Modo: ANOMALY_ONLY selecionado. Removendo 'No-Anomaly' ---")
        df_train_raw = df_train_raw[df_train_raw['anomaly_class'] != 'No-Anomaly'].copy()
    
    # 2. SEPARAÇÃO E BALANCEAMENTO (usando o mapa de 12 classes)
    
    final_df_list = []
    class_names = df_train_raw['anomaly_class'].unique()
    
    # Separar dados Reais (Originais) de dados Aumentados
    df_original = df_train_raw[~df_train_raw['image_filepath'].str.contains('augmented_images', na=False)].copy()
    df_augmented = df_train_raw[df_train_raw['image_filepath'].str.contains('augmented_images', na=False)].copy()
    
    print(f"Total Original (Raw): {len(df_original)}, Total Aumentado (Raw): {len(df_augmented)}")

    default_target = 20_000 

    for class_name in class_names:
        # Pega o alvo específico do mapa de 12 classes.
        target_count = target_count_map.get(class_name, default_target) 
        
        df_orig_class = df_original[df_original['anomaly_class'] == class_name]
        df_aug_class = df_augmented[df_augmented['anomaly_class'] == class_name]
        
        current_count = len(df_orig_class)
        
        if current_count >= target_count:
            # Caso 1: Undersampling (pegar amostras ORIGINAIS aleatórias)
            df_sampled = df_orig_class.sample(n=target_count, random_state=42)
            final_df_list.append(df_sampled)
            print(f"  -> {class_name}: Undersampled (Originals) to {target_count}")
            
        else:
            # Caso 2: Oversampling Limitado (pegar todos os ORIGINAIS + AUGMENTADOS)
            
            # A) Adicionar TODOS os dados ORIGINAIS
            final_df_list.append(df_orig_class)
            
            # B) Calcular quantos dados AUMENTADOS faltam
            needed_from_aug = target_count - current_count
            
            if needed_from_aug > 0:
                # C) Amostrar aleatoriamente do pool aumentado (Limitado)
                n_to_sample = min(needed_from_aug, len(df_aug_class))
                
                if n_to_sample > 0:
                    df_sampled_aug = df_aug_class.sample(n=n_to_sample, random_state=42)
                    final_df_list.append(df_sampled_aug)
                    print(f"  -> {class_name}: Added {n_to_sample} Augmented (Total: {current_count + n_to_sample})")
                else:
                    print(f"  -> {class_name}: Faltam {needed_from_aug}, mas não há dados aumentados suficientes.")

    # Combinar todos os dataframes amostrados/selecionados
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    # 3. MAPEAMENTO BINÁRIO (modo BINARY)
    if mode == 'BINARY':
        # Requer sub-amostragem final se o total de 'Anomaly' for maior que BINARY_TARGET_COUNT
        final_df = map_to_binary_label(final_df)
        print("\n--- Mapeamento Binário aplicado no conjunto de treino balanceado. ---")
        
        # Sub-amostragem adicional da classe 'Anomaly' e 'No-Anomaly' para o alvo BINARY_TARGET_COUNT
        
        df_anomaly = final_df[final_df['anomaly_class'] == 'Anomaly']
        df_no_anomaly = final_df[final_df['anomaly_class'] == 'No-Anomaly']
        
        n_anomaly = min(BINARY_TARGET_COUNT, len(df_anomaly))
        df_anomaly_sampled = df_anomaly.sample(n=n_anomaly, random_state=42)

        n_no_anomaly = min(BINARY_TARGET_COUNT, len(df_no_anomaly))
        df_no_anomaly_sampled = df_no_anomaly.sample(n=n_no_anomaly, random_state=42)
        
        final_df = pd.concat([df_anomaly_sampled, df_no_anomaly_sampled], ignore_index=True)
        print(f"--- Sub-amostragem binária aplicada para {BINARY_TARGET_COUNT} amostras por classe. ---")
        
    return final_df


# =========================================================================
# 4. CLASSES GLOBAIS E EXTRATORES (EXISTENTES)
# =========================================================================

class ResizeWithPadding:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        width, height = img.size
        target_w, target_h = self.size
        ratio = min(target_w / width, target_h / height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new(img.mode, self.size, color=0)
        new_img.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return new_img.convert('RGB') 

def create_transforms(target_size):
    from torchvision import transforms
    return transforms.Compose([
        ResizeWithPadding(target_size),
        transforms.ToTensor(),
    ])

class AugmentedSolarPanelDataset(Dataset):
    def __init__(self, df, base_dir, aug_dir):
        self.df = df
        self.base_dir = base_dir
        self.aug_dir = aug_dir
        self.transform = create_transforms(IMAGE_SIZE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_full = row['image_filepath']
        label = row['label_encoded']
        
        if 'augmented_images' in img_path_full:
            full_path = os.path.join(self.aug_dir, os.path.basename(img_path_full))
        else:
            full_path = img_path_full
            
        # O AlexNet espera 3 canais RGB, mesmo que a imagem seja em tons de cinza.
        # Ao usar self.transform (que chama ResizeWithPadding.convert('RGB')),
        # garantimos que o input é RGB.
        image = Image.open(full_path).convert('L') # Abre como Grayscale (L)
        image = self.transform(image) # Converte para RGB no transform
        
        return image, label

class FeatureCacheDataset(Dataset):
    #Dataset que carrega features de um arquivo .npy (Memória Mapeada) e labels de um tensor.
    def __init__(self, features_path, labels):
        # Carrega o array de features usando memória mapeada (não carrega tudo na RAM)
        self.features = np.load(features_path, mmap_mode='r') 
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Converte o slice de feature para tensor float32
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = self.labels[idx]
        return x, y


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class UltraDeepDNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(UltraDeepDNNClassifier, self).__init__()
        
        self.layer_stack = nn.Sequential(
            # Camada 1: 4096 -> 3072
            nn.Linear(input_size, 3072),
            nn.BatchNorm1d(3072), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.55), 

            # Camada 2: 3072 -> 2048
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            # Camada 3: 2048 -> 1024
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.45),

            # Camada 4: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            # Camada 5: 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),
            
            # Camada 6: 256 -> 128
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            # Camada 7: 128 -> 64
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            
            # Camada de Saída
            nn.Linear(64, num_classes) 
        )

    def forward(self, x):
        return self.layer_stack(x)
        
def build_extractors():
    # Modelos baseados em HuggingFace/transformers
    resnet_model = ResNetModel.from_pretrained("microsoft/resnet-50").to(DEVICE)
    resnet_model.eval() 
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
    vit_model.eval()
    mobilenet_model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224").to(DEVICE)
    mobilenet_model.eval()
    
    # AlexNet (torchvision.models)
    alexnet_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features.to(DEVICE)
    alexnet_model.eval()
    
    return [resnet_model, vit_model, mobilenet_model, alexnet_model]

@torch.no_grad() 
def extract_and_cache_features(data_loader, models, split_name, total_samples):

    # 1. Preparação do Cache
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    model_names = ["ResNet", "ViT", "MobileNetV2", "AlexNet"] 
    ensemble_cache_path = os.path.join(FEATURE_CACHE_DIR, f'features_{split_name}_{CLASS_SUBSET_MODE}_ENSEMBLE.npy')
    individual_cache_paths = []
    
    # 1.1 Checar se o cache final já existe
    if os.path.exists(ensemble_cache_path):
        print(f"\nCache de features do ENSEMBLE encontrado em {ensemble_cache_path}. Pulando extração.")
        return ensemble_cache_path
        
    print(f"\nIniciando extração e escrita batch-wise para {split_name} (Total: {total_samples})...")
        
    # 1.2 Extração por Modelo (escrita direta no disco)
    for model_idx, model in enumerate(models):
        model_name = model_names[model_idx]
        model_cache_path = os.path.join(FEATURE_CACHE_DIR, f'features_{split_name}_{CLASS_SUBSET_MODE}_{model_name}.npy')
        individual_cache_paths.append(model_cache_path)

        # 1.2.1 Se o arquivo individual já existe, pular a extração
        if os.path.exists(model_cache_path):
            print(f"  -> Cache individual de {model_name} encontrado. Pulando extração.")
            continue
            
        # 1.2.2 Determinar o tamanho do feature vector (Executa um batch)
        data_loader_iter = iter(data_loader)
        initial_inputs, _ = next(data_loader_iter)
        initial_inputs = initial_inputs.to(DEVICE)
        
        outputs = model(initial_inputs)
        if model_name in ["ResNet", "MobileNetV2"]:
            features = outputs.last_hidden_state.mean(dim=[2, 3]) 
        elif model_name == "ViT":
            features = outputs.last_hidden_state[:, 0, :]
        elif model_name == "AlexNet":
            features = outputs.view(outputs.size(0), -1) 
        
        feature_size = features.shape[1]
        
        # 1.2.3 Cria o array de memória mapeada para este modelo
        memmap_features = np.lib.format.open_memmap(
            model_cache_path, 
            mode='w+', 
            dtype='float32', 
            shape=(total_samples, feature_size)
        )
        
        print(f"  -> Modelo {model_name} | Feature Size: {feature_size}")
        
        # 1.2.4 Extrai features e escreve no memmap (iniciando do zero)
        current_idx = 0
        for inputs, _ in tqdm(data_loader, desc=f"Writing {model_name} Features"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            if model_name in ["ResNet", "MobileNetV2"]:
                features = outputs.last_hidden_state.mean(dim=[2, 3]) 
            elif model_name == "ViT":
                features = outputs.last_hidden_state[:, 0, :]
            elif model_name == "AlexNet":
                features = outputs.view(outputs.size(0), -1) 
            
            batch_size = features.shape[0]
            
            # Escreve diretamente no disco
            memmap_features[current_idx:current_idx + batch_size] = features.cpu().numpy()
            current_idx += batch_size
        
        # Garante que os dados sejam gravados no disco
        memmap_features.flush()
        
        del memmap_features
        torch.cuda.empty_cache()
        
    # 2. Concatenação Final do Ensemble (Mapeada para evitar estouro de RAM)
    
    # 2.1 Calcula o tamanho total do ensemble
    total_ensemble_size = sum(np.lib.format.open_memmap(p, mode='r').shape[1] for p in individual_cache_paths)
    
    print(f"\nConcatenando features (Input Size: {total_ensemble_size})...")
    
    # 2.2 Cria o arquivo final de memória mapeada (ENSEMBLE)
    ensemble_memmap = np.lib.format.open_memmap(
        ensemble_cache_path, 
        mode='w+', 
        dtype='float32', 
        shape=(total_samples, total_ensemble_size)
    )
    
    # 2.3 Concatena coluna por coluna (lendo do disco e escrevendo no disco)
    col_idx = 0
    for model_path in individual_cache_paths:
        # Abre o arquivo individual com mmap_mode='r' para não carregar na RAM
        model_data = np.load(model_path, mmap_mode='r')
        size = model_data.shape[1]
        
        # Escreve o bloco no ensemble mapeado (lendo do disco e escrevendo no disco)
        ensemble_memmap[:, col_idx:col_idx + size] = model_data
        col_idx += size
        
        # Libera a referência ao modelo individual (o arquivo permanece no disco)
        del model_data
        
    ensemble_memmap.flush()
    
    print(f"Ensemble features salvas com sucesso em: {ensemble_cache_path}")
    return ensemble_cache_path


def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')
        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
            
    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)
    best_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        
        # Validação
        val_loss = validate_model(model, val_loader, criterion)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict() 
        
        if early_stopper.early_stop(val_loss):
            print(f"\n[Early Stopping] Parando na Época {epoch} após {EARLY_STOPPING_PATIENCE} épocas sem melhora.")
            model.load_state_dict(best_weights) 
            break
            
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    print("\nTraining finished.")

def evaluate_classifier(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    return accuracy, report, all_preds, all_labels

# =========================================================================
# 5. EXECUÇÃO PRINCIPAL
# =========================================================================

if __name__ == '__main__':
    
    # 1. Configurar Logging
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(LOG_FILENAME)
    
    # 2. CARREGAR DADOS BRUTOS (ORIGINAL + TODOS OS AUGMENTADOS)
    print("Carregando DataFrames de Treino (Raw Augmented), Validação e Teste...")
    try:
        df_train_raw = pd.read_json(DF_TRAIN_PATH, orient='records', lines=True)
        df_val = pd.read_json(DF_VAL_PATH, orient='records', lines=True)
        df_test = pd.read_json(DF_TEST_PATH, orient='records', lines=True)
    except FileNotFoundError:
        print("ERRO: Certifique-se de que o script data_augmentation.py foi executado com sucesso e os arquivos existem em augmented_data_output.")
        exit()
        
    # 2.1 FILTRAGEM DOS DATAFRAMES DE VALIDAÇÃO E TESTE
    if CLASS_SUBSET_MODE == 'ANOMALY_ONLY':
        df_val = df_val[df_val['anomaly_class'] != 'No-Anomaly'].copy()
        df_test = df_test[df_test['anomaly_class'] != 'No-Anomaly'].copy()
    elif CLASS_SUBSET_MODE == 'BINARY':
        # Mapeamento Binário dos conjuntos de Validação e Teste
        df_val = map_to_binary_label(df_val)
        df_test = map_to_binary_label(df_test)

    # 3. APLICAR LÓGICA DE EQUALIZAÇÃO E UNDERSAMPLING LIMITADO
    df_train_final = remake_balanced_train_df(df_train_raw, TARGET_SAMPLES_PER_CLASS_MAP, CLASS_SUBSET_MODE)
    
    # 3.1 CONFIGURAÇÕES FINAIS DE PESO E NOMEAÇÃO
    
    current_target_map = TARGET_SAMPLES_PER_CLASS_MAP.copy()
    current_weight_map = CLASS_WEIGHTS_MAP.copy()
    
    if CLASS_SUBSET_MODE == 'BINARY':
        print("\n--- Aplicando Configurações Binárias Finais ---")
        
        # O total da classe 'Anomaly' agora é soma dos 11 alvos de anomalia
        anomaly_count = df_train_final[df_train_final['anomaly_class'] == 'Anomaly'].shape[0]
        no_anomaly_count = df_train_final[df_train_final['anomaly_class'] == 'No-Anomaly'].shape[0]
        
        current_target_map = {
            'Anomaly': anomaly_count, 
            'No-Anomaly': no_anomaly_count, 
        }
        current_weight_map = BINARY_WEIGHTS_MAP.copy()

    print(f"\nTotal de Amostras de Treino FINAL (Após Equalização): {len(df_train_final)}")
    print("Distribuição Final de Classes de Treino:")
    print(df_train_final['anomaly_class'].value_counts())


    # 4. PREPARAR LABELS (Codificação e Pesos)
    le = LabelEncoder()
    # Fit no conjunto completo (filtrado/mapeado) para garantir o mapeamento de todas as classes ativas
    df_temp = pd.concat([df_train_final, df_val, df_test])
    le.fit(df_temp['anomaly_class']) 
    class_names = le.classes_
    
    # Prepara os pesos usando o mapa de pesos ATUAL
    class_to_weight = {name: current_weight_map.get(name, 1.0) for name in class_names}
    sorted_class_names = le.classes_
    weights = [class_to_weight[name] for name in sorted_class_names]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    # Aplica a codificação
    df_train_final['label_encoded'] = le.transform(df_train_final['anomaly_class'])
    df_val['label_encoded'] = le.transform(df_val['anomaly_class'])
    df_test['label_encoded'] = le.transform(df_test['anomaly_class'])

    y_train = torch.tensor(df_train_final['label_encoded'].values, dtype=torch.long)
    y_val = torch.tensor(df_val['label_encoded'].values, dtype=torch.long)
    y_test = torch.tensor(df_test['label_encoded'].values, dtype=torch.long)

    print(f"Total classes: {len(class_names)}. Train samples (Final): {len(df_train_final)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    print(f"Pesos de Classe Aplicados (na ordem do LabelEncoder): {class_weights_tensor.tolist()}")

    # 5. CRIAR DATALOADERS PARA IMAGENS E EXTRAIR/CACHEAR CARACTERÍSTICAS
    train_dataset_img = AugmentedSolarPanelDataset(df_train_final, BASE_DIR, AUG_IMAGES_DIR)
    val_dataset_img = AugmentedSolarPanelDataset(df_val, BASE_DIR, AUG_IMAGES_DIR)
    test_dataset_img = AugmentedSolarPanelDataset(df_test, BASE_DIR, AUG_IMAGES_DIR)

    train_loader_img = DataLoader(train_dataset_img, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader_img = DataLoader(val_dataset_img, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader_img = DataLoader(test_dataset_img, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    feature_extractors = build_extractors()

    # Extrair e salvar features em cache
    X_train_cache_path = extract_and_cache_features(train_loader_img, feature_extractors, 'train', len(df_train_final))
    X_val_cache_path = extract_and_cache_features(val_loader_img, feature_extractors, 'val', len(df_val))
    X_test_cache_path = extract_and_cache_features(test_loader_img, feature_extractors, 'test', len(df_test))
    
    # O tamanho do vetor de features é lido do cache para determinar o INPUT_SIZE
    INPUT_SIZE = np.load(X_train_cache_path, mmap_mode='r').shape[1]
    print(f"INPUT_SIZE (Features Ensemble): {INPUT_SIZE}")

    # 6. PREPARAR DATALOADERS PARA O CLASSIFICADOR USANDO O CACHE
    train_dataset_dnn = FeatureCacheDataset(X_train_cache_path, y_train)
    val_dataset_dnn = FeatureCacheDataset(X_val_cache_path, y_val)
    test_dataset_dnn = FeatureCacheDataset(X_test_cache_path, y_test)

    train_loader_dnn = DataLoader(train_dataset_dnn, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_dnn = DataLoader(val_dataset_dnn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader_dnn = DataLoader(test_dataset_dnn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # 7. TREINAMENTO DO CLASSIFICADOR 
    NUM_CLASSES = len(class_names) 

    print("\n" + "="*50)
    print(f"INICIANDO TREINAMENTO DO CLASSIFICADOR ({CLASS_SUBSET_MODE} classes)")
    print("="*50)

    model = UltraDeepDNNClassifier(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
    
    # Usar a função de perda com pesos de classe
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )

    train_and_validate(model, train_loader_dnn, val_loader_dnn, criterion, optimizer, scheduler, EPOCHS)

    # 8. AVALIAÇÃO, SALVAMENTO E MÉTRICAS
    accuracy, report, all_preds, all_labels = evaluate_classifier(model, test_loader_dnn, class_names)

    print("\n" + "="*50)
    print(f"CLASSIFICATION RESULTS ({CLASS_SUBSET_MODE} classes)")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Detailed Classification Report:")
    print(report)

    # Gerar e salvar as matrizes de confusão
    save_confusion_matrix(all_labels, all_preds, class_names)
    
    MODEL_FILENAME = f'ultradeep_dnn_{CLASS_SUBSET_MODE}_aug.pth'
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"\nFinal DNN model salvo como: {MODEL_FILENAME}")