import os
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageEnhance 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 0. CONFIGURAÇÃO GLOBAL E PARÂMETROS DE OUTPUT
# =========================================================================

BASE_DIR = "./InfraredSolarModules" 
OUTPUT_DIR = "./augmented_data_output" 
AUG_IMAGES_DIR = os.path.join(OUTPUT_DIR, "augmented_images") 
DF_TRAIN_PATH = os.path.join(OUTPUT_DIR, "df_train_augmented.json")
DF_VAL_PATH = os.path.join(OUTPUT_DIR, "df_validation.json")
DF_TEST_PATH = os.path.join(OUTPUT_DIR, "df_test.json")
NEW_METADATA_PATH = os.path.join(OUTPUT_DIR, "generated_metadata.json")

# =========================================================================
# NOVOS PARÂMETROS DE AUGMENTATION (DO COLEGA)
# =========================================================================

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

# Parâmetros de SALVAMENTO FINAL (Target do usuário: 40x24)
FINAL_SAVE_H = 40
FINAL_SAVE_W = 24

# Parâmetros do VAE 
# H/W devem ser múltiplos de 16. Proporcional a 40x24 (5:3).
VAE_IMAGE_H = 160
VAE_IMAGE_W = 96 
VAE_LATENT_DIM = PARAMETROS_AUG["dimensao_latente"]
VAE_BATCH_SIZE = 128
VAE_EPOCHS = 70 

# Parâmetros da última camada do Encoder para cálculo do Flatten/Reshape
VAE_FINAL_H = 10 
VAE_FINAL_W = 6 
VAE_FLATTENED_SIZE = 256 * VAE_FINAL_H * VAE_FINAL_W 

MAX_COUNT = 20_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================================================================
# 1. FUNÇÃO DE PREPARAÇÃO DE DADOS (COPIADA)
# =========================================================================

def prepare_data(base_dir):
    # Carrega dados e realiza a separação Treino/Validação/Teste.
    metadata_path = os.path.join(base_dir, "module_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERRO FATAL: Arquivo de metadados não encontrado em {metadata_path}.")
        exit()

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df = df.rename(columns={"index": "id"})
    df["image_filepath"] = df["image_filepath"].apply(lambda x: os.path.join(base_dir, x))

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['anomaly_class'])
    class_names = le.classes_

    df_train, df_temp = train_test_split(
        df, test_size=0.2, random_state=50135, stratify=df['label_encoded']
    )
    
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=50135, stratify=df_temp['label_encoded']
    )
    
    return df_train, df_val, df_test, class_names


# =========================================================================
# 2. CLASSES E FUNÇÕES GLOBAIS
# =========================================================================

class VAEImageDataset(Dataset):
    def __init__(self, df, base_dir):
        self.image_paths = df['image_filepath'].tolist() 
        self.anomaly_classes = df['anomaly_class'].tolist()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((VAE_IMAGE_H, VAE_IMAGE_W)), 
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, self.anomaly_classes[idx]


# =========================================================================
# 3. IMPLEMENTAÇÃO DO VAE COM ARQUITETURA CORRIGIDA
# =========================================================================

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(VAE_FLATTENED_SIZE, latent_dim)
        self.fc_log_var = nn.Linear(VAE_FLATTENED_SIZE, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, VAE_FLATTENED_SIZE)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid() 
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 256, VAE_FINAL_H, VAE_FINAL_W) 
        return self.decoder_conv(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train_vae(model, dataloader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for i, data in enumerate(tqdm(dataloader, desc=f"VAE Epoch {epoch}")):
            img, _ = data 
            img = img.to(DEVICE)
            
            optimizer.zero_grad()
            recon_img, mu, log_var = model(img)
            loss = vae_loss_function(recon_img, img, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"VAE Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.4f}")
    print("VAE Training Complete.")


# =========================================================================
# 4. GERAÇÃO E SALVAMENTO DE IMAGENS
# =========================================================================

def generate_and_save_images(vae_model, class_name, num_needed, save_dir, metadata_list):
    vae_model.eval()
    
    with torch.no_grad():
        for i in range(num_needed):
            z = torch.randn(1, VAE_LATENT_DIM).to(DEVICE)
            generated_img_tensor = vae_model.decode(z)
            
            generated_img_np = (generated_img_tensor.cpu().squeeze().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(generated_img_np, mode='L')
            
            # 2. Redimensiona para o tamanho de salvamento final (40x24)
            final_img = img#.resize((FINAL_SAVE_W, FINAL_SAVE_H)) # PIL usa (W, H)
            
            filename = f"{class_name}_{i:04d}_vae.jpeg"
            save_path = os.path.join(save_dir, filename)
            final_img.save(save_path)
            
            metadata_list.append({
                "id": f"aug_vae_{class_name}_{i}",
                "image_filepath": os.path.join("augmented_images", filename), 
                "anomaly_class": class_name
            })

def apply_brightness_adjustment(img, factor_str):   
    img_np = np.array(img, dtype=np.int16) # Usar int16 para permitir valores fora de 0-255
    
    if factor_str == '+30':
        img_np += 30
    elif factor_str == '-30':
        img_np -= 30
        
    # Saturar os valores para garantir que fiquem entre 0 e 255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np, mode='L')


def apply_three_way_augmentation(df_class, class_name, save_dir, metadata_list):
    
    if len(df_class) == 0:
        return

    paths = df_class['image_filepath'].values 
    
    # Lista de tipos de aumento geométrico e de brilho
    aug_operations = [
        # Base: Original
        ("orig", None, 0),
        
        # Aumentos Geométricos Simples
        ("flip_h", Image.FLIP_LEFT_RIGHT, 0),
        ("flip_v", Image.FLIP_TOP_BOTTOM, 0),
        ("rot180", None, 180)
    ]
    
    brightness_factors = [
        ("orig", None),
        ("plus30", "+30"),
        ("minus30", "-30")
    ]
    
    idx_counter = 0 # Contador para IDs únicos de imagens aumentadas
    
    for path in paths:
        try:
            # 1. Abre e redimensiona a imagem original
            img_original_L = Image.open(path).convert('L')
            img_original_resized = img_original_L#.resize((FINAL_SAVE_W, FINAL_SAVE_H))
            
            # 2. Aplicar Aumentos Geométricos e de Brilho
            for geo_type, transpose_mode, rotate_angle in aug_operations:
                
                # Cópia geométrica base
                geo_img = img_original_resized.copy()
                
                if transpose_mode is not None:
                     geo_img = img_original_resized.transpose(transpose_mode)
                elif rotate_angle != 0:
                     geo_img = img_original_resized.rotate(rotate_angle, expand=False) 
                
                # Aplicar Variações de Brilho à Imagem Geométrica (3x: Orig, +30, -30)
                for br_tag, br_factor in brightness_factors:
                    
                    final_img = geo_img.copy()
                    
                    # Se o fator de brilho for aplicado, ajuste a imagem
                    if br_factor is not None:
                        final_img = apply_brightness_adjustment(geo_img, br_factor)
                        
                    # O nome da operação combinada
                    combined_tag = f"{geo_type}_{br_tag}" if br_tag != 'orig' else geo_type
                    
                    # Se o tipo geométrico for 'orig' e o brilho também for 'orig', 
                    # é a imagem original sem aumento, que já está no DataFrame base (df_train).
                    # Apenas salvamos as versões aumentadas (8 novas)
                    if geo_type == 'orig' and br_tag == 'orig':
                        continue 
                        
                    filename = f"{class_name}_{idx_counter:04d}_{combined_tag}.jpeg"
                    save_path = os.path.join(save_dir, filename)
                    final_img.save(save_path)
                    
                    metadata_list.append({
                        "id": f"aug_{combined_tag}_{class_name}_{idx_counter}",
                        "image_filepath": os.path.join("augmented_images", filename),
                        "anomaly_class": class_name
                    })
                    idx_counter += 1
                        
        except Exception as e:
            print(f"Erro ao aumentar imagem {path}: {e}")


# =========================================================================
# 5. EXECUÇÃO DO DATA AUGMENTATION OFFLINE
# =========================================================================

def execute_augmentation_pipeline():
    
    os.makedirs(AUG_IMAGES_DIR, exist_ok=True)
    
    df_train, df_val, df_test, class_names = prepare_data(BASE_DIR)
    
    # 1. Treinamento do VAE 
    print("\nIniciando Treinamento do VAE")
    vae_model = VAE(VAE_LATENT_DIM).to(DEVICE)
    train_vae_dataset = VAEImageDataset(df_train, BASE_DIR)
    vae_dataloader = DataLoader(train_vae_dataset, batch_size=VAE_BATCH_SIZE, shuffle=True)
    train_vae(vae_model, vae_dataloader, VAE_EPOCHS)
    
    print("\nIniciando Geração e Equalização de Classes...")
    
    # Separar os metadados aumentados em duas listas: Simples e VAE
    simple_aug_metadata = []
    vae_aug_metadata = []

    # Linhas comentadas para gerar apenas imagens com o VAE
    """
    # 2. Executar Aumento Simples Exaustivo (3x geom x 3x brilho = 8 novas por original)
    print(f"Aplicando Aumento Simples Exaustivo (8x) e salvando em {FINAL_SAVE_H}x{FINAL_SAVE_W}...")
    for class_name in class_names:
        df_class_original = df_train[df_train['anomaly_class'] == class_name].copy()
        
        # O aumento simples gera 8 NOVAS amostras por original
        apply_three_way_augmentation(df_class_original, class_name, AUG_IMAGES_DIR, simple_aug_metadata)
        
    df_simple_augmented = pd.DataFrame(simple_aug_metadata)
    
    # 3. Concatenar o DF de Treino Original com o DF Simplesmente Aumentado (Base para Contagem)
    # df_train contém a imagem original (não aumentada)
    df_train_base_augmented = pd.concat([
        df_train[['id', 'image_filepath', 'anomaly_class']], 
        df_simple_augmented
    ], ignore_index=True)
    """
    df_train_base_augmented = df_train
    class_counts = df_train_base_augmented['anomaly_class'].value_counts()
    max_count = MAX_COUNT
    print(f"\nAnálise de Classes de Treino (Máximo: {max_count}):\n{class_counts}")


    # 4. Preencher o restante com o VAE
    print(f"\nPreenchendo o restante com dados gerados pelo VAE (Gerado em 160x96, Salvo em {FINAL_SAVE_H}x{FINAL_SAVE_W})...")
    for class_name in class_names:
        current_count = class_counts.get(class_name, 0)
        num_needed = max_count - current_count
        
        if num_needed > 0:
            print(f"\nClasse: {class_name} | Faltam {num_needed} (Gerando com VAE...)")
            
            generate_and_save_images(vae_model, class_name, num_needed, AUG_IMAGES_DIR, vae_aug_metadata)
        else:
            print(f"Classe: {class_name} | Já atingiu ou excedeu o máximo ({current_count}).")

    # 5. Salvar Novos Metadados e DataFrames Finais
    new_metadata_list = simple_aug_metadata + vae_aug_metadata
    
    with open(NEW_METADATA_PATH, 'w') as f:
        json.dump(new_metadata_list, f, indent=4)
    
    df_vae_augmented = pd.DataFrame(vae_aug_metadata)
    
    # df_train_final é a soma do original + simples + VAE
    df_train_final = pd.concat([df_train_base_augmented, df_vae_augmented], ignore_index=True)
    
    df_train_final.to_json(DF_TRAIN_PATH, orient='records', lines=True)
    df_val.to_json(DF_VAL_PATH, orient='records', lines=True)
    df_test.to_json(DF_TEST_PATH, orient='records', lines=True)
    
    print("\nData Augmentation Offline CONCLUÍDO.")

if __name__ == '__main__':
    execute_augmentation_pipeline()