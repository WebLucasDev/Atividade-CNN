import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Carregar dados
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Visualizar algumas imagens do dataset original
plt.figure(figsize=(12, 8))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'{class_names[y_train[i]]}')
    plt.axis('off')
plt.suptitle('Exemplos do Fashion MNIST Dataset')
plt.tight_layout()
plt.show()

# Normalizar dados
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Incluir canal de cor
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Transformar em problema de classificação binária: Frio vs Calor
# Roupas de CALOR (verão): T-shirt/top (0), Dress (3), Sandal (5), Shirt (6), Sneaker (7)
# Roupas de FRIO (inverno): Trouser (1), Pullover (2), Coat (4), Bag (8), Ankle boot (9)

def transform_to_binary(y):
    """
    Transforma as classes originais em classificação binária:
    0 - Frio (Trouser, Pullover, Coat, Bag, Ankle boot)
    1 - Calor (T-shirt, Dress, Sandal, Shirt, Sneaker)
    """
    # Classes de calor (verão)
    summer_classes = [0, 3, 5, 6, 7]  # T-shirt, Dress, Sandal, Shirt, Sneaker
    
    binary_labels = np.zeros(len(y))
    for i, label in enumerate(y):
        if label in summer_classes:
            binary_labels[i] = 1  # Calor
        else:
            binary_labels[i] = 0  # Frio
    
    return binary_labels

# Converter labels para classificação binária
y_train_binary = transform_to_binary(y_train)
y_test_binary = transform_to_binary(y_test)

# Verificar distribuição das classes
unique, counts = np.unique(y_train_binary, return_counts=True)
print("Distribuição das classes:")
print(f"Frio (0): {counts[0]} amostras")
print(f"Calor (1): {counts[1]} amostras")

# Visualizar exemplos da nova classificação
plt.figure(figsize=(15, 6))
frio_indices = np.where(y_train_binary == 0)[0][:6]
calor_indices = np.where(y_train_binary == 1)[0][:6]

for i, idx in enumerate(frio_indices):
    plt.subplot(2, 6, i+1)
    plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Frio: {class_names[y_train[idx]]}')
    plt.axis('off')

for i, idx in enumerate(calor_indices):
    plt.subplot(2, 6, i+7)
    plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Calor: {class_names[y_train[idx]]}')
    plt.axis('off')

plt.suptitle('Exemplos de Classificação Binária: Frio vs Calor')
plt.tight_layout()
plt.show()

# Criar modelo CNN
def create_cnn_model():
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        
        # Camadas totalmente conectadas
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Saída binária
    ])
    
    return model

# Criar e compilar o modelo
model = create_cnn_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Exibir arquitetura do modelo
model.summary()

# Treinamento
print("Iniciando treinamento...")
history = model.fit(
    x_train, y_train_binary,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test_binary),
    verbose=1
)

# Avaliar modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test_binary, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")

# Fazer predições
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test_binary, y_pred, 
                          target_names=['Frio', 'Calor']))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_binary, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Frio', 'Calor'], 
            yticklabels=['Frio', 'Calor'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Plotar histórico de treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

# Exemplos de predições
plt.figure(figsize=(15, 10))
sample_indices = np.random.choice(len(x_test), 20, replace=False)

for i, idx in enumerate(sample_indices):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    
    pred_prob = y_pred_prob[idx][0]
    pred_class = 'Calor' if pred_prob > 0.5 else 'Frio'
    true_class = 'Calor' if y_test_binary[idx] == 1 else 'Frio'
    
    # Cor do título: verde se correto, vermelho se incorreto
    color = 'green' if pred_class == true_class else 'red'
    
    plt.title(f'Real: {true_class}\nPred: {pred_class} ({pred_prob:.2f})', 
              color=color, fontsize=9)
    plt.axis('off')

plt.suptitle('Exemplos de Predições (Verde=Correto, Vermelho=Incorreto)')
plt.tight_layout()
plt.show()

# Análise por classe original
def analyze_by_original_class():
    print("\nAnálise detalhada por classe original:")
    print("="*50)
    
    for original_class in range(10):
        mask = y_test == original_class
        if np.sum(mask) > 0:
            class_predictions = y_pred[mask]
            true_binary = y_test_binary[mask]
            accuracy = np.mean(class_predictions == true_binary)
            
            expected_label = 1 if original_class in [0, 3, 5, 6, 7] else 0
            expected_name = "Calor" if expected_label == 1 else "Frio"
            
            print(f"{class_names[original_class]:12} -> {expected_name:5} | "
                  f"Acurácia: {accuracy:.3f} | "
                  f"Amostras: {np.sum(mask):4}")

analyze_by_original_class()

print(f"\nResumo:")
print(f"Acurácia geral: {test_accuracy:.4f}")
print(f"Total de parâmetros: {model.count_params():,}")