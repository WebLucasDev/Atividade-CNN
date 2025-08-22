# CNN para Classificação de Roupas de Verão

Este projeto usa **Inteligência Artificial** para identificar se uma roupa é adequada para o **verão** ou **inverno**.

## O que o projeto faz?

- Analisa imagens de roupas do dataset Fashion MNIST
- Classifica as roupas em duas categorias:
  - ** VERÃO**: Camisetas, vestidos, sandálias, camisas, tênis
  - ** INVERNO**: Calças, suéteres, casacos, bolsas, botas
- Usa uma **Rede Neural Convolucional (CNN)** para aprender os padrões

## Como rodar o projeto?

### 1. Preparar o ambiente
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 2. Instalar bibliotecas
```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn jupyter ipykernel
```

### 3. Rodar o código
```bash
# No Jupyter Notebook
jupyter notebook

# Ou direto no Python
python fashion_mnist_cnn.py
```

## Resultados esperados

- **Acurácia**: ~90%+ de precisão
- **Visualizações**: Gráficos mostrando o desempenho
- **Exemplos**: Imagens com predições corretas e incorretas

## Arquivos do projeto

- `atividade_cnn.ipynb` - Notebook original da atividade
- `fashion_mnist_cnn.py` - Código completo da CNN
- `requirements.txt` - Lista de bibliotecas necessárias
- `README.md` - Este arquivo

## Como funciona?

1. **Carrega** 60.000 imagens de roupas para treinar
2. **Transforma** o problema de 10 classes em apenas 2 (verão vs inverno)
3. **Treina** uma rede neural para reconhecer padrões nas imagens
4. **Testa** com 10.000 imagens novas
5. **Mostra** os resultados com gráficos e estatísticas

## Classificação das roupas

| Verão (1) | Inverno (0) |
|-----------|-------------|
| Camiseta | Calça |
| Vestido | Suéter |
| Sandália | Casaco |
| Camisa | Bolsa |
| Tênis | Bota |

## Tecnologias usadas

- **Python** - Linguagem de programação
- **TensorFlow/Keras** - Para criar a rede neural
- **Matplotlib/Seaborn** - Para gráficos
- **NumPy** - Para cálculos matemáticos
- **Scikit-learn** - Para métricas de avaliação
