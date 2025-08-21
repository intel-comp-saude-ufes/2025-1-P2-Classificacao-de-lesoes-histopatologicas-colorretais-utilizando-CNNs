# Classificação de Lesões Histopatológicas Colorretais Utilizando Redes Neurais Convolucionais

## Descrição

Este estudo apresenta o desenvolvimento e a avaliação de um pipeline focado na classificação de lesões em imagens histopatológicas utilizando Redes Neurais Convolucionais (CNNs). O objetivo principal é explorar e comparar o desempenho de diferentes arquiteturas de CNN para classificar corretamente imagens de pólipos colorretais, diferenciando entre lesões benignas (hiperplásicos - HP) e pré-cancerosas (adenomas serrilhados sésseis - SSA).

Vídeo da apresentação do trabalho: 

O artigo completo pode ser encontrado no arquivo `Classificação de Lesões Histopatológicas Colorretais Utilizando CNNS.pdf`, presente no repositório, ou pelo link a seguir: https://drive.google.com/file/d/1BDhrf9lpXj7ANGhh8Okk0065XQqHLWTi/view?usp=sharing

## Objetivos

* Construir um pipeline completo para a classificação de imagens histopatológicas usando PyTorch e CNNs.
* Comparar o desempenho de uma arquitetura customizada (SimpleCNN) com modelos pré-treinados de ponta (ResNet50, VGG16 e DenseNet121) por meio de *transfer learning* e *fine-tuning*.
* Avaliar o impacto de técnicas de *ensemble* de modelos para melhorar a qualidade da classificação.
* Demonstrar a viabilidade do uso de CNNs para automatizar e auxiliar a prática clínica na detecção precoce de lesões malignas.

## Conjunto de Dados

O conjunto de dados utilizado foi o **MHIST (Minimalist Histopathology Image Analysis Dataset)**.

* **Fonte:** Desenvolvido pelo Departamento de Patologia e Medicina Laboratorial do Centro Médico Dartmouth-Hitchcock.
* **Composição:** 3.152 imagens de pólipos colorretais com tamanho de 224x224 pixels, sendo 2.162 (68,59%) da classe HP e 990 (31,41%) da classe SSA.
* **Tarefa:** Classificação binária entre pólipos benignos (Hiperplásicos - HP) e pré-cancerosos (Adenomas Serrilhados Sésseis - SSA).
* **Link para a página oficial o dataset**: https://bmirds.github.io/MHIST/

## Metodologia

O pipeline implementado no notebook `Trab2_IntelCompSaúde_Classificacao_MHIST.ipynb` segue os seguintes passos:

1.  **Importação e Configuração:**
    * Importação das bibliotecas necessárias, incluindo PyTorch, Torchvision, Scikit-learn, Pandas, Seaborn, Pillow, Numpy, TQDM e Matplotlib.
    * Configuração de sementes para garantir a reprodutibilidade dos experimentos.

2.  **Carregamento e Pré-processamento:**
    * Criação de um *Dataset* customizado para o carregamento das imagens e rótulos.
    * Aplicação de *data augmentation* no conjunto de treino para aumentar a robustez do modelo, incluindo técnicas como *RandomResizedCrop*, *RandomHorizontalFlip*, *RandomRotation* e *ColorJitter*.

3.  **Arquiteturas de CNN:**
    * **SimpleCNN:** Uma arquitetura de CNN customizada e desenvolvida como modelo base.
    * **Modelos Pré-treinados:** Utilização de arquiteturas renomadas via *Transfer Learning* e *Fine-tuning*:
        * ResNet50
        * VGG16
        * DenseNet121

4.  **Treinamento e Avaliação:**
    * Análise da complexidade dos modelos (número de parâmetros).
    * Treinamento e ajuste fino (*fine-tuning*) dos modelos.
    * Avaliação de desempenho utilizando métricas como acurácia, AUC, curvas de aprendizado e matriz de confusão.
    * Avaliação de técnicas de ensemble

## Resultados

* O melhor modelo individual foi a **ResNet50**, que alcançou uma **acurácia de 85%** e **AUC de 91.1%, embora apresentou sinais de ovefitting**.
* A abordagem de **ensemble**, utilizando *softmax* ponderado dos modelos, superou o desempenho individual, atingindo **86% de acurácia** e **AUC de 93.12%**.
* Os resultados demonstram a eficácia das CNNs e do *transfer learning* na classificação de imagens histopatológicas, com desempenho comparável a outros trabalhos na literatura.

## Como Reproduzir

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/intel-comp-saude-ufes/2025-1-P2-Classificacao-de-lesoes-histopatologicas-colorretais-utilizando-CNNs.git
    cd 2025-1-P2-Classificacao-de-lesoes-histopatologicas-colorretais-utilizando-CNNs
    ```

2.  **Instale as dependências:**
    O projeto foi desenvolvido em Python 3.12.11 e faz uso de bibliotecas do ecossistema Python voltadas para ciência de dados e deep learning. Para reproduzir o ambiente, certifique-se de instalar as seguintes dependências nas versões indicadas:
    ```
    torch       : 2.6.0+cu124
    torchvision : 0.21.0+cu124
    pandas      : 2.2.3
    numpy       : 1.26.4
    scikit-learn: 1.2.2
    matplotlib  : 3.7.2
    seaborn     : 0.12.2
    tqdm        : 4.67.1
    pillow      : 11.3.0
    ```
    É necessário importar também as bibliotecas `os` e `random`, nativas do Python.

4.  **Execute o Notebook:**
    Abra e execute o notebook `Trab2_IntelCompSaúde_Classificacao_MHIST.ipynb` em um ambiente com suporte a Jupyter (como Jupyter Lab ou Google Colab) e acesso a uma GPU para acelerar o treinamento.
5.  **Ajuste:**
    É importante ressaltar que o dataset foi importado de forma particular pelo kaggle, mas, como apresentado no início desse README, é possível conseguir os dados do site oficial do dataset (https://bmirds.github.io/MHIST/). Portanto, é necessária a adaptação do código para importar as imagens e as anotações de rótulos da base MHIST conforme a necessidade.
    Atualize URLs dos CSVs, se necessário.

## Autores

* Antonio Borssato - antonio.borssato@edu.ufes.br
* Lucas Alves - lucas.o.alves@edu.ufes.br
* Rodrigo Fardin - rodrigo.fardin@edu.ufes.br
