# README - Gerador de GIF Hipnótico com Transformações Lineares

## Abordagem do Projeto

Para começar este trabalho, o foco foi entender o básico de transformações lineares aplicadas em imagens (rotação, escala e cisalhamento). Sempre tivemos dificuldade e bloqueio com matemática, então no início tudo pareceu muito confuso. 

Fomos insistindo, pedimos ajuda ao monitor para entender melhor a proposta da atividade extra **e também usamos IA para pedir explicações detalhadas passo a passo** sobre cada etapa - tanto da parte matemática quanto da implementação em código. Com isso, conseguimos dividir o problema em partes menores:

1. **Carregar GIF e separar frames** (Pillow/ImageSequence)
2. **Implementar as matrizes** de rotação, escala e cisalhamento do zero
3. **Mapeamento inverso** com interpolação vizinho mais próximo (NumPy + einsum)
4. **Animação progressiva** usando funções seno/cosseno para efeito cíclico
5. **Interface gráfica** (Tkinter) para controle dos parâmetros
6. **Salvamento do GIF final** mantendo duration e loop originais

## Decisões de Implementação

- **Matemática**: Implementamos todas as matrizes 2x2 manualmente sem usar funções prontas de rotação/transformação
- **Eficiência**: Usamos `np.einsum` para aplicar a matriz inversa a todos os pixels de uma vez (vetorização)
- **Interpolação**: Vizinho mais próximo com `np.round()` e máscara para limites da imagem
- **GUI**: Adicionamos sliders para tornar o programa interativo e reutilizável
- **Efeito extra**: Transformação linear no canal azul dos pixels para efeito visual adicional

## Conceitos Matemáticos Utilizados

- **Matrizes 2x2** para transformações lineares afins
- **Mapeamento inverso** para evitar buracos na imagem
- **Composição de matrizes** (ordem: cisalhamento → escala → rotação)
- **Centro da imagem** como origem das transformações
- **Funções trigonométricas** (sin/cos) para animação suave e cíclica

## Desafios Enfrentados

1. **Entender mapeamento inverso** - tivemos que visualizar graficamente para "sentir" o conceito
2. **Coordenadas centralizadas** - descobrir que precisávamos subtrair o centro da imagem
3. **Performance** - loops duplos eram lentos, resolvemos com vetorização NumPy
4. **GIF animado** - manter duration original e detectar se era realmente animado
5. **Matriz não-inversível** - tratamento de exceção com np.linalg.inv()

A interface gráfica foi a cereja do bolo - permite testar diferentes intensidades sem mexer no código!
