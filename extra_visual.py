import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageSequence
import math
import os


# ==========================================
# PARTE 1: TRANSFORMAÇÕES LINEARES (MATEMÁTICA)
# ==========================================

# Gera a matriz de rotação 2x2 a partir de um ângulo em graus
def gerar_matriz_rotacao(angulo_graus):
    rad = math.radians(angulo_graus)  # converte graus para radianos
    return np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad),  math.cos(rad)]
    ])


# Gera a matriz de escala 2x2 a partir dos fatores de escala em x e y
def gerar_matriz_escala(sx, sy):
    return np.array([[sx, 0], [0, sy]])


# Gera a matriz de cisalhamento 2x2 a partir dos fatores em x e y
def gerar_matriz_cisalhamento(cx, cy):
    return np.array([[1, cx], [cy, 1]])


# Aplica uma transformação linear em uma imagem usando mapeamento inverso
def aplicar_transformacao_geometrica(imagem_pil, matriz_transformacao):
    # Converte a imagem para array NumPy (matriz de pixels)
    img_array = np.array(imagem_pil)
    altura, largura = img_array.shape[:2]

    # Cria um array de saída com o mesmo tamanho da imagem original
    saida_array = np.zeros_like(img_array)

    # Calcula o centro da imagem para usar como origem das transformações
    centro_x, centro_y = largura / 2, altura / 2

    # Calcula a matriz inversa para fazer mapeamento inverso (destino -> origem)
    try:
        matriz_inversa = np.linalg.inv(matriz_transformacao)
    except np.linalg.LinAlgError:
        # Se a matriz não for inversível, retorna a imagem original
        return imagem_pil

    # Cria matrizes com as coordenadas (x, y) de todos os pixels de saída
    y_coords, x_coords = np.indices((altura, largura))

    # Desloca as coordenadas para que o centro da imagem seja (0, 0)
    x_shifted = x_coords - centro_x
    y_shifted = y_coords - centro_y

    # Junta as coordenadas em um vetor 2x(altura)x(largura)
    coords_vetor = np.stack([x_shifted, y_shifted])

    # Aplica a matriz inversa a todas as coordenadas de uma vez (mapeamento inverso)
    coords_origem = np.einsum('ij,jkl->ikl', matriz_inversa, coords_vetor)

    # Converte as coordenadas de volta para o sistema original da imagem
    src_x = np.round(coords_origem[0] + centro_x).astype(int)
    src_y = np.round(coords_origem[1] + centro_y).astype(int)

    # Máscara para pegar apenas os pontos que caem dentro da imagem de origem (vizinho mais próximo)
    mascara = (src_x >= 0) & (src_x < largura) & (src_y >= 0) & (src_y < altura)

    # Copia os pixels da imagem de origem para a saída usando as coordenadas mapeadas
    saida_array[y_coords[mascara], x_coords[mascara]] = img_array[src_y[mascara], src_x[mascara]]

    # Converte o array de volta para imagem PIL
    return Image.fromarray(saida_array)


# ==========================================
# PARTE 2: INTERFACE GRÁFICA (GUI) COM TKINTER
# ==========================================

class AplicativoGIF:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador de GIF Hipnótico - Trabalho Final")
        self.root.geometry("500x600")
        self.caminho_arquivo = ""

        # Título principal da janela
        tk.Label(root, text="Configuração da Animação", font=("Arial", 16, "bold")).pack(pady=10)

        # Área de seleção de arquivo GIF
        frame_arquivo = tk.Frame(root)
        frame_arquivo.pack(pady=10)

        # Botão para escolher o GIF original
        self.btn_selecionar = tk.Button(
            frame_arquivo,
            text="1. Selecionar GIF Original",
            command=self.selecionar_arquivo,
            bg="#dddddd"
        )
        self.btn_selecionar.pack(side=tk.LEFT, padx=5)

        # Rótulo que mostra o nome do arquivo selecionado
        self.lbl_arquivo = tk.Label(frame_arquivo, text="Nenhum arquivo selecionado", fg="red")
        self.lbl_arquivo.pack(side=tk.LEFT)

        # Slider da intensidade da rotação
        tk.Label(root, text="Intensidade da Rotação (Graus):").pack(anchor="w", padx=20)
        self.scale_rotacao = tk.Scale(root, from_=0, to=180, orient=tk.HORIZONTAL, length=400)
        self.scale_rotacao.set(20)  # valor padrão
        self.scale_rotacao.pack(pady=5)

        # Slider da intensidade do efeito de zoom (escala pulsante)
        tk.Label(root, text="Intensidade do Pulsar (Zoom):").pack(anchor="w", padx=20)
        self.scale_zoom = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.scale_zoom.set(30)  # valor padrão (30%)
        self.scale_zoom.pack(pady=5)

        # Slider da intensidade da tonalidade azul
        tk.Label(root, text="Intensidade do Efeito Azul:").pack(anchor="w", padx=20)
        self.scale_cor = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=400)
        self.scale_cor.set(50)  # valor padrão
        self.scale_cor.pack(pady=5)

        # Botão para começar o processamento do GIF
        self.btn_gerar = tk.Button(
            root,
            text="2. GERAR GIF HIPNÓTICO",
            command=self.processar,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            state=tk.DISABLED  # começa desabilitado até selecionar o arquivo
        )
        self.btn_gerar.pack(pady=30, ipadx=20, ipady=10)

        # Barra de status na parte inferior da janela
        self.lbl_status = tk.Label(root, text="Aguardando...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    # Função para abrir o seletor de arquivo e guardar o caminho do GIF
    def selecionar_arquivo(self):
        arquivo = filedialog.askopenfilename(filetypes=[("Arquivos GIF", "*.gif")])
        if arquivo:
            self.caminho_arquivo = arquivo
            # Atualiza o texto com o nome do arquivo escolhido
            self.lbl_arquivo.config(text=os.path.basename(arquivo), fg="green")
            # Habilita o botão de gerar GIF
            self.btn_gerar.config(state=tk.NORMAL)

    # Função principal que processa o GIF e gera o novo GIF transformado
    def processar(self):
        # Lê os valores dos sliders da interface
        max_rot = self.scale_rotacao.get()
        max_zoom = self.scale_zoom.get() / 100.0  # converte 30 em 0.3
        fator_cor = self.scale_cor.get()

        # Define o caminho do arquivo de saída na mesma pasta do original
        pasta = os.path.dirname(self.caminho_arquivo)
        nome_saida = os.path.join(pasta, "resultado_customizado.gif")

        try:
            gif_original = Image.open(self.caminho_arquivo)

            # Confere se o arquivo realmente é um GIF animado
            if not getattr(gif_original, "is_animated", False):
                messagebox.showerror("Erro", "A imagem selecionada não é um GIF animado!")
                return

            # Converte cada frame para RGBA e guarda em uma lista
            frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif_original)]
            novos_frames = []
            total = len(frames)

            # Atualiza a barra de status
            self.lbl_status.config(text="Processando... Por favor, aguarde. (A janela pode congelar levemente)")
            self.root.update()

            # Percorre todos os frames e aplica as transformações progressivas
            for i, frame in enumerate(frames):
                # t é o "tempo" normalizado ao longo do ciclo (0 a 2π)
                t = (i / total) * 2 * math.pi

                # A rotação varia de acordo com o seno ao longo do tempo
                angulo = max_rot * math.sin(t)

                # A escala também varia com o seno, criando o efeito de pulsar (zoom)
                escala = 1.0 + max_zoom * math.sin(t * 2)

                # Cisalhamento suave baseado em cosseno
                shear = 0.1 * math.cos(t)

                # Matriz composta: primeiro cisalhamento, depois escala, depois rotação
                M_composta = gerar_matriz_cisalhamento(shear, 0) @ \
                             gerar_matriz_escala(escala, escala) @ \
                             gerar_matriz_rotacao(angulo)

                # Aplica a transformação geométrica ao frame atual
                novo_frame = aplicar_transformacao_geometrica(frame, M_composta)

                # Efeito de cor azul progressivo
                if fator_cor > 0:
                    dados = np.array(novo_frame)
                    # A intensidade do azul varia com o valor do slider e o seno do tempo
                    add_blue = int(fator_cor * abs(math.sin(t)))

                    canal_b = dados[:, :, 2].astype(np.int16) + add_blue
                    dados[:, :, 2] = np.clip(canal_b, 0, 255).astype(np.uint8)

                    novo_frame = Image.fromarray(dados)

                # Adiciona o frame transformado à lista de novos frames
                novos_frames.append(novo_frame)

                # Atualiza a barra de status a cada 5 frames para manter a interface responsiva
                if i % 5 == 0:
                    self.lbl_status.config(text=f"Processando frame {i}/{total}...")
                    self.root.update()

            # Salva todos os frames como um novo GIF animado
            novos_frames[0].save(
                nome_saida,
                save_all=True,
                append_images=novos_frames[1:],
                duration=gif_original.info.get('duration', 100),
                loop=0
            )

            self.lbl_status.config(text="Concluído!")
            messagebox.showinfo("Sucesso!", f"GIF criado com sucesso em:\n{nome_saida}")

        except Exception as e:
            # Tratamento genérico de erro para evitar travar a aplicação
            messagebox.showerror("Erro Crítico", f"Ocorreu um erro:\n{e}")
            self.lbl_status.config(text="Erro.")


# --- INICIALIZAÇÃO DA APLICAÇÃO ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoGIF(root)
    root.mainloop()
