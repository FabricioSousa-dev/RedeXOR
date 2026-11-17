"""Rede neural para aprender XOR, sem bibliotecas externas.

XOR vale 1 quando as duas entradas são diferentes.
Rede com duas entradas, dois neurônios ocultos e uma saída.
"""

import math
import random

TAXA_APRENDIZADO = 0.8
EPOCAS = 10_000
SEMENTE = 0


def sigmoide(soma):
    """Transforma a soma do neurônio em um valor entre 0 e 1."""
    return 1.0 / (1.0 + math.exp(-soma))


def derivada_sigmoide(ativacao):
    """A inclinação da sigmoide pode ser calculada pelo valor que ela devolveu."""
    return ativacao * (1.0 - ativacao)


class RedeXor:
    def __init__(self, semente=SEMENTE):
        sorteio = random.Random(semente)

        # Matriz 2 x 2. Cada linha pertence a um neurônio oculto.
        # A coluna 0 guarda o peso de A e a coluna 1, o de B.
        # pesos_ocultos[1][0] liga a primeira entrada ao segundo neurônio oculto.
        self.pesos_ocultos = [[0.0, 0.0], [0.0, 0.0]]

        # Um bias por neurônio oculto, somado antes de aplicar a sigmoide.
        self.bias_ocultos = [0.0, 0.0]

        # A saída é um único neurônio. pesos_saida[0] liga o primeiro oculto
        # à saída e pesos_saida[1] liga o segundo.
        self.pesos_saida = [0.0, 0.0]
        self.bias_saida = 0.0

        # Resultado de cada oculto, usado na saída e no ajuste.
        self.ativacoes_ocultas = [0.0, 0.0]

        # Os pesos começam entre -1 e 1. Os bias começam em zero.
        for neuronio in range(2):
            for indice_entrada in range(2):
                self.pesos_ocultos[neuronio][indice_entrada] = 2 * sorteio.random() - 1
            self.pesos_saida[neuronio] = 2 * sorteio.random() - 1

    def propagar(self, entrada):
        """Forward propagation. entrada[0] é A e entrada[1] é B."""
        for neuronio in range(2):
            soma_oculta = self.bias_ocultos[neuronio]
            for indice_entrada in range(len(entrada)):
                soma_oculta += entrada[indice_entrada] * self.pesos_ocultos[neuronio][indice_entrada]
            self.ativacoes_ocultas[neuronio] = sigmoide(soma_oculta)

        soma_saida = self.bias_saida
        for neuronio in range(2):
            soma_saida += self.ativacoes_ocultas[neuronio] * self.pesos_saida[neuronio]
        return sigmoide(soma_saida)

    def ajustar(self, entrada, esperado):
        saida = self.propagar(entrada)
        erro = saida - esperado

        # Backpropagation. A perda deste exemplo é 0,5 * erro^2.
        # Cada delta mede quanto a perda varia com a soma do neurônio.
        delta_saida = erro * derivada_sigmoide(saida)

        # Um delta por oculto. Usa os pesos da saída antes de alterá-los.
        deltas_ocultos = [
            delta_saida * self.pesos_saida[n] * derivada_sigmoide(self.ativacoes_ocultas[n])
            for n in range(2)
        ]

        # No peso, o gradiente é o delta vezes o valor que chega pela ligação.
        # No bias, o gradiente é o próprio delta, pois ele entra somado.
        # O sinal de menos escolhe o sentido em que a perda tende a diminuir.
        for neuronio in range(2):
            delta_oculto = deltas_ocultos[neuronio]
            for indice_entrada in range(len(entrada)):
                self.pesos_ocultos[neuronio][indice_entrada] -= (
                    TAXA_APRENDIZADO * delta_oculto * entrada[indice_entrada]
                )
            self.bias_ocultos[neuronio] -= TAXA_APRENDIZADO * delta_oculto
            self.pesos_saida[neuronio] -= (
                TAXA_APRENDIZADO * delta_saida * self.ativacoes_ocultas[neuronio]
            )
        self.bias_saida -= TAXA_APRENDIZADO * delta_saida


def main():
    # Cada linha é um exemplo com os valores de A e B.
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # Na mesma ordem das linhas de `entradas`.
    esperados = [0, 1, 1, 0]

    rede = RedeXor()

    # Uma época é uma passagem pelos quatro exemplos.
    for _ in range(EPOCAS):
        for entrada, esperado in zip(entradas, esperados):
            rede.ajustar(entrada, esperado)

    print("A XOR B | Esperado | Saída    | Previsto")
    for entrada, esperado in zip(entradas, esperados):
        saida = rede.propagar(entrada)
        # A partir de 0,5, a saída decimal é interpretada como 1.
        previsto = 1 if saida >= 0.5 else 0
        print(f"{entrada[0]} XOR {entrada[1]} | {esperado:8d} | {saida:.6f} | {previsto:8d}")


if __name__ == "__main__":
    main()