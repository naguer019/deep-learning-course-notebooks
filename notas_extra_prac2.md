### Explicación de la Formulación Matricial

Lo que se introduce en la parte 3.1 es cómo escribir una red neuronal matemáticamente usando **matrices y vectores** en lugar de escribir una ecuación separada para cada neurona individual.

Esta forma matricial es exactamente como PyTorch y TensorFlow procesan los datos 'por debajo', ya que las GPU están optimizadas para hacer multiplicaciones de matrices muy rápido.

Vamos a analizar las dos ecuaciones sabiendo que nuestra red tiene:

- $D_i$ características de entrada (input size).
- $D$ neuronas ocultas (hidden units).
- $D_o$ valores de salida (output size).

**Primera ecuación (De la entrada a la capa oculta):**
$$ \mathbf{h} = \text{a}[\boldsymbol{\theta}_0 + \boldsymbol{\Theta}\mathbf{x}] $$

- $\mathbf{x}$: Es un **vector columna** que contiene los $D_i$ valores de entrada.
- $\boldsymbol{\Theta}$: Es la **matriz de pesos** (weights). Conecta los $D_i$ inputs con las $D$ neuronas ocultas. En álgebra lineal, para transformar un vector de tamaño $D_i$ en un vector de tamaño $D$, la matriz debe ser de tamaño $D \times D_i$ (Siempre es: `destino × origen`).
- $\boldsymbol{\theta}_0$: Es el **vector de sesgos** (biases) de la capa oculta. Le suma un valor particular temporal a cada una de las $D$ neuronas ocultas, por lo que su dimensión es $D \times 1$.
- $\text{a}[...]$: Es la **función de activación** (como ReLU o Sigmoid). Se aplica elemento a elemento al resultado, introduciendo la 'no linealidad'.
- $\mathbf{h}$: Es el **vector resultante** con las activaciones de las neuronas ocultas, de dimensión $D \times 1$.

**Segunda ecuación (De la capa oculta a la salida):**
$$ \mathbf{y} = \boldsymbol{\phi}\_0 + \boldsymbol{\Phi}\mathbf{h} $$

- Esto hace exactamente lo mismo que el paso anterior, pero ahora toma como "entrada" el vector $\mathbf{h}$ que obtuvimos recién.
- $\boldsymbol{\Phi}$: Es la **nueva matriz de pesos** para llegar a la salida. Como vamos de $D$ componentes ocultos a $D_o$ componentes de salida, su tamaño es $D_o \times D$.
- $\boldsymbol{\phi}_0$: Es el **vector de sesgos** para las neuronas de salida, de tamaño $D_o \times 1$.
- $\mathbf{y}$: Es el **vector de predicción final**, de tamaño $D_o \times 1$.

---

### Completando las Tablas

Para completar las tablas formalmente, vamos a indicar sus dimensiones estándar con la notación de filas por columnas (`filas × columnas`).

#### **Part A — `ShallowNet1D` with $D$ hidden units (Section 1)**

En este caso, entra solo **1 número** ($D_i = 1$) y sale solo **1 número** ($D_o = 1$). La capa oculta tiene $D$ unidades.

| Symbol                  | Meaning                  | Dimensions   |
| ----------------------- | ------------------------ | ------------ |
| $\mathbf{x}$            | Input                    | $1 \times 1$ |
| $\boldsymbol{\theta}_0$ | Hidden biases            | $D \times 1$ |
| $\boldsymbol{\Theta}$   | Input-to-hidden weights  | $D \times 1$ |
| $\mathbf{h}$            | Hidden activations       | $D \times 1$ |
| $\boldsymbol{\phi}_0$   | Output bias              | $1 \times 1$ |
| $\boldsymbol{\Phi}$     | Hidden-to-output weights | $1 \times D$ |
| $\mathbf{y}$            | Output                   | $1 \times 1$ |

**Total number of parameters (as a formula in $D$):** `3D + 1`

> **¿De dónde sale esta fórmula?**
>
> - Capa 1: Hay $D$ pesos (en la matriz $\boldsymbol{\Theta}$) + $D$ sesgos (en el vector $\boldsymbol{\theta}_0$).
> - Capa 2: Hay $D$ pesos (en la matriz $\boldsymbol{\Phi}$) + $1$ sesgo (en el vector $\boldsymbol{\phi}_0$).
> - Suma total = $D + D + D + 1 = 3D + 1$.

---

#### **Part B — `ShallowNet` on FashionMNIST with $D$ hidden units (Section 3)**

En este caso, la entrada es una imagen aplanada de **784 píxeles** ($D_i = 784$) y la salida son las probabilidades logarítmicas de las **10 clases de prendas de ropa** ($D_o = 10$). La capa oculta tiene $D$ unidades.

| Symbol                  | Meaning                  | Dimensions     |
| ----------------------- | ------------------------ | -------------- |
| $\mathbf{x}$            | Flattened image          | $784 \times 1$ |
| $\boldsymbol{\theta}_0$ | Hidden biases            | $D \times 1$   |
| $\boldsymbol{\Theta}$   | Input-to-hidden weights  | $D \times 784$ |
| $\mathbf{h}$            | Hidden activations       | $D \times 1$   |
| $\boldsymbol{\phi}_0$   | Output bias              | $10 \times 1$  |
| $\boldsymbol{\Phi}$     | Hidden-to-output weights | $10 \times D$  |
| $\mathbf{y}$            | Class logits             | $10 \times 1$  |

**Total number of parameters for $D = 256$:** `203,530`

> **¿Cómo llegamos a ese número?** Vamos a aplicar las dimensiones de las matrices que calculamos sabiendo que D=256:
>
> - Capa 1 pesos ($\boldsymbol{\Theta}$): $256 \times 784 = 200,704$ parámetros
> - Capa 1 sesgos ($\boldsymbol{\theta}_0$): $256$ parámetros
> - Capa 2 pesos ($\boldsymbol{\Phi}$): $10 \times 256 = 2,560$ parámetros
> - Capa 2 sesgos ($\boldsymbol{\phi}_0$): $10$ parámetros
> - Total = $200,704 + 256 + 2,560 + 10 = \mathbf{203,530}$

¡Espero que esto te ayude a ver clarísimamente cómo se arman estas redes matricialmente! ¿Tienes alguna otra duda con estas ecuaciones?

Created notas_extra_prac2.md

¡Excelente pregunta! Y la respuesta es **sí, absolutamente afectan**.

Justamente, esos números son la razón por la cual la tabla de la **Parte B** es diferente a la de la Parte A.

Cuando en la letra del problema te dicen _"Each image is flattened to **784** features; there are **10** output classes"_, te están dando los valores exactos para las variables de entrada y salida:

- $D_i = 784$ (el tamaño de la entrada)
- $D_o = 10$ (el tamaño de la salida)

Por lo tanto, en la tabla de la Parte B **no debes poner** dimensiones genéricas como "$D_i \times 1$", ni poner "$1 \times 1$" como en la primera tabla. **Debes poner exactamente los números 784 y 10** en los lugares correspondientes.

Es decir, en tu Jupyter Notebook, la tabla que debes completar debería verse exactamente así, usando los números (como te pasé en el mensaje anterior):

- $\mathbf{x}$ pasa a ser de $784 \times 1$ (porque entran 784 píxeles).
- $\boldsymbol{\Theta}$ pasa a ser de $D \times 784$ (porque conecta los 784 inputs hacia las $D$ neuronas ocultas).
- $\boldsymbol{\phi}_0$ pasa a ser de $10 \times 1$ (porque hay 10 clases de salida).
- $\boldsymbol{\Phi}$ pasa a ser de $10 \times D$ (porque conecta las $D$ ocultas hacia las 10 salidas).
- $\mathbf{y}$ pasa a ser de $10 \times 1$ (porque el resultado final son 10 valores).

En resumen: sí afectan, y debes incluir explícitamente el `784` y el `10` al rellenar esa segunda tabla para que esté correcta 😉.

# 3.1

¡Sí, **está perfecto**! 🎉

El código que escribiste cumple con todos los requisitos del ejercicio. Aquí te detallo por qué está excelente:

1. **La estructura de la red (`__init__`)**:
   Definiste correctamente la capa oculta con `nn.Linear(input_dim, hidden_dim)`, seguida de un `nn.ReLU()`, y finalmente la capa de salida con `nn.Linear(hidden_dim, output_dim)`. No pusiste `Softmax` al final, lo cual es exactamente lo que pide la letra (porque como bien dice, `CrossEntropyLoss` ya lo aplica implícitamente).
2. **El "aplanado" (Flattening) en el `forward`**:
   La línea `x = x.view(x.size(0), -1)` es la forma clásica y más correcta en PyTorch de lograr esto. - `x.size(0)` mantiene intacta la dimensión del batch (la cantidad de imágenes). - `1` le dice a PyTorch que multiplique todas las dimensiones restantes (`1 * 28 * 28 = 784`) para que las condense en un solo vector. Así pasas de `(batch, 1, 28, 28)` a `(batch, 784)`.

**Un pequeño tip adicional (opcional):**
Otra forma equivalente y un poco más moderna en PyTorch de hacer ese aplanado es usar `x = x.flatten(start_dim=1)`. ¡Pero usar `view` como tú lo hiciste está perfecto y es 100% válido profesionalmente!

## Como funciona .view

En PyTorch, la función `.view()` sirve para **cambiar la forma (dimensiones) de un tensor sin cambiar los datos que contiene adentro**. Imagínala como si tuvieras una caja de 12 huevos: puedes verlos como 2 filas de 6, 3 filas de 4, o una sola fila larga de 12. Los huevos (los datos) son los mismos, solo cambia cómo los "ves".

Vamos a desglosar exactamente qué hace esta línea específica:
`x = x.view(x.size(0), -1)`

### 1. El primer parámetro: `x.size(0)`

Este término obtiene el tamaño de la primera dimensión del tensor original. En deep learning, la dimensión 0 **siempre es el tamaño del batch** (es decir, cuántas imágenes estás procesando a la vez, por ejemplo, 32 o 64 imágenes).
Al poner `x.size(0)` ahí, le estás diciendo a PyTorch: _"Mantén el número de imágenes intacto. No me mezcles píxeles de una imagen con otra."_

### 2. El segundo parámetro: `1`

Este es un "truco de magia" muy usado en PyTorch. El `-1` significa: _"PyTorch, fíjate cuántos datos quedan y calcula esta dimensión por mí automáticamente"_.

### Viendo el "Antes y Después" en tu red:

Tu red recibe un tensor `x` con forma 4D (4 dimensiones): `(batch, canales, alto, ancho)`.
Por ejemplo, si tienes 32 imágenes de FashionMNIST, su forma inicial es:
`(32, 1, 28, 28)`

Al aplicar `x.view(x.size(0), -1)`:

- PyTorch ve que `x.size(0)` es **32**. Así que la nueva forma empezará en `(32, ...)`.
- PyTorch ve el `1` y dice: _"A ver, si originalmente tenía 32 × 1 × 28 × 28 elementos, y ahora quiero mantener 32 filas, ¿cuántos elementos me quedan por fila? 1 × 28 × 28, ¡que es 784!"_
- Por lo tanto, PyTorch automáticamente agrupa todo lo que sobra y transforma (alisa) las dimensiones restantes en una sola dimensión de tamaño **784**.

La nueva forma del tensor será en 2D: `(32, 784)`.

### ¿Por qué lo necesitamos?

Las capas lineales como `nn.Linear(input_dim, hidden_dim)` (que conectan las neuronas una a una) **requieren vectores planos (1D)** para cada imagen. No saben qué hacer con imágenes cuadradas en 2D (matrices `28x28`). Entonces, tu `view` se encarga de estirar ese cuadrado de la imagen en una tira larga de características antes de que entre a la red.
