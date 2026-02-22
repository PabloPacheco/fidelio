# Finite Volume Method (FVM) Solver – Problemas Estacionarios en 2D

Este programa resuelve **problemas estacionarios de difusión** en dos dimensiones utilizando el **Método de Volúmenes Finitos (FVM)** sobre **mallas no estructuradas de elementos triangulares**.

El código está diseñado para:

* Manejar mallas triangulares no ortogonales.
* Resolver ecuaciones elípticas estacionarias.
* Incluir condiciones de borde tipo Dirichlet, Neumann y Robin.
* Incorporar correcciones por no ortogonalidad (difusión cruzada).

## Cálculo de gradiente

### Gradiente en $f$

**Gradiente normal en mallas ortogonales**

$$
(\nabla \phi \cdot \mathbf{n})_f = \left. \frac{\partial \phi}{\partial n} \right|_f = \frac{\phi_F - \phi_C}{\|\mathbf{r}_F - \mathbf{r}_C\|} = \frac{\phi_F - \phi_C}{d_{CF}}
$$

Vector unitario en la dirección CF

$$
\mathbf{e} = \frac{\mathbf{r}_F - \mathbf{r}_C}{\|\mathbf{r}_F - \mathbf{r}_C\|} = \frac{\mathbf{d}_{CF}}{d_{CF}}
$$

**Gradiente en dirección $e$ (malla no ortogonal)

$$
(\nabla \phi \cdot \mathbf{e})_f = \left. \frac{\partial \phi}{\partial e} \right|_f = \frac{\phi_F - \phi_C}{\|\mathbf{r}_F - \mathbf{r}_C\|} = \frac{\phi_F - \phi_C}{d_{CF}}
$$

### Difusión cruzada

Descomposición del vector de superficie

$$
\mathbf{S}_f = \mathbf{E}_f + \mathbf{T}_f
$$

Descomposición del flujo de difusión en mallas no ortogonales

$$
(\nabla \phi)_f \cdot \mathbf{S}_f = E_f \frac{\phi_F - \phi_C}{d_{CF}} + (\nabla \phi)_f \cdot \mathbf{T}_f
$$

Cálculo de $\mathbf{E}_f$

- Enfoque de corrección mínima

$$
\mathbf{E}_f = (\mathbf{e} \cdot \mathbf{S}_f) \mathbf{e} = (S_f \cos \theta) \mathbf{e}
$$

-  Enfoque sobre-relajado
$$
\mathbf{E}_f = \frac{\mathbf{S}_f \cdot \mathbf{S}_f}{\mathbf{e} \cdot \mathbf{S}_f} \mathbf{e}
$$

### Gradiente en $C$

Descomposición del vector de superficie

$$
\mathbf{S}_f = \mathbf{E}_f + \mathbf{T}_f
$$

Descomposición del flujo de difusión en mallas no ortogonales

$$
(\nabla \phi)_f \cdot \mathbf{S}_f = E_f \frac{\phi_F - \phi_C}{d_{CF}} + (\nabla \phi)_f \cdot \mathbf{T}_f
$$

Cálculo de $\mathbf{E}_f$

- Enfoque de corrección mínima

$$
\mathbf{E}_f = (\mathbf{e} \cdot \mathbf{S}_f) \mathbf{e} = (S_f \cos \theta) \mathbf{e}
$$

-  Enfoque sobre-relajado
$$
\mathbf{E}_f = \frac{\mathbf{S}_f \cdot \mathbf{S}_f}{\mathbf{e} \cdot \mathbf{S}_f} \mathbf{e}
$$



## Forma general de los coeficientes

**Forma algebraica general para mallas no ortogonales**
$$
a_C \phi_C + \sum_{F \in NB(C)} a_F \phi_F = b_C
$$

**Coeficientes para mallas no ortogonales**

$$
a_C = \sum_{f \in nb(C)} F_{Cf} + S_p V_C
$$

$$
a_F = F_{Ff}
$$


$$
b_C = S_c V_C + \sum_{f \in nb(C)} (\Gamma \nabla \phi)_f \cdot \mathbf{T}_f
$$


## Término difusivo

Se muestram los valores de $F_{Cf}$ y $F_{Ff}$ que aportan la discretización del término difusivo.

**Flujo de difusión en mallas no ortogonales - Forma expandida**
$$
\sum_{f \in nb(C)} (J_{\phi,D} \cdot \mathbf{S}_f) = \sum_{f \in nb(C)} \Gamma_f D_f (\phi_C - \phi_F) + \sum_{f \in nb(C)} (\Gamma \nabla \phi)_f \cdot \mathbf{T}_f 
$$

$$
\left(\sum_{f \in nb(C)} F_{Cf}\right)\phi_C + \sum_{f \in nb(C)} (F_{Ff} \phi_F) + \sum_{f \in nb(C)} F_{Vf}
$$

**Coeficiente de difusión geométrica para mallas no ortogonales**
$$
D_f = \frac{E_f}{d_{CF}}
$$

**Forma algebraica general para mallas no ortogonales**
$$
a_C \phi_C + \sum_{F \in NB(C)} a_F \phi_F = b_C
$$

**Coeficientes para mallas no ortogonales**

$$
a_F = F_{Ff} = -\Gamma_f D_f
$$

$$
a_C = \sum_{f \in nb(C)} F_{Cf} =  \sum_{f \in nb(C)} \Gamma_f D_f
$$

$$
b_C = Q_{\phi,C} V_C + \sum_{f \in nb(C)} (\Gamma \nabla \phi)_f \cdot \mathbf{T}_f
$$



## Condiciones de borde: 

Se detalla los términos $F_{Cf}$ y $F_{Ff}$ que aportan cada tipo de condición de borde. Además, se muestra cómo se calcula el valor de $\phi_b$ en el borde, dependiendo de cada condición.

### Newman

**Flujo a través de la frontera**

$$
(-\Gamma \nabla\phi)_{b} \cdot \mathbf{n} = q_{b}
$$

donde $q_{b}$ es el flujo de calor por unidad de área a través de la frontera, $\mathbf{n}$ vector normal a la frontera.

**Flujo de $\phi$ por difusión $D$ a través de una frontera $b$**

$$
J_{\phi,D,b} \cdot \mathbf{S}_{b} = (-\Gamma \nabla\phi)_{b} \cdot S_{b}\mathbf{n} = q_{b}S_{b} = F_{Cb}\phi_{C} + F_{Vb}
$$


$$
F_{Cb} = 0, \quad F_{Vb} = q_{b}S_{b}
$$

Cálculo de $\phi_b$ (variable en el borde bajo condición de Newman)

$$
\phi_{b} = \frac{\Gamma_{b}\,D_{b}\,\phi_{C} - q_{b}}{\Gamma_{b}\, D_{b}}
$$

$$
D_{b} = \frac{S_{b}}{d_{Cb}}
$$

### Dirichlet


**Discretización del flujo en la frontera para condición de Dirichlet**

$$
J_{\phi,D,b} \cdot \mathbf{S}_b = -\Gamma_b (\nabla \phi)_b \cdot \mathbf{S}_b = -\Gamma_b \frac{\phi_b - \phi_C}{d_{Cb}} E_b - \Gamma_b (\nabla \phi)_b \cdot \mathbf{T}_b = F_{Cb} \phi_C + F_{Vb}
$$

**Linealización en la frontera para Dirichlet - Malla no ortogonal**
$$
F_{Cb} = \Gamma_b D_b, \quad \quad F_{Vb} = -\Gamma_b D_b \phi_b - \Gamma_b (\nabla \phi)_b \cdot \mathbf{T}_b, \quad \quad D_b = \frac{E_b}{d_{Cb}}
$$

### Robin


#### Malla no ortogonal

**Balance de flujo en la frontera para condición mixta**
$$
J_{\phi,D,b} \cdot \mathbf{S}_b = -\Gamma_b \left[ \frac{\phi_b - \phi_C}{d_{Cb}} E_b + (\nabla \phi)_b \cdot \mathbf{T}_b \right] = -h_{\infty} (\phi_{\infty} - \phi_b) S_b
$$

**Flujo linealizado en la frontera para condición mixta**
$$
J_{\phi,D,b} \cdot \mathbf{S}_b = -\left[ \frac{h_{\infty} S_b \frac{\Gamma_b E_b}{d_{Cb}}}{h_{\infty} S_b + \frac{\Gamma_b E_b}{d_{Cb}}} \right] (\phi_{\infty} - \phi_C) - \frac{h_{\infty} S_b \Gamma_b (\nabla \phi)_b \cdot \mathbf{T}_b}{h_{\infty} S_b + \frac{\Gamma_b E_b}{d_{Cb}}} = F_{Cb} \phi_C + F_{Vb}
$$

**Coeficientes de linealización para frontera mixta**
$$
F_{Cb} = \frac{h_{\infty} S_b \frac{\Gamma_b E_b}{d_{Cb}}}{h_{\infty} S_b + \frac{\Gamma_b E_b}{d_{Cb}}}, \quad F_{Vb} = -F_{Cb} \phi_{\infty} - \frac{h_{\infty} S_b \Gamma_b (\nabla \phi)_b \cdot \mathbf{T}_b}{h_{\infty} S_b + \frac{\Gamma_b E_b}{d_{Cb}}}
$$


**Cálculo de $\phi$ en la frontera para condición mixta**
$$
\phi_b = \frac{h_{\infty} S_b \phi_{\infty} + \frac{\Gamma_b E_b}{d_{Cb}} \phi_C - \Gamma_b (\nabla \phi)_b \cdot \mathbf{T}_b}{h_{\infty} S_b + \frac{\Gamma_b E_b}{d_{Cb}}}
$$





#### Malla ortogonal

**Flujo en la frontera para condición mixta**

$$
J_{\phi,D,b} \cdot \mathbf{S}_{b} = -\left[ \frac{h_{\infty}(\Gamma_{\phi,b}/d_{xb})}{h_{\infty} + (\Gamma_{\phi,b}/d_{xb})} \right] S_{b}(\phi_{\infty} - \phi_{C}) = F_{Cb}\phi_{C} + F_{Vb}
$$

**Coeficientes de linealización para condición mixta**
$$
F_{Cb} = R_{eq}, \quad F_{Vb} = -R_{eq}\phi_{\infty}
$$


**Cálculo de $\phi$ en la frontera para condición mixta/Robin**

$$
\phi_{b} = \frac{h_{\infty}\phi_{\infty} + (\Gamma_{\phi,b}/d_{xb})\phi_{C}}{h_{\infty} + (\Gamma_{\phi,b}/d_{xb})}
$$

## $\phi_f$

$$
\phi_f = g_f \phi_F + (1-g_f)\phi_C
$$


$$
g_f = \frac{V_C}{V_C + V_F}
$$

$$
g_f =
\frac{\vec d_{Cf}\cdot \hat n_f}
{\vec d_{Cf}\cdot \hat n_f + \vec d_{fF}\cdot \hat n_f}
$$

* $ \vec d_{Cf} $  → `mesh.dCf`
* $ \vec d_{fF} $  → `mesh.dfF`
* $ \mathbf{S}_f $ → normal orientada
* $ \hat n_f = \mathbf{S}_f / |\mathbf{S}_f| $





## $\nabla \phi_f$

$$
\nabla \phi_f =
g_f \nabla \phi_F +
(1-g_f)\nabla \phi_C
$$

con

$$
g_f =
\frac{|\vec d_{Cf}|}
{|\vec d_{Cf}| + |\vec d_{fF}|}
$$

donde:

* $ \vec d_{Cf} $  → `mesh.dCf`
* $ \vec d_{fF} $  → `mesh.dfF`





$$
\nabla \phi_f = \nabla \phi_C
$$

$$
\nabla \phi_f =
g_f \nabla \phi_F
+
(1-g_f)\nabla \phi_C
$$

$$
g_f = \frac{|d_{Cf}|}{|d_{Cf}| + |d_{fF}|}
$$



$$
-\Gamma_b \frac{\phi_b - \phi_C}{d_{Cb}} E_b - \Gamma_b (\nabla\phi)_b \cdot \mathbf{T}_b = S_p \phi_C + S_c
$$


$$
\phi_b = \phi_C - \frac{d_{Cb}}{\Gamma_b E_b} \left( S_p \phi_C + S_c + \Gamma_b (\nabla\phi)_b \cdot \mathbf{T}_b \right)
$$






