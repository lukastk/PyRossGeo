# PyRossGeo configuration files

**General note of caution:** <i>At the moment PyRossGeo does not have any checks in place to detect formatting errors in the configuration files. Any mistakes in the configuration files may lead to uncaught exceptions or unexpected simulation behaviour. This will be rectified in the future.</i>

- [PyRossGeo configuration files](#pyrossgeo-configuration-files)
  - [`model.json` (Local infection dynamics)](#modeljson-local-infection-dynamics)
  - [`node_parameters.csv` (Defining the parameters of the infection model)](#nodeparameterscsv-defining-the-parameters-of-the-infection-model)
  - [`cnode_parameters.csv` (Defining the parameters of the infection model in the commuterverse)](#cnodeparameterscsv-defining-the-parameters-of-the-infection-model-in-the-commuterverse)
  - [`contact_matrices.json` (Defining the contact matrices)](#contactmatricesjson-defining-the-contact-matrices)
  - [`node_cmatrices.csv` (Assigning contact matrices at each node)](#nodecmatricescsv-assigning-contact-matrices-at-each-node)
  - [`cnode_cmatrices.csv` (Assigning contact matrices at each commuter node)](#cnodecmatricescsv-assigning-contact-matrices-at-each-commuter-node)
  - [`node_populations.csv` (Populating the nodes)](#nodepopulationscsv-populating-the-nodes)
  - [`cnode_populations.csv` (Populating the commuter nodes)](#cnodepopulationscsv-populating-the-commuter-nodes)

## `model.json` (Local infection dynamics)

The infection dynamics at each geographical node is defined using the
`model.json` configuration file. This Using it, we can define any variant of the
age-bracketed SIR model (e.g. SEIR, SEAIR, SEAI5R).

Below are examples of `model.json` files defining the SIR and
SEAIR models:

**Example:** SIR model

```json
{
    "classes" : ["S", "I", "R"],

    "S" : {
        "linear"    : [],
        "infection" : [ ["I", "-beta"] ]
    },

    "I" : {
        "linear"    : [ ["I", "-gamma"] ],
        "infection" : [ ["I", "beta"] ]
    },

    "R" : {
        "linear"    : [ ["I", "gamma"] ],
        "infection" : []
    }
}

```

This corresponds to

$$
\begin{aligned}
\dot{S}^\mu & = - \beta \sum_\nu C_{\mu \nu} \frac{I^\nu}{N^\nu} S^\mu \\
\dot{I}^\mu & = \beta \sum_\nu C_{\mu \nu} \frac{I^\nu}{N^\nu} - \gamma I \\
\dot{R}^\mu & = \gamma I
\end{aligned}
$$

where the $\mu$ corresponds the age bracket. Indices giving the home
and locations $(i,j)$ have been omitted.

**Example:** SEAIR model

```json
{
    "classes" : ["S", "E", "A", "I", "R"],

    "S" : {
        "linear"    : [],
        "infection" : [ ["I", "-betaI"], ["A", "-betaA"] ]
    },

    "E" : {
        "linear"    : [ ["E", "-gammaE"] ],
        "infection" : [ ["I", "betaI"], ["A", "betaA"] ]
    },

    "A" : {
        "linear"    : [ ["E", "gammaE"], ["A", "-gammaA"] ],
        "infection" : []
    },

    "I" : {
        "linear"    : [ ["A", "gammaA"], ["I", "-gammaI"] ],
        "infection" : []
    },

    "R" : {
        "linear"    : [ ["I", "gammaI"] ],
        "infection" : []
    }
}

```

This corresponds to

$$
\begin{aligned}
\dot{S}^\mu & = - \lambda^\mu(t) S^\mu  \\
\dot{E}^\mu & = \lambda^\mu(t) S^\mu - \gamma_E E \\
\dot{A}^\mu & = \gamma_E E - \gamma_A A \\
\dot{I}^\mu & =  \gamma_A A  - \gamma_I I \\
\dot{R}^\mu & = \gamma_I I
\end{aligned}
$$

where

$$
\lambda^\mu(t) = \sum_\nu \left( \beta_I C^I_{\mu\nu} \frac{I^\nu}{N^\nu} + \beta_A C^A_{\mu\nu} \frac{A^\nu}{N^\nu} \right)
$$

The superscripts in the contact matrices $C^A_{\mu\nu}$ signify the fact
that each infection class can have its own contact structure. This accomodates for the fact that, for example, asymptomatic infecteds and
symptomatic infects have different social behaviour.

We will go through each component of the `model.json` configuration file in order:

-  The list `"classes" : ["S", "E", "A", "I", "R"]` defines the epidemiological
classes of the model. <i>The order in which they are written are important</i>, as this ordering must be maintained consistently with all other configuration files. Each model requires the presence of a susceptible class. This class
will always be the first element of the list `classes`, regardless
of whether it is labelled as `S` or not.
- The dynamics of each class is defined by a key-value pair
    ```json
    "E" : {
        "linear"    : [ ["E", "-gammaE"] ],
        "infection" : [ ["I", "betaI"], ["A", "betaA"] ]
    },
    ```
  - In order, this reads out as: $\dot{E}^\mu = -\gamma_E E + \beta_I \sum_\nu C^I_{\mu \nu} \frac{I^\nu}{N^\nu} S^\mu + \beta_A \sum_\nu C^A_{\mu \nu} \frac{A^\nu}{N^\nu} S^\mu$.
  - The linear terms for each epidemic class is defined by the lists of lists:
    ```json
    "linear"    : [ ["E", "-gammaE"] ]
    ```
    Eeach pair in `linear` corresponds to the linear coupling 
    with the class and the coupling constant respectively. So
    `["E", "-gammaE"]` corresponds to the term $-\gamma_E E$ in
    the equation for $\dot{E}$. The minus sign in front of `gammaE`
    signifies that the negative of the coefficient should be used.
  - The infection terms are defined in a similar manner. Each pair
    in `infection` corresponds to the non-linear coupling with $S$
    and the coupling constant respectively. So `["I", "betaI"]`
    corresponds to the term $\beta_I \sum_\nu C^I_{\mu \nu} \frac{I^\nu}{N^\nu} S$.

Note that we do not need to define the age-structure in `model.json`.
The contact matrices by which each infection term will interact
with the susceptible S class is defined in [`node_cmatrices.csv`](#nodecmatricescsv-assigning-contact-matrices-at-each-node) and [`cnode_cmatrices.csv`](#cnodecmatricescsv-assigning-contact-matrices-at-each-commuter-node).



## `node_parameters.csv` (Defining the parameters of the infection model)

The values of the parameters defined in [`model.json`](#modeljson-local-infection-dynamics) are given in the `node_parameters.csv` configuration file. The corresponding configuration file for
the commuterverse is given in [`cnode_parameters.csv`](#cnodeparameterscsv-defining-the-parameters-of-the-infection-model-in-the-commuterverse).

For each node $(\alpha,i,j)$, which represents age, home and location respectively, `node_parameters.csv` sets the model coefficients.

Consider the SEAIR model

$$
\begin{aligned}
\dot{S}^\mu & = - \lambda^\mu(t) S^\mu  \\
\dot{E}^\mu & = \lambda^\mu(t) S^\mu - \gamma_E E \\
\dot{A}^\mu & = \gamma_E E - \gamma_A A \\
\dot{I}^\mu & =  \gamma_A A  - \gamma_I I \\
\dot{R}^\mu & = \gamma_I I
\end{aligned}
$$

where

$$
\lambda^\mu(t) = \sum_\nu \left( \beta_I C^I_{\mu\nu} \frac{I^\nu}{N^\nu} + \beta_A C^A_{\mu\nu} \frac{A^\nu}{N^\nu} \right).
$$

and the index $\mu$ represents the age-bracket. Indices giving the home
and locations $(i,j)$ have been omitted.

Below is an example of a `node_parameters.csv` file for the SEAIR model.

**Example:** Model parameters for the SEAIR model

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>Location</th>
            <th>Age</th>
            <th>betaI</th>
            <th>betaA</th>
            <th>gammaE</th>
            <th>gammaA</th>
            <th>gammaI</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ALL</td>
            <td>ALL</td>
            <td>ALL</td>
            <td>0.02</td>
            <td>0.04</td>
            <td>0.03</td>
            <td>0.05</td>
            <td>0.02</td>
        </tr>
        <tr>
            <td>0</td>
            <td>1</td>
            <td>0</td>
            <td>0.06</td>
            <td>0.08</td>
            <td>0.03</td>
            <td>0.05</td>
            <td>0.02</td>
        </tr>
        <tr>
            <td>2</td>
            <td>2</td>
            <td>ALL</td>
            <td>0.01</td>
            <td>0.01</td>
            <td>0.05</td>
            <td>0.05</td>
            <td>0.07</td>
        </tr>
    </tbody>
</table>
</div>



The `Home`, `Location` and `Age` columns corresponds to $(\alpha, i, j)$ respectively. 

## `cnode_parameters.csv` (Defining the parameters of the infection model in the commuterverse)

## `contact_matrices.json` (Defining the contact matrices)

## `node_cmatrices.csv` (Assigning contact matrices at each node)

## `cnode_cmatrices.csv` (Assigning contact matrices at each commuter node)

## `node_populations.csv` (Populating the nodes)

## `cnode_populations.csv` (Populating the commuter nodes)