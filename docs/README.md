# PyRossGeo configuration files

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

<p align="center"><img src="https://rawgit.com/in	git@github.com:lukastk/PyRossGeo/master/svgs/c9abd835ffc3684b6c25c3f9849c346f.svg?invert_in_darkmode" align=middle width=175.80585pt height=117.28397999999999pt/></p>


<!--
<img style="text-align: center;" src="https://render.githubusercontent.com/render/math?math=
<p align="center"><img src="https://rawgit.com/in	git@github.com:lukastk/PyRossGeo/master/svgs/585113cec5630ff1b5886026d1551dee.svg?invert_in_darkmode" align=middle width=57.774255pt height=15.1337505pt/></p>">

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;<p align="center"><img src="https://rawgit.com/in	git@github.com:lukastk/PyRossGeo/master/svgs/be5c2c6047c80f747e1648afbfea5792.svg?invert_in_darkmode" align=middle width=841.25415pt height=117.28397999999999pt/></p>" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;<p align="center"><img src="https://rawgit.com/in	git@github.com:lukastk/PyRossGeo/master/svgs/be5c2c6047c80f747e1648afbfea5792.svg?invert_in_darkmode" align=middle width=841.25415pt height=117.28397999999999pt/></p>" title="<p align="center"><img src="https://rawgit.com/in	git@github.com:lukastk/PyRossGeo/master/svgs/bf0ddd7c5ad1f0f9b6861508bd279c80.svg?invert_in_darkmode" align=middle width=175.80585pt height=117.28397999999999pt/></p>" /></a>
-->

**Example:** SEAIR model

```json
{
    "classes" : ["S", "E", "A", "I", "R"],

    "S" : {
        "linear"    : [],
        "nonlinear" : [ ["I", "-betaI"], ["A", "-betaA"] ]
    },

    "E" : {
        "linear"    : [ ["E", "-gammaE"] ],
        "nonlinear" : [ ["I", "betaI"], ["A", "betaA"] ]
    },

    "A" : {
        "linear"    : [ ["E", "gammaE"], ["A", "-gammaA"] ],
        "nonlinear" : []
    },

    "I" : {
        "linear"    : [ ["A", "gammaA"], ["I", "-gammaI"] ],
        "nonlinear" : []
    },

    "R" : {
        "linear"    : [ ["I", "gammaI"] ],
        "nonlinear" : []
    }
}

```


We will go through each component of the `model.json` configuration file in order:

-  The list `"classes" : ["S", "E", "A", "I", "R"]` defines the epidemiological
classes of the model. <i>The order in which they are written are important</i>,
as this ordering must be maintained consistently with all other configuration files.
- The dynamics of each class is defined by a key-value pair
    ```json
    "I" : {
        "linear"    : [ ["I", "-gamma"] ],
        "infection" : [ ["I", "beta"] ]
    },
    ```
- 

Each model requires the presence of a susceptible class. This class
will always be the first element of the list `classes`, regardless
of whether it is labelled as `S` or not.

Note that we do not need to define the age-structure in `model.json`.
The contact matrices by which each infection term will interact
with the susceptible S class is defined in

<img style="text-align: center;" src="https://render.githubusercontent.com/render/math?math=x">

## `node_parameters.csv` (Defining the parameters of the infection model)

## `cnode_parameters.csv` (Defining the parameters of the infection model in the commuterverse)

## `contact_matrices.json` (Defining the contact matrices)

## `node_cmatrices.csv` (Assigning contact matrices at each node)

## `cnode_cmatrices.csv` (Assigning contact matrices at each commuter node)

## `node_populations.csv` (Populating the nodes)

## `cnode_populations.csv` (Populating the commuter nodes)