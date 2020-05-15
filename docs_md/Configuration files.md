# Configuration files

**General note of caution:** <i>At the moment PyRossGeo does not have any checks in place to detect formatting errors in the configuration files. Any mistakes in the configuration files may lead to uncaught exceptions or unexpected simulation behaviour. This will be rectified in the future.</i>

**Note:** <i>PyRossGeo allows for a great deal of configuration, 
but in practice the parameter space is kept small and manageable.
The true volume of the parameter space for the model can be found in
[model specification](./model.pdf)</i>

**Table of contents:**

- [Configuration files](#configuration-files)
  - [`model.json`](#modeljson)
  - [`node_parameters.csv`](#node_parameterscsv)
  - [`cnode_parameters.csv`](#cnode_parameterscsv)
  - [`contact_matrices.json`](#contact_matricesjson)
  - [`node_cmatrices.csv`](#node_cmatricescsv)
  - [`cnode_cmatrices.csv`](#cnode_cmatricescsv)
  - [`node_populations.csv`](#node_populationscsv)
  - [`cnode_populations.csv`](#cnode_populationscsv)
  - [`commuter_networks.csv`](#commuter_networkscsv)

## `model.json`

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

<p align="center"><img src="svgs/f60cde6602a325f67daa9089cce63474.svg?invert_in_darkmode" align=middle width=197.70299999999997pt height=117.28397999999999pt/></p>

where the <img src="svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.904950000000003pt height=14.155350000000013pt/> corresponds the age bracket. Indices giving the home
and locations <img src="svgs/aa20264597f5a63b51587e0581c48f2c.svg?invert_in_darkmode" align=middle width=33.46497000000001pt height=24.65759999999998pt/> have been omitted.

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

<p align="center"><img src="svgs/c7ca94df7d0969dd950f82e279b447cf.svg?invert_in_darkmode" align=middle width=161.56403999999998pt height=128.83794pt/></p>

where

<p align="center"><img src="svgs/50e99d01b3ca04fdf496d651babfcb3c.svg?invert_in_darkmode" align=middle width=276.75615pt height=42.73994999999999pt/></p>

The superscripts in the contact matrices <img src="svgs/bec9a3cba8019a312901175bcdde150a.svg?invert_in_darkmode" align=middle width=27.099105000000005pt height=27.656969999999987pt/> signify the fact
that each infection class can have its own contact structure. This allows for us to take into the account that, for example, asymptomatic infecteds and
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
  - In order, this reads out as: <img src="svgs/38eb20f19bf4c6b091ed12ada70d5c38.svg?invert_in_darkmode" align=middle width=368.74315499999994pt height=30.267599999999987pt/>.
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
with the susceptible S class is defined in [`node_cmatrices.csv`](#nodecmatricescsv) and [`cnode_cmatrices.csv`](#cnodecmatricescsv).



## `node_parameters.csv`

The values of the parameters defined in [`model.json`](#modeljson-local-infection-dynamics) are given in the `node_parameters.csv` configuration file. The corresponding configuration file for
the commuterverse is given in [`cnode_parameters.csv`](#cnodeparameterscsv).

For each node <img src="svgs/4f79053d87a6fb03853a78308c8abba5.svg?invert_in_darkmode" align=middle width=51.34734000000001pt height=24.65759999999998pt/>, which represents age, home and location respectively, `node_parameters.csv` sets the model coefficients.

Consider the SEAIR model

<p align="center"><img src="svgs/42a668e8fd65daad98789d7553788668.svg?invert_in_darkmode" align=middle width=153.57111pt height=128.83794pt/></p>

where

<p align="center"><img src="svgs/73eff1363547c26b3b405699cc179342.svg?invert_in_darkmode" align=middle width=284.06235pt height=42.73994999999999pt/></p>

and the index <img src="svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.904950000000003pt height=14.155350000000013pt/> represents the age-bracket. Indices giving the home
and locations <img src="svgs/aa20264597f5a63b51587e0581c48f2c.svg?invert_in_darkmode" align=middle width=33.46497000000001pt height=24.65759999999998pt/> have been omitted.

Below is an example of a `node_parameters.csv` file for the SEAIR model.

**Example:** Parameters for the SEAIR model

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
            <td>ALL</td>
            <td>HOME</td>
            <td>ALL</td>
            <td>0.05</td>
            <td>0.01</td>
            <td>0.07</td>
            <td>0.05</td>
            <td>0.01</td>
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
<br>


The `Home`, `Location` and `Age` columns corresponds to <img src="svgs/1b711759d62c78b9d4b4a2c031f451b4.svg?invert_in_darkmode" align=middle width=51.34734000000001pt height=24.65759999999998pt/> respectively, and the subsequent columns give the parameter values for the model. The ordering of the first three columns can not be switched, but the columns for the model parameters can be given in any order.

Keywords:
- The keyword `ALL` matches all values of <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.576500000000003pt height=14.155350000000013pt/>, <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/> and <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710483000000004pt height=21.683310000000006pt/>.
- The keyword `HOME` copies the value of the Home column.

Each row is read sequentially, meaning that although the first row in
the example sets the parameters for all nodes, the subsequent nodes
overwrite these values for specific nodes.

The following row sets the model parameters for residents of 0 who are at location 1, and belong to age-bracket 0:

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
            <td>0</td>
            <td>1</td>
            <td>0</td>
            <td>0.06</td>
            <td>0.08</td>
            <td>0.03</td>
            <td>0.05</td>
            <td>0.02</td>
        </tr>
    </tbody>
</table>

The following row sets the model parameters for residents of 2 who are at location 2, and belong to any age-bracket.

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

The following row sets parameters for all nodes <img src="svgs/1b711759d62c78b9d4b4a2c031f451b4.svg?invert_in_darkmode" align=middle width=51.34734000000001pt height=24.65759999999998pt/>, for which <img src="svgs/cd479494c39d95dc8012b12ea2a67946.svg?invert_in_darkmode" align=middle width=35.291355pt height=21.683310000000006pt/>, using the keyword `HOME`.

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
            <td>HOME</td>
            <td>ALL</td>
            <td>0.05</td>
            <td>0.01</td>
            <td>0.07</td>
            <td>0.05</td>
            <td>0.01</td>
        </tr>
    </tbody>
</table>

Lastly, the following row sets the model parameters for all nodes:

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
    </tbody>
</table>

## `cnode_parameters.csv`

The model parameters for each commuterverse node is given in `cnode_parameters.csv`. For each commuter node <img src="svgs/a6c3b8d83b10ce5e1aced9038f6944d8.svg?invert_in_darkmode" align=middle width=85.99338pt height=24.65759999999998pt/>, which represents age, home and origin, and destination respectively, `cnode_parameters.csv` sets the model coefficients.

This configuration file works in largely the same way as [`node_parameters.csv`](#nodeparameterscsv), and so ground covered there will not be repeated in this section.

**Example:** Parameters for the SEAIR model at the commuter nodes

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>From</th>
            <th>To</th>
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
            <td>ALL</td>
            <td>0.02</td>
            <td>0.04</td>
            <td>0.03</td>
            <td>0.05</td>
            <td>0.02</td>
        </tr>
        <tr>
            <td>ALL</td>
            <td>HOME</td>
            <td>ALL</td>
            <td>ALL</td>
            <td>0.31</td>
            <td>0.04</td>
            <td>0.01</td>
            <td>0.02</td>
            <td>0.07</td>
        </tr>
        <tr>
            <td>0</td>
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

Keywords:
- The keyword `ALL` matches all values of <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.576500000000003pt height=14.155350000000013pt/>, <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/> and <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710483000000004pt height=21.683310000000006pt/>.
- The keyword `HOME` copies the value of the Home column.

## `contact_matrices.json`

All contact matrices used in the simulation are defined in `contact_matrices.json`. An example is shown below.

**Example:** Contact matrices with three age groups

```json
{
    "C_home" : [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ],

    "C_away" : [
        [1,   0.5, 0  ],
        [0.5, 1,   0.5],
        [0,   0.5, 1  ]
    ],

    "C_transport" : [
        [3, 4, 2],
        [5, 4, 2],
        [2, 3, 2]
    ]
}
```

The row and column indices of the arrays coresponds to <img src="svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.904950000000003pt height=14.155350000000013pt/> and <img src="svgs/b49211c7e49541e500c32b4d56d354dc.svg?invert_in_darkmode" align=middle width=9.166740000000003pt height=14.155350000000013pt/> in <img src="svgs/d993d287d4cf042f9a6b318dd4a76b87.svg?invert_in_darkmode" align=middle width=27.099105000000005pt height=22.46574pt/> respectively.

Contact matrices for every infection class, for every node and commuter node, can be specified using the [`node_populations.csv`](#nodepopulationscsv) and [`cnode_populations.csv`](#cnodepopulationscsv) configuration files.

**Note:** Do not add unused contact matrices to the configuration file,
as this will affect performance negatively.

## `node_cmatrices.csv`

The contact matrices to be used at each node is set
using the `node_cmatrices.csv` configuration file. Each row assigns a
contact matrix, as defined in [`contact_matrices.json`](#contactmatricesjson), to a specific node.

**Example:** Contact matrices at each node for the SEAIR model

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>Location</th>
            <th>S</th>
            <th>E</th>
            <th>A</th>
            <th>I</th>
            <th>R</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ALL</td>
            <td>ALL</td>
            <td></td>
            <td></td>
            <td>C_awayA</td>
            <td>C_awayI</td>
            <td></td>
        </tr>
        <tr>
            <td>ALL</td>
            <td>HOME</td>
            <td></td>
            <td></td>
            <td>C_homeA</td>
            <td>C_homeI</td>
            <td></td>
        </tr>
        <tr>
            <td>3</td>
            <td>ALL</td>
            <td></td>
            <td></td>
            <td>C_away3A</td>
            <td>C_away3I</td>
            <td></td>
        </tr>
        <tr>
            <td>3</td>
            <td>3</td>
            <td></td>
            <td></td>
            <td>C_home3A</td>
            <td>C_home3I</td>
            <td></td>
        </tr>
    </tbody>
</table>

The `Home` and `Location` columns corresponds to <img src="svgs/e8873e227619b7a62ee7eb981ef1faea.svg?invert_in_darkmode" align=middle width=33.46497000000001pt height=24.65759999999998pt/> respectively, and the subsequent columns give the specific contact matrix to be used for each
infection class. Note that there is a column for each epidemic class SEAIR, and
not just for the infection classes A and I. This is necessary, albeit
superfluous, and will be changed in future updates.

Keywords:
- The keyword `ALL` matches all values of <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/> and <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710483000000004pt height=21.683310000000006pt/>.
- The keyword `HOME` copies the value of the Home column.

Each row is read sequentially, meaning that although the first row in
the example sets the parameters for all nodes, the subsequent nodes
overwrite these values for specific nodes.

In the example above we have used the `ALL` keyword to first set default
contact matrices `C_awayA` and `C_awayI` for all nodes. In the second row we use the `HOME` keyword to set contact matrices `C_homeA` and `C_homeI` for people at home. Finally, in the third row we specifically set the contact matrix of that residents of location 3.

## `cnode_cmatrices.csv`

The contact matrices to be used at each commuter node is set
using the `cnode_cmatrices.csv` configuration file. Each row assigns a
contact matrix, as defined in [`contact_matrices.json`](#contactmatricesjson), to a specific node.

As `cnode_cmatrices.csv` works in a similar way to [`node_populations.csv`](#nodepopulationscsv), see the latter for a more 
detailed description.

**Example:** Contact matrices at each commuter node for the SEAIR model

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>From</th>
            <th>To</th>
            <th>S</th>
            <th>E</th>
            <th>A</th>
            <th>I</th>
            <th>R</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ALL</td>
            <td>ALL</td>
            <td>ALL</td>
            <td></td>
            <td></td>
            <td>C_transport</td>
            <td>C_transport</td>
            <td></td>
        </tr>
        <tr>
            <td>ALL</td>
            <td>HOME</td>
            <td>ALL</td>
            <td></td>
            <td></td>
            <td>C_goingToWork</td>
            <td>C_goingToWork</td>
            <td></td>
        </tr>
        <tr>
            <td>ALL</td>
            <td>ALL</td>
            <td>HOME</td>
            <td></td>
            <td></td>
            <td>C_goingHome</td>
            <td>C_goingHome</td>
            <td></td>
        </tr>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>3</td>
            <td></td>
            <td></td>
            <td>C_goingToWork1</td>
            <td>C_goingToWork1</td>
            <td></td>
        </tr>
    </tbody>
</table>

## `node_populations.csv`

Each node <img src="svgs/1b711759d62c78b9d4b4a2c031f451b4.svg?invert_in_darkmode" align=middle width=51.34734000000001pt height=24.65759999999998pt/> is populated using the `node_populations.csv`
configuration file. The structure of this file is straight-forward, and
is best explained by looking at the example below.

**Example:** SIR populations at nodes

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>Location</th>
            <th>S0</th>
            <th>I0</th>
            <th>R0</th>
            <th>S1</th>
            <th>I2</th>
            <th>R2</th>
            <th>S3</th>
            <th>I3</th>
            <th>R3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>100</td>
            <td>1</td>
            <td>0</td>
            <td>0</td>
            <td>200</td>
            <td>0</td>
            <td>50</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>200</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>300</td>
            <td>0</td>
            <td>100</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>20</td>
            <td>0</td>
        </tr>
    </tbody>
</table>

The `Home` and `Location` columns corresponds to <img src="svgs/e8873e227619b7a62ee7eb981ef1faea.svg?invert_in_darkmode" align=middle width=33.46497000000001pt height=24.65759999999998pt/> respectively.
If <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/> is the number of epidemic classes, then the <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/> subsequent columns
are the populations of each epidemic class for age group 0. The next <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/>
are the populations for age group 1, and so on.

The labels for the populations of each epidemic class is cosmetic, and
serve no function other than to improve legibility. They could be renamed
`S(child)` and `S(adult)` and so on.

Consider the example above. In the first row we have seeded the nodes
<img src="svgs/9abbb6714b8c02331691b394070d84b3.svg?invert_in_darkmode" align=middle width=48.99015pt height=24.65759999999998pt/>, <img src="svgs/8fed2e01cacc57c032e549c9bf81b905.svg?invert_in_darkmode" align=middle width=48.99015pt height=24.65759999999998pt/> and <img src="svgs/0f1b91f67a29bbd7b291d3e9c21e0c15.svg?invert_in_darkmode" align=middle width=48.99015pt height=24.65759999999998pt/> with 200, 300 and a 100 susceptibles
respectively, and we have put one infected in <img src="svgs/9abbb6714b8c02331691b394070d84b3.svg?invert_in_darkmode" align=middle width=48.99015pt height=24.65759999999998pt/>. In other words

<p align="center"><img src="svgs/23f17a0483f7944e4a1ff857212da905.svg?invert_in_darkmode" align=middle width=91.586715pt height=98.40039pt/></p>

In the third row, we have put 20 infected residents of node 0 into 
location 1, so

<p align="center"><img src="svgs/a9a0564344a970e7e31b5f5110c56bb3.svg?invert_in_darkmode" align=middle width=85.07994pt height=18.312359999999998pt/></p>

Note that in most cases we would not have rows like the third row in the
example above. The feature of allowing initial populations to be
displaced from their homes is included for completeness rather than for
utility.

## `cnode_populations.csv`

In most cases we would want the initial populations of each commuter node
to be 0, but the feature of allowing non-zero initial populations is
included for completeness. The configuration file `cnode_populations.csv`
works very similarly to [`node_populations.csv`](#nodepopulationscsv), so see the latter for more
detailed instructions on how the file works.

**Example:** SIR populations at commuter nodes

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>From</th>
            <th>To</th>
            <th>S0</th>
            <th>I0</th>
            <th>R0</th>
            <th>S1</th>
            <th>I2</th>
            <th>R2</th>
            <th>S3</th>
            <th>I3</th>
            <th>R3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>1</td>
            <td>100</td>
            <td>1</td>
            <td>0</td>
            <td>0</td>
            <td>200</td>
            <td>0</td>
            <td>50</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>0</td>
            <td>1</td>
            <td>0</td>
            <td>200</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>300</td>
            <td>0</td>
            <td>0</td>
            <td>100</td>
            <td>0</td>
        </tr>
    </tbody>
</table>

## `commuter_networks.csv`

The commuting network is defined using the `commuter_networks.csv`
configuration file. Each row in the file defines a commute between
two nodes, occuring at a given time.

**Example:** SEIR commuter network

<table>
    <thead>
        <tr>
            <th>Home</th>
            <th>From</th>
            <th>To</th>
            <th>Age</th>
            <th># to move</th>
            <th>% to move</th>
            <th>t1</th>
            <th>t2</th>
            <th>ct1</th>
            <th>ct2</th>
            <th>Allow S</th>
            <th>Allow E</th>
            <th>Allow I</th>
            <th>Allow R</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>1</td>
            <td>0</td>
            <td>100</td>
            <td>-1</td>
            <td>7</td>
            <td>8</td>
            <td>8</td>
            <td>9</td>
            <td>1</td>
            <td>1</td>
            <td>0</td>
            <td>1</td>
        </tr>
        <tr>
            <td>0</td>
            <td>1</td>
            <td>0</td>
            <td>0</td>
            <td>-1</td>
            <td>1.0</td>
            <td>17.5</td>
            <td>18.5</td>
            <td>18.5</td>
            <td>19.5</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
            <td>2</td>
            <td>0</td>
            <td>1</td>
            <td>-1</td>
            <td>0.5</td>
            <td>6</td>
            <td>8</td>
            <td>9</td>
            <td>10</td>
            <td>1</td>
            <td>1</td>
            <td>0</td>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
            <td>2</td>
            <td>0</td>
            <td>1</td>
            <td>-1</td>
            <td>1.0</td>
            <td>19</td>
            <td>20</td>
            <td>21</td>
            <td>22</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
        </tr>
    </tbody>
</table>


The home <img src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/>, origin <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710483000000004pt height=21.683310000000006pt/>, destination <img src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075495000000004pt height=22.831379999999992pt/> and age-group <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.576500000000003pt height=14.155350000000013pt/> are given in the `Home`, 
`From`, `To` and `Age` columns respectively.

The amount of people
that are to be transported is either given as an absolute number, using
`# to move`, or as a percentage using `% to move`. When either of the
two is specified, the other must be set to `-1`.

The departure window is given by `t1` and `t2`. People will be moving
from the origin node to the commuting node between these two times.
The transport is modeled using a Gaussian pulse function (see the [model description](model.pdf)
for more details).

After leaving the origin node <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710483000000004pt height=21.683310000000006pt/>, people are then moved to the commuter
node <img src="svgs/eca87f87ecdf13e2dcc9d71ffb2287e5.svg?invert_in_darkmode" align=middle width=85.99338pt height=24.65759999999998pt/>. The departure window for movement from the 
commuter node to the destination node <img src="svgs/22f7d56420ef6a94b85737024596aff7.svg?invert_in_darkmode" align=middle width=52.712385pt height=24.65759999999998pt/> is given by
`ct1` and `ct2`.

Restriction on what classes can commute can be set using the `Allow O` column,
where `O` stands for a given epidemiological class. If there are <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/> different
epidemiological classes, then there should be <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.867000000000003pt height=14.155350000000013pt/> columns `Allow O1`, `Allow O2`,
..., `Allow On` in the configuration file.

In the example above, the first row sets that 100% of the residents of 0,
who are currently at 0, and are of age-group 0, should leave for location 1, between 7 and 8 o'clock.
They will then depart from the commuter node to the destination 1 between 8 and 9 o'clock.
The row also specifies that all classes except the infecteds <img src="svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.515980000000004pt height=22.46574pt/> are allowed to move.

The second row specifies the return from location 1 back home. Note that
we allow movement of all classes, including the infected class. This is
because we would want all classes to return back home, regardless of
whether they have been infected during during their time at work or not.

Similarly, the third and fourth row describes the commute of the population
of location 2. In this case, we have specified the forward movement using
percentages instead of absolute numbers. We have said that 50% of the population
resident at 2, who are at 2, and are of age-group 1, should move to location 0.
                
In general it is advisable to use absolute numbers for departing populations
rather than percentages, as this leads to more predictable results. Returning
populations should always be set to 100%, in order to avoid a migration
between nodes over time.

**Important notes:**

- The times <img src="svgs/4fd7e1029c498a5313f07c76d75fcbb3.svg?invert_in_darkmode" align=middle width=43.059555pt height=24.65759999999998pt/> and <img src="svgs/b156133b4caf917f2a53dfd833a4cf09.svg?invert_in_darkmode" align=middle width=57.287175000000005pt height=24.65759999999998pt/> must *not* overlap. If they do
  this will lead to the situation where populations get stuck in the 
  commuterverse.
- Currently, there is an issue when defining commuter networks where there
  are paths with more than 2 steps. Although it is possible
  to define paths such as <img src="svgs/b1424bbb39a8b6fdc4d3b594f416babe.svg?invert_in_darkmode" align=middle width=109.58871000000002pt height=21.18732pt/>, this should be avoided. The
  reaons for this is that it will lead to migration if there are other
  commuting patterns that share intermediate nodes such as <img src="svgs/4bfb1d8d79ee2249687ab0aca6ad1230.svg?invert_in_darkmode" align=middle width=109.58871000000002pt height=21.18732pt/>.
  This will be rectified in a future iteration of PyRossGeo.