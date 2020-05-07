# PyRossGeo configuration files

## `model.json` (Local infection dynamics)

The infection dynamics at each geographical node is defined using the
`model.json` configuration file. It can define any variant of the
age-bracketed SIR 

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

```json
{
    "classes" : ["S", "I", "R"],

    "S" : {
        "linear"    : [],
        "nonlinear" : [ ["I", "-beta"] ]
    },

    "I" : {
        "linear"    : [ ["I", "-gamma"] ],
        "nonlinear" : [ ["I", "beta"] ]
    },

    "R" : {
        "linear"    : [ ["I", "gamma"] ],
        "nonlinear" : []
    }
}

```