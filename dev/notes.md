## Contact matrices idea

Look at Eq. 2 of https://arxiv.org/pdf/2003.12055.pdf.

The contact matrix of a certain node should be how they *contribute* to the SIR dynamics (not how they experience it). So consider we have
people who live at j, and then also people who live at i but work at j. The lambda factor should then be

C(j,j)\*I_jj + C(i,j)\*I_ij

This way, by stopping people from going to work, we have a natural way of reduing the work contact matrices.

To prevent this from slowing down the code, you can implement that the lambda calculation of a node will only proceed if the infections of
a particular node exceeds 1 (or a smaller number)

## readme-tex

python -m readme2tex --output "Configuration files.md" "Configuration files.tex.md" --username lukastk  --project PyRossGeo --nocdn

## s

- Start simulation with initial conditions from a few weeks ago, with infection rates and contact matrices during normal times (no lockdown.)
- Show then that what has ensued in reality is not as bad as results from simulation.
- Then run again with adjusted contact matrix, to try to predict the evolution of the 

Synthetic data:
- Do the above, with synthetic data.

## Docs stuff

- Add area to configuration docs
- Use $\beta_{ij}^\alpha-..."
- Change "age-structure" to contact-structure
- Introduce terminology
  - Node = n(a,i,j)
  - Commuter node = cn(a,i,j,k)
  - Geographic node (geonode) = gn(a,j)
  - Communities aren't nodes

## Todo (main):

- Make a tool that makes sure that `model.json` adds to 0.

- Stop commuter network on weekends.

- Allow each transport to cycle on a time-scale different from 24h. So for example, 48h.

- Allow one-off transport.

- Allow for constant rates in `model.json`

- Time-dependent contact-matrices. You could have a `cmatrix_schedule.json` where each schedule can be multiplied by a constant factor at different times of the day. There'd be a loop at each time-step checking which factor to use for which contact matrix.
  - We should also be able to schedule what contact matrix to use at what
    time. So for example when to use C_home_and_school and when to just use
    C_home.

- Currently every single contact matrix is comptued at every single node. You should have it so that we know exactly which contact matrices are used at which node, and only compute these.

- In tutorial 3, I set `C_away[0,0] = 70000`, and the resulting simulation blew up.

- In reality, a full lockdown scenario would be the following: Each
family only interacts with itself. Meaning that there is a very small pool
of people that each person can interact with. The C*beta should be set
to account for this.

- When it comes to the geographical model, this is one thing that it can
do that no other can: When we look at very coarse grained network, it's
not overly hard to get nodes where the population of infecteds is quite small such
that stochastic dynamic overtake. In which case, it is possible that the infetion
dies out. This is how the geographic model can model lockdown better than any else.

- Add over save options. Just output network data, for example,
or any of the others.

## Todo:

- Make it so that SIR dynamics cant overstep
   - Basically see if `dt*(dX_state[si+oi]+lambda)<X_state[si+oi]`.
   - I'm leaving this out for now, to see if it's necessary.
- Fix contact matrices
- Delta movement
- SEIR
- Make separate commuterverses for each schedule
  - Optimize tau calculation
- Instead of computing lambda and tau, you should simply apply them directly.
  - So have an F and G calculation section

- Put some checks into the initialisation to help users
  - The number of age_groups is gotten from the contact matrix. `age_groups`
    then has to match with the number of columns in `node_populations_dat`,
    this should be checked, and if they don't match the user should get a warning.

- The network video in https://gitlab.com/camsofties/covid19/-/issues/14#note_328344827
showed that a node in central london was inreasing.

- I could potentially just force all the people to go back home.

- Run simulations for just 20 days to see what happens.

- Simulation runner
  - Creates a folder with the data files
  - Creates py notebook inside
  - Copies the simulation code
  - You can name the simulation
  - Either copy over the commuter networks files etc or note down their name

- Estimate betas for commuting from: https://www.medrxiv.org/content/10.1101/2020.04.04.20053058v1

## Interventions

Interventions:

- In the Blair slides, they note that only complete lockdowns have caused R
 to drop below 1, we can test that.

## Notes

---

I've noticed that some weird things might happen when the infection numbers are
really low in the London network. I started out with I=1, and S=8000 in a node,
and due to the system parameters the I population quickly decayed.  But since
there always remained a small number of I remaining in the node, there was
still a decay in S.

We might need to implement a cutoff in the infection. So if I<1, then it cant infect.

---

Cyclical routes:

I think one issue might be that if we specify 50 people moving from 0 to 1,
and 50 people moving back, any errors that occured in the forward movement will
be propagated when going back. (So let's say only 49 people went to 1, and
then in the back journey there'll be -1 people at 1).

Possible solution:

- At the end of ct2, simply dump the remaining bit of N0 into the destination node. T
might have to store the amount of people its actually moved (so we'll have to pass dt
into geoSIR as well), so it can calculate this properly.

- Whenever subtracting using T, check that dt*T wont make the term negative. If
it is, then simply subtract the remainder and set it to 0. Could do this for lambda and tau as well.


--

Another error in cyclical routes:

Let's say we have two sets of people living in i, who are moving around the network
in a cyclical route. Set 1 goes 1->2->3->1 and set 2 goes 1->3->1. Let's also
say that their journeys from 3 to 1 coincide, and let's say 1 started their 3->1
journey before 2 did. Then this will cause a bug. When 2 starts their journey,
the commuting algorithm will note down the current population of the commuterverse
and aim to empty it by the end of the commuting window. Inevitably, the commuterverse
population will become negative.

The only way to fix this is by having separate commuterverses for each schedule.

I wont implement this right now, but I will in the future.

For now, i'll assume that we can't have coinciding commuting windows in transport.
So the scenario described above is not allowed.

The plus side of this is that now we can empty the commuterverses perfectly.
By ct2, we can simply empty out the contents of the commuterverse.

Having separate commuterverses for each commute will increase the complexity of the
simulation. On the other hand, this will be mitigated by the fact that we more easily
know when a commuterverse needs to be simulated now. In the tau matrix calculation,
we can skip a calculation if the T is not active.

---

Right now, the return from a node is modelled by having that 100% of people at
the node return. This is an issue if two different sets of people return to
to i from j. We could have a system of this sort:

Instead of individual Ts, we would have *paths*. The initial departure (movement
away from i) can be given in terms of percentage or move_N from i, but then after
that the path will have kept track of how many people actually left, and will
make sure that the right amount of people move from place to place along the
path.

---

Right now I'm prohibiting infection dynamics if the population is smaller than 0.

```python
if _Ns[age_b] > 1: # No infections can occur if there are fewer than one person at node
    _lambdas[age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]
```

This is not particularly disruptive, as it only affects nodes where there aren't any people in anyway.
We could potentially modify it to

```python
if _Is[age_b] > 1: # No infections can occur if there are fewer than one person at node
    _lambdas[age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]
```

and so only allowing infections if the infected class has more than 1. This would be modelling decision
rather than a fix to a numerical error.

It could, for example, affect SEIR dynamics to a large degree. If we don't allow I smaller than 1 to infect,
then the infection might never spread, as there might not be enough E converting to I during a work day.