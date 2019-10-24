# Kokkos Tuning

This is a design document describing the motivation, ideas, design, and prototype implementation of the Kokkos Tuning System

## Motivation

Currently, Kokkos makes a lot of decisions about tuning parameters (CUDA block sizes, different kernel implementations) 
by picking an option that results in the best performance for the widest array of applications and architectures at the 
time the choice is made. This approach leaves performance on the table, and appears increasingly untenable as the number 
of architectures and applications grows.

The Kokkos team would like to instead open up the ability to set the parameters as part of the tooling system so that
these parameters can be tuned for individual applications across all the architectures they might run on. In order to match the
feel of past Kokkos tooling efforts, we'd like to achieve this with a callback system.

## Ideas

A Kokkos Tuning system should be as small as is wise while achieving the following goals

1. Expose to tools enough data about the _context_ of the running application to tune intelligently
2. Expose to tools enough data about tuning parameters that they might know how to optimize what they're asked to
3. Expose to applications an interface that they might inform a tool about their current application context
4. Expose to tools the results of their choices
5. No perturbation of Kokkos Core when this system is disabled

Shared among the first three of these goals is a need for some way to describe the semantics of variables (tuning parameters, context variables)
internal to Kokkos or an application to an outside tool. 

### Semantics of Variables

I think it's best to talk about the semantics of variables with concrete examples.

Suppose Kokkos wants a tool to choose a block size for it. Suppose all the application context is perfectly understood, that the tool knows
that the application has 10,000,000 particles active and that it's running a kernel called "make_particles_go," which is a parallel_for in
the "cuda" execution space. The tool needs to know several things about what a block size _is_ for this to be generic and practical

1. Is it an integer value? A float? A string? (Type)
2. Relatedly, what are the mathematical semantics which are valid for it? Is it something 
for which a list can be sorted? Do the distances between items in a sorted list make sense?
If I divide two values, does the ratio have some meaning? (semantics)
3. What are the valid choices for this value? Is a block size of -128 okay? How about 7? (candidates)

Semantics (as always) are likely the source of the most confusion here, so a bit of detail is good. Here I'm leaning heavily on the field
of statistics to enable tools to do intelligent searching. If ordering doesn't make sense, if a value is "categorical", the only thing
a tool can do is try all possible values for a tuning value. If they're ordered (ordinal), the search can take advantage of this by 
using the concept of a directional search. If the distances between elements matter (interval data) you can cheat with things like
bisection. Finally if ratios matter you can play games where you increase by a factor of 10 in your searches.

Claim: the combination of context, candidates, semantics, and types gives a tool enough to intelligently explore the search space of 
tuning parameters

### Context

Suppose a tool perfectly understands what a block size is. To effectively tune one, it needs to know something about the application.

In a trivial case, the tool knows absolutely nothing other than candidate values, and tries to make a choice that optimizes across all
invocations of kernels. This isn't _that_ far from what Kokkos does now, so it's not unreasonable for this to produce decent results. 
That said, we could quickly add some context from Kokkos
